#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

#ifdef PCLPRECESS_USE_OPENMP
#include <omp.h>
#endif

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using Point2T = pcl::PointXY;
using Cloud2T = pcl::PointCloud<Point2T>;
using ColorPointT = pcl::PointXYZRGB;
using ColorCloudT = pcl::PointCloud<ColorPointT>;

constexpr double kPi = 3.14159265358979323846;

struct Options {
    // Edit these defaults to configure the program.
    std::string input_path = "../data/sample_plane_7x7.pcd";
    int k_neighbors = 30;
    double angle_threshold_deg = 120.0;
    double voxel_leaf = 0.0;
    int sor_mean_k = 30;
    double sor_stddev = 1.0;
    double radius = 0.0;
    int radius_min_neighbors = 3;
    double edge_radius = 0.0;
    double edge_radius_factor = 3.0;
    int edge_min_neighbors = 3;
    double cluster_tolerance = 0.0;
    int min_cluster_size = 5;
    int max_cluster_size = std::numeric_limits<int>::max();
    int normal_k = 36;
    double normal_angle_deg = 35.0;
    double region_radius = 0.0;
    double region_radius_factor = 4.0;
    double plane_distance_factor = 2.0;
    bool use_boundary_barrier = false;
    double boundary_block_factor = 0.0;
    int min_region_points = 10;
    double trajectory_spacing = 0.0;
    double trajectory_spacing_factor = 8.0;
    double trajectory_support_radius = 0.0;
    double trajectory_support_factor = 5.0;
    double spray_offset = 0.0;
    bool exterior_only = true;
    double exterior_tolerance = 0.0;
    double exterior_tolerance_factor = 10.0;
    int threads = 0;
    int max_visualized_trajectory_lines = 0;
    bool visualize = false;
    std::string out_boundary = "boundary_points.pcd";
    std::string out_colored = "boundary_colored.pcd";
    std::string out_segmented = "boundary_segmented.pcd";
    std::string out_trajectory = "spray_trajectory.pcd";
    std::string out_trajectory_csv = "spray_trajectory.csv";
    std::string out_preprocessed = "preprocessed_cloud.pcd";
};

struct SurfaceRegion {
    std::vector<int> indices;
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
};

struct SprayPathPoint {
    PointT point;
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    int region_id = -1;
    int line_id = -1;
    int segment_id = -1;
};

#ifdef _WIN32
std::string chooseInputCloudFile() {
    char file_name[MAX_PATH] = {};
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = "Point cloud files (*.pcd;*.ply)\0*.pcd;*.ply\0PCD files (*.pcd)\0*.pcd\0PLY files (*.ply)\0*.ply\0All files (*.*)\0*.*\0";
    ofn.lpstrFile = file_name;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    ofn.lpstrTitle = "Select input point cloud";
    if (GetOpenFileNameA(&ofn) == TRUE) {
        return std::string(file_name);
    }
    return {};
}
#else
std::string chooseInputCloudFile() {
    return {};
}
#endif

Options makeOptions() {
    Options options;
    const std::string selected_path = chooseInputCloudFile();
    if (!selected_path.empty()) {
        options.input_path = selected_path;
        options.visualize = true;
    }
    if (options.input_path.empty()) {
        throw std::runtime_error("Options::input_path must be set");
    }
    if (options.k_neighbors < 3) {
        throw std::runtime_error("Options::k_neighbors must be at least 3");
    }
    if (options.angle_threshold_deg <= 0.0 || options.angle_threshold_deg >= 360.0) {
        throw std::runtime_error("Options::angle_threshold_deg must be in (0, 360)");
    }
    if (options.voxel_leaf < 0.0) {
        throw std::runtime_error("Options::voxel_leaf must be non-negative");
    }
    if (options.sor_mean_k < 0) {
        throw std::runtime_error("Options::sor_mean_k must be non-negative");
    }
    if (options.sor_stddev <= 0.0) {
        throw std::runtime_error("Options::sor_stddev must be positive");
    }
    if (options.radius < 0.0) {
        throw std::runtime_error("Options::radius must be non-negative");
    }
    if (options.radius_min_neighbors < 1) {
        throw std::runtime_error("Options::radius_min_neighbors must be positive");
    }
    if (options.edge_radius < 0.0) {
        throw std::runtime_error("Options::edge_radius must be non-negative");
    }
    if (options.edge_radius_factor <= 0.0) {
        throw std::runtime_error("Options::edge_radius_factor must be positive");
    }
    if (options.edge_min_neighbors < 1) {
        throw std::runtime_error("Options::edge_min_neighbors must be positive");
    }
    if (options.cluster_tolerance < 0.0) {
        throw std::runtime_error("Options::cluster_tolerance must be non-negative");
    }
    if (options.min_cluster_size < 1) {
        throw std::runtime_error("Options::min_cluster_size must be positive");
    }
    if (options.max_cluster_size < options.min_cluster_size) {
        throw std::runtime_error("Options::max_cluster_size must be greater than or equal to Options::min_cluster_size");
    }
    if (options.normal_k < 6) {
        throw std::runtime_error("Options::normal_k must be at least 6");
    }
    if (options.normal_angle_deg <= 0.0 || options.normal_angle_deg >= 90.0) {
        throw std::runtime_error("Options::normal_angle_deg must be in (0, 90)");
    }
    if (options.region_radius < 0.0) {
        throw std::runtime_error("Options::region_radius must be non-negative");
    }
    if (options.region_radius_factor <= 0.0) {
        throw std::runtime_error("Options::region_radius_factor must be positive");
    }
    if (options.plane_distance_factor <= 0.0) {
        throw std::runtime_error("Options::plane_distance_factor must be positive");
    }
    if (options.boundary_block_factor < 0.0) {
        throw std::runtime_error("Options::boundary_block_factor must be non-negative");
    }
    if (options.min_region_points < 1) {
        throw std::runtime_error("Options::min_region_points must be positive");
    }
    if (options.trajectory_spacing < 0.0) {
        throw std::runtime_error("Options::trajectory_spacing must be non-negative");
    }
    if (options.trajectory_spacing_factor <= 0.0) {
        throw std::runtime_error("Options::trajectory_spacing_factor must be positive");
    }
    if (options.trajectory_support_radius < 0.0) {
        throw std::runtime_error("Options::trajectory_support_radius must be non-negative");
    }
    if (options.trajectory_support_factor <= 0.0) {
        throw std::runtime_error("Options::trajectory_support_factor must be positive");
    }
    if (options.exterior_tolerance < 0.0) {
        throw std::runtime_error("Options::exterior_tolerance must be non-negative");
    }
    if (options.exterior_tolerance_factor <= 0.0) {
        throw std::runtime_error("Options::exterior_tolerance_factor must be positive");
    }
    if (options.threads < 0) {
        throw std::runtime_error("Options::threads must be non-negative");
    }
    if (options.max_visualized_trajectory_lines < 0) {
        throw std::runtime_error("Options::max_visualized_trajectory_lines must be non-negative");
    }

    const std::filesystem::path input_parent = std::filesystem::absolute(options.input_path).parent_path();
    auto normalizeOutputPath = [&](const std::string& path) -> std::string {
        const std::filesystem::path fs_path(path);
        if (fs_path.is_absolute()) {
            return fs_path.string();
        }
        return (input_parent / fs_path).string();
    };

    options.out_boundary = normalizeOutputPath(options.out_boundary);
    options.out_colored = normalizeOutputPath(options.out_colored);
    options.out_segmented = normalizeOutputPath(options.out_segmented);
    options.out_trajectory = normalizeOutputPath(options.out_trajectory);
    options.out_trajectory_csv = normalizeOutputPath(options.out_trajectory_csv);
    options.out_preprocessed = normalizeOutputPath(options.out_preprocessed);
    return options;
}

CloudT::Ptr loadCloud(const std::string& path) {
    auto cloud = std::make_shared<CloudT>();
    const std::filesystem::path fs_path(path);
    const std::string ext = fs_path.extension().string();

    int result = -1;
    if (ext == ".pcd" || ext == ".PCD") {
        result = pcl::io::loadPCDFile<PointT>(path, *cloud);
    } else if (ext == ".ply" || ext == ".PLY") {
        result = pcl::io::loadPLYFile<PointT>(path, *cloud);
    } else {
        throw std::runtime_error("Only .pcd and .ply files are supported: " + path);
    }

    if (result < 0 || cloud->empty()) {
        throw std::runtime_error("Failed to read a non-empty point cloud: " + path);
    }
    cloud->is_dense = false;
    return cloud;
}

CloudT::Ptr preprocessCloud(const CloudT::Ptr& input, const Options& options) {
    auto finite_cloud = std::make_shared<CloudT>();
    std::vector<int> finite_indices;
    pcl::removeNaNFromPointCloud(*input, *finite_cloud, finite_indices);
    finite_cloud->is_dense = false;

    CloudT::Ptr current = finite_cloud;

    if (options.voxel_leaf > 0.0) {
        auto voxel_cloud = std::make_shared<CloudT>();
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(current);
        const float leaf = static_cast<float>(options.voxel_leaf);
        voxel.setLeafSize(leaf, leaf, leaf);
        voxel.filter(*voxel_cloud);
        voxel_cloud->is_dense = false;
        current = voxel_cloud;
    }

    if (options.sor_mean_k > 0 && static_cast<int>(current->size()) > options.sor_mean_k) {
        auto sor_cloud = std::make_shared<CloudT>();
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(current);
        sor.setMeanK(options.sor_mean_k);
        sor.setStddevMulThresh(options.sor_stddev);
        sor.filter(*sor_cloud);
        sor_cloud->is_dense = false;
        current = sor_cloud;
    }

    if (options.radius > 0.0) {
        auto radius_cloud = std::make_shared<CloudT>();
        pcl::RadiusOutlierRemoval<PointT> radius_filter;
        radius_filter.setInputCloud(current);
        radius_filter.setRadiusSearch(options.radius);
        radius_filter.setMinNeighborsInRadius(options.radius_min_neighbors);
        radius_filter.filter(*radius_cloud);
        radius_cloud->is_dense = false;
        current = radius_cloud;
    }

    return current;
}

Eigen::Vector3f pointVector(const PointT& point) {
    return Eigen::Vector3f(point.x, point.y, point.z);
}

void configureParallelism(const Options& options) {
#ifdef PCLPRECESS_USE_OPENMP
    if (options.threads > 0) {
        omp_set_num_threads(options.threads);
    }
    std::cout << "OpenMP enabled, max threads: " << omp_get_max_threads() << '\n';
#else
    (void)options;
    std::cout << "OpenMP not enabled; running single-threaded hot loops\n";
#endif
}

bool estimateTangentFrame(
    const CloudT& cloud,
    const std::vector<int>& neighbor_indices,
    Eigen::Vector3f* normal,
    Eigen::Vector3f* u_axis,
    Eigen::Vector3f* v_axis) {
    if (neighbor_indices.size() < 3) {
        return false;
    }

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    int valid_count = 0;
    for (const int index : neighbor_indices) {
        const PointT& p = cloud[index];
        if (!pcl::isFinite(p)) {
            continue;
        }
        centroid += pointVector(p);
        ++valid_count;
    }
    if (valid_count < 3) {
        return false;
    }
    centroid /= static_cast<float>(valid_count);

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const int index : neighbor_indices) {
        const PointT& p = cloud[index];
        if (!pcl::isFinite(p)) {
            continue;
        }
        const Eigen::Vector3f centered = pointVector(p) - centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= static_cast<float>(valid_count);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    *normal = solver.eigenvectors().col(0).normalized();
    *u_axis = solver.eigenvectors().col(2).normalized();
    *v_axis = normal->cross(*u_axis).normalized();

    if (!normal->allFinite() || !u_axis->allFinite() || !v_axis->allFinite()) {
        return false;
    }
    return true;
}

double maxAngularGapOnTangentPlane(
    const CloudT& cloud,
    int query_index,
    const std::vector<int>& neighbor_indices,
    const Eigen::Vector3f& u_axis,
    const Eigen::Vector3f& v_axis) {
    std::vector<double> angles;
    angles.reserve(neighbor_indices.size());

    const Eigen::Vector3f query = pointVector(cloud[query_index]);
    for (const int neighbor_index : neighbor_indices) {
        if (neighbor_index == query_index) {
            continue;
        }
        const PointT& neighbor = cloud[neighbor_index];
        if (!pcl::isFinite(neighbor)) {
            continue;
        }

        const Eigen::Vector3f direction = pointVector(neighbor) - query;
        const float x = direction.dot(u_axis);
        const float y = direction.dot(v_axis);
        if (std::abs(x) < 1.0e-8F && std::abs(y) < 1.0e-8F) {
            continue;
        }

        double angle = std::atan2(static_cast<double>(y), static_cast<double>(x));
        if (angle < 0.0) {
            angle += 2.0 * kPi;
        }
        angles.push_back(angle);
    }

    if (angles.size() < 2) {
        return 2.0 * kPi;
    }

    std::sort(angles.begin(), angles.end());

    double max_gap = 0.0;
    for (size_t i = 1; i < angles.size(); ++i) {
        max_gap = std::max(max_gap, angles[i] - angles[i - 1]);
    }
    max_gap = std::max(max_gap, angles.front() + 2.0 * kPi - angles.back());
    return max_gap;
}

std::vector<int> detectBoundaryIndices(const CloudT::Ptr& cloud, const Options& options) {
    const int search_k = std::min<int>(options.k_neighbors + 1, static_cast<int>(cloud->size()));
    const int point_count = static_cast<int>(cloud->size());
    const double threshold_rad = options.angle_threshold_deg * kPi / 180.0;
    std::vector<std::vector<int>> thread_results;

#ifdef PCLPRECESS_USE_OPENMP
    const int thread_count = std::max(1, omp_get_max_threads());
#else
    const int thread_count = 1;
#endif
    thread_results.resize(static_cast<size_t>(thread_count));

#ifdef PCLPRECESS_USE_OPENMP
#pragma omp parallel
#endif
    {
#ifdef PCLPRECESS_USE_OPENMP
        const int thread_id = omp_get_thread_num();
#else
        const int thread_id = 0;
#endif
        pcl::KdTreeFLANN<PointT> tree;
        tree.setInputCloud(cloud);
        std::vector<int> neighbor_indices(search_k);
        std::vector<float> neighbor_distances(search_k);
        auto& local_boundary = thread_results[static_cast<size_t>(thread_id)];
        local_boundary.reserve(cloud->size() / static_cast<size_t>(thread_count * 10 + 1));

#ifdef PCLPRECESS_USE_OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
    for (int i_signed = 0; i_signed < point_count; ++i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        const PointT& query = (*cloud)[i];
        if (!pcl::isFinite(query)) {
            continue;
        }

        const int found = tree.nearestKSearch(query, search_k, neighbor_indices, neighbor_distances);
        if (found < 4) {
            continue;
        }
        neighbor_indices.resize(found);

        Eigen::Vector3f normal;
        Eigen::Vector3f u_axis;
        Eigen::Vector3f v_axis;
        if (!estimateTangentFrame(*cloud, neighbor_indices, &normal, &u_axis, &v_axis)) {
            neighbor_indices.resize(search_k);
            continue;
        }

        const double max_gap = maxAngularGapOnTangentPlane(
            *cloud,
            static_cast<int>(i),
            neighbor_indices,
            u_axis,
            v_axis);

        if (max_gap > threshold_rad) {
            local_boundary.push_back(static_cast<int>(i));
        }
        neighbor_indices.resize(search_k);
    }
    }

    std::vector<int> boundary_indices;
    size_t total = 0;
    for (const auto& local : thread_results) {
        total += local.size();
    }
    boundary_indices.reserve(total);
    for (const auto& local : thread_results) {
        boundary_indices.insert(boundary_indices.end(), local.begin(), local.end());
    }
    std::sort(boundary_indices.begin(), boundary_indices.end());
    return boundary_indices;
}

double estimateBoundarySpacing(const CloudT::Ptr& cloud, const std::vector<int>& boundary_indices) {
    if (boundary_indices.size() < 3) {
        return 0.0;
    }

    auto boundary_cloud = std::make_shared<CloudT>();
    boundary_cloud->reserve(boundary_indices.size());
    for (const int index : boundary_indices) {
        if (index >= 0 && static_cast<size_t>(index) < cloud->size()) {
            boundary_cloud->push_back((*cloud)[index]);
        }
    }
    if (boundary_cloud->size() < 3) {
        return 0.0;
    }

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(boundary_cloud);

    std::vector<double> nearest_distances;
    nearest_distances.reserve(boundary_cloud->size());
    std::vector<int> ids(2);
    std::vector<float> distances(2);

    for (size_t i = 0; i < boundary_cloud->size(); ++i) {
        ids.assign(2, 0);
        distances.assign(2, 0.0F);
        if (tree.nearestKSearch((*boundary_cloud)[i], 2, ids, distances) >= 2 && distances[1] > 1.0e-12F) {
            nearest_distances.push_back(std::sqrt(static_cast<double>(distances[1])));
        }
    }

    if (nearest_distances.empty()) {
        return 0.0;
    }

    std::sort(nearest_distances.begin(), nearest_distances.end());
    return nearest_distances[nearest_distances.size() / 2];
}

double estimateCloudSpacing(const CloudT::Ptr& cloud) {
    if (cloud->size() < 3) {
        return 0.0;
    }

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);

    std::vector<double> nearest_distances;
    nearest_distances.reserve(std::min<size_t>(cloud->size(), 2000));
    std::vector<int> ids(2);
    std::vector<float> distances(2);
    const size_t stride = std::max<size_t>(1, cloud->size() / 2000);

    for (size_t i = 0; i < cloud->size(); i += stride) {
        ids.assign(2, 0);
        distances.assign(2, 0.0F);
        if (tree.nearestKSearch((*cloud)[i], 2, ids, distances) >= 2 && distances[1] > 1.0e-12F) {
            nearest_distances.push_back(std::sqrt(static_cast<double>(distances[1])));
        }
    }

    if (nearest_distances.empty()) {
        return 0.0;
    }

    std::sort(nearest_distances.begin(), nearest_distances.end());
    return nearest_distances[nearest_distances.size() / 2];
}

std::vector<int> removeIsolatedBoundaryIndices(
    const CloudT::Ptr& cloud,
    const std::vector<int>& boundary_indices,
    const Options& options) {
    if (boundary_indices.size() < static_cast<size_t>(options.edge_min_neighbors + 1)) {
        return boundary_indices;
    }

    auto boundary_cloud = std::make_shared<CloudT>();
    boundary_cloud->reserve(boundary_indices.size());
    std::vector<int> original_indices;
    original_indices.reserve(boundary_indices.size());

    for (const int index : boundary_indices) {
        if (index >= 0 && static_cast<size_t>(index) < cloud->size()) {
            boundary_cloud->push_back((*cloud)[index]);
            original_indices.push_back(index);
        }
    }
    if (boundary_cloud->empty()) {
        return {};
    }

    const double spacing = estimateBoundarySpacing(cloud, original_indices);
    double radius = options.edge_radius;
    if (radius <= 0.0) {
        radius = spacing > 0.0 ? spacing * options.edge_radius_factor : 1.0;
    }
    if (radius <= 0.0) {
        return original_indices;
    }

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(boundary_cloud);

    std::vector<int> filtered_indices;
    filtered_indices.reserve(original_indices.size());
    std::vector<int> neighbors;
    std::vector<float> distances;

    for (size_t i = 0; i < boundary_cloud->size(); ++i) {
        neighbors.clear();
        distances.clear();
        const int found = tree.radiusSearch((*boundary_cloud)[i], radius, neighbors, distances);
        if (found >= options.edge_min_neighbors + 1) {
            filtered_indices.push_back(original_indices[i]);
        }
    }

    std::cout << "Boundary outlier filter:\n"
              << "  spacing: " << spacing << '\n'
              << "  radius: " << radius << '\n'
              << "  min boundary neighbors: " << options.edge_min_neighbors << '\n'
              << "  removed isolated points: " << (original_indices.size() - filtered_indices.size()) << '\n';

    if (filtered_indices.empty() && !original_indices.empty()) {
        std::cout << "  warning: filter removed all boundary points, keeping original result\n";
        return original_indices;
    }
    return filtered_indices;
}

std::vector<Eigen::Vector3f> estimatePointNormals(const CloudT::Ptr& cloud, const Options& options) {
    std::vector<Eigen::Vector3f> normals(cloud->size(), Eigen::Vector3f::Zero());
    if (cloud->size() < 6) {
        return normals;
    }

    Eigen::Vector3f cloud_center = Eigen::Vector3f::Zero();
    for (const auto& point : *cloud) {
        cloud_center += pointVector(point);
    }
    cloud_center /= static_cast<float>(cloud->size());

    const int search_k = std::min<int>(options.normal_k, static_cast<int>(cloud->size()));
    const int point_count = static_cast<int>(cloud->size());

#ifdef PCLPRECESS_USE_OPENMP
#pragma omp parallel
#endif
    {
        pcl::KdTreeFLANN<PointT> tree;
        tree.setInputCloud(cloud);
        std::vector<int> ids(search_k);
        std::vector<float> distances(search_k);

#ifdef PCLPRECESS_USE_OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
    for (int i_signed = 0; i_signed < point_count; ++i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        ids.assign(search_k, 0);
        distances.assign(search_k, 0.0F);
        const int found = tree.nearestKSearch((*cloud)[i], search_k, ids, distances);
        if (found < 6) {
            continue;
        }
        ids.resize(found);

        Eigen::Vector3f normal;
        Eigen::Vector3f u_axis;
        Eigen::Vector3f v_axis;
        if (estimateTangentFrame(*cloud, ids, &normal, &u_axis, &v_axis)) {
            const Eigen::Vector3f outward = pointVector((*cloud)[i]) - cloud_center;
            if (outward.dot(normal) < 0.0F) {
                normal = -normal;
            }
            normals[i] = normal;
        }
        ids.resize(search_k);
    }
    }

    return normals;
}

std::vector<std::uint8_t> buildBoundaryBarrierMask(
    const CloudT::Ptr& cloud,
    const std::vector<int>& boundary_indices,
    double radius) {
    std::vector<std::uint8_t> barrier(cloud->size(), 0);
    if (boundary_indices.empty() || radius <= 0.0) {
        for (const int index : boundary_indices) {
            if (index >= 0 && static_cast<size_t>(index) < barrier.size()) {
                barrier[static_cast<size_t>(index)] = 1;
            }
        }
        return barrier;
    }

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);
    std::vector<int> neighbors;
    std::vector<float> distances;

    for (const int index : boundary_indices) {
        if (index < 0 || static_cast<size_t>(index) >= cloud->size()) {
            continue;
        }
        neighbors.clear();
        distances.clear();
        tree.radiusSearch((*cloud)[static_cast<size_t>(index)], radius, neighbors, distances);
        for (const int neighbor : neighbors) {
            if (neighbor >= 0 && static_cast<size_t>(neighbor) < barrier.size()) {
                barrier[static_cast<size_t>(neighbor)] = 1;
            }
        }
    }
    return barrier;
}

std::vector<SurfaceRegion> segmentSurfaceRegions(
    const CloudT::Ptr& cloud,
    const std::vector<Eigen::Vector3f>& normals,
    const std::vector<int>& boundary_indices,
    const Options& options,
    std::vector<int>* point_region_ids) {
    point_region_ids->assign(cloud->size(), -1);
    std::vector<SurfaceRegion> regions;
    if (cloud->empty()) {
        return regions;
    }

    const double spacing = estimateCloudSpacing(cloud);
    double grow_radius = options.region_radius;
    if (grow_radius <= 0.0) {
        grow_radius = spacing > 0.0 ? spacing * options.region_radius_factor : 1.0;
    }
    const double barrier_radius = options.use_boundary_barrier
        ? (spacing > 0.0 ? spacing : grow_radius) * options.boundary_block_factor
        : 0.0;
    const std::vector<std::uint8_t> barrier_mask = options.use_boundary_barrier
        ? buildBoundaryBarrierMask(cloud, boundary_indices, barrier_radius)
        : std::vector<std::uint8_t>(cloud->size(), 0);
    const double normal_dot_threshold = std::cos(options.normal_angle_deg * kPi / 180.0);

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);

    struct NormalGroup {
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        int count = 0;
    };

    std::vector<NormalGroup> normal_groups;
    std::vector<int> normal_group_ids(cloud->size(), -1);
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (barrier_mask[i] != 0 || normals[i].isZero(1.0e-6F)) {
            continue;
        }

        Eigen::Vector3f normal = normals[i].normalized();
        int best_group = -1;
        double best_alignment = -1.0;
        for (size_t g = 0; g < normal_groups.size(); ++g) {
            const double alignment = static_cast<double>(normal.dot(normal_groups[g].normal.normalized()));
            if (alignment > best_alignment) {
                best_alignment = alignment;
                best_group = static_cast<int>(g);
            }
        }

        if (best_group >= 0 && best_alignment >= normal_dot_threshold) {
            Eigen::Vector3f group_normal = normal_groups[static_cast<size_t>(best_group)].normal.normalized();
            if (normal.dot(group_normal) < 0.0F) {
                normal = -normal;
            }
            normal_groups[static_cast<size_t>(best_group)].normal =
                (normal_groups[static_cast<size_t>(best_group)].normal * static_cast<float>(normal_groups[static_cast<size_t>(best_group)].count) + normal).normalized();
            normal_groups[static_cast<size_t>(best_group)].count += 1;
            normal_group_ids[i] = best_group;
        } else {
            NormalGroup group;
            group.normal = normal;
            group.count = 1;
            normal_group_ids[i] = static_cast<int>(normal_groups.size());
            normal_groups.push_back(group);
        }
    }

    std::vector<std::uint8_t> visited(cloud->size(), 0);
    std::vector<int> neighbors;
    std::vector<float> distances;

    for (size_t seed = 0; seed < cloud->size(); ++seed) {
        if (visited[seed] != 0 || normal_group_ids[seed] < 0) {
            continue;
        }

        SurfaceRegion region;
        const int seed_group = normal_group_ids[seed];
        Eigen::Vector3f running_normal = normal_groups[static_cast<size_t>(seed_group)].normal.normalized();
        std::queue<int> pending;
        pending.push(static_cast<int>(seed));
        visited[seed] = 1;

        while (!pending.empty()) {
            const int current = pending.front();
            pending.pop();
            region.indices.push_back(current);

            neighbors.clear();
            distances.clear();
            tree.radiusSearch((*cloud)[static_cast<size_t>(current)], grow_radius, neighbors, distances);

            for (const int neighbor : neighbors) {
                if (neighbor < 0) {
                    continue;
                }
                const size_t ni = static_cast<size_t>(neighbor);
                if (visited[ni] != 0 || normal_group_ids[ni] != seed_group) {
                    continue;
                }

                Eigen::Vector3f candidate_normal = normals[ni].normalized();
                if (candidate_normal.dot(running_normal) < 0.0F) {
                    candidate_normal = -candidate_normal;
                }

                visited[ni] = 1;
                pending.push(neighbor);
                running_normal += candidate_normal;
            }
        }

        if (region.indices.size() >= static_cast<size_t>(options.min_region_points)) {
            region.normal = running_normal.normalized();
            const int region_id = static_cast<int>(regions.size());
            for (const int index : region.indices) {
                (*point_region_ids)[static_cast<size_t>(index)] = region_id;
            }
            regions.push_back(std::move(region));
        }
    }

    std::cout << "Surface normal segmentation:\n"
              << "  spacing: " << spacing << '\n'
              << "  grow radius: " << grow_radius << '\n'
              << "  boundary barrier radius: " << barrier_radius << '\n'
              << "  normal tolerance: " << options.normal_angle_deg << " deg\n"
              << "  dominant normal groups: " << normal_groups.size() << '\n'
              << "  regions: " << regions.size() << '\n';
    return regions;
}

CloudT::Ptr extractByIndices(const CloudT::Ptr& cloud, const std::vector<int>& indices) {
    auto output = std::make_shared<CloudT>();
    output->reserve(indices.size());
    for (const int index : indices) {
        output->push_back((*cloud)[index]);
    }
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = false;
    return output;
}

std::array<std::uint8_t, 3> paletteColor(size_t index) {
    static const std::array<std::array<std::uint8_t, 3>, 12> colors{{
        {{230, 80, 70}}, {{70, 160, 230}}, {{80, 190, 110}}, {{230, 180, 60}},
        {{190, 100, 230}}, {{70, 210, 205}}, {{240, 130, 60}}, {{150, 190, 70}},
        {{220, 90, 150}}, {{120, 130, 240}}, {{90, 200, 150}}, {{210, 150, 90}}
    }};
    return colors[index % colors.size()];
}

double cross2D(const Eigen::Vector2f& origin, const Eigen::Vector2f& a, const Eigen::Vector2f& b) {
    const Eigen::Vector2f oa = a - origin;
    const Eigen::Vector2f ob = b - origin;
    return static_cast<double>(oa.x() * ob.y() - oa.y() * ob.x());
}

std::vector<Eigen::Vector2f> convexHull2D(std::vector<Eigen::Vector2f> points) {
    if (points.size() <= 3) {
        return points;
    }

    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
        if (a.x() == b.x()) {
            return a.y() < b.y();
        }
        return a.x() < b.x();
    });

    std::vector<Eigen::Vector2f> hull;
    hull.reserve(points.size() * 2);
    for (const auto& p : points) {
        while (hull.size() >= 2 && cross2D(hull[hull.size() - 2], hull.back(), p) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    const size_t lower_size = hull.size();
    for (auto it = points.rbegin() + 1; it != points.rend(); ++it) {
        while (hull.size() > lower_size && cross2D(hull[hull.size() - 2], hull.back(), *it) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(*it);
    }

    if (!hull.empty()) {
        hull.pop_back();
    }
    return hull;
}

bool pointInPolygon(const Eigen::Vector2f& p, const std::vector<Eigen::Vector2f>& polygon) {
    if (polygon.size() < 3) {
        return false;
    }

    bool inside = false;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const Eigen::Vector2f& pi = polygon[i];
        const Eigen::Vector2f& pj = polygon[j];
        const bool intersects = ((pi.y() > p.y()) != (pj.y() > p.y())) &&
            (p.x() < (pj.x() - pi.x()) * (p.y() - pi.y()) / ((pj.y() - pi.y()) + 1.0e-12F) + pi.x());
        if (intersects) {
            inside = !inside;
        }
    }
    return inside;
}

std::vector<SprayPathPoint> generateSprayTrajectories(
    const CloudT::Ptr& cloud,
    const std::vector<SurfaceRegion>& regions,
    const Options& options) {
    std::vector<SprayPathPoint> trajectory;
    if (cloud->empty() || regions.empty()) {
        return trajectory;
    }

    const double cloud_spacing = estimateCloudSpacing(cloud);
    const double spacing = options.trajectory_spacing > 0.0
        ? options.trajectory_spacing
        : std::max(cloud_spacing * options.trajectory_spacing_factor, cloud_spacing + 1.0e-6);
    const double support_radius = options.trajectory_support_radius > 0.0
        ? options.trajectory_support_radius
        : std::max(cloud_spacing * options.trajectory_support_factor, cloud_spacing + 1.0e-6);
    const float spacing_f = static_cast<float>(spacing);
    const float support_radius_f = static_cast<float>(support_radius);
    const float max_segment_gap = static_cast<float>(spacing * 1.75);
    const float offset_f = static_cast<float>(options.spray_offset);

    for (size_t region_id = 0; region_id < regions.size(); ++region_id) {
        const SurfaceRegion& region = regions[region_id];
        if (region.indices.size() < 3 || region.normal.isZero(1.0e-6F)) {
            continue;
        }

        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        for (const int index : region.indices) {
            center += pointVector((*cloud)[static_cast<size_t>(index)]);
        }
        center /= static_cast<float>(region.indices.size());

        Eigen::Vector3f normal = region.normal.normalized();
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const int index : region.indices) {
            const Eigen::Vector3f delta = pointVector((*cloud)[static_cast<size_t>(index)]) - center;
            covariance += delta * delta.transpose();
        }
        covariance /= static_cast<float>(region.indices.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        if (solver.info() != Eigen::Success) {
            continue;
        }

        Eigen::Vector3f u_axis = solver.eigenvectors().col(2).normalized();
        u_axis = (u_axis - u_axis.dot(normal) * normal).normalized();
        if (!u_axis.allFinite() || u_axis.isZero(1.0e-6F)) {
            u_axis = normal.unitOrthogonal().normalized();
        }
        Eigen::Vector3f v_axis = normal.cross(u_axis).normalized();

        std::vector<Eigen::Vector2f> projected;
        projected.reserve(region.indices.size());
        float min_u = std::numeric_limits<float>::max();
        float max_u = -std::numeric_limits<float>::max();
        float min_v = std::numeric_limits<float>::max();
        float max_v = -std::numeric_limits<float>::max();

        for (const int index : region.indices) {
            const Eigen::Vector3f delta = pointVector((*cloud)[static_cast<size_t>(index)]) - center;
            Eigen::Vector2f uv(delta.dot(u_axis), delta.dot(v_axis));
            projected.push_back(uv);
            min_u = std::min(min_u, uv.x());
            max_u = std::max(max_u, uv.x());
            min_v = std::min(min_v, uv.y());
            max_v = std::max(max_v, uv.y());
        }

        const std::vector<Eigen::Vector2f> boundary = convexHull2D(projected);
        if (boundary.size() < 3) {
            continue;
        }

        auto projected_cloud = std::make_shared<Cloud2T>();
        projected_cloud->reserve(projected.size());
        for (const auto& uv : projected) {
            Point2T p;
            p.x = uv.x();
            p.y = uv.y();
            projected_cloud->push_back(p);
        }
        pcl::KdTreeFLANN<Point2T> projected_tree;
        projected_tree.setInputCloud(projected_cloud);
        std::vector<int> support_indices;
        std::vector<float> support_distances;

        int line_id = 0;
        int segment_id = 0;
        for (float v = min_v; v <= max_v + spacing_f * 0.5F; v += spacing_f, ++line_id) {
            std::vector<std::vector<Eigen::Vector2f>> line_segments;
            std::vector<Eigen::Vector2f> current_segment;
            Eigen::Vector2f previous_valid(0.0F, 0.0F);
            bool has_previous_valid = false;

            for (float u = min_u; u <= max_u + spacing_f * 0.5F; u += spacing_f) {
                Eigen::Vector2f uv(u, v);
                bool valid = false;
                if (!pointInPolygon(uv, boundary)) {
                    valid = false;
                } else {
                    Point2T query;
                    query.x = uv.x();
                    query.y = uv.y();
                    support_indices.clear();
                    support_distances.clear();
                    valid = projected_tree.radiusSearch(query, support_radius_f, support_indices, support_distances, 1) > 0;
                }

                if (valid) {
                    if (has_previous_valid && (uv - previous_valid).norm() > max_segment_gap && !current_segment.empty()) {
                        line_segments.push_back(current_segment);
                        current_segment.clear();
                    }
                    current_segment.push_back(uv);
                    previous_valid = uv;
                    has_previous_valid = true;
                } else {
                    if (!current_segment.empty()) {
                        line_segments.push_back(current_segment);
                        current_segment.clear();
                    }
                    has_previous_valid = false;
                }
            }
            if (!current_segment.empty()) {
                line_segments.push_back(current_segment);
            }

            if (line_id % 2 == 1) {
                std::reverse(line_segments.begin(), line_segments.end());
                for (auto& segment : line_segments) {
                    std::reverse(segment.begin(), segment.end());
                }
            }

            for (const auto& segment : line_segments) {
                if (segment.size() < 2) {
                    continue;
                }
                const int current_segment_id = segment_id++;
                for (const auto& uv : segment) {
                    const Eigen::Vector3f p3 = center + uv.x() * u_axis + uv.y() * v_axis + offset_f * normal;
                    SprayPathPoint path_point;
                    path_point.point = PointT(p3.x(), p3.y(), p3.z());
                    path_point.normal = normal;
                    path_point.region_id = static_cast<int>(region_id);
                    path_point.line_id = line_id;
                    path_point.segment_id = current_segment_id;
                    trajectory.push_back(path_point);
                }
            }
        }
    }

    std::cout << "Spray trajectory generation:\n"
              << "  spacing: " << spacing << '\n'
              << "  support radius: " << support_radius << '\n'
              << "  normal offset: " << options.spray_offset << '\n'
              << "  trajectory points: " << trajectory.size() << '\n';
    return trajectory;
}

ColorCloudT::Ptr makeColoredCloud(const CloudT::Ptr& cloud, const std::vector<int>& boundary_indices) {
    std::vector<std::uint8_t> is_boundary(cloud->size(), 0);
    for (const int index : boundary_indices) {
        if (index >= 0 && static_cast<size_t>(index) < is_boundary.size()) {
            is_boundary[static_cast<size_t>(index)] = 1;
        }
    }

    auto colored = std::make_shared<ColorCloudT>();
    colored->reserve(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        const PointT& src = (*cloud)[i];
        ColorPointT dst;
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        if (is_boundary[i] != 0) {
            dst.r = 255;
            dst.g = 40;
            dst.b = 40;
        } else {
            dst.r = 170;
            dst.g = 170;
            dst.b = 170;
        }
        colored->push_back(dst);
    }
    colored->width = static_cast<std::uint32_t>(colored->size());
    colored->height = 1;
    colored->is_dense = false;
    return colored;
}

ColorCloudT::Ptr makeSegmentedCloud(
    const CloudT::Ptr& cloud,
    const std::vector<int>& boundary_indices,
    const std::vector<int>& point_region_ids,
    const Options& options) {
    std::vector<std::uint8_t> is_boundary(cloud->size(), 0);
    for (const int index : boundary_indices) {
        if (index >= 0 && static_cast<size_t>(index) < is_boundary.size()) {
            is_boundary[static_cast<size_t>(index)] = 1;
        }
    }

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);

    const double spacing = estimateCloudSpacing(cloud);
    double ownership_radius = options.edge_radius;
    if (ownership_radius <= 0.0) {
        ownership_radius = spacing > 0.0 ? spacing * std::max(2.5, options.edge_radius_factor) : 1.0;
    }

    auto segmented = std::make_shared<ColorCloudT>();
    segmented->reserve(cloud->size());

    std::vector<int> neighbors;
    std::vector<float> distances;

    for (size_t i = 0; i < cloud->size(); ++i) {
        const PointT& src = (*cloud)[i];
        ColorPointT dst;
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;

        if (is_boundary[i] != 0) {
            std::vector<int> adjacent_regions;
            neighbors.clear();
            distances.clear();
            tree.radiusSearch(src, ownership_radius, neighbors, distances);
            for (const int neighbor : neighbors) {
                if (neighbor < 0) {
                    continue;
                }
                const int rid = point_region_ids[static_cast<size_t>(neighbor)];
                if (rid < 0) {
                    continue;
                }
                if (std::find(adjacent_regions.begin(), adjacent_regions.end(), rid) == adjacent_regions.end()) {
                    adjacent_regions.push_back(rid);
                }
            }

            if (adjacent_regions.size() >= 2) {
                dst.r = 255;
                dst.g = 35;
                dst.b = 35;
            } else if (adjacent_regions.size() == 1) {
                const auto c = paletteColor(static_cast<size_t>(adjacent_regions.front()));
                dst.r = static_cast<std::uint8_t>(std::min<int>(255, c[0] + 45));
                dst.g = static_cast<std::uint8_t>(std::min<int>(255, c[1] + 45));
                dst.b = static_cast<std::uint8_t>(std::min<int>(255, c[2] + 45));
            } else {
                dst.r = 255;
                dst.g = 255;
                dst.b = 255;
            }
        } else {
            const int rid = i < point_region_ids.size() ? point_region_ids[i] : -1;
            if (rid >= 0) {
                const auto c = paletteColor(static_cast<size_t>(rid));
                dst.r = c[0];
                dst.g = c[1];
                dst.b = c[2];
            } else {
                dst.r = 115;
                dst.g = 115;
                dst.b = 115;
            }
        }

        segmented->push_back(dst);
    }

    segmented->width = static_cast<std::uint32_t>(segmented->size());
    segmented->height = 1;
    segmented->is_dense = false;
    return segmented;
}

std::vector<pcl::PointIndices> clusterBoundaryPoints(const CloudT::Ptr& boundary_cloud, const Options& options) {
    std::vector<pcl::PointIndices> clusters;
    if (boundary_cloud->empty()) {
        return clusters;
    }

    if (options.cluster_tolerance <= 0.0) {
        pcl::PointIndices all;
        all.indices.resize(boundary_cloud->size());
        std::iota(all.indices.begin(), all.indices.end(), 0);
        clusters.push_back(std::move(all));
        return clusters;
    }

    auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
    tree->setInputCloud(boundary_cloud);

    pcl::EuclideanClusterExtraction<PointT> extraction;
    extraction.setClusterTolerance(options.cluster_tolerance);
    extraction.setMinClusterSize(options.min_cluster_size);
    extraction.setMaxClusterSize(options.max_cluster_size);
    extraction.setSearchMethod(tree);
    extraction.setInputCloud(boundary_cloud);
    extraction.extract(clusters);
    return clusters;
}

void visualizeResults(
    const CloudT::Ptr& cloud,
    const CloudT::Ptr& boundary_cloud,
    const ColorCloudT::Ptr& segmented_cloud,
    const std::vector<SprayPathPoint>& trajectory,
    const Options& options) {
    pcl::visualization::PCLVisualizer viewer("Point Cloud Boundary Segmentation");
    viewer.setBackgroundColor(0.03, 0.03, 0.035);

    if (segmented_cloud && !segmented_cloud->empty()) {
        viewer.addPointCloud<ColorPointT>(segmented_cloud, "segmented_cloud");
    } else {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> source_color(cloud, 170, 170, 170);
        viewer.addPointCloud<PointT>(cloud, source_color, "source_cloud");
    }
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        1.0,
        segmented_cloud && !segmented_cloud->empty() ? "segmented_cloud" : "source_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<PointT> boundary_color(boundary_cloud, 255, 40, 40);
    viewer.addPointCloud<PointT>(boundary_cloud, boundary_color, "boundary_points");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        5.0,
        "boundary_points");

    auto trajectory_cloud = std::make_shared<CloudT>();
    trajectory_cloud->reserve(trajectory.size());
    for (const auto& path_point : trajectory) {
        trajectory_cloud->push_back(path_point.point);
    }
    if (!trajectory_cloud->empty()) {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> path_color(trajectory_cloud, 255, 230, 30);
        viewer.addPointCloud<PointT>(trajectory_cloud, path_color, "spray_trajectory_points");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            2.0,
            "spray_trajectory_points");

        size_t drawable_segments = 0;
        for (size_t i = 1; i < trajectory.size(); ++i) {
            if (trajectory[i].region_id == trajectory[i - 1].region_id &&
                trajectory[i].line_id == trajectory[i - 1].line_id) {
                ++drawable_segments;
            }
        }
        const size_t max_lines = static_cast<size_t>(options.max_visualized_trajectory_lines);
        const size_t line_stride = (max_lines > 0 && drawable_segments > max_lines)
            ? static_cast<size_t>(std::ceil(static_cast<double>(drawable_segments) / static_cast<double>(max_lines)))
            : 1;
        size_t segment_counter = 0;

        for (size_t i = 1; i < trajectory.size(); ++i) {
            if (trajectory[i].region_id != trajectory[i - 1].region_id ||
                trajectory[i].line_id != trajectory[i - 1].line_id ||
                trajectory[i].segment_id != trajectory[i - 1].segment_id) {
                continue;
            }
            if (max_lines > 0 && (segment_counter++ % line_stride) != 0) {
                continue;
            }
            if (max_lines == 0) {
                ++segment_counter;
            }
            const std::string line_id = "spray_line_" + std::to_string(i);
            viewer.addLine<PointT>(trajectory[i - 1].point, trajectory[i].point, 1.0, 0.85, 0.05, line_id);
            viewer.setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                3.0,
                line_id);
        }
        if (line_stride > 1) {
            std::cout << "Viewer sampled spray line segments by stride " << line_stride
                      << " to keep interaction responsive\n";
        }
    }

    viewer.addCoordinateSystem(50.0);
    viewer.initCameraParameters();
    viewer.resetCamera();

    std::cout << "Visualization is running. Close the PCL window or press q in the viewer to exit.\n";
    while (!viewer.wasStopped()) {
        viewer.spinOnce(16);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    std::cout << "Visualization closed.\n";
}

void saveClouds(
    const CloudT::Ptr& preprocessed_cloud,
    const CloudT::Ptr& boundary_cloud,
    const ColorCloudT::Ptr& colored_cloud,
    const ColorCloudT::Ptr& segmented_cloud,
    const std::vector<SprayPathPoint>& trajectory,
    const Options& options) {
    std::filesystem::create_directories(std::filesystem::path(options.out_preprocessed).parent_path());
    std::filesystem::create_directories(std::filesystem::path(options.out_boundary).parent_path());
    std::filesystem::create_directories(std::filesystem::path(options.out_colored).parent_path());
    std::filesystem::create_directories(std::filesystem::path(options.out_segmented).parent_path());
    std::filesystem::create_directories(std::filesystem::path(options.out_trajectory).parent_path());
    std::filesystem::create_directories(std::filesystem::path(options.out_trajectory_csv).parent_path());

    if (pcl::io::savePCDFileBinary(options.out_preprocessed, *preprocessed_cloud) < 0) {
        throw std::runtime_error("Failed to save " + options.out_preprocessed);
    }
    if (pcl::io::savePCDFileBinary(options.out_boundary, *boundary_cloud) < 0) {
        throw std::runtime_error("Failed to save " + options.out_boundary);
    }
    if (pcl::io::savePCDFileBinary(options.out_colored, *colored_cloud) < 0) {
        throw std::runtime_error("Failed to save " + options.out_colored);
    }
    if (pcl::io::savePCDFileBinary(options.out_segmented, *segmented_cloud) < 0) {
        throw std::runtime_error("Failed to save " + options.out_segmented);
    }

    auto trajectory_cloud = std::make_shared<CloudT>();
    trajectory_cloud->reserve(trajectory.size());
    for (const auto& path_point : trajectory) {
        trajectory_cloud->push_back(path_point.point);
    }
    trajectory_cloud->width = static_cast<std::uint32_t>(trajectory_cloud->size());
    trajectory_cloud->height = 1;
    trajectory_cloud->is_dense = false;
    if (!trajectory_cloud->empty()) {
        if (pcl::io::savePCDFileBinary(options.out_trajectory, *trajectory_cloud) < 0) {
            throw std::runtime_error("Failed to save " + options.out_trajectory);
        }
    } else {
        std::cout << "Trajectory is empty; skipped trajectory PCD: " << options.out_trajectory << '\n';
    }

    std::ofstream csv(options.out_trajectory_csv);
    if (!csv) {
        throw std::runtime_error("Failed to save " + options.out_trajectory_csv);
    }
    csv << "region,line,segment,x,y,z,nx,ny,nz\n";
    for (const auto& path_point : trajectory) {
        csv << path_point.region_id << ','
            << path_point.line_id << ','
            << path_point.segment_id << ','
            << path_point.point.x << ','
            << path_point.point.y << ','
            << path_point.point.z << ','
            << path_point.normal.x() << ','
            << path_point.normal.y() << ','
            << path_point.normal.z() << '\n';
    }
}

}  // namespace

int main() {
    try {
        const Options options = makeOptions();
        configureParallelism(options);
        const CloudT::Ptr raw_cloud = loadCloud(options.input_path);
        const CloudT::Ptr cloud = preprocessCloud(raw_cloud, options);

        std::cout << "Loaded " << raw_cloud->size() << " points from " << options.input_path << '\n';
        std::cout << "Preprocessed points: " << cloud->size() << '\n';
        const std::vector<int> raw_boundary_indices = detectBoundaryIndices(cloud, options);
        const std::vector<int> boundary_indices = removeIsolatedBoundaryIndices(cloud, raw_boundary_indices, options);
        const CloudT::Ptr boundary_cloud = extractByIndices(cloud, boundary_indices);
        const ColorCloudT::Ptr colored_cloud = makeColoredCloud(cloud, boundary_indices);
        const std::vector<pcl::PointIndices> clusters = clusterBoundaryPoints(boundary_cloud, options);
        const std::vector<Eigen::Vector3f> normals = estimatePointNormals(cloud, options);
        std::vector<int> point_region_ids;
        const std::vector<SurfaceRegion> surface_regions = segmentSurfaceRegions(
            cloud,
            normals,
            boundary_indices,
            options,
            &point_region_ids);
        const ColorCloudT::Ptr segmented_cloud = makeSegmentedCloud(
            cloud,
            boundary_indices,
            point_region_ids,
            options);
        const std::vector<SprayPathPoint> trajectory = generateSprayTrajectories(
            cloud,
            surface_regions,
            options);

        saveClouds(cloud, boundary_cloud, colored_cloud, segmented_cloud, trajectory, options);

        std::cout << "Raw boundary points: " << raw_boundary_indices.size() << '\n';
        std::cout << "Boundary points after outlier filter: " << boundary_cloud->size() << '\n';
        std::cout << "Boundary clusters: " << clusters.size() << '\n';
        for (size_t i = 0; i < clusters.size(); ++i) {
            std::cout << "  cluster " << i << ": " << clusters[i].indices.size() << " points\n";
        }
        std::cout << "Surface regions: " << surface_regions.size() << '\n';
        const size_t max_regions_to_print = std::min<size_t>(surface_regions.size(), 30);
        for (size_t i = 0; i < max_regions_to_print; ++i) {
            const Eigen::Vector3f n = surface_regions[i].normal;
            std::cout << "  region " << i << ": " << surface_regions[i].indices.size()
                      << " points, normal=[" << n.x() << ", " << n.y() << ", " << n.z() << "]\n";
        }
        if (surface_regions.size() > max_regions_to_print) {
            std::cout << "  ... " << (surface_regions.size() - max_regions_to_print)
                      << " more regions omitted from console output\n";
        }
        std::cout << "Saved: " << options.out_preprocessed << '\n';
        std::cout << "Saved: " << options.out_boundary << '\n';
        std::cout << "Saved: " << options.out_colored << '\n';
        std::cout << "Saved: " << options.out_segmented << '\n';
        std::cout << "Saved: " << options.out_trajectory << '\n';
        std::cout << "Saved: " << options.out_trajectory_csv << '\n';

        if (options.visualize) {
            visualizeResults(cloud, boundary_cloud, segmented_cloud, trajectory, options);
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
