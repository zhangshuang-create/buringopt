#include "SprayTrajectory.h"

#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace spraytraj {
namespace {

using Point2T = pcl::PointXY;
using Cloud2T = pcl::PointCloud<Point2T>;

constexpr double kPi = 3.14159265358979323846;

struct InternalOptions {
    int sor_mean_k = 30;
    double sor_stddev = 1.0;
    double radius = 0.0;
    int radius_min_neighbors = 3;
    int normal_k = 36;
    int region_neighbors = 30;
    double region_smoothness_deg = 35.0;
    double region_curvature_threshold = 1.0;
    int min_region_size = 10;
    int max_region_size = std::numeric_limits<int>::max();
    double path_spacing_factor = 8.0;
    double support_radius_factor = 1.8;
    double max_segment_gap_factor = 1.75;
    double boundary_band_factor = 1.5;
    double virtual_sample_factor = 0.5;
};

Eigen::Vector3f toVector(const PointT& point) {
    return Eigen::Vector3f(point.x, point.y, point.z);
}

Eigen::Vector3f normalFromRegion(
    const NormalCloudT::Ptr& normals,
    const pcl::PointIndices& region);

double estimateCloudSpacing(const CloudT::Ptr& cloud) {
    if (!cloud || cloud->size() < 3) {
        return 0.0;
    }

    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);

    std::vector<double> nearest;
    nearest.reserve(std::min<size_t>(cloud->size(), 2000));
    std::vector<int> ids(2);
    std::vector<float> distances(2);
    const size_t stride = std::max<size_t>(1, cloud->size() / 2000);

    for (size_t i = 0; i < cloud->size(); i += stride) {
        ids.assign(2, 0);
        distances.assign(2, 0.0F);
        if (tree.nearestKSearch((*cloud)[i], 2, ids, distances) >= 2 && distances[1] > 1.0e-12F) {
            nearest.push_back(std::sqrt(static_cast<double>(distances[1])));
        }
    }

    if (nearest.empty()) {
        return 0.0;
    }
    std::sort(nearest.begin(), nearest.end());
    return nearest[nearest.size() / 2];
}

CloudT::Ptr filterNoise(const CloudT::Ptr& input, const Options& options) {
    if (!input || input->empty()) {
        throw std::runtime_error("Input point cloud must be non-empty");
    }

    auto finite = std::make_shared<CloudT>();
    std::vector<int> finite_indices;
    pcl::removeNaNFromPointCloud(*input, *finite, finite_indices);
    finite->is_dense = false;

    CloudT::Ptr current = finite;
    const InternalOptions internal;

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

    if (internal.sor_mean_k > 0 && static_cast<int>(current->size()) > internal.sor_mean_k) {
        auto sor_cloud = std::make_shared<CloudT>();
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(current);
        sor.setMeanK(internal.sor_mean_k);
        sor.setStddevMulThresh(internal.sor_stddev);
        sor.filter(*sor_cloud);
        sor_cloud->is_dense = false;
        current = sor_cloud;
    }

    if (internal.radius > 0.0) {
        auto radius_cloud = std::make_shared<CloudT>();
        pcl::RadiusOutlierRemoval<PointT> radius_filter;
        radius_filter.setInputCloud(current);
        radius_filter.setRadiusSearch(internal.radius);
        radius_filter.setMinNeighborsInRadius(internal.radius_min_neighbors);
        radius_filter.filter(*radius_cloud);
        radius_cloud->is_dense = false;
        current = radius_cloud;
    }

    return current;
}

NormalCloudT::Ptr estimateNormals(const CloudT::Ptr& cloud, const Options& options) {
    (void)options;
    auto normals = std::make_shared<NormalCloudT>();
    if (!cloud || cloud->empty()) {
        return normals;
    }

    pcl::NormalEstimation<PointT, NormalT> estimation;
    auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
    estimation.setInputCloud(cloud);
    estimation.setSearchMethod(tree);
    const InternalOptions internal;
    estimation.setKSearch(std::max(6, std::min(internal.normal_k, static_cast<int>(cloud->size()))));
    estimation.compute(*normals);

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& point : *cloud) {
        center += toVector(point);
    }
    center /= static_cast<float>(cloud->size());

    for (size_t i = 0; i < normals->size(); ++i) {
        Eigen::Vector3f normal(normals->at(i).normal_x, normals->at(i).normal_y, normals->at(i).normal_z);
        if (!normal.allFinite() || normal.isZero(1.0e-6F)) {
            continue;
        }
        normal.normalize();
        if ((toVector((*cloud)[i]) - center).dot(normal) < 0.0F) {
            normal = -normal;
        }
        normals->at(i).normal_x = normal.x();
        normals->at(i).normal_y = normal.y();
        normals->at(i).normal_z = normal.z();
    }

    return normals;
}

std::vector<pcl::PointIndices> segmentByNormals(
    const CloudT::Ptr& cloud,
    const NormalCloudT::Ptr& normals,
    const Options& options) {
    (void)options;
    std::vector<pcl::PointIndices> regions;
    if (!cloud || cloud->empty() || !normals || normals->size() != cloud->size()) {
        return regions;
    }

    pcl::RegionGrowing<PointT, NormalT> region_growing;
    auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
    const InternalOptions internal;
    region_growing.setSearchMethod(tree);
    region_growing.setInputCloud(cloud);
    region_growing.setInputNormals(normals);
    region_growing.setMinClusterSize(std::max(1, internal.min_region_size));
    region_growing.setMaxClusterSize(internal.max_region_size);
    region_growing.setNumberOfNeighbours(std::max(1, internal.region_neighbors));
    region_growing.setSmoothnessThreshold(internal.region_smoothness_deg * kPi / 180.0);
    region_growing.setCurvatureThreshold(internal.region_curvature_threshold);
    region_growing.extract(regions);

    return regions;
}

std::array<std::uint8_t, 3> paletteColor(size_t index) {
    static const std::array<std::array<std::uint8_t, 3>, 12> colors{{
        {{230, 80, 70}}, {{70, 160, 230}}, {{80, 190, 110}}, {{230, 180, 60}},
        {{190, 100, 230}}, {{70, 210, 205}}, {{240, 130, 60}}, {{150, 190, 70}},
        {{220, 90, 150}}, {{120, 130, 240}}, {{90, 200, 150}}, {{210, 150, 90}}
    }};
    return colors[index % colors.size()];
}

ColorCloudT::Ptr makeColoredRegions(
    const CloudT::Ptr& cloud,
    const std::vector<pcl::PointIndices>& regions,
    const std::vector<int>& boundary_indices) {
    auto colored = std::make_shared<ColorCloudT>();
    if (!cloud) {
        return colored;
    }

    std::vector<int> region_ids(cloud->size(), -1);
    for (size_t region_id = 0; region_id < regions.size(); ++region_id) {
        for (const int index : regions[region_id].indices) {
            if (index >= 0 && static_cast<size_t>(index) < region_ids.size()) {
                region_ids[static_cast<size_t>(index)] = static_cast<int>(region_id);
            }
        }
    }

    std::vector<std::uint8_t> is_boundary(cloud->size(), 0);
    for (const int index : boundary_indices) {
        if (index >= 0 && static_cast<size_t>(index) < is_boundary.size()) {
            is_boundary[static_cast<size_t>(index)] = 1;
        }
    }

    colored->reserve(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        ColorPointT out;
        out.x = (*cloud)[i].x;
        out.y = (*cloud)[i].y;
        out.z = (*cloud)[i].z;
        if (is_boundary[i] != 0) {
            out.r = 255;
            out.g = 35;
            out.b = 35;
        } else if (region_ids[i] >= 0) {
            const auto color = paletteColor(static_cast<size_t>(region_ids[i]));
            out.r = color[0];
            out.g = color[1];
            out.b = color[2];
        } else {
            out.r = 120;
            out.g = 120;
            out.b = 120;
        }
        colored->push_back(out);
    }
    colored->width = static_cast<std::uint32_t>(colored->size());
    colored->height = 1;
    colored->is_dense = false;
    return colored;
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

void pushPoint2D(Cloud2T& cloud, const Eigen::Vector2f& uv) {
    Point2T point;
    point.x = uv.x();
    point.y = uv.y();
    cloud.push_back(point);
}

Cloud2T::Ptr makeVirtualSupportCloud2D(
    const std::vector<Eigen::Vector2f>& projected,
    double sample_spacing) {
    auto support_cloud = std::make_shared<Cloud2T>();
    if (projected.empty()) {
        return support_cloud;
    }

    support_cloud->reserve(projected.size() * 4);
    for (const Eigen::Vector2f& uv : projected) {
        pushPoint2D(*support_cloud, uv);
    }

    if (projected.size() < 2 || sample_spacing <= 1.0e-9) {
        return support_cloud;
    }

    pcl::KdTreeFLANN<Point2T> tree;
    tree.setInputCloud(support_cloud);
    const double neighbor_radius = sample_spacing * 3.0;
    std::vector<int> neighbors;
    std::vector<float> distances;

    for (size_t i = 0; i < projected.size(); ++i) {
        Point2T query;
        query.x = projected[i].x();
        query.y = projected[i].y();
        neighbors.clear();
        distances.clear();
        tree.radiusSearch(query, neighbor_radius, neighbors, distances);

        for (const int neighbor : neighbors) {
            if (neighbor < 0 || static_cast<size_t>(neighbor) <= i) {
                continue;
            }
            const Eigen::Vector2f a = projected[i];
            const Eigen::Vector2f b = projected[static_cast<size_t>(neighbor)];
            const double length = static_cast<double>((b - a).norm());
            if (length <= sample_spacing || length > neighbor_radius) {
                continue;
            }

            const int steps = static_cast<int>(std::floor(length / sample_spacing));
            for (int step = 1; step < steps; ++step) {
                const float t = static_cast<float>(step) / static_cast<float>(steps);
                pushPoint2D(*support_cloud, a + t * (b - a));
            }
        }
    }

    support_cloud->width = static_cast<std::uint32_t>(support_cloud->size());
    support_cloud->height = 1;
    support_cloud->is_dense = false;
    return support_cloud;
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

double distanceToSegment2D(
    const Eigen::Vector2f& p,
    const Eigen::Vector2f& a,
    const Eigen::Vector2f& b) {
    const Eigen::Vector2f ab = b - a;
    const float denom = ab.squaredNorm();
    if (denom <= 1.0e-12F) {
        return static_cast<double>((p - a).norm());
    }
    const float t = std::max(0.0F, std::min(1.0F, (p - a).dot(ab) / denom));
    return static_cast<double>((p - (a + t * ab)).norm());
}

double distanceToPolygonBoundary2D(
    const Eigen::Vector2f& p,
    const std::vector<Eigen::Vector2f>& polygon) {
    if (polygon.empty()) {
        return std::numeric_limits<double>::max();
    }

    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < polygon.size(); ++i) {
        const Eigen::Vector2f& a = polygon[i];
        const Eigen::Vector2f& b = polygon[(i + 1) % polygon.size()];
        best = std::min(best, distanceToSegment2D(p, a, b));
    }
    return best;
}

std::vector<int> boundaryFromRegionHulls(
    const CloudT::Ptr& cloud,
    const std::vector<pcl::PointIndices>& regions,
    const NormalCloudT::Ptr& normals,
    const Options& options) {
    std::vector<int> boundary_indices;
    if (!cloud || cloud->empty() || !normals || normals->size() != cloud->size()) {
        return boundary_indices;
    }

    const InternalOptions internal;
    const double estimated_spacing = estimateCloudSpacing(cloud);
    const double point_spacing = options.point_spacing > 0.0 ? options.point_spacing : estimated_spacing;
    const double boundary_band = std::max(point_spacing * internal.boundary_band_factor, point_spacing + 1.0e-6);
    std::vector<std::uint8_t> is_boundary(cloud->size(), 0);

    for (const pcl::PointIndices& region : regions) {
        if (region.indices.size() < 3) {
            continue;
        }

        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        for (const int index : region.indices) {
            center += toVector((*cloud)[static_cast<size_t>(index)]);
        }
        center /= static_cast<float>(region.indices.size());

        const Eigen::Vector3f normal = normalFromRegion(normals, region);
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const int index : region.indices) {
            const Eigen::Vector3f delta = toVector((*cloud)[static_cast<size_t>(index)]) - center;
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
        const Eigen::Vector3f v_axis = normal.cross(u_axis).normalized();

        std::vector<Eigen::Vector2f> projected;
        projected.reserve(region.indices.size());
        for (const int index : region.indices) {
            const Eigen::Vector3f delta = toVector((*cloud)[static_cast<size_t>(index)]) - center;
            projected.emplace_back(delta.dot(u_axis), delta.dot(v_axis));
        }

        const std::vector<Eigen::Vector2f> hull = convexHull2D(projected);
        if (hull.size() < 3) {
            continue;
        }

        for (size_t i = 0; i < projected.size(); ++i) {
            if (distanceToPolygonBoundary2D(projected[i], hull) <= boundary_band) {
                const int index = region.indices[i];
                if (index >= 0 && static_cast<size_t>(index) < is_boundary.size()) {
                    is_boundary[static_cast<size_t>(index)] = 1;
                }
            }
        }
    }

    for (size_t i = 0; i < is_boundary.size(); ++i) {
        if (is_boundary[i] != 0) {
            boundary_indices.push_back(static_cast<int>(i));
        }
    }
    return boundary_indices;
}

Eigen::Vector3f normalFromRegion(
    const NormalCloudT::Ptr& normals,
    const pcl::PointIndices& region) {
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (const int index : region.indices) {
        const NormalT& normal = (*normals)[static_cast<size_t>(index)];
        Eigen::Vector3f n(normal.normal_x, normal.normal_y, normal.normal_z);
        if (!n.allFinite() || n.isZero(1.0e-6F)) {
            continue;
        }
        n.normalize();
        if (!sum.isZero(1.0e-6F) && n.dot(sum) < 0.0F) {
            n = -n;
        }
        sum += n;
    }
    if (sum.isZero(1.0e-6F)) {
        return Eigen::Vector3f::UnitZ();
    }
    return sum.normalized();
}

}  // namespace

PreparedCloud preprocessBoundarySegment(const CloudT::Ptr& input, const Options& options) {
    PreparedCloud result;
    result.filtered = filterNoise(input, options);
    result.normals = estimateNormals(result.filtered, options);
    result.regions = segmentByNormals(result.filtered, result.normals, options);
    result.boundary_indices = boundaryFromRegionHulls(result.filtered, result.regions, result.normals, options);
    result.point_region_ids.assign(result.filtered->size(), -1);

    for (size_t region_id = 0; region_id < result.regions.size(); ++region_id) {
        for (const int index : result.regions[region_id].indices) {
            if (index >= 0 && static_cast<size_t>(index) < result.point_region_ids.size()) {
                result.point_region_ids[static_cast<size_t>(index)] = static_cast<int>(region_id);
            }
        }
    }

    result.colored_regions = makeColoredRegions(result.filtered, result.regions, result.boundary_indices);
    return result;
}

std::vector<TrajectoryPoint> generateTrajectory(const CloudT::Ptr& input, const Options& options) {
    return generateTrajectory(preprocessBoundarySegment(input, options), options);
}

std::vector<TrajectoryPoint> generateTrajectory(const PreparedCloud& prepared, const Options& options) {
    std::vector<TrajectoryPoint> trajectory;
    const CloudT::Ptr& cloud = prepared.filtered;
    if (!cloud || cloud->size() < 3 || !prepared.normals || prepared.normals->size() != cloud->size()) {
        return trajectory;
    }

    const InternalOptions internal;
    const double estimated_spacing = estimateCloudSpacing(cloud);
    const double point_spacing = options.point_spacing > 0.0 ? options.point_spacing : estimated_spacing;
    const double spacing = options.path_spacing > 0.0
        ? options.path_spacing
        : std::max(point_spacing * internal.path_spacing_factor, point_spacing + 1.0e-6);
    const double support_radius = std::max(spacing * internal.support_radius_factor, point_spacing * 2.5 + 1.0e-6);

    const float spacing_f = static_cast<float>(spacing);
    const float support_radius_f = static_cast<float>(support_radius);
    const float max_segment_gap = static_cast<float>(spacing * internal.max_segment_gap_factor);
    const float offset_f = static_cast<float>(options.spray_offset);

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& point : *cloud) {
        center += toVector(point);
    }
    center /= static_cast<float>(cloud->size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& point : *cloud) {
        const Eigen::Vector3f delta = toVector(point) - center;
        covariance += delta * delta.transpose();
    }
    covariance /= static_cast<float>(cloud->size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return trajectory;
    }

    Eigen::Vector3f normal = solver.eigenvectors().col(0).normalized();
    Eigen::Vector3f u_axis = solver.eigenvectors().col(2).normalized();
    Eigen::Vector3f v_axis = normal.cross(u_axis).normalized();
    if (!v_axis.allFinite() || v_axis.isZero(1.0e-6F)) {
        v_axis = normal.unitOrthogonal().normalized();
    }
    u_axis = v_axis.cross(normal).normalized();

    std::vector<Eigen::Vector2f> projected;
    projected.reserve(cloud->size());
    float min_u = std::numeric_limits<float>::max();
    float max_u = -std::numeric_limits<float>::max();
    float min_v = std::numeric_limits<float>::max();
    float max_v = -std::numeric_limits<float>::max();

    for (const auto& point : *cloud) {
        const Eigen::Vector3f delta = toVector(point) - center;
        Eigen::Vector2f uv(delta.dot(u_axis), delta.dot(v_axis));
        projected.push_back(uv);
        min_u = std::min(min_u, uv.x());
        max_u = std::max(max_u, uv.x());
        min_v = std::min(min_v, uv.y());
        max_v = std::max(max_v, uv.y());
    }

    const std::vector<Eigen::Vector2f> hull = convexHull2D(projected);
    if (hull.size() < 3) {
        return trajectory;
    }

    auto projected_cloud = std::make_shared<Cloud2T>();
    projected_cloud->reserve(projected.size());
    for (const Eigen::Vector2f& uv : projected) {
        pushPoint2D(*projected_cloud, uv);
    }
    pcl::KdTreeFLANN<Point2T> projected_tree;
    projected_tree.setInputCloud(projected_cloud);

    std::vector<int> support_indices;
    std::vector<float> support_distances;
    int segment_id = 0;
    int line_id = 0;

    for (float v = min_v; v <= max_v + spacing_f * 0.5F; v += spacing_f, ++line_id) {
        std::vector<std::vector<int>> line_segments;
        std::vector<int> current_segment;
        Eigen::Vector2f previous_valid(0.0F, 0.0F);
        bool has_previous_valid = false;

        for (float u = min_u; u <= max_u + spacing_f * 0.5F; u += spacing_f) {
            const Eigen::Vector2f uv(u, v);
            int nearest_index = -1;
            bool valid = false;
            if (pointInPolygon(uv, hull)) {
                Point2T query;
                query.x = uv.x();
                query.y = uv.y();
                support_indices.clear();
                support_distances.clear();
                if (projected_tree.radiusSearch(query, support_radius_f, support_indices, support_distances, 1) > 0) {
                    nearest_index = support_indices.front();
                    valid = nearest_index >= 0 && static_cast<size_t>(nearest_index) < cloud->size();
                }
            }

            if (valid) {
                if (has_previous_valid && (uv - previous_valid).norm() > max_segment_gap && !current_segment.empty()) {
                    line_segments.push_back(current_segment);
                    current_segment.clear();
                }
                current_segment.push_back(nearest_index);
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
            int last_index = -1;
            for (const int nearest_index : segment) {
                if (nearest_index == last_index) {
                    continue;
                }
                last_index = nearest_index;
                Eigen::Vector3f n(
                    prepared.normals->at(static_cast<size_t>(nearest_index)).normal_x,
                    prepared.normals->at(static_cast<size_t>(nearest_index)).normal_y,
                    prepared.normals->at(static_cast<size_t>(nearest_index)).normal_z);
                if (!n.allFinite() || n.isZero(1.0e-6F)) {
                    n = normal;
                } else {
                    n.normalize();
                }
                const Eigen::Vector3f p3 = toVector((*cloud)[static_cast<size_t>(nearest_index)]) + offset_f * n;
                TrajectoryPoint point;
                point.point = PointT(p3.x(), p3.y(), p3.z());
                point.normal = n;
                point.region_id = prepared.point_region_ids.size() == cloud->size()
                    ? prepared.point_region_ids[static_cast<size_t>(nearest_index)]
                    : 0;
                point.line_id = line_id;
                point.segment_id = current_segment_id;
                trajectory.push_back(point);
            }
        }
    }

    return trajectory;
}

}  // namespace spraytraj
