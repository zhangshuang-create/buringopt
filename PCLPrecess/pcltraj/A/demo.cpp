#include "SprayTrajectory.h"

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace {

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

spraytraj::CloudT::Ptr loadCloud(const std::string& path) {
    auto cloud = std::make_shared<spraytraj::CloudT>();
    const std::filesystem::path fs_path(path);
    const std::string ext = fs_path.extension().string();
    int result = -1;
    if (ext == ".pcd" || ext == ".PCD") {
        result = pcl::io::loadPCDFile<spraytraj::PointT>(path, *cloud);
    } else if (ext == ".ply" || ext == ".PLY") {
        result = pcl::io::loadPLYFile<spraytraj::PointT>(path, *cloud);
    } else {
        throw std::runtime_error("Only .pcd and .ply are supported");
    }
    if (result < 0 || cloud->empty()) {
        throw std::runtime_error("Failed to load cloud: " + path);
    }
    cloud->is_dense = false;
    return cloud;
}

void saveTrajectoryCsv(const std::string& path, const std::vector<spraytraj::TrajectoryPoint>& trajectory) {
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("Failed to save " + path);
    }
    csv << "region,line,segment,x,y,z,nx,ny,nz\n";
    for (const auto& point : trajectory) {
        csv << point.region_id << ','
            << point.line_id << ','
            << point.segment_id << ','
            << point.point.x << ','
            << point.point.y << ','
            << point.point.z << ','
            << point.normal.x() << ','
            << point.normal.y() << ','
            << point.normal.z() << '\n';
    }
}

spraytraj::CloudT::Ptr trajectoryToCloud(const std::vector<spraytraj::TrajectoryPoint>& trajectory) {
    auto cloud = std::make_shared<spraytraj::CloudT>();
    cloud->reserve(trajectory.size());
    for (const auto& point : trajectory) {
        cloud->push_back(point.point);
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

void visualizeResult(
    const spraytraj::PreparedCloud& prepared,
    const std::vector<spraytraj::TrajectoryPoint>& trajectory) {
    pcl::visualization::PCLVisualizer viewer("Spray Trajectory Preview");
    viewer.setBackgroundColor(0.03, 0.03, 0.035);

    if (prepared.colored_regions && !prepared.colored_regions->empty()) {
        viewer.addPointCloud<spraytraj::ColorPointT>(prepared.colored_regions, "regions");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            2.0,
            "regions");
    }

    auto boundary_cloud = std::make_shared<spraytraj::CloudT>();
    if (prepared.filtered) {
        boundary_cloud->reserve(prepared.boundary_indices.size());
        for (const int index : prepared.boundary_indices) {
            if (index >= 0 && static_cast<size_t>(index) < prepared.filtered->size()) {
                boundary_cloud->push_back((*prepared.filtered)[static_cast<size_t>(index)]);
            }
        }
    }
    boundary_cloud->width = static_cast<std::uint32_t>(boundary_cloud->size());
    boundary_cloud->height = 1;
    boundary_cloud->is_dense = false;
    if (!boundary_cloud->empty()) {
        pcl::visualization::PointCloudColorHandlerCustom<spraytraj::PointT> boundary_color(
            boundary_cloud,
            255,
            35,
            35);
        viewer.addPointCloud<spraytraj::PointT>(boundary_cloud, boundary_color, "boundary");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            5.0,
            "boundary");
    }

    const spraytraj::CloudT::Ptr trajectory_cloud = trajectoryToCloud(trajectory);
    if (!trajectory_cloud->empty()) {
        pcl::visualization::PointCloudColorHandlerCustom<spraytraj::PointT> trajectory_color(
            trajectory_cloud,
            255,
            230,
            30);
        viewer.addPointCloud<spraytraj::PointT>(trajectory_cloud, trajectory_color, "trajectory_points");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            3.0,
            "trajectory_points");

        for (size_t i = 1; i < trajectory.size(); ++i) {
            if (trajectory[i].region_id != trajectory[i - 1].region_id ||
                trajectory[i].line_id != trajectory[i - 1].line_id ||
                trajectory[i].segment_id != trajectory[i - 1].segment_id) {
                continue;
            }
            const std::string line_id = "trajectory_line_" + std::to_string(i);
            viewer.addLine<spraytraj::PointT>(
                trajectory[i - 1].point,
                trajectory[i].point,
                1.0,
                0.85,
                0.05,
                line_id);
            viewer.setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                3.0,
                line_id);
        }
    }

    viewer.addCoordinateSystem(50.0);
    viewer.initCameraParameters();
    viewer.resetCamera();

    std::cout << "Visualization is running. Close the PCL window or press q to exit.\n";
    while (!viewer.wasStopped()) {
        viewer.spinOnce(16);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input_file;
        if (argc >= 2) {
            input_file = argv[1];
        } else {
            input_file = chooseInputCloudFile();
        }

        if (input_file.empty()) {
            std::cerr << "No input point cloud selected.\n";
            return 1;
        }

        spraytraj::Options options;
        const auto cloud = loadCloud(input_file);
        const spraytraj::PreparedCloud prepared = spraytraj::preprocessBoundarySegment(cloud, options);
        const std::vector<spraytraj::TrajectoryPoint> trajectory =
            spraytraj::generateTrajectory(prepared, options);

        const std::filesystem::path input_path(input_file);
        const std::filesystem::path output_dir = input_path.parent_path();
        const std::string colored_path = (output_dir / "A_colored_regions.pcd").string();
        const std::string csv_path = (output_dir / "A_trajectory.csv").string();
        pcl::io::savePCDFileBinary(colored_path, *prepared.colored_regions);
        saveTrajectoryCsv(csv_path, trajectory);

        std::cout << "Filtered points: " << prepared.filtered->size() << '\n';
        std::cout << "Boundary points: " << prepared.boundary_indices.size() << '\n';
        std::cout << "Regions: " << prepared.regions.size() << '\n';
        std::cout << "Trajectory points: " << trajectory.size() << '\n';
        std::cout << "Saved: " << colored_path << '\n';
        std::cout << "Saved: " << csv_path << '\n';

        visualizeResult(prepared, trajectory);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
