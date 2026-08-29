#pragma once

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <vector>

namespace spraytraj {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using ColorPointT = pcl::PointXYZRGB;
using ColorCloudT = pcl::PointCloud<ColorPointT>;

struct Options {
    double point_spacing = 0.0;
    double path_spacing = 0.0;
    double spray_offset = 0.0;
    double voxel_leaf = 0.0;
};

struct TrajectoryPoint {
    PointT point;
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    int region_id = -1;
    int line_id = -1;
    int segment_id = -1;
};

struct PreparedCloud {
    CloudT::Ptr filtered;
    NormalCloudT::Ptr normals;
    std::vector<int> boundary_indices;
    std::vector<pcl::PointIndices> regions;
    std::vector<int> point_region_ids;
    ColorCloudT::Ptr colored_regions;
};

// The first three stages in one function:
// filtering -> boundary recognition -> normal-based region segmentation.
PreparedCloud preprocessBoundarySegment(const CloudT::Ptr& input, const Options& options = Options{});

// Main function for robot path planning input:
// scanned point cloud -> ordered spray trajectory points.
std::vector<TrajectoryPoint> generateTrajectory(const CloudT::Ptr& input, const Options& options = Options{});

// Reuse this overload when you want to inspect/save the intermediate result.
std::vector<TrajectoryPoint> generateTrajectory(const PreparedCloud& prepared, const Options& options = Options{});

}  // namespace spraytraj
