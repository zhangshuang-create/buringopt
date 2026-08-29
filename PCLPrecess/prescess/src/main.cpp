#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <ShObjIdl.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using PointT = pcl::PointXYZRGB;
using Cloud = pcl::PointCloud<PointT>;

// 所有常用处理参数集中在这里修改。
struct Parameters {
    // RealSense exports XYZ coordinates in metres.  Convert once at input so
    // every processing stage and the saved target PLY consistently use mm.
    double input_metres_to_millimetres = 1000.0;
    int mean_k = 50;
    double stddev_mul_thresh = 1.0;
    double plane_distance_threshold = 10.0;
    int ransac_max_iterations = 1000;
    double cluster_tolerance = 20.0;
    int cluster_min_size = 30;
    double point_size = 2.0;
    double coordinate_axis_scale = 100.0;
    int viewer_sleep_ms = 16;
};

struct PlaneRemovalResult {
    Cloud::Ptr target{new Cloud};
    pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
    pcl::ModelCoefficients::Ptr coefficients{new pcl::ModelCoefficients};
};

struct ClusterSelectionResult {
    Cloud::Ptr largest{new Cloud};
    std::size_t cluster_count = 0;
    std::size_t largest_cluster_size = 0;
};

class ComApartment {
public:
    ComApartment() {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED |
                                                       COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            initialized_ = true;
        } else if (hr != RPC_E_CHANGED_MODE) {
            throw std::runtime_error("COM 初始化失败，HRESULT=" +
                                     std::to_string(static_cast<unsigned long>(hr)));
        }
    }
    ~ComApartment() {
        if (initialized_) {
            CoUninitialize();
        }
    }
    ComApartment(const ComApartment&) = delete;
    ComApartment& operator=(const ComApartment&) = delete;

private:
    bool initialized_ = false;
};

template <typename T>
class ComPtr {
public:
    ~ComPtr() {
        if (pointer_ != nullptr) {
            pointer_->Release();
        }
    }
    T** put() { return &pointer_; }
    T* operator->() const { return pointer_; }
    T* get() const { return pointer_; }
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ComPtr() = default;

private:
    T* pointer_ = nullptr;
};

bool isAsciiPath(const fs::path& path) {
    for (const wchar_t character : path.native()) {
        if (static_cast<unsigned int>(character) > 127U) {
            return false;
        }
    }
    return true;
}

fs::path asciiStagingDirectory() {
    fs::path candidate = fs::temp_directory_path() / L"PCLPreprocessAscii";
    if (!isAsciiPath(candidate)) {
        wchar_t windows_directory[MAX_PATH]{};
        const UINT length = GetWindowsDirectoryW(windows_directory, MAX_PATH);
        if (length == 0 || length >= MAX_PATH) {
            throw std::runtime_error("无法找到可供 PCL 使用的 ASCII 临时目录");
        }
        candidate = fs::path(windows_directory) / L"Temp" / L"PCLPreprocessAscii";
    }
    std::error_code error;
    fs::create_directories(candidate, error);
    if (error || !isAsciiPath(candidate)) {
        throw std::runtime_error("无法创建 ASCII 临时目录: " + error.message());
    }
    return candidate;
}

fs::path uniqueStagingPath(const wchar_t* role) {
    const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::wstring name = std::wstring(role) + L"_" +
                              std::to_wstring(GetCurrentProcessId()) + L"_" +
                              std::to_wstring(ticks) + L".ply";
    return asciiStagingDirectory() / name;
}

class TemporaryFile {
public:
    explicit TemporaryFile(fs::path path) : path_(std::move(path)) {}
    ~TemporaryFile() {
        std::error_code ignored;
        fs::remove(path_, ignored);
    }
    const fs::path& path() const { return path_; }
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;

private:
    fs::path path_;
};

// 返回空路径表示用户主动取消，不作为程序内部异常。
fs::path selectPlyFile() {
    ComApartment apartment;
    ComPtr<IFileOpenDialog> dialog;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(dialog.put()));
    if (FAILED(hr)) {
        throw std::runtime_error("无法创建 Windows 文件选择窗口");
    }

    const COMDLG_FILTERSPEC filters[] = {{L"PLY 点云 (*.ply)", L"*.ply"}};
    dialog->SetFileTypes(1, filters);
    dialog->SetFileTypeIndex(1);
    dialog->SetDefaultExtension(L"ply");
    dialog->SetTitle(L"选择 PLY 点云文件");
    DWORD options = 0;
    if (SUCCEEDED(dialog->GetOptions(&options))) {
        dialog->SetOptions(options | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST |
                           FOS_FORCEFILESYSTEM | FOS_STRICTFILETYPES);
    }

    hr = dialog->Show(nullptr);
    if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
        return {};
    }
    if (FAILED(hr)) {
        throw std::runtime_error("文件选择窗口执行失败");
    }

    ComPtr<IShellItem> item;
    if (FAILED(dialog->GetResult(item.put()))) {
        throw std::runtime_error("无法取得所选文件");
    }
    PWSTR raw_path = nullptr;
    if (FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &raw_path)) || raw_path == nullptr) {
        throw std::runtime_error("无法取得所选文件的完整路径");
    }
    const fs::path selected(raw_path);
    CoTaskMemFree(raw_path);
    if (_wcsicmp(selected.extension().c_str(), L".ply") != 0) {
        throw std::runtime_error("只允许选择 .ply 文件");
    }
    return selected;
}

Cloud::Ptr loadPlyCloud(const fs::path& input_path) {
    auto cloud = Cloud::Ptr(new Cloud);
    int result = -1;
    if (isAsciiPath(input_path)) {
        result = pcl::io::loadPLYFile<PointT>(input_path.string(), *cloud);
    } else {
        // PCL 1.12~1.14 的窄字符文件接口可能无法打开中文路径，先以宽字符 API 复制。
        TemporaryFile staging(uniqueStagingPath(L"input"));
        std::error_code error;
        fs::copy_file(input_path, staging.path(), fs::copy_options::overwrite_existing, error);
        if (error) {
            throw std::runtime_error("复制中文路径输入文件到临时目录失败: " + error.message());
        }
        result = pcl::io::loadPLYFile<PointT>(staging.path().string(), *cloud);
    }
    if (result < 0) {
        throw std::runtime_error("PLY 读取失败");
    }
    if (cloud->empty()) {
        throw std::runtime_error("输入点云为空");
    }
    return cloud;
}

void scaleCoordinates(Cloud& cloud, double scale) {
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        throw std::invalid_argument("Point-cloud coordinate scale must be positive and finite");
    }
    for (PointT& point : cloud.points) {
        point.x = static_cast<float>(static_cast<double>(point.x) * scale);
        point.y = static_cast<float>(static_cast<double>(point.y) * scale);
        point.z = static_cast<float>(static_cast<double>(point.z) * scale);
    }
    for (int axis = 0; axis < 3; ++axis) {
        cloud.sensor_origin_[axis] = static_cast<float>(
            static_cast<double>(cloud.sensor_origin_[axis]) * scale);
    }
}

Cloud::Ptr removeStatisticalOutliers(const Cloud::ConstPtr& input,
                                     const Parameters& parameters) {
    if (input->size() <= static_cast<std::size_t>(parameters.mean_k)) {
        throw std::runtime_error("点数不足，无法使用当前 MeanK 执行统计离群点剔除");
    }
    auto filtered = Cloud::Ptr(new Cloud);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(input);
    sor.setMeanK(parameters.mean_k);
    sor.setStddevMulThresh(parameters.stddev_mul_thresh);
    sor.filter(*filtered);
    if (filtered->size() < 3) {
        throw std::runtime_error("离群点处理后点数不足，至少需要 3 个点");
    }
    return filtered;
}

PlaneRemovalResult removeLargestPlane(const Cloud::ConstPtr& input,
                                      const Parameters& parameters) {
    PlaneRemovalResult result;
    pcl::SACSegmentation<PointT> segmentation;
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(parameters.plane_distance_threshold);
    segmentation.setMaxIterations(parameters.ransac_max_iterations);
    segmentation.setInputCloud(input);
    segmentation.segment(*result.inliers, *result.coefficients);

    if (result.inliers->indices.empty() || result.coefficients->values.size() < 4) {
        throw std::runtime_error("RANSAC 未检测到有效平面");
    }

    pcl::ExtractIndices<PointT> extractor;
    extractor.setInputCloud(input);
    extractor.setIndices(result.inliers);
    extractor.setNegative(true);
    extractor.filter(*result.target);
    if (result.target->empty()) {
        throw std::runtime_error("删除最大平面后目标点云为空");
    }
    return result;
}

ClusterSelectionResult keepLargestCluster(const Cloud::ConstPtr& input,
                                          const Parameters& parameters) {
    if (input->size() < static_cast<std::size_t>(parameters.cluster_min_size)) {
        throw std::runtime_error("删除平面后的点数小于聚类最小点数");
    }

    auto tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(input);

    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<PointT> extraction;
    extraction.setClusterTolerance(parameters.cluster_tolerance);
    extraction.setMinClusterSize(parameters.cluster_min_size);
    extraction.setMaxClusterSize(static_cast<int>(std::min<std::size_t>(
        input->size(), static_cast<std::size_t>(std::numeric_limits<int>::max()))));
    extraction.setSearchMethod(tree);
    extraction.setInputCloud(input);
    extraction.extract(clusters);
    if (clusters.empty()) {
        throw std::runtime_error("欧式聚类未找到满足参数要求的点簇");
    }

    const auto largest = std::max_element(
        clusters.begin(), clusters.end(),
        [](const pcl::PointIndices& left, const pcl::PointIndices& right) {
            return left.indices.size() < right.indices.size();
        });

    auto largest_indices = pcl::PointIndices::Ptr(new pcl::PointIndices(*largest));
    ClusterSelectionResult result;
    result.cluster_count = clusters.size();
    result.largest_cluster_size = largest->indices.size();
    pcl::ExtractIndices<PointT> extractor;
    extractor.setInputCloud(input);
    extractor.setIndices(largest_indices);
    extractor.setNegative(false);
    extractor.filter(*result.largest);
    if (result.largest->empty()) {
        throw std::runtime_error("最大点簇提取结果为空");
    }
    return result;
}

void savePlyCloud(const fs::path& output_path, const Cloud::ConstPtr& cloud) {
    if (cloud->empty()) {
        throw std::runtime_error("拒绝保存空点云");
    }
    int result = -1;
    if (isAsciiPath(output_path)) {
        result = pcl::io::savePLYFileBinary(output_path.string(), *cloud);
    } else {
        TemporaryFile staging(uniqueStagingPath(L"output"));
        result = pcl::io::savePLYFileBinary(staging.path().string(), *cloud);
        if (result >= 0) {
            std::error_code error;
            fs::copy_file(staging.path(), output_path, fs::copy_options::overwrite_existing,
                          error);
            if (error) {
                throw std::runtime_error("复制结果到中文输出路径失败: " + error.message());
            }
        }
    }
    if (result < 0) {
        throw std::runtime_error("PLY 保存失败");
    }
}

void visualizeClouds(const Cloud::ConstPtr& before, const Cloud::ConstPtr& after,
                     const fs::path& output_path, const Parameters& parameters) {
    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>(
        "Point Cloud Preprocessing");
    int left = 0;
    int right = 1;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, left);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, right);
    viewer->setBackgroundColor(0.02, 0.02, 0.04, left);
    viewer->setBackgroundColor(0.02, 0.02, 0.04, right);

    // 直接使用 PLY 中的 RGB 字段，不再覆盖为统一蓝色/橙色。
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> before_rgb(before);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> after_rgb(after);
    if (!before_rgb.isCapable() || !after_rgb.isCapable()) {
        throw std::runtime_error("点云不包含可用的 RGB 颜色字段");
    }
    if (!viewer->addPointCloud<PointT>(before, before_rgb, "before_cloud", left) ||
        !viewer->addPointCloud<PointT>(after, after_rgb, "after_cloud", right)) {
        throw std::runtime_error("PCLVisualizer 添加点云失败");
    }
    const bool view_setup_ok =
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, parameters.point_size,
            "before_cloud", left) &&
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, parameters.point_size,
            "after_cloud", right) &&
        viewer->addText("Before", 15, 15, 22, 0.35, 0.65, 1.0,
                        "before_title", left) &&
        viewer->addText("After", 15, 15, 22, 1.0, 0.65, 0.2,
                        "after_title", right) &&
        viewer->addText("Ctrl+S: Save target PLY", 15, 45, 16, 0.85, 0.85, 0.85,
                        "save_status", right);
    if (!view_setup_ok) {
        throw std::runtime_error("PCLVisualizer 视图属性设置失败");
    }
    viewer->addCoordinateSystem(parameters.coordinate_axis_scale, "before_axes", left);
    viewer->addCoordinateSystem(parameters.coordinate_axis_scale, "after_axes", right);
    viewer->resetCamera();

    auto last_save = std::make_shared<std::chrono::steady_clock::time_point>(
        std::chrono::steady_clock::time_point::min());
    viewer->registerKeyboardCallback(
        [viewer, after, output_path, last_save](
            const pcl::visualization::KeyboardEvent& event) {
            const std::string& key = event.getKeySym();
            const bool is_save_key = key == "s" || key == "S" ||
                                     event.getKeyCode() == 's' ||
                                     event.getKeyCode() == 'S';
            if (!event.keyDown() || !event.isCtrlPressed() || !is_save_key) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (*last_save != std::chrono::steady_clock::time_point::min() &&
                now - *last_save < std::chrono::milliseconds(500)) {
                return;
            }
            *last_save = now;
            try {
                savePlyCloud(output_path, after);
                std::wcout << L"Ctrl+S 保存成功: " << output_path << L'\n';
                viewer->updateText("Saved target PLY", 15, 45, 16,
                                   0.3, 1.0, 0.3, "save_status");
            } catch (const std::exception& error) {
                std::cerr << "Ctrl+S 保存失败: " << error.what() << '\n';
                viewer->updateText("Save failed - see console", 15, 45, 16,
                                   1.0, 0.25, 0.25, "save_status");
            }
        });

    while (!viewer->wasStopped()) {
        viewer->spinOnce(10);
        std::this_thread::sleep_for(std::chrono::milliseconds(parameters.viewer_sleep_ms));
    }
}

int main() {
    try {
        const Parameters parameters;
        const fs::path input_path = selectPlyFile();
        if (input_path.empty()) {
            std::wcout << L"用户取消了文件选择。\n";
            return 0;
        }

        std::wcout << L"输入文件路径: " << input_path << L'\n';
        const Cloud::Ptr original = loadPlyCloud(input_path);
        scaleCoordinates(*original, parameters.input_metres_to_millimetres);
        std::cout << "Input XYZ converted from metres to millimetres (x"
                  << parameters.input_metres_to_millimetres << ")\n";
        std::cout << "原始点数: " << original->size() << '\n';

        const Cloud::Ptr denoised = removeStatisticalOutliers(original, parameters);
        std::cout << "离群点剔除数量: " << original->size() - denoised->size() << '\n';
        std::cout << "离群点剔除后的点数: " << denoised->size() << '\n';

        PlaneRemovalResult plane = removeLargestPlane(denoised, parameters);
        std::cout << "最大平面点数: " << plane.inliers->indices.size() << '\n';
        std::cout << "平面方程 a、b、c、d: " << plane.coefficients->values[0] << ", "
                  << plane.coefficients->values[1] << ", "
                  << plane.coefficients->values[2] << ", "
                  << plane.coefficients->values[3] << '\n';
        std::cout << "删除平面后的点数: " << plane.target->size() << '\n';

        ClusterSelectionResult clusters = keepLargestCluster(plane.target, parameters);
        std::cout << "检测到的有效点簇数量: " << clusters.cluster_count << '\n';
        std::cout << "最大点簇点数: " << clusters.largest_cluster_size << '\n';
        std::cout << "最终目标点数: " << clusters.largest->size() << '\n';

        const fs::path output_path =
            input_path.parent_path() / (input_path.stem().wstring() + L"_target.ply");
        std::wcout << L"待保存路径: " << output_path << L'\n';
        std::cout << "处理完成，请在可视化窗口中按 Ctrl+S 保存目标点云。\n";

        visualizeClouds(original, clusters.largest, output_path, parameters);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        MessageBoxA(nullptr, error.what(), "Point Cloud Preprocessing Error",
                    MB_OK | MB_ICONERROR);
        return 1;
    }
}
