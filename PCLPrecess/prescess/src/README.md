# PLY 点云预处理程序

## 处理流程

程序通过 Windows `IFileOpenDialog` 选择一个 PLY 文件，使用 PCL 读取为 `PointXYZRGB` 彩色点云；随后依次进行统计离群点剔除、RANSAC 最大平面检测、最大平面删除和欧式聚类，只保留点数最多的一簇。处理完成后在一个 PCLVisualizer 窗口的左右视口中按 PLY 原始颜色显示处理前后点云；让窗口获得焦点并按 `Ctrl+S`，即可把最大点簇以保留 RGB 字段的二进制 PLY 保存到输入目录下的 `原文件名_target.ply`。

轨迹中不包含体素降采样、直通滤波或其他普通滤波。

## 项目结构

```text
src/
├─ CMakeLists.txt
├─ main.cpp
├─ README.md
└─ build/                 # CMake 构建目录
   └─ Release/
      └─ PlyPointCloudPreprocess.exe
```

## 配置与编译

在“x64 Native Tools Command Prompt for VS 2022”中执行：

```bat
cd /d D:\SprayCode\PCLPrecess\prescess\src
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DPCL_DIR="C:\path\to\PCL\lib\cmake\pcl"
cmake --build build --config Release
```

`PCL_DIR` 必须指向包含 `PCLConfig.cmake` 的目录。也可以先设置环境变量：

```bat
set PCL_DIR=C:\path\to\PCL\lib\cmake\pcl
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

常见安装布局示例：

```text
C:\Program Files\PCL 1.14.1\cmake
C:\Program Files\PCL 1.14.1\lib\cmake\pcl
C:\vcpkg\installed\x64-windows\share\pcl
```

若使用 vcpkg，应传入工具链文件，而不是手工填写各依赖目录：

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
```

## Visual Studio 运行

打开 `build\PlyPointCloudPreprocess.sln`，将 `PlyPointCloudPreprocess` 设为启动项目，将配置切换为 `Release | x64`，按 `Ctrl+F5` 运行。程序启动后会弹出原生 PLY 文件选择窗口。

## 中文路径兼容

Windows 文件窗口和程序内部路径均使用宽字符。PCL 1.12～1.14 的 PLY 接口接收窄字符串，部分构建无法直接打开中文路径。因此程序在遇到非 ASCII 输入路径时，先通过 `std::filesystem` 复制到 ASCII 临时目录再读取；保存时先写入 ASCII 临时文件，再通过宽字符文件系统路径复制到目标目录。临时文件由 RAII 自动清理。

## DLL 缺失处理

Debug/Release 构建都会自动将 PCL 和通过 CMake 导入目标发现的 VTK 等运行库复制到对应输出目录。修改 CMakeLists 后必须重新执行配置和编译命令，不能只运行旧 exe。Debug 版 PCL DLL 通常在文件名末尾带 `d`，例如 `pcl_segmentationd.dll`。

如果自定义 PCL 包没有向 CMake 提供运行库位置，仍提示缺少 `pcl_*.dll`、`vtk*.dll`、`boost_*.dll`、`flann*.dll` 或 `OpenNI2.dll`，可将 PCL 及其第三方依赖的 `bin` 目录加入 `PATH`，然后重新启动 Visual Studio。例如：

```bat
set PATH=C:\path\to\PCL\bin;C:\path\to\PCL\3rdParty\VTK\bin;C:\path\to\PCL\3rdParty\FLANN\bin;C:\path\to\PCL\3rdParty\OpenNI2\Tools;%PATH%
```

也可以把所需 DLL 复制到 `build\Release`，但维护 `PATH` 通常更方便。必须保证应用程序、PCL 和全部依赖都是相同架构，例如全部为 x64，并且 Debug/Release 运行库匹配。

## 参数调整

处理参数集中定义在 `main.cpp` 顶部的 `Parameters` 结构体中：

- `mean_k = 50`
- `stddev_mul_thresh = 1.0`
- `plane_distance_threshold = 0.01`
- `ransac_max_iterations = 1000`
- `cluster_tolerance = 0.02`
- `cluster_min_size = 30`
- `point_size = 2.0`
- `coordinate_axis_scale = 0.1`
- `viewer_sleep_ms = 16`

`plane_distance_threshold` 和 `cluster_tolerance` 的单位都与输入点云坐标单位一致。
