cmake_minimum_required(VERSION 3.20)

project(PlyPointCloudPreprocess VERSION 1.0 LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "This program requires the native Windows file dialog.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL 1.12 REQUIRED COMPONENTS
    common
    io
    filters
    search
    kdtree
    segmentation
    visualization
)

add_executable(PlyPointCloudPreprocess main.cpp)

target_include_directories(PlyPointCloudPreprocess PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(PlyPointCloudPreprocess PRIVATE ${PCL_LIBRARY_DIRS})
# PCL 1.12 的配置文件可能以 -Dxxx 形式提供这些定义；沿用其推荐用法。
add_definitions(${PCL_DEFINITIONS})
target_compile_options(PlyPointCloudPreprocess PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_link_libraries(PlyPointCloudPreprocess PRIVATE
    ${PCL_LIBRARIES}
    ole32
    shell32
)

# 把运行时 DLL 自动部署到 exe 同目录，避免脱离 PCL 环境变量后无法启动。
if(WIN32)
    find_file(PCL_COMMON_RUNTIME_RELEASE
        NAMES pcl_common.dll
        HINTS
            "${PCL_DIR}/../bin"
            "${PCL_DIR}/../../bin"
            "${PCL_DIR}/../../../bin"
        NO_DEFAULT_PATH
    )
    if(PCL_COMMON_RUNTIME_RELEASE)
        get_filename_component(PCL_RUNTIME_DIR "${PCL_COMMON_RUNTIME_RELEASE}" DIRECTORY)
        file(GLOB PCL_RUNTIME_DLLS CONFIGURE_DEPENDS "${PCL_RUNTIME_DIR}/*.dll")
        # 同时部署 Release 和末尾带 d 的 Debug DLL，便于直接从 VS 启动任一配置。
        foreach(PCL_RUNTIME_DLL IN LISTS PCL_RUNTIME_DLLS)
            add_custom_command(TARGET PlyPointCloudPreprocess POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${PCL_RUNTIME_DLL}"
                        "$<TARGET_FILE_DIR:PlyPointCloudPreprocess>"
                VERBATIM
            )
        endforeach()

        # 某些官方/第三方 PCL 1.12 包的 pcl_io.dll 仍声明依赖 OpenNI2.dll，
        # 即使当前项目没有启用 OpenNI 采集功能也必须部署该 DLL。
        get_filename_component(PCL_INSTALL_ROOT "${PCL_RUNTIME_DIR}" DIRECTORY)
        find_file(OPENNI2_RUNTIME_RELEASE
            NAMES OpenNI2.dll
            HINTS
                "${PCL_INSTALL_ROOT}/3rdParty/OpenNI2/Redist"
                "${PCL_INSTALL_ROOT}/3rdParty/OpenNI2/Tools"
            NO_DEFAULT_PATH
        )
        if(OPENNI2_RUNTIME_RELEASE)
            add_custom_command(TARGET PlyPointCloudPreprocess POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${OPENNI2_RUNTIME_RELEASE}"
                        "$<TARGET_FILE_DIR:PlyPointCloudPreprocess>"
                VERBATIM
            )
        endif()
    else()
        message(WARNING "pcl_common.dll not found; add the PCL bin directory to PATH.")
    endif()

    # 复制由导入目标声明的 VTK 等传递运行库（CMake 3.21+）。
    add_custom_command(TARGET PlyPointCloudPreprocess POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:PlyPointCloudPreprocess>
                "$<TARGET_FILE_DIR:PlyPointCloudPreprocess>"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
endif()

# 新版 VTK/PCL 组合可能要求自动初始化渲染模块。
if(COMMAND vtk_module_autoinit AND VTK_LIBRARIES)
    vtk_module_autoinit(
        TARGETS PlyPointCloudPreprocess
        MODULES ${VTK_LIBRARIES}
    )
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT
             PlyPointCloudPreprocess)
