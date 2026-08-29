cmake_minimum_required(VERSION 3.16)

project(PclTrajBasic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Convenience hint for this Windows workstation. On Linux/ROS, install PCL
# normally and CMake will find it through the system package path.
if(NOT PCL_DIR AND EXISTS "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/cmake/PCLConfig.cmake")
    set(PCL_DIR "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/cmake" CACHE PATH "PCL config directory")
endif()

find_package(PCL REQUIRED COMPONENTS common io kdtree search filters segmentation visualization)
find_package(OpenMP)

add_executable(pcltraj_basic
    basic.cpp
)

target_include_directories(pcltraj_basic PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcltraj_basic PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(pcltraj_basic PRIVATE ${PCL_DEFINITIONS})

if(WIN32)
    target_link_libraries(pcltraj_basic PRIVATE comdlg32)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(pcltraj_basic PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(pcltraj_basic PRIVATE PCLPRECESS_USE_OPENMP)
endif()

if(MSVC)
    target_compile_options(pcltraj_basic PRIVATE /W4 /permissive-)
else()
    target_compile_options(pcltraj_basic PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    set(PCL_RUNTIME_DIRS
        "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/bin"
        "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/3rdParty/Boost/lib"
        "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/3rdParty/FLANN/bin"
        "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/3rdParty/Qhull/bin"
        "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/3rdParty/OpenNI2/Redist"
        "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/3rdParty/VTK/bin"
    )

    set(PCL_RUNTIME_PATH "")
    foreach(runtime_dir IN LISTS PCL_RUNTIME_DIRS)
        if(EXISTS "${runtime_dir}")
            list(APPEND PCL_RUNTIME_PATH "${runtime_dir}")
            file(GLOB runtime_dlls CONFIGURE_DEPENDS "${runtime_dir}/*.dll")
            foreach(runtime_dll IN LISTS runtime_dlls)
                add_custom_command(TARGET pcltraj_basic POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${runtime_dll}"
                        "$<TARGET_FILE_DIR:pcltraj_basic>"
                    VERBATIM)
            endforeach()
        endif()
    endforeach()

    if(PCL_RUNTIME_PATH)
        list(JOIN PCL_RUNTIME_PATH "\\;" PCL_RUNTIME_PATH_JOINED)
        set_target_properties(pcltraj_basic PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=${PCL_RUNTIME_PATH_JOINED}\\;%PATH%")
    endif()
endif()
