cmake_minimum_required(VERSION 3.16)

project(SprayTrajectoryA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT PCL_DIR AND EXISTS "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/cmake/PCLConfig.cmake")
    set(PCL_DIR "E:/prebuilt/Windows/common/vs2019/PCL1.12.1/cmake" CACHE PATH "PCL config directory")
endif()

find_package(PCL REQUIRED COMPONENTS common io kdtree search filters features segmentation visualization)

add_library(spray_trajectory
    SprayTrajectory.cpp
)

target_include_directories(spray_trajectory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS})
target_link_libraries(spray_trajectory PUBLIC ${PCL_LIBRARIES})
target_compile_definitions(spray_trajectory PUBLIC ${PCL_DEFINITIONS})

add_executable(spraytraj_demo
    demo.cpp
)

target_link_libraries(spraytraj_demo PRIVATE spray_trajectory)

if(WIN32)
    target_link_libraries(spraytraj_demo PRIVATE comdlg32)
endif()

if(MSVC)
    target_compile_options(spray_trajectory PRIVATE /W4 /permissive-)
    target_compile_options(spraytraj_demo PRIVATE /W4 /permissive-)
else()
    target_compile_options(spray_trajectory PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(spraytraj_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
