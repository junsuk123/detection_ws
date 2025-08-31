cmake_minimum_required(VERSION 3.14)
project(euclidean_cluster_gpu)

find_package(ament_cmake_auto REQUIRED)

# Use modern CUDA support instead of FindCUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

ament_auto_find_build_dependencies()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif()

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

include_directories(
    include
    SYSTEM
)

# CUDA compilation flags
set_property(TARGET CUDA::cudart PROPERTY IMPORTED_GLOBAL TRUE)

# Create CUDA library
add_library(euclidean_cluster_gpu_lib SHARED
    lib/euclidean_cluster_cuda.cu
    src/label_compress.cpp
)

target_link_libraries(euclidean_cluster_gpu_lib
    CUDA::cudart
    CUDA::cuda_driver
)

target_include_directories(euclidean_cluster_gpu_lib SYSTEM PRIVATE
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${vision_msgs_INCLUDE_DIRS}
    ${visualization_msgs_INCLUDE_DIRS}
    ${geometry_msgs_INCLUDE_DIRS}
    ${std_msgs_INCLUDE_DIRS}
)

# Set CUDA properties
set_target_properties(euclidean_cluster_gpu_lib PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "75;86;87;89"
)

# Main library
ament_auto_add_library(euclidean_cluster_gpu SHARED
    src/euclidean_cluster_gpu_node.cpp
)

target_link_libraries(euclidean_cluster_gpu
    euclidean_cluster_gpu_lib
)

# Create the executable directly
add_executable(euclidean_cluster_gpu_node
    src/euclidean_cluster_gpu_node.cpp
)

target_link_libraries(euclidean_cluster_gpu_node
    euclidean_cluster_gpu_lib
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${std_msgs_LIBRARIES}
)

target_include_directories(euclidean_cluster_gpu_node PRIVATE
    include
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${vision_msgs_INCLUDE_DIRS}
    ${visualization_msgs_INCLUDE_DIRS}
    ${geometry_msgs_INCLUDE_DIRS}
    ${std_msgs_INCLUDE_DIRS}
)

# Install the executable
install(TARGETS euclidean_cluster_gpu_node
    DESTINATION lib/${PROJECT_NAME}
)

# Register as component for launch files
rclcpp_components_register_node(euclidean_cluster_gpu
    PLUGIN "EuclideanClusterGpuNode"
    EXECUTABLE euclidean_cluster_gpu_node_component
)

install(DIRECTORY launch config
    DESTINATION share/${PROJECT_NAME}
)

ament_auto_package()

# Set ROS_DISTRO macros
set(ROS_DISTRO $ENV{ROS_DISTRO})
if(${ROS_DISTRO} STREQUAL "rolling")
    add_compile_definitions(ROS_DISTRO_ROLLING)
elseif(${ROS_DISTRO} STREQUAL "foxy")
    add_compile_definitions(ROS_DISTRO_FOXY)
elseif(${ROS_DISTRO} STREQUAL "galactic")
    add_compile_definitions(ROS_DISTRO_GALACTIC)
elseif(${ROS_DISTRO} STREQUAL "humble")
    add_compile_definitions(ROS_DISTRO_HUMBLE)
endif()
