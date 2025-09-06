cmake_minimum_required(VERSION 3.8)
project(range_image_clustering)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# PCL 경고 메시지 제거를 위한 정책 설정
cmake_policy(SET CMP0074 NEW)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# 라이브러리 빌드
add_library(range_image_clustering src/range_image_clustering.cpp)
target_include_directories(range_image_clustering PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(
  range_image_clustering
  "rclcpp"
  "sensor_msgs"
  "pcl_conversions"
  "visualization_msgs"
)
target_link_libraries(range_image_clustering ${PCL_LIBRARIES})

# 노드 실행 파일 빌드
add_executable(range_image_clustering_node src/range_image_clustering_node.cpp)
target_include_directories(range_image_clustering_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(
  range_image_clustering_node
  "rclcpp"
  "sensor_msgs"
  "pcl_conversions"
  "visualization_msgs"
)
target_link_libraries(range_image_clustering_node
  range_image_clustering
  ${PCL_LIBRARIES}
)

# ROI 시각화 노드 빌드
add_executable(roi_publisher_node src/roi_publisher_node.cpp)
target_include_directories(roi_publisher_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(
  roi_publisher_node
  "rclcpp" 
  "visualization_msgs"
)

# 설치 대상 지정
install(TARGETS
  range_image_clustering
  range_image_clustering_node
  roi_publisher_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ament_package는 마지막에 호출해야 함
ament_package()
