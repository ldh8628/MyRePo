# SPDX-License-Identifier: BSD-2-Clause
cmake_minimum_required(VERSION 3.16)
project(hdl_graph_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
else()
  add_compile_options(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
endif()

find_package(ament_cmake REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_PKG_CONFIG REQUIRED IMPORTED_TARGET yaml-0.1)
find_package(PCL REQUIRED COMPONENTS common filters io registration sample_consensus search)
list(APPEND CMAKE_MODULE_PATH "${PCL_DIR}/Modules")
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system date_time iostreams serialization)
find_package(libusb REQUIRED)
find_package(Qhull REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(ndt_omp REQUIRED)

find_package(OpenMP)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/FloorCoeffs.msg
  msg/PrefilteringDebug.msg
  msg/ScanMatchingOdometryDebug.msg
  msg/ScanMatchingStatus.msg
  srv/DumpGraph.srv
  srv/LoadGraph.srv
  srv/SaveMap.srv
  DEPENDENCIES geometry_msgs std_msgs
)

foreach(pyext_target
  ${PROJECT_NAME}__rosidl_typesupport_c__pyext
  ${PROJECT_NAME}__rosidl_typesupport_fastrtps_c__pyext
  ${PROJECT_NAME}__rosidl_typesupport_introspection_c__pyext
)
  if(TARGET ${pyext_target} AND TARGET Python::Python)
    target_link_libraries(${pyext_target} PRIVATE Python::Python)
  endif()
endforeach()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(common_deps
  geometry_msgs
  message_filters
  nav_msgs
  pcl_conversions
  pcl_ros
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_eigen
  tf2_geometry_msgs
  tf2_ros
)

add_library(registrations
  src/hdl_graph_slam/registrations.cpp
)
target_include_directories(registrations PUBLIC
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/../fast_gicp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../ndt_omp/include
)
target_link_libraries(registrations
  ${PCL_LIBRARIES}
)
target_link_directories(registrations PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../install/fast_gicp/lib
)
ament_target_dependencies(registrations
  rclcpp
)
target_link_libraries(registrations
  fast_gicp
  ndt_omp::ndt_omp
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(registrations OpenMP::OpenMP_CXX)
endif()

set(g2o_libraries
  g2o_core
  g2o_stuff
  g2o_types_slam3d
  g2o_types_slam3d_addons
  g2o_solver_pcg
  g2o_solver_cholmod
  g2o_solver_csparse
  g2o_csparse_extension
)

add_library(hdl_graph_slam_core
  src/hdl_graph_slam/graph_slam.cpp
  src/hdl_graph_slam/information_matrix_calculator.cpp
  src/hdl_graph_slam/keyframe.cpp
  src/hdl_graph_slam/map_cloud_generator.cpp
  src/g2o/robust_kernel_io.cpp
)
target_link_libraries(hdl_graph_slam_core
  registrations
  ${PCL_LIBRARIES}
  Boost::filesystem
  ${g2o_libraries}
)
ament_target_dependencies(hdl_graph_slam_core
  rclcpp
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hdl_graph_slam_core OpenMP::OpenMP_CXX)
endif()

add_executable(scan_matching_odometry_node
  src/hdl_graph_slam_ros2/scan_matching_odometry_node.cpp
)
target_link_libraries(scan_matching_odometry_node
  registrations
  ${PCL_LIBRARIES}
)
ament_target_dependencies(scan_matching_odometry_node ${common_deps})
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(scan_matching_odometry_node "${cpp_typesupport_target}")

add_executable(prefiltering_node
  src/hdl_graph_slam_ros2/prefiltering_node.cpp
)
target_link_libraries(prefiltering_node
  ${PCL_LIBRARIES}
)
ament_target_dependencies(prefiltering_node ${common_deps})
target_link_libraries(prefiltering_node "${cpp_typesupport_target}")

add_executable(floor_detection_node
  src/hdl_graph_slam_ros2/floor_detection_node.cpp
)
target_link_libraries(floor_detection_node
  ${PCL_LIBRARIES}
)
ament_target_dependencies(floor_detection_node ${common_deps})
target_link_libraries(floor_detection_node "${cpp_typesupport_target}")

add_executable(simple_graph_slam_node
  src/hdl_graph_slam_ros2/simple_graph_slam_node.cpp
)
target_link_libraries(simple_graph_slam_node
  ${PCL_LIBRARIES}
)
ament_target_dependencies(simple_graph_slam_node ${common_deps})

add_executable(graph_slam_node
  src/hdl_graph_slam_ros2/graph_slam_node.cpp
)
target_link_libraries(graph_slam_node
  hdl_graph_slam_core
  ${PCL_LIBRARIES}
)
ament_target_dependencies(graph_slam_node ${common_deps})
target_link_libraries(graph_slam_node "${cpp_typesupport_target}")

add_executable(map2odom_publisher
  src/hdl_graph_slam_ros2/map2odom_publisher.cpp
)
ament_target_dependencies(map2odom_publisher
  geometry_msgs
  rclcpp
  tf2_ros
)

install(TARGETS
  registrations
  hdl_graph_slam_core
  prefiltering_node
  scan_matching_odometry_node
  floor_detection_node
  simple_graph_slam_node
  graph_slam_node
  map2odom_publisher
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# upstream also installs rviz/, but this port carries only the runtime nodes
# (see package.xml) and has no rviz/ directory — listing it fails the install step
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
