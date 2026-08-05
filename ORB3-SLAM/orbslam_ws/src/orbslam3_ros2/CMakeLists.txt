cmake_minimum_required(VERSION 3.8)
  project(orbslam3_ros2)

  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()

  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(ament_cmake REQUIRED)

  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(message_filters REQUIRED)

  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  if((NOT OPENGL_opengl_LIBRARY OR OPENGL_opengl_LIBRARY MATCHES "NOTFOUND") AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/libOpenGL.so.0")
    set(OPENGL_opengl_LIBRARY "$ENV{CONDA_PREFIX}/lib/libOpenGL.so.0" CACHE FILEPATH "OpenGL library" FORCE)
  endif()
  find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
  find_package(Pangolin REQUIRED)
  find_package(Boost REQUIRED COMPONENTS serialization)
  find_package(OpenSSL REQUIRED)
  find_package(spdlog REQUIRED)
  find_package(console_bridge REQUIRED)

  find_library(GLDISPATCH_LIB
    NAMES GLdispatch
    PATHS
      /usr/lib/x86_64-linux-gnu
      $ENV{CONDA_PREFIX}/lib
  )
  if(NOT GLDISPATCH_LIB)
    message(FATAL_ERROR "Required GLdispatch library not found")
  endif()

  find_package(statistics_msgs REQUIRED)
  find_package(libstatistics_collector REQUIRED)

  get_filename_component(DEFAULT_ORB_SLAM3_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../ORB_SLAM3"
    ABSOLUTE
  )
  set(ORB_SLAM3_ROOT
    "${DEFAULT_ORB_SLAM3_ROOT}"
    CACHE PATH "Path to ORB_SLAM3 source directory"
  )

  set(ORB_SLAM3_INCLUDE_DIRS
    ${ORB_SLAM3_ROOT}
    ${ORB_SLAM3_ROOT}/include
    ${ORB_SLAM3_ROOT}/include/CameraModels
    ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  )

  set(ORB_SLAM3_LIB
    ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
  )

  set(DBOW2_LIB
    ${ORB_SLAM3_ROOT}/lib/libDBoW2.so
  )

  set(G2O_LIB
    ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so
  )

  foreach(required_file
    ${ORB_SLAM3_LIB}
    ${DBOW2_LIB}
    ${G2O_LIB}
  )
    if(NOT EXISTS ${required_file})
      message(FATAL_ERROR "Required ORB-SLAM3 library not found: ${required_file}")
    endif()
  endforeach()

  add_executable(rgbd_node
    src/rgbd_node.cpp
  )

  target_include_directories(rgbd_node PRIVATE
    ${ORB_SLAM3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )

  ament_target_dependencies(rgbd_node
    rclcpp
    sensor_msgs
    geometry_msgs
    nav_msgs
    std_msgs
    tf2
    tf2_ros
    message_filters
    statistics_msgs
    libstatistics_collector
  )

  target_link_libraries(rgbd_node
    ${ORB_SLAM3_LIB}
    ${DBOW2_LIB}
    ${G2O_LIB}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Boost::serialization
    OpenSSL::Crypto
    spdlog::spdlog
    console_bridge
    ${GLDISPATCH_LIB}
  )

  target_link_directories(rgbd_node PRIVATE
    $ENV{CONDA_PREFIX}/lib
  )

  if(TARGET Pangolin::pangolin)
    target_link_libraries(rgbd_node Pangolin::pangolin)
  elseif(TARGET pangolin)
    target_link_libraries(rgbd_node pangolin)
  else()
    target_link_directories(rgbd_node PRIVATE
      $ENV{CONDA_PREFIX}/lib
    )
    target_link_libraries(rgbd_node
      pango_display
      pango_opengl
      pango_windowing
      pango_vars
      pango_core
    )
  endif()

  set_target_properties(rgbd_node PROPERTIES
    BUILD_RPATH "${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib;$ENV{CONDA_PREFIX}/lib"
    INSTALL_RPATH "${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib;$ENV{CONDA_PREFIX}/lib"
  )

  install(TARGETS rgbd_node
    DESTINATION lib/${PROJECT_NAME}
  )

  install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
  )

  install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
  )

  ament_package()
