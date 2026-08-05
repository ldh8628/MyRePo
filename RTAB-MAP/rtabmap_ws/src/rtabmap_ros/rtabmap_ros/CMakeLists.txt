cmake_minimum_required(VERSION 3.5)
project(rtabmap_ros)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
   DESTINATION share/${PROJECT_NAME}
   PATTERN "__pycache__" EXCLUDE
)

ament_package()
