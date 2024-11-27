cmake_minimum_required(VERSION 3.8)
project(yolo_world_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(msg_files
  "msg/PixelPose.msg"
  "msg/BbPixelPose.msg"
  "msg/BbPixelPoseArray.msg"
  "msg/ObjectPose.msg"
  "msg/ObjectPoseArray.msg"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
