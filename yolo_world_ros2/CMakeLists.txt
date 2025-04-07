cmake_minimum_required(VERSION 3.8)
project(yolo_world_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(PROGRAMS
  ${PROJECT_NAME}/${PROJECT_NAME}.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ${PROJECT_NAME}
)
install(PROGRAMS
  ${PROJECT_NAME}/pose_transformer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pose_transformer
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
