cmake_minimum_required(VERSION 3.8)
project(posetrack)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(CURL REQUIRED)
find_package(message_filters REQUIRED)
find_package(rcpputils REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# 寻找OpenCV库，使用自定义的OpenCV_DIR
# set(OpenCV_DIR /usr/lib/aarch64-linux-gnu)
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_VERSION}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# define dnn lib path
set(DNN_PATH "/usr/include/dnn")

set(DNN_LIB_PATH "/usr/lib/")

include_directories(${DNN_PATH})
link_directories(${DNN_LIB_PATH})


# 寻找Eigen3
set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)

# bytetrack_lib
add_library(bytetrack_lib SHARED
  src/bytetrack/BYTETracker.cpp
  src/bytetrack/kalmanFilter.cpp
  src/bytetrack/lapjv.cpp
  src/bytetrack/STrack.cpp
  src/bytetrack/utils.cpp
  )

target_link_libraries(bytetrack_lib Eigen3::Eigen)


add_executable(yoloposetrack src/yoloposetrack.cc)


ament_target_dependencies(yoloposetrack
   rclcpp
   sensor_msgs
   cv_bridge
   vision_msgs
   image_transport
   geometry_msgs
   message_filters
   rcpputils
)
target_link_libraries(yoloposetrack 
  bytetrack_lib
  ${OpenCV_LIBS} opencv_world
  dnn
  pthread
  rt
  dl
  )


install(TARGETS yoloposetrack  bytetrack_lib
  DESTINATION lib/${PROJECT_NAME})
 

#  install(DIRECTORY launch
#       DESTINATION share/${PROJECT_NAME}/launch
#  )
 
 install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)
 
 

target_include_directories(yoloposetrack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(yoloposetrack PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
