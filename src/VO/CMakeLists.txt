cmake_minimum_required(VERSION 2.8.3)
project(loc_VO)

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" )

set(LEV_ROS_DEPS_PATH "../robot_localization")

find_package(catkin REQUIRED COMPONENTS
  diagnostic_msgs
  diagnostic_updater
  eigen_conversions
  geographic_msgs
  geometry_msgs
  message_filters
  message_generation
  nav_msgs
  roscpp
  roslint
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
  tf2_ros
  xmlrpcpp

  cv_bridge
  )

find_package( OpenCV REQUIRED )
find_package( GFLAGS REQUIRED )
# find_package( spdlog REQUIRED )
# find_package( readerwriterqueue REQUIRED )
find_package( Boost REQUIRED )

# Attempt to find Eigen using its own CMake module.
# If that fails, fall back to cmake_modules package.
find_package(Eigen3)
set(EIGEN_PACKAGE EIGEN3)
if(NOT EIGEN3_FOUND)
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
  set(EIGEN_PACKAGE Eigen)
endif()

add_definitions(-DEIGEN_NO_DEBUG -DEIGEN_MPL2_ONLY)

roslint_cpp()

###################################
## catkin specific configuration ##
###################################

add_message_files(FILES
  LocationalImage.msg
  LocationalImage2.msg
  LocationalImage3.msg
  LocationalImage4.msg
  Test.msg
)

generate_messages(DEPENDENCIES
        std_msgs
        sensor_msgs
        nav_msgs
        geographic_msgs
        geometry_msgs
        )

catkin_package(
  INCLUDE_DIRS
    include

  CATKIN_DEPENDS
    rospy
    cv_bridge
    opencv2

    robot_localization
    cmake_modules
    diagnostic_msgs
    diagnostic_updater
    eigen_conversions
    geographic_msgs
    geometry_msgs
    message_filters
    message_runtime
    nav_msgs
    roscpp
    sensor_msgs
    std_msgs
    std_srvs
    tf2
    tf2_geometry_msgs
    tf2_ros
    xmlrpcpp
  DEPENDS ${EIGEN_PACKAGE}
)

###########
## Build ##
###########

include_directories(include src ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${ZeroMQ_INCLUDE_DIRS} ${Spinnaker_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIRS} ${LEV_ROS_DEPS_PATH}/spdlog/include ${LEV_ROS_DEPS_PATH}/readerwriterqueue)

set(BUILD_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_FLAGS}")

add_executable(rtvo src/rtvo.cpp)
add_dependencies(rtvo ${PROJECT_NAME}_gencpp)
target_link_libraries(rtvo ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${ZeroMQ_LIBRARIES} ${Spinnaker_LIBRARIES})

