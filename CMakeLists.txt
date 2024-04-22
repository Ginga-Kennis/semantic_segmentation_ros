cmake_minimum_required(VERSION 3.0.2)
project(semantic_segmentation_ros)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  GetSegmentedImage.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

