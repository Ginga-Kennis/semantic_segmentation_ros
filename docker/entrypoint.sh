#!/bin/bash
set -e

# Setup ROS environment
source /opt/ros/noetic/setup.bash

# Source the catkin workspace setup file
source /root/catkin_ws/devel/setup.bash

exec "$@"
