#!/bin/bash
set -e

# Setup ROS environment
source /opt/ros/noetic/setup.bash

exec "$@"
