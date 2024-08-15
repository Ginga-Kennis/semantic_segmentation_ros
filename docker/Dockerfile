# Use NVIDIA's CUDA image with cuDNN
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set noninteractive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Install basic utilities and add ROS repository
RUN apt-get update && apt-get install -y \
    lsb-release \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

###################
# ROS Instllation #
###################

# Setup your computer to accept software from packages.ros.org.
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# Add the ROS GPG key to verify package authenticity
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Install ROS Noetic
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS environment to be sourced automatically in bash
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Install additional ROS tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-catkin-tools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

###################
#      SETUP      #
###################

# Create a catkin workspace
RUN mkdir -p /root/catkin_ws/src

# Set the working directory to the catkin workspace
WORKDIR /root/catkin_ws/src

# Clone the semantic_segmentation_ros repository
RUN git clone https://github.com/Ginga-Kennis/semantic_segmentation_ros.git

# Set the working directory to the cloned repository
WORKDIR /root/catkin_ws/src/semantic_segmentation_ros

# Install Python dependencies from the repository's requirements.txt
RUN pip install -r requirements.txt

# Build the catkin workspace using catkin build
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build"

# Set entrypoint
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
