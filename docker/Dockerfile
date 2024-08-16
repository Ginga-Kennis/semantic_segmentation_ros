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
#   Create User   #
###################
# Note: Adjust these ARG values to match the user environment on your host system. 
ARG USERNAME=oinarisan
ARG USER_UID=1000
ARG USER_GID=1000

# Create a new group and user based on the provided arguments.
# The user is also given a home directory, and a .config directory is created within it.
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# Set up sudo
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
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
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/$USERNAME/.bashrc

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
# SETUP Workspace #
###################
# Switch to user before creating the workspace
USER $USERNAME

# Create a catkin workspace
RUN mkdir -p /home/$USERNAME/catkin_ws/src

# Build the catkin workspace using catkin build
WORKDIR /home/$USERNAME/catkin_ws/
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build"

############################
# Install Python libraries #
############################
# Install specified Python libraries
RUN pip3 install --no-cache-dir \
    catkin-pkg \
    numpy \
    pytorch-ignite \
    matplotlib \
    protobuf==4.25.3 \
    tensorboard \
    pyyaml \
    tqdm \
    torch \
    segmentation-models-pytorch

# Set entrypoint
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
