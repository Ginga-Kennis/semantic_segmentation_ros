# Semantic Segmentation ROS
This repository contains a ROS package designed for semantic segmentation.
![Demo Animation](assets/readme/deeplabv3.gif)

## Installation
The following instructions were tested with `python3.8` on Ubuntu 20.04.

Clone the repository into the `src` folder of a catkin workspace.

```
git clone https://github.com/Ginga-Kennis/semantic_segmentation_ros.git
```

Create and activate a new virtual environment.

```
cd /path/to/semantic_segmentation_ros
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies within the activated virtual environment.

```
pip install -r requirements.txt
```

Build and source the catkin workspace,

```
catkin build vgn
source /path/to/catkin_ws/devel/setup.zsh
```

## Network Training
```
python3 scripts/train.py [--datadir] [--logdir] [--model] [--batch-size] [--val-split] [--lr] [--epochs]
```

Training and validation metrics are logged to TensorBoard and can be accessed with

```
tensorboard --logdir=log/training
```

## RealSense inference
This package contains an example of realtime semantic segmentation with Intel Realsense D435.
```
roslaunch semantic_segmentation_ros realsense_semantic_segmentation.launch
```
