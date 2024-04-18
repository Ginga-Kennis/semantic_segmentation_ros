# Semantic Segmentation ROS
This repository contains a ROS package designed for semantic segmentation.
![Demo Animation](assets/readme/deeplabv3plus_demo.gif)

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
catkin build semantic_segmentation_ros
source /path/to/catkin_ws/devel/setup.bash
```

## Network Training
To train the network on a custom dataset, follow these steps:
* Place your images in the `assets/data/img`  directory and the JSON files created with LabelMe in the `assets/data/ann` directory.   
Ensure that each image has a corresponding JSON file with a matching name.
* Configure the `config/train.json` file according to your training preferences and dataset specifications.  
```
python3 scripts/train.py [--config]
```

Training and validation metrics are logged to TensorBoard and can be accessed with

```
tensorboard --logdir=log/training
```

## RealSense Inference
This package contains an example of realtime semantic segmentation with Intel Realsense.  
Configure the `config/realsense.yaml` file to suit your environment.
```
roslaunch semantic_segmentation_ros realsense_semantic_segmentation.launch
```
