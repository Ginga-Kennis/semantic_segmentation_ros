#!/usr/bin/env python3

from pathlib import Path
import rospy
from rospy.timer import Timer

from semantic_segmentation_ros.detection import SemanticSegmentation

class SemanticSegmentationServer:
    def __init__(self):
        self.model_path = Path(rospy.get_param("~model_path"))
        self.semantic_segmentation = SemanticSegmentation(self.model_path)
        rospy.loginfo(self.model_path)

if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_server")
    server = SemanticSegmentationServer()
    rospy.spin()