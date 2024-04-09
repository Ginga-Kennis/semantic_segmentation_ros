#!/usr/bin/env python3

from pathlib import Path
import rospy
from sensor_msgs.msg import Image

from semantic_segmentation_ros.detection import SemanticSegmentation

class SemanticSegmentationServer:
    def __init__(self):
        self.image_topic = rospy.get_param("~camera")
        self.model_path = Path(rospy.get_param("~model_path"))
        self.semantic_segmentation = SemanticSegmentation(self.model_path)
        rospy.loginfo(self.model_path)
        rospy.loginfo(self.image_topic)

        self.subscriber = rospy.Subscriber(self.image_topic, Image, self.image_callback)

    def image_callback(self, msg):
        rospy.loginfo("Received an image!")

if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_server")
    server = SemanticSegmentationServer()
    rospy.spin()