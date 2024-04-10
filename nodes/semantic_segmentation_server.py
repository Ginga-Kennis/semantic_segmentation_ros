#!/usr/bin/env python3

import rospy
import cv_bridge
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image

from semantic_segmentation_ros.detection import SemanticSegmentation

class SemanticSegmentationServer:
    def __init__(self):
        self.load_parameters()
        self.init_topics()
        self.init_semantic_segmentation_server()
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.loginfo("Semantic Segmentation Server is ready")
    
    def load_parameters(self):
        self.color_topic = rospy.get_param("~color_topic")
        self.model_path = Path(rospy.get_param("~model_path"))
        rospy.loginfo(self.color_topic)
        rospy.loginfo(self.model_path)

    def init_topics(self):
        rospy.Subscriber(self.color_topic, Image, self.sensor_cb)

    def init_semantic_segmentation_server(self):
        self.ss_server =  SemanticSegmentation(self.model_path)

    def sensor_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        y_pred = self.ss_server.predict(img)
        print(y_pred.shape)



if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_server")
    server = SemanticSegmentationServer()
    rospy.spin()