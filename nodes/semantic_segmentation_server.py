#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image

from semantic_segmentation_ros.detection import SemanticSegmentation
from semantic_segmentation_ros.rviz import Visualizer

class SemanticSegmentationServer:
    def __init__(self):
        self.load_parameters()
        self.init_pubsub()
        
        self.ss_server = SemanticSegmentation(self.model_name, self.encoder_name, self.encoder_weights, self.in_channels, self.classes, self.model_path)
        self.vis = Visualizer(self.classes)
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.loginfo("Semantic Segmentation Server is ready")
    
    def load_parameters(self):
        self.color_topic = rospy.get_param("~camera/color_topic")
        self.model_name = rospy.get_param("~model/model_name")
        self.encoder_name = rospy.get_param("~model/encoder_name")
        self.encoder_weights = rospy.get_param("~model/encoder_weights")
        self.in_channels = rospy.get_param("~model/in_channels")
        self.classes = rospy.get_param("~model/classes")
        self.model_path = Path(rospy.get_param("~model/model_path"))

    def init_pubsub(self):
        self.segmentation_pub = rospy.Publisher("segmentation_image", Image, queue_size=1)
        rospy.Subscriber(self.color_topic, Image, self.sensor_cb)

    def sensor_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        y_pred = self.ss_server.predict(img)
        seg_img = self.vis.pred_to_image(y_pred)
        seg_msg = self.cv_bridge.cv2_to_imgmsg(seg_img, "bgr8")
        self.segmentation_pub.publish(seg_msg)


if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_server")
    server = SemanticSegmentationServer()
    rospy.spin()