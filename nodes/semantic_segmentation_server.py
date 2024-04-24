#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
import torch
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image

from semantic_segmentation_ros.detection import SemanticSegmentation
from semantic_segmentation_ros.rviz import Visualizer
from semantic_segmentation_ros.srv import GetSegmentedImage, GetSegmentedImageResponse

class SemanticSegmentationServer:
    def __init__(self):
        self.load_parameters()
        self.init_pubsub()
        self.init_services()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_bridge = cv_bridge.CvBridge()
        self.segmentation_model = SemanticSegmentation(self.model_name, self.encoder_name, self.encoder_weights, self.in_channels, self.classes, self.model_path, self.device)
        self.vis = Visualizer(self.classes)

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
        self.segmentation_mask_pub = rospy.Publisher("segmentation_mask", Image, queue_size=1)
        rospy.Subscriber(self.color_topic, Image, self.rgb_image_callback)

    def init_services(self):
        rospy.Service("get_segmentation_image", GetSegmentedImage, self.get_segmentation_image)

    def rgb_image_callback(self, msg):
        try:
            mask_pred = self.segmentation_model.predict(self.cv_bridge.imgmsg_to_cv2(msg, "rgb8").transpose(2,0,1).astype(np.float32))

            # publish mask
            self.latest_mask_pred = self.cv_bridge.cv2_to_imgmsg(mask_pred.astype(np.uint8), "mono8")
            self.segmentation_mask_pub.publish(self.latest_mask_pred)

            # publish image
            self.vis.publish_segmented_image(mask_pred) 
            
        except Exception as e:
            rospy.logerr(f"Failed to Process Image : {e}")

    def get_segmentation_image(self, req):
        if self.latest_mask_pred is None:
            rospy.logwarn("No Segmented Image Available")
            return GetSegmentedImageResponse()
        return GetSegmentedImageResponse(self.latest_mask_pred)


if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_server")
    server = SemanticSegmentationServer()
    rospy.spin()