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
        self.model_path = Path(rospy.get_param("~model/model_path"))
        self.encoder_name = rospy.get_param("~model/encoder_name")
        self.encoder_weights = rospy.get_param("~model/encoder_weights")
        self.in_channels = rospy.get_param("~model/in_channels")
        self.classes = rospy.get_param("~model/classes")
        rospy.loginfo(self.color_topic)
        rospy.loginfo(self.model_path)

    def init_topics(self):
        self.segmentation_pub = rospy.Publisher("segmentation_image", Image, queue_size=10)
        rospy.Subscriber(self.color_topic, Image, self.sensor_cb)

    def init_semantic_segmentation_server(self):
        self.ss_server =  SemanticSegmentation(self.model_path, self.encoder_name, self.encoder_weights, self.in_channels, self.classes)

    def sensor_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        y_pred = self.ss_server.predict(img)
        seg_img = self.pred_to_color(y_pred)
        # カラーのセグメンテーションマップをROSメッセージに変換して配信
        seg_msg = self.cv_bridge.cv2_to_imgmsg(seg_img, "bgr8")
        self.segmentation_pub.publish(seg_msg)

    def pred_to_color(self, y_pred):
        class_indices = np.argmax(y_pred, axis=0)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]])
        seg_img = np.zeros((class_indices.shape[0], class_indices.shape[1], 3), dtype=np.uint8)
        for class_id in range(y_pred.shape[0]):
            seg_img[class_indices == class_id] = colors[class_id % len(colors)]
        return seg_img



if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_server")
    server = SemanticSegmentationServer()
    rospy.spin()