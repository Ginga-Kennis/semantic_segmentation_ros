import rospy
import cv2
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image

class Visualizer:
    def __init__(self,classes):
        self.classes = classes - 1 # Exclude background
        self.cv_bridge = cv_bridge.CvBridge()
        self.colors = self.generate_colors()
        self.create_segmented_image_publisher()

    def generate_colors(self):
        hues = np.linspace(0, 180, self.classes, endpoint=False, dtype=np.uint8)
        saturation = 255 * np.ones_like(hues, dtype=np.uint8)
        value = 255 * np.ones_like(hues, dtype=np.uint8)
        hsv_colors = np.stack([hues, saturation, value], axis=-1)
        rgb_colors = cv2.cvtColor(hsv_colors.reshape(1, -1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return np.vstack([rgb_colors, np.array([0, 0, 0], dtype=np.uint8)])
    
    def create_segmented_image_publisher(self):
        self.segmented_image_pub = rospy.Publisher("segmentation_image", Image, queue_size=1)

    def publish_segmented_image(self, mask_pred):
        seg_img = self.colors[mask_pred]
        msg = self.cv_bridge.cv2_to_imgmsg(seg_img, "bgr8")
        self.segmented_image_pub.publish(msg)
    