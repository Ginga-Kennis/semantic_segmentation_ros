#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.save_path = 'img'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.count = 0

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            if self.count % 30*5 == 0:  # 30fpsの場合、1秒に1枚保存
                img_filename = os.path.join(self.save_path, f'image_{rospy.get_time()}.jpeg')
                cv2.imwrite(img_filename, cv_image)
                rospy.loginfo('Saved image: %s', img_filename)
            self.count += 1

if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)
    ic = ImageSaver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
