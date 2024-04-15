import cv2
import numpy as np

class Visualizer:
    def __init__(self,classes):
        self.classes = classes - 1 # Exclude background
        self.colors = self.generate_colors()

    def generate_colors(self):
        hues = np.linspace(0, 180, self.classes, endpoint=False, dtype=np.uint8)
        saturation = 255 * np.ones_like(hues, dtype=np.uint8)
        value = 255 * np.ones_like(hues, dtype=np.uint8)
        hsv_colors = np.stack([hues, saturation, value], axis=-1)
        rgb_colors = cv2.cvtColor(hsv_colors.reshape(1, -1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return np.vstack([rgb_colors, np.array([0, 0, 0], dtype=np.uint8)])

    def pred_to_image(self, y_pred):
        class_indices = np.argmax(y_pred, axis=0)
        seg_img = self.colors[class_indices]
        return seg_img
    

