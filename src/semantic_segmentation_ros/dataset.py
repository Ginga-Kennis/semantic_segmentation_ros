import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset

LABELS = ['circle', 'square', 'star', 'triangle']

class SegDataset(Dataset):
    def __init__(self, dir):
        self.img_dir = os.path.join(dir, "img")
        self.ann_dir = os.path.join(dir, "ann")
        self.imgs = list(sorted(os.listdir(self.img_dir)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.ann_dir, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))

        img = get_image(img_path)
        mask = get_mask(img, mask_path)

        return img, mask
    
def get_image(img_path):
    img = cv2.imread(img_path).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1))
     

def get_mask(img, mask_path):
    with open(mask_path) as handle:
        data = json.load(handle)
    shape_dicts = data["shapes"]

    channels = []

    # class in mask
    cls = [x['label'] for x in shape_dicts]
    
    # points of each class
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]

    # dictionary where key:label, value:points
    label2poly = dict(zip(cls, poly))

    # define background
    _, height, width = img.shape
    background = np.zeros(shape=(height, width), dtype=np.float32)

    for label in LABELS:
        
        blank = np.zeros(shape=(height, width), dtype=np.float32)
        
        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], 255)
            cv2.fillPoly(background, [label2poly[label]], 255)
            
        channels.append(blank)
    _, thresh = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
    channels.append(thresh)

    Y = np.stack(channels, axis=0)
    return Y