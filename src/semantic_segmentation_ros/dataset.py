import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset

labels = ['circle', 'square', 'star', 'triangle']

class SegDataset(Dataset):
    def __init__(self, dir):
        self.img_dir = os.path.join(dir, "img")
        self.ann_dir = os.path.join(dir, "ann")
        self.imgs = list(sorted(os.listdir(self.img_dir)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        shape_dicts = get_poly(ann_path)
        mask = create_multi_masks(img,shape_dicts)

        return img, mask
    
def get_poly(ann_path):
    with open(ann_path) as handle:
        data = json.load(handle)
    return data['shapes']

def create_multi_masks(im, shape_dicts):
    height, width, _ = im.shape
    channels = []
    cls = [x['label'] for x in shape_dicts]
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]
    label2poly = dict(zip(cls, poly))
    background = np.zeros(shape=(height, width), dtype=np.float32)

    for i, label in enumerate(labels):
        
        blank = np.zeros(shape=(height, width), dtype=np.float32)
        
        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], 255)
            cv2.fillPoly(background, [label2poly[label]], 255)
            
        channels.append(blank)
    _, thresh = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
    channels.append(thresh)

    Y = np.stack(channels, axis=0)
    return Y