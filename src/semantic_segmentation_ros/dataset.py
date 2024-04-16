import os
import random
import cv2
import json
import torch
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import hflip, vflip, rotate

from semantic_segmentation_ros.utils import vis_img_mask

class SegDataset(Dataset):
    def __init__(self, path, labels, augmentations):
        self.labels = labels
        self.img_path = os.path.join(path, "img")
        self.ann_path = os.path.join(path, "ann")
        self.imgs = list(sorted(os.listdir(self.img_path)))

        self.augmentations = augmentations

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.imgs[idx])
        mask_path = os.path.join(self.ann_path, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))

        img = torch.tensor(get_image(img_path), dtype=torch.float32)
        mask = torch.tensor(get_mask(img, mask_path, self.labels), dtype=torch.float32)  # [num_classes, height, width]

        if self.augmentations["enabled"]:
            aug_img, aug_mask = apply_transforms(img, mask, self.augmentations)
            vis_img_mask(img, mask, aug_img, aug_mask)
        return aug_img, aug_mask
    
def apply_transforms(img, mask, augmentations):
    # Horizontal flip 
    if random.random() > augmentations["flip_ud"]:
        img = hflip(img)
        mask = hflip(mask)
    
    # Vertical flip 
    if random.random() > augmentations["flip_lr"]:
        img = vflip(img)
        mask = vflip(mask)

    # Rotate randomly 
    angle = random.uniform(augmentations["rotate"][0], augmentations["rotate"][1])
    img = rotate(img, angle)
    mask = rotate(mask, angle, fill=0)  # Use fill=0 for other mask channels
    # Update the background channel specifically if rotation creates empty space
    background = (mask[-1, :, :] == 0).float()  # Update background channel with empty spaces
    mask[-1, :, :] = background

    return img, mask

    
def get_image(img_path):
    img = cv2.imread(img_path).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1))
     

def get_mask(img, mask_path, labels):
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

    for label in labels:
        
        blank = np.zeros(shape=(height, width), dtype=np.float32)
        
        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], 1)
            cv2.fillPoly(background, [label2poly[label]], 255)
            
        channels.append(blank)
    _, thresh = cv2.threshold(background, 127, 1, cv2.THRESH_BINARY_INV)
    channels.append(thresh)

    Y = np.stack(channels, axis=0)
    return Y