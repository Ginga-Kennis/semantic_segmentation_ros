import os
import cv2
import json
import torch
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from torchvision.transforms import v2

class SegDataset(Dataset):
    def __init__(self, path, labels, augmentations):
        self.labels = labels
        self.img_path = os.path.join(path, "img")
        self.ann_path = os.path.join(path, "ann")
        self.imgs = list(sorted(os.listdir(self.img_path)))

        self.augment = augmentations["enabled"]
        self.img_transform, self.mask_transform = build_transforms(augmentations)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_path, self.imgs[idx])
        mask_path = os.path.join(self.ann_path, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))

        img = torch.tensor(get_image(img_path))
        mask = torch.tensor(get_mask(img, mask_path, self.labels))

        if self.augment == True:
            img, mask = apply_augmentation(img, mask, self.img_transform, self.mask_transform)

        return img, mask
    
def build_transforms(augmentations):
    img_transform_list = []
    mask_transform_list = []

    if augmentations['flip_lr'] > 0:
        img_transform_list.append(v2.RandomHorizontalFlip(p=augmentations['flip_lr']))
        mask_transform_list.append(v2.RandomHorizontalFlip(p=augmentations['flip_lr']))
    if augmentations['flip_ud'] > 0:
        img_transform_list.append(v2.RandomVerticalFlip(p=augmentations['flip_ud']))
        mask_transform_list.append(v2.RandomVerticalFlip(p=augmentations['flip_ud']))
    if augmentations['rotate'] != 0:
        img_transform_list.append(v2.RandomRotation(degrees=augmentations['rotate']))
        mask_transform_list.append(v2.RandomRotation(degrees=augmentations['rotate']))
    if augmentations['brightness_multiply'] is not None:
        img_transform_list.append(v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5))  # Adjust sharpness as a proxy
        img_transform_list.append(v2.ColorJitter(brightness=augmentations['brightness_multiply']))
    if augmentations['sigma'][0] != 0:
        img_transform_list.append(v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=augmentations['sigma'])], p=augmentations['apply_prob']))

    return v2.Compose(img_transform_list), v2.Compose(mask_transform_list)

def apply_augmentation(img, mask, img_transform, mask_transform):
    img_transformed = img_transform(img)
    mask_transformed = mask_transform(mask)  # マスクは整数型で保持することが必要な場合が多いため、変換後に型を調整するかもしれません。
    return img_transformed, mask_transformed
    
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