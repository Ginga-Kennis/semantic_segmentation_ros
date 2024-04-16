import os
import random
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors

from semantic_segmentation_ros.utils import vis_img_mask

class SegDataset(Dataset):
    def __init__(self, path, labels, augmentations):
        self.labels = labels
        self.img_path = os.path.join(path, "img")
        self.ann_path = os.path.join(path, "ann")
        self.imgs = list(sorted(os.listdir(self.img_path)))

        self.augment = augmentations["enabled"]
        self.trans = build_transform(augmentations)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.imgs[idx])
        mask_path = os.path.join(self.ann_path, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))

        img = torch.tensor(get_image(img_path), dtype=torch.float32)
        mask = torch.tensor(get_mask(img, mask_path, self.labels), dtype=torch.float32)  # [num_classes, height, width]

        if self.augment == True:
            aug_img, aug_mask = self.trans(tv_tensors.Image(img), tv_tensors.Mask(mask))

            # TensorをNumPy配列に変換
            aug_mask = aug_mask.data.numpy()
            
            # 変換後のマスクに背景チャンネルを追加
            # 変換によって作成された余白は0であることを利用する
            # チャンネルが0のピクセルは背景として1で塗りつぶす
            background_channel = np.all(aug_mask == 0, axis=0).astype(np.float32)
            
            # 新しい背景チャンネルを追加
            aug_mask = np.concatenate((background_channel[None, :], aug_mask), axis=0)

            # Tensorに戻す
            aug_img = aug_img.data  # Tensorに戻す
            aug_mask = torch.tensor(aug_mask, dtype=torch.float32)
            # vis_img_mask(img, mask, aug_img, aug_mask)
        return aug_img, aug_mask
    
def build_transform(augmentations):
    trans = transforms.Compose([
        transforms.RandomRotation(augmentations["rotate"])
    ])
    return trans
    
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

    for label in labels:
        
        blank = np.zeros(shape=(height, width), dtype=np.float32)
        
        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], 1)
            
        channels.append(blank)

    Y = np.stack(channels, axis=0)
    return Y