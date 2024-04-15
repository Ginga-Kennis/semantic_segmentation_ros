import os
import cv2
import json
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from torch.utils.data import Dataset

class SegDataset(Dataset):
    def __init__(self, path, labels, augmentations):
        self.labels = labels
        self.img_path = os.path.join(path, "img")
        self.ann_path = os.path.join(path, "ann")
        self.imgs = list(sorted(os.listdir(self.img_path)))

        self.augmentations = augmentations

        if self.augmentations['enabled']:
            self.seq = iaa.Sequential([
                iaa.Fliplr(self.augmentations['flip_lr']),  # Horizontal flips
                iaa.Flipud(self.augmentations['flip_ud']),  # Vertical flips
                iaa.Multiply(self.augmentations['brightness_multiply']),  # Change brightness
                iaa.Affine(rotate=self.augmentations['rotate']),  # Rotate
                iaa.Sometimes(
                    self.augmentations['apply_prob'],
                    iaa.GaussianBlur(sigma=self.augmentations['sigma'])
                )  # Apply Gaussian blur
            ], random_order=True)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_path, self.imgs[idx])
        mask_path = os.path.join(self.ann_path, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))

        img = get_image(img_path)
        mask = get_mask(img, mask_path, self.labels)

        if self.augmentations['enabled']:
            img, mask = apply_augment(img, mask, self.seq)

        return img, mask
    
def apply_augment(img, mask, seq):
    ia.seed(1)
    seq_det = seq.to_deterministic()  
    img_aug = np.transpose(img, (1, 2, 0))  # Convert (channel, height, width) to (height, width, channel) 
    img_aug = seq_det.augment_image(img_aug)
    mask_aug = np.array([seq_det.augment_image(channel) for channel in mask])
    img_aug = np.transpose(img_aug, (2, 0, 1))  # Convert (height, width, channel) to (channel, height, width) 
    return img_aug, mask_aug
    
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