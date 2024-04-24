import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors

from semantic_segmentation_ros.utils import get_image, get_labelme_mask, vis_img_mask

class SegDataset(Dataset):
    """A dataset class for semantic segmentation which handles data loading, augmentations, and preprocessing.

    Inputs:
        path (str) - Base path to the dataset.
        labels (dict) - Class labels.
        augmentations (dict) - Dictionary specifying augmentation parameters.
        is_train (bool) - Specifies if the dataset is for training. Defaults to True.
    """

    def __init__(self, path: str, labels: list, augmentations: dict, is_train: bool = True):
        self.labels = labels
        self.img_path = os.path.join(path, "img")
        self.ann_path = os.path.join(path, "ann")
        self.imgs = list(sorted(os.listdir(self.img_path)))
        self.anns = list(sorted(os.listdir(self.ann_path)))

        self.augment = augmentations["enable"]
        self.trans = build_transform(augmentations)

        self.is_train = is_train

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple:
        """
        Fetches the image and mask tensors by index.

        Inputs: idx (int) - The index of the data item.
        Outputs: tuple - Tuple containing the image (Tensor) and mask (Tensor).
        """
        img_path = os.path.join(self.img_path, self.imgs[idx])
        mask_path = os.path.join(self.ann_path, self.anns[idx])

        img = torch.tensor(get_image(img_path), dtype=torch.float32)
        mask = torch.tensor(get_labelme_mask(img, mask_path, self.labels), dtype=torch.float32)  # [num_classes, height, width]

        if self.is_train == True and self.augment == True:
            img, mask = self.trans(tv_tensors.Image(img), tv_tensors.Mask(mask))

        mask = add_background(mask)

        return img, mask
    
def build_transform(augmentations: dict) -> transforms.Compose:
    """
    Builds a composed torchvision transformation based on augmentation settings.

    Inputs: augmentations (dict) - Augmentation settings as a dictionary.
    Outputs: transforms.Compose - A torchvision Compose object containing all the transformations.
    """
    transform_list = []

    if augmentations["affine"]["enable"]:
        transform_list.append(transforms.RandomAffine(degrees=augmentations["affine"]["rotate"], translate=augmentations["affine"]["translate"], scale=augmentations["affine"]["scale"], shear=augmentations["affine"]["shear"]))
    
    if augmentations["perspective"]["enable"]:
        transform_list.append(transforms.RandomPerspective(distortion_scale=augmentations["perspective"]["distortion_scale"], p=augmentations["perspective"]["p"]))
    
    if augmentations["elastic"]["enable"]:
        transform_list.append(transforms.ElasticTransform(alpha=augmentations["elastic"]["alpha"], sigma=augmentations["elastic"]["sigma"]))
    
    trans = transforms.Compose(transform_list)
    return trans
    
def add_background(mask: torch.Tensor) -> torch.Tensor:
    """
    Adds a background channel to the mask tensor.

    Inputs: mask (torch.Tensor) - The original mask tensor without the background channel.
    Outputs: torch.Tensor - Updated mask tensor with the background channel added.
    """
    background = torch.all(mask == 0, dim=0, keepdim=True)
    return torch.cat((mask, background), dim=0)
