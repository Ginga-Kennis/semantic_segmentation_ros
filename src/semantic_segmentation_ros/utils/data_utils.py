import json

import cv2
import numpy as np
import torch


def get_rgb_img_tensor(img_path: str) -> torch.tensor:
    """
    Read an image from the given path and convert it from BGR to RGB format.

    Args:
    img_path (str): The file path to the image.

    Returns:
    torch.Tensor: The image data as a torch tensor in RGB format (Channel, Height, Width).
    """
    img = cv2.imread(img_path).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return torch.tensor(np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1)), dtype=torch.float32)


def get_bgr_img_tensor(img_path: str) -> torch.tensor:
    """
    Read an image from the given path and return it in BGR format.

    Input:
    img_path (str): The file path to the image.

    Output:
    torch.Tensor: The image data as a torch tensor in BGR format (Channel, Height, Width).
    """
    img = cv2.imread(img_path).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)


def get_labelme_mask_tensor(mask_path: str, labels: list) -> torch.tensor:
    """
    Generate segmentation masks from a LabelMe JSON file for specified labels.

    Input:
    mask_path (str): The file path to the LabelMe JSON file.
    labels (list): A list of string labels for which masks are to be created.

    Output:
    torch.Tensor: A torch tensor of masks where each channel corresponds to one label (Class, Height, Width).
    """
    with open(mask_path) as handle:
        data = json.load(handle)
    shape_dicts = data["shapes"]

    channels = []

    # dictionary where key:label, value:list of points
    label2poly = {}
    for x in shape_dicts:
        if x["label"] not in label2poly:
            label2poly[x["label"]] = []
        label2poly[x["label"]].append(np.array(x["points"], dtype=np.int32))

    # define background
    height = data["imageHeight"]
    width = data["imageWidth"]

    for label in labels:
        blank = np.zeros(shape=(height, width), dtype=np.float32)

        if label in label2poly:
            for poly_points in label2poly[label]:
                cv2.fillPoly(blank, [poly_points], 1)

        channels.append(blank)

    y = np.stack(channels, axis=0)
    return torch.tensor(y, dtype=torch.float32)


def get_coco_mask():
    pass
