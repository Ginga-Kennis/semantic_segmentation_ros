import torch

from semantic_segmentation_ros.models.unet import Unet
from semantic_segmentation_ros.models.deeplabv3 import DeeplabV3

def get_model(name):
    models = {
        "unet": Unet(3,5),
        "deeplabv3" : DeeplabV3(5)
    }
    return models[name.lower()]

def load_network(path, device):
    model_name = "unet"
    net = get_model(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net

def load_model(path, device):
    model_name = path.stem.split("_")[0]
    net = get_model(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net