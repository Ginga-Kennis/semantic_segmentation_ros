import torch

from semantic_segmentation_ros.models.unet import Unet

def get_network(name):
    models = {
        "unet": Unet(3,5),
    }
    return models[name.lower()]

def load_network(path, device):
    model_name = "unet"
    net = get_network(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net