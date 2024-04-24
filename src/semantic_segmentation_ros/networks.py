import torch
import segmentation_models_pytorch as smp

def get_model(model_name, encoder_name, encoder_weights, in_channels, classes):
    models = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus
    }

    if model_name not in models:
        raise ValueError(f"Unknown model type: {model}")
    
    model = models[model_name]
    return model(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)

def load_model(model_name, encoder_name, encoder_weights, in_channels, classes, path, device):
    model = get_model(model_name, encoder_name, encoder_weights, in_channels, classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model