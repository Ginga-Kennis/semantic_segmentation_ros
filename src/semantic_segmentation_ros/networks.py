import torch
import segmentation_models_pytorch as smp

def create_model(model, encoder_name, encoder_weights, in_channels, classes):
    if model == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model}")
    return model

def get_model(model, encoder_name, encoder_weights, in_channels, classes):
    return create_model(model, encoder_name, encoder_weights, in_channels, classes)

def load_model(path, encoder_name, encoder_weights, in_channels, classes, device):
    model = path.stem.split("_")[0]
    net = get_model(model, encoder_name, encoder_weights, in_channels, classes).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net