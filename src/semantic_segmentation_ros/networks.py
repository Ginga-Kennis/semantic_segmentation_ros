import segmentation_models_pytorch as smp
import torch


def get_model(model_name: str, encoder_name: str, encoder_weights: str, in_channels: int, classes: int) -> torch.nn.Module:
    """
    Retrieve a segmentation model with the specified parameters.

    Input:
        model_name (str): Name of the model type (e.g., "unet", "unetplusplus", "deeplabv3", "deeplabv3plus").
        encoder_name (str): Name of the model encoder (e.g., "resnet50").
        encoder_weights (str): Pre-trained weights for the encoder (e.g., "imagenet").
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        classes (int): Number of classes for the output segmentation mask.

    Returns:
        torch.nn.Module: An instance of the specified segmentation model.
    """
    models = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "deeplabv3": smp.DeepLabV3, "deeplabv3plus": smp.DeepLabV3Plus}

    if model_name not in models:
        raise ValueError(f"Unknown model type: {model_name}")

    model = models[model_name]
    return model(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)


def load_model(model_name: str, encoder_name: str, encoder_weights: str, in_channels: int, classes: int, path: str, device: str) -> torch.nn.Module:
    """
    Load a model from a specified path and move it to a given device.

    Input:
        model_name (str): Name of the model type.
        encoder_name (str): Name of the model encoder.
        encoder_weights (str): Pre-trained weights for the encoder.
        in_channels (int): Number of input channels.
        classes (int): Number of classes for the output segmentation mask.
        path (str): Path to the saved model weights file.
        device (str): Device to load the model onto ("cpu" or "cuda").

    Returns:
        torch.nn.Module: The loaded model, moved to the specified device.
    """
    model = get_model(model_name, encoder_name, encoder_weights, in_channels, classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
