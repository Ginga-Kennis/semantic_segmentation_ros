import torch
import segmentation_models_pytorch as smp

unet = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=5,                      # model output channels (number of classes in your dataset)
)

unetplusplus = smp.UnetPlusPlus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=5,                      
)

deeplabv3 = smp.DeepLabV3(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=5,                      
)

deeplabv3plus = smp.DeepLabV3Plus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=5,                      
)

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

def load_model(path, device):
    model = path.stem.split("_")[0]
    net = get_model(model).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net