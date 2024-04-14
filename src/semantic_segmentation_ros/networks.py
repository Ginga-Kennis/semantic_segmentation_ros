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

def create_model(model_name, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=5):
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    return model

def get_model(model_name, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=5):
    return create_model(model_name, encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)

def load_model(path, device):
    model_name = path.stem.split("_")[0]
    net = get_model(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net