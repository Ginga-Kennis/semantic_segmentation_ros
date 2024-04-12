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

def get_model(name):
    models = {
        "unet": unet,
        "unetplusplus" : unetplusplus,
        "deeplabv3" : deeplabv3,
        "deeplabv3plus" : deeplabv3plus
    }
    return models[name.lower()]

def load_model(path, device):
    model_name = path.stem.split("_")[0]
    net = get_model(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net