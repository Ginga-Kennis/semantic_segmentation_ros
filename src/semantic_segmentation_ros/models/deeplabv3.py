import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3

class DeeplabV3(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.model = deeplabv3.deeplabv3_resnet101(pretrained=True)
        self.model.classifier = deeplabv3.DeepLabHead(2048, out_channels)

    def forward(self,x):
        return self.model(x)["out"]