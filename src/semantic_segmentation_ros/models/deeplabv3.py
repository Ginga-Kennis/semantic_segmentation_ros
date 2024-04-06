import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet101

class DeeplabV3(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.model = deeplabv3_resnet101(weights=None, num_classes=5)

    def forward(self,x):
        return self.model(x)["out"]