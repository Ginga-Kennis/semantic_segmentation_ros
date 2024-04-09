import numpy as np
import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

class MeanIoU(Metric):
    def __init__(self, num_classes, output_transform=lambda x: x, device="cpu"):
        self.num_classes = num_classes
        self.ious = None
        super(MeanIoU, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.ious = []

    # per batch
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        iou = calc_iou(y_pred, y, self.num_classes)
        self.ious.append(iou)

    # per epoch
    def compute(self):
        if len(self.ious) == 0:
            raise NotComputableError("MeanIoU must have at least one example before it can be computed.")
        return np.nanmean(self.ious)



def calc_iou(y_pred, y, num_classes):
    y_pred = torch.argmax(y_pred, dim=1)
    y = torch.argmax(y, dim=1)

    ious = []
    for cls in range(num_classes):
        true_positive = ((y_pred == cls) & (y == cls)).sum().item()
        false_positive = ((y_pred == cls) & (y != cls)).sum().item()
        false_negative = ((y_pred != cls) & (y == cls)).sum().item()

        intersection = true_positive
        union = true_positive + false_positive + false_negative
        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union
        ious.append(iou)
    return np.nanmean(ious)

