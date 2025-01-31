import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class MeanIoU(Metric):
    def __init__(self, num_classes, output_transform=lambda x: x, device="cpu"):
        self.num_classes = num_classes
        self.device = device
        self.ious = torch.tensor([], device=self.device)
        super(MeanIoU, self).__init__(output_transform=output_transform, device=self.device)

    def reset(self):
        self.ious = torch.tensor([], device=self.device)

    # per batch
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        iou = calc_iou(y_pred, y, self.num_classes, self.device)
        self.ious = torch.cat((self.ious, iou.unsqueeze(0)), dim=0)

    # per epoch
    def compute(self):
        if self.ious.nelement() == 0:
            raise NotComputableError("MeanIoU must have at least one example before it can be computed.")
        return torch.nanmean(self.ious).item()


def calc_iou(y_pred, y, num_classes, device):
    y_pred = torch.argmax(y_pred, dim=1)
    y = torch.argmax(y, dim=1)

    ious = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        true_positive = ((y_pred == cls) & (y == cls)).float().sum(dim=[1, 2])
        false_positive = ((y_pred == cls) & (y != cls)).float().sum(dim=[1, 2])
        false_negative = ((y_pred != cls) & (y == cls)).float().sum(dim=[1, 2])

        intersection = true_positive
        union = true_positive + false_positive + false_negative
        iou = torch.where(union > 0, intersection / union, torch.tensor(float("nan"), device=device))
        ious[cls] = torch.nanmean(iou)

    return torch.nanmean(ious)
