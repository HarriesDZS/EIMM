import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = input_flat * target_flat

        loss = (2*intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        loss = 1 - loss

        return loss