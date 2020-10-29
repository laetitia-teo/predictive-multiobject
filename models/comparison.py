import torch
import torch.nn as nn

class PairwiseL2(nn.Module):

    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x, y):
        sqdiff = (x - y).pow(2) / (2 * self.scale**2)
        return sqdiff.sum(-1).mean(-1)
