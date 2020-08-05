"""
This module defines some utility models, such as image decoders, vaes and
distance readers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Conv

class InterpolateDecoder(nn.Module):
    """
    Vanilla decoder for images.

    Uses the binary decomposition of the final size to decide on the final
    number of layers of the net.
    """
    def __init__(self, in_ch, inter_ch, out_ch, out_size):
        super().__init__()

        f_in = in_ch
        f_out = inter_ch
        binary = format(out_size)[:1]

        self.layers = []
        
        for i, bit in enumerate(binary):
            if bit:
                # TODO: add *2 interpolation
                # TODO: add 1-padding
                layers.append(nn.Conv2d(f_in, f_out, 3, 1))
            else:
                # TODO: add *2 interpolation
                layers.append(nn.Conv2d(f_in, f_out, 3, 1))
            
            layers.append(nn.ReLU())

            f_in = f_inter
            if i == len(binary) - 2:
                f_out = out_ch

        layers.pop(-1) # remove last relu

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SpatialBroadcastDecoder(nn.Module):
    """
    Spatial Broadcast Decoder for images.

    Same number of layers as InterpolateDecoder.
    """
    def __init__(self, in_ch, inter_ch, out_ch, out_size):
        super().__init__()

        f_in = in_ch
        f_out = inter_ch
        binary = format(out_size)[:1]

        self.out_size = out_size

        self.layers = []

        for i, _ in enumerate(range(len(binary))):
            layers.append(nn.Conv2d(f_in, f_out, 3, 1))
            layers.append(nn.ReLU())

            f_in = f_inter
            if i == len(binary) - 2:
                f_out = out_ch

        layers.pop(-1) # remove last relu

        self.net = nn.Sequential(*layers)

        # define grid
        a = torch.linspace(-1, 1, out_size)
        grid = torch.stack(torch.meshgrid(a, a), 0)
        self.register_buffer('grid', grid)

    def forward(self, x):
        # x is vector
        b, f = x.shape
        x = x.expand(b, f, self.out_size, self.out_size)
        # concatenate x-y grid with broadcast input vector
        x = torch.cat(
            [x, self.grid.expand(b, 2, self.out_size, self.out_size)],
            1)
        
        return self.net(x)