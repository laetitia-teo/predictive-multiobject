"""
This module defines some utility models, such as image decoders, vaes and
distance readers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models import Conv

### Misc

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        lin = layer_sizes.pop(0)

        for i, ls in enumerate(layer_sizes):
            layers.append(nn.Linear(lin, ls))
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            ls = lin

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

### Encoders

class MaxpoolEncoder(nn.Module):
    """
    Simple CNN encoder with maxpooling.

    Use the image size to decide on the number of layers and the padding.
    """
    def __init__(self, in_ch, inter_ch, out_ch, in_size, stop=0):
        super().__init__()

        f_in = in_ch
        f_out = inter_ch
        binary = format(in_size, 'b')[1:]

        layers = []

        for i, bit in enumerate(binary):

            if i >= len(binary) - stop:
                break

            layers.append(
                nn.Conv2d(f_in, f_out, 3, padding=1, padding_mode='reflect'))
            
            layers.append(nn.MaxPool2d(2))
            if bit == '1':
                layers.append(nn.ReflectionPad2d((-1, 0, -1, 0)))

            layers.append(nn.ReLU())

            f_in = inter_ch
            if i == len(binary) - 2:
                f_out = out_ch

        # remove last relu
        layers.pop(-1)
        layers.append(nn.Flatten(1, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

### Decoders

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
        binary = format(out_size, 'b')[1:]

        layers = []
        
        for i, bit in enumerate(binary):

            layers.append(nn.Upsample(scale_factor=2))
            if bit == '1':
                # fast-exponentiation-like, grow size by 1
                layers.append(nn.ReflectionPad2d((1, 0, 1, 0)))

            layers.append(
                nn.Conv2d(f_in, f_out, 3, padding=1, padding_mode='reflect'))
            
            layers.append(nn.ReLU())

            f_in = inter_ch
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

        f_in = in_ch + 2 # add positions
        f_out = inter_ch
        binary = format(out_size, 'b')[1:]

        self.out_size = out_size

        layers = []

        for i, _ in enumerate(range(len(binary))):
            layers.append(
                nn.Conv2d(f_in, f_out, 3, padding=1, padding_mode='reflect'))
            layers.append(nn.ReLU())

            f_in = inter_ch
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
        b, f, _, _ = x.shape
        x = x.expand(b, f, self.out_size, self.out_size)
        # concatenate x-y grid with broadcast input vector
        x = torch.cat(
            [x, self.grid.expand(b, 2, self.out_size, self.out_size)],
            1)
        
        return self.net(x)

### full vaes

class VAE(nn.Module):
    """
    Base class for vaes.
    """
    def __init__(self, zdim, inter_ch, img_size, encoder_type, decoder_type):
        super().__init__()

        self.zdim = zdim

        self.encoder = encoder_type(3, inter_ch, zdim, img_size)
        self.decoder = decoder_type(zdim, inter_ch, 3, img_size)

        self.lin1 = nn.Linear(zdim, 2 * zdim)

        # TODO: maybe change this
        self.scale = 1.

    def forward(self, img):
        B, _, _, _ = img.shape

        # encode to normal
        enc = self.encoder(img)
        # vec = enc.flatten(1, 3) # not necessary ?
        # model the log-variance instead of std
        mus, logvar = self.lin1(enc).chunk(2, -1)

        # sample
        eps = torch.normal(torch.zeros(self.zdim), torch.ones(self.zdim))
        std = (0.5 * logvar).exp()
        z = mus + eps * std
        # create spatial dims
        z = z.expand(1, 1, B, self.zdim).permute(2, 3, 0, 1)

        decoded = self.decoder(z)

        # compute loss
        # scale for reconstruction ?
        reconstruction_loss = F.mse_loss(img, decoded) / self.scale
        kl_loss = 0.5 * (logvar.exp() - 1 + mus**2 - logvar).sum()

        loss = (reconstruction_loss + kl_loss) / B

        return decoded, loss

    def generate(self):
        """
        Generate an image from the latent N(0, 1) distrib
        """
        z = torch.normal(torch.zeros(self.zdim), torch.ones(self.zdim))
        return self.decoder(z)

class UpscaleVAE(VAE):
    """
    Classical Vae with upscaling.
    """
    def __init__(self, zdim, inter_ch, img_size):
        super().__init__(zdim, inter_ch, img_size,
                         MaxpoolEncoder, InterpolateDecoder)

class BroadcastVAE(VAE):
    """
    VAE with spatial broadcast decoder.
    """
    def __init__(self, zdim, inter_ch, img_size):
        super().__init__(zdim, inter_ch, img_size,
                         MaxpoolEncoder, SpatialBroadcastDecoder)

### Tests

img = torch.rand(10, 3, 50, 50)
conv = MaxpoolEncoder(3, 32, 32, 50)
vae1 = UpscaleVAE(32, 32, 50)
vae2 = BroadcastVAE(32, 32, 50)