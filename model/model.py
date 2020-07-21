"""
This module defines the different components of the model + their interaction.
"""

import torch
import torch.nn as nn

### perception modules

class SimpleConv_Perception(nn.Module):

    def __init__(self, K):
        
        super().__init__()
        self.K = K

        conv1 = nn.Conv2d(3, 32, 3)
        conv2 = nn.Conv2d(32, 32, 3)
        conv3 = nn.Conv2d(32, 32, 3)
        conv4 = nn.Conv2d(32, self.K, 3)

        self.net = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            conv4)

    def forward(self, x):
        return self.net(x)

### memory modules

class ParallelLSTM_Memory(nn.Module):

    def __init__(self, K):

        super().__init__()
        self.K = K

    def forward(self, x):
        return

class RMC_Memory(nn.Module):

    def __init__(self, K):

        super().__init__()
        self.K = K

    def forward(self, x):
        return

### distance modules

