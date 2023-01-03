

from typing import Tuple, Dict, ByteString
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = 1,
                 activation: bool = True,
                 ) -> None:
        super().__init__()
        layer = [
            nn.Conv2d(
                in_channel, out_channel,
                kernel_size=kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False),
            nn.GroupNorm(1, out_channel)  # Equivalent to LayerNorm
        ]
        if (activation):
            layer += [nn.PReLU(out_channel, 0.2)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        layer = [
            ConvBlock(in_channel, in_channel, 3, 1, 1, activation=True),
            ConvBlock(in_channel, in_channel, 3, 1, 1, activation=False),
            nn.BatchNorm2d(in_channel)
        ]
        self.block = nn.Sequential(*layer)
        self.activation = nn.PReLU(in_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        return self.activation(skip + self.block(x))
