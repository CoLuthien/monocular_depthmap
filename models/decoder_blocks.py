
from typing import Tuple, Dict, ByteString, List
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from models.basic_blocks import *


class DepthDecoder(nn.Module):
    def __init__(self, in_: int = 16) -> None:
        super().__init__()
        # (N, in_ * 16, W / 16, H / 16) : input
        dim = in_
        self.d0 = nn.Sequential(*[
            nn.PixelShuffle(2),
            ConvBlock(dim // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        ])
        self.d1 = nn.Sequential(*[
            nn.PixelShuffle(2),
            ConvBlock(dim, dim, 3, 1, 1),
            ResBlock(dim),
        ])
        self.d2 = nn.Sequential(*[
            nn.PixelShuffle(2),
            ConvBlock(dim * 2, dim * 2, 3, 1, 1),
            ResBlock(dim * 2),
        ])
        self.d3 = nn.Sequential(*[
            nn.PixelShuffle(2),
            ConvBlock(dim * 4, dim * 4, 3, 1, 1),
            ResBlock(dim * 4),
        ])
        self.d4 = nn.Sequential(*[
            nn.PixelShuffle(2),
            ConvBlock(dim * 4, dim * 8, 3, 1, 1),
            ResBlock(dim * 8)
        ])

        self.mods = [self.d0, self.d1, self.d2, self.d3, self.d4]

    def forward(self, x) -> torch.Tensor:
        s0 = x[0]
        s1 = x[1]
        s2 = x[2]
        s3 = x[3]
        s4 = x[4]
        u3 = self.d4(s4)
        u2 = self.d3(torch.cat([s3, u3], 1))
        u1 = self.d2(torch.cat([s2, u2], 1))
        u0 = self.d1(torch.cat([s1, u1], 1))
        o0 = self.d0(torch.cat([s0, u0], 1))
        return o0


class PoseDecoder(nn.Module):
    def __init__(self, in_features: int = 2) -> None:
        super().__init__()
        squeezer = [
            nn.LazyConv2d(128, 1),
            nn.Tanh()
        ]

        convs = [
            ResBlock(256),
            ResBlock(256),
            nn.Conv2d(256, 2, 3, 1, 1),
            nn.PReLU(2)
        ]
        linear = [
            nn.LazyLinear(128, bias=True),
            nn.LazyLinear(16)
        ]

        self.squeezer = nn.Sequential(*squeezer)
        self.conv = nn.Sequential(*convs)
        self.linear = nn.Sequential(*linear)
        self.in_features = in_features

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        squeeze = list(map(lambda f: self.squeezer(f), x))

        feature = torch.cat(squeeze, dim=1)
        b, c, _, _ = feature.size()

        feature = self.conv(feature).flatten(2, 3)

        out = self.linear(feature).view(b, 2, 4, 4)
        return out.split(1, dim=1)
