
from __future__ import absolute_import, division, print_function
from typing import Callable, List, Tuple, Union, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import networkx as nx
from collections import OrderedDict

from models.base.basic_blocks import *
from models.base.blocks import *
from models.base.graph import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

"""
b   0   1   2  3   4
c: 32  64 128 256 512 encoder output
   32  64 128 256     decoder output 
w: 128 64 32  16  8 ** 2


"""


class Decoder(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.d4 = nn.Sequential(
            *[
                nn.PixelShuffle(2),
                ConvBlock(dim // 4, dim // 4, 3, 1, 1, g=4),
                ResBlock(dim//4),
                ConvBlock(dim // 4, dim // 2, 3, 1, 1, g=4),
                ResBlock(dim // 2)
            ]
        )
        self.pyramid_pool = SpatialPyramidPooling(pool_type='avg')

        dim3 = dim // 4  # input: 512, output 128
        self.d3 = self.make_block(dim3)

        dim2 = dim // 8  # input: 256, output 64
        self.d2 = self.make_block(dim2)
        self.att2 = ChannelAttention(8, 8)

        dim1 = dim // 16  # input: 128, output 32
        self.d1 = self.make_block(dim1)

        dim0 = dim // 32  # input: 64, output 16
        self.d0 = self.make_block(dim0)
        self.att0 = ChannelAttention(4, 16)

        self.out = nn.Sequential(
            *[
                ConvBlock(dim0, 3, 3, 1, 1),
                ResBlock(3),
                nn.Conv2d(3, 1, 1, bias=False, padding_mode='reflect'),
                nn.Sigmoid()
            ]
        )

    def make_block(self, in_dim: int):
        block = [
            nn.PixelShuffle(2),
            ConvBlock(in_dim, in_dim, 3, 1, 1),
            ResBlock(in_dim),
            ConvBlock(in_dim, in_dim, 3, 1, 1),
            ResBlock(in_dim)
        ]
        return nn.Sequential(*block)

    def forward(self, x: List[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        e0, e1, e2, e3, e4 = x
        d4 = self.d4(e4)

        d3 = self.d3(torch.cat([e3, d4], dim=1))  # output: 128

        d2 = self.d2(torch.cat([e2, d3], dim=1))  # 64
        d2 = self.att2(d2, ref)

        d1 = self.d1(torch.cat([e1, d2], dim=1))  # 32

        d0 = self.d0(torch.cat([e0, d1], dim=1))  # 16
        d0 = self.att0(d0, ref)

        out = self.out(d0)
        return out


class PoseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        squeezer = [
            nn.LazyConv2d(128, 3),
            nn.ReLU(inplace=True),
        ]

        convs = [
            ConvBlock(256, 256, 3, 1, 1),
            ResBlock(256),
            ConvBlock(256, 256, 3, 1, 1),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.ReLU(inplace=True),
        ]
        linear = [
            nn.LazyLinear(256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16, bias=False),
        ]

        self.squeezer = nn.Sequential(*squeezer)
        self.conv = nn.Sequential(*convs)
        self.linear = nn.Sequential(*linear)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        squeeze = list(map(lambda f: self.squeezer(f), x))

        feature = torch.cat(squeeze, dim=1)

        feature = self.conv(feature).flatten(2, 3)

        out = self.linear(feature).view(-1, 4, 4)
        return out
