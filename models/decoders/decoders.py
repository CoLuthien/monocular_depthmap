
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
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

"""
b  0   1    2   3 4
c: 32 64 128 256 512
   32 64 128 256
w: 128 64 32 16 8 ** 2
adj : (32, 64, 128, 256, 512 ) ** 2

"""


class Decoder(nn.Module):
    def __init__(self, dim: int = 512, width: int = 16, height: int = 16) -> None:
        super().__init__()

        self.to_adjacency = AdjacencyMatrix()

        self.d4 = nn.Sequential(
            *[
                nn.PixelShuffle(2),
                ConvBlock(dim // 4, dim // 4, 3, 1, 1),
                ResBlock(dim//4),
                ConvBlock(dim // 4, dim // 2, 3, 1, 1),
                ResBlock(dim // 2)
            ]
        )

        dim3 = dim // 4
        self.d3 = self.make_block(dim3)
        dim2 = dim // 8
        self.d2 = self.make_block(dim2)
        dim1 = dim // 16
        self.d1 = self.make_block(dim1)
        dim0 = dim // 32
        self.d0 = self.make_block(dim0)

        # feature similarity adjacency matrix * patch relation * weight
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=height, p2=width),
            nn.Linear(3 * height * width, 64),
        )
        self.gc3 = GraphConv(64, 64)

        self.out = nn.Sequential(
            *[
                ConvBlock(dim0, 3, 3, 1, 1),
                ResBlock(3),
                ConvBlock(3, 1, 3, 1, 1)
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

        d2 = self.d2(torch.cat([e2, d3], dim=1)) # 64

        d1 = self.d1(torch.cat([e1, d2], dim=1)) # 32

        d0 = self.d0(torch.cat([e0, d1], dim=1)) # 16
        out = self.out(d0)
        return out
