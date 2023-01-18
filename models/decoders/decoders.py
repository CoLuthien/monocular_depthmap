
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
                ConvBlock(dim // 4, dim // 4, 3, 1, 1),
                ResBlock(dim//4),
                ConvBlock(dim // 4, dim // 2, 3, 1, 1),
                ResBlock(dim // 2)
            ]
        )
        self.pyramid_pool = SpatialPyramidPooling()

        dim3 = dim // 4  # input: 512, output 128
        self.d3 = self.make_block(dim3)
        self.a3 = FeatureConnection(4, 8)  # b x 128 x 128
        self.gc3 = GraphConv(21, 1)

        dim2 = dim // 8  # input: 256, output 64
        self.d2 = self.make_block(dim2)

        dim1 = dim // 16  # input: 128, output 32
        self.d1 = self.make_block(dim1)
        self.a1 = FeatureConnection(2, 8)
        self.gc1 = GraphConv(21, 1)

        dim0 = dim // 32  # input: 64, output 16
        self.d0 = self.make_block(dim0)

        self.out = nn.Sequential(
            *[
                ConvBlock(dim0, 3, 3, 1, 1),
                ResBlock(3),
                ConvBlock(3, 1, 3, 1, 1, act=False),
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

        a1 = self.a1(ref)

        d3 = self.d3(torch.cat([e3, d4], dim=1))  # output: 128
        a3 = self.a3(ref)  # b x 128 x 128
        p3 = self.pyramid_pool(d3)  # b x 128 x 21
        f3 = self.gc3(a3, p3)
        d3 = torch.einsum('bchw,bcx-> bchw', d3, f3)

        d2 = self.d2(torch.cat([e2, d3], dim=1))  # 64

        d1 = self.d1(torch.cat([e1, d2], dim=1))  # 32
        a1 = self.a1(ref)  # b x 32 x 32
        p1 = self.pyramid_pool(d1)  # b x 128 x 21
        f1 = self.gc1(a1, p1)
        d1 = torch.einsum('bchw,bcx-> bchw', d1, f1)

        d0 = self.d0(torch.cat([e0, d1], dim=1))  # 16
        out = self.out(d0)
        return out
