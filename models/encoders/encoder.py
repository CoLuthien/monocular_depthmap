
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


class Encoder(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()

        self.e0 = self.make_block(in_dim, dim)
        self.e1 = self.make_block(dim, dim * 2)
        self.e2 = self.make_block(dim * 2, dim * 4)
        self.e3 = self.make_block(dim * 4, dim * 8)
        self.e4 = self.make_block(dim * 8, dim * 16)

    def make_block(self, in_dim: int, dim: int) -> List[nn.Module]:
        block = [
            ConvBlock(in_dim, dim // 2, 4, 2, 1),
            ResBlock(dim // 2),
            ConvBlock(dim // 2, dim, 3, 1, 1),
            ResBlock(dim),
        ]
        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        return [e0, e1, e2, e3, e4]
