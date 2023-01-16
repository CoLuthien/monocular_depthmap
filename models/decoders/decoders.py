
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

"""
b
c: 32 64 128 256 512
w: 128 64 32 16 8 ** 2
adj : b x 1 x (32, 64, 128, 256, 512 ) ** 2

"""


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.spatial_pool = SpatialPyramidPooling()
        self.to_adjacency = AdjacencyMatrix()

    def forward(self, x: List[torch.Tensor]):
        for f in x:
            pool = self.spatial_pool(f)
            adj = self.to_adjacency(pool)
            print (pool.shape, adj.shape)
