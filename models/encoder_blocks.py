
from typing import Tuple, Dict, ByteString, List
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from models.basic_blocks import *


class EncoderBlock(nn.Module):
    def __init__(self, img_chan: int = 3, dim: int = 16) -> None:
        super().__init__()
        # input (N, 3, W, H)
        self.e0 = nn.Sequential(*[
            ConvBlock(img_chan, dim, 4, 2, 1),  # maintain
            ResBlock(dim),
        ])
        self.e1 = nn.Sequential(*[
            ConvBlock(dim, dim * 2, 4, 2, 1),  # /2
            ResBlock(dim * 2),
        ])
        self.e2 = nn.Sequential(*[
            ConvBlock(dim * 2, dim * 4, 4, 2, 1),  # /4
            ResBlock(dim * 4),
        ])
        self.e3 = nn.Sequential(*[
            ConvBlock(dim * 4, dim * 8, 4, 2, 1),  # / 2
            ResBlock(dim * 8),
        ])
        self.e4 = nn.Sequential(*[
            ConvBlock(dim * 8, dim * 16, 4, 2, 1),
            ResBlock(dim * 16)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s0 = self.e0(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        return [s0, s1, s2, s3, s4]


class ImageDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        """
        maps 3d point to 
        """

        layer = [
            ResBlock(3),
            ResBlock(3),
            ResBlock(3),
        ]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor, pose: torch.Tensor):
        return self.block(x)


