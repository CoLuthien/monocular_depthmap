
import numpy as np

import torch
import torch.nn as nn

from models.base.basic_blocks import *
from models.base.blocks import *


class ChannelAttention(nn.Module):
    def __init__(self, patch_per_direction: int, n_label: int, pooling_level=(1, 2, 4)) -> None:
        super().__init__()

        self.pooling = SpatialPyramidPooling('avg', pooling_level)
        self.adj = FeatureConnection(patch_per_direction, n_label)
        self.out_channel = patch_per_direction ** 2
        self.gc = GCNet(21, 128, self.out_channel)

    def forward(self, latent_vector: torch.Tensor, image_patch: torch.Tensor) -> torch.Tensor:
        adjacency = self.adj(image_patch)  # b x 128 x 128
        pooled_feature = self.pooling(latent_vector)  # b x 128 x 21
        attention = self.gc(adjacency, pooled_feature)

        return torch.einsum('bchw,bxc-> bchw', latent_vector, attention)
