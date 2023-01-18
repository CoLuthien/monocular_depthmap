
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp

from torchvision.transforms import Resize
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class GraphConv(Module):
    def __init__(self, inp, outp, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = inp
        self.out_features = outp
        self.weight = nn.Linear(inp, outp, bias=bias)

    def forward(self, channel_conn: torch.Tensor, img_feature: torch.Tensor):
        # input: (b, c, c), (b, c, inp)
        # out : (b, c, outp)
        support = self.weight(img_feature)
        output = channel_conn @ support

        return F.softmax(output, dim=1)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_type='max', levels=(1, 2, 4)):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_type = pool_type
        self.levels = levels
        self.poolers = []
        for level in levels:
            if pool_type == 'max':
                self.poolers.append(nn.AdaptiveMaxPool2d(
                    output_size=(level, level)))
            elif pool_type == 'avg':
                self.poolers.append(nn.AdaptiveAvgPool2d(
                    output_size=(level, level)))
            else:
                raise ValueError('Invalid pool_type: {}'.format(pool_type))

    def forward(self, x):
        b, c, h, w = x.size()
        out = []
        for pooler in self.poolers:
            out.append(pooler(x).view(b, c, -1))
        out = torch.cat(out, dim=2)
        return out


class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# class AdjacencyMatrix(nn.Module):
    # def __init__(self):
        #super(AdjacencyMatrix, self).__init__()
        #self.spatial_pool = SpatialPyramidPooling()

    # def normalize(self, x: torch.Tensor):
        #rinv = x.sum(dim=0).float_power(-1)
        #rinv.nan_to_num(0, 0, 0)
        #rinv = rinv.diag().to(dtype=torch.float32)
        #x = torch.einsum('bij,i->bij', x, rinv)
        # return x.to(dtype=torch.float32)

    # def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        # """
        # input: b x c x f, 3d tensor
        # output: b x c x c 3d tensor
        # """
        #feature_a = self.spatial_pool(feature_a)
        #feature_b = self.spatial_pool(feature_b)
        #dot = torch.einsum('bij,bkj -> bik', feature_a, feature_b)
        #b, c, r = dot.size()
        #dot = dot + torch.eye(c).to(dot.device)
        # return self.normalize(dot)

class AdjacencyMatrix(nn.Module):
    def __init__(self, n_bins):
        super(AdjacencyMatrix, self).__init__()
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor):
        # Bin the values of the max pooled feature map
        b, c, h, w = x.size()
        bin = x.flatten().view(b, 1, -1)
        y = x.flatten().view(b, -1, 1)

        adj = torch.isclose(bin, y, 1e-1).float()

        return adj


class FeatureConnection(nn.Module):
    def __init__(self, patch_per_direction: int, feature_count: int, bin_count: int = 32, resize_to: int = 256) -> None:
        super().__init__()
        patch_size = resize_to // patch_per_direction

        # p ** 2 patch, f feature per patch => (p**2) * f object
        self.block = nn.Sequential(
            Resize(resize_to),
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(3 * patch_size ** 2, feature_count),
            Rearrange('b h w c -> b c h w'),
            AdjacencyMatrix(bin_count)
        )

    def forward(self, batched_img: torch.Tensor) -> torch.Tensor:
        """
        input: (b, c, h, w) batched tensor image
        output: (b, 1, patch_count**2 * feature_count,patch_count**2 * feature_count)
        """

        return self.block(batched_img)
