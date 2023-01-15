
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


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        k: int,
        s: int = 1,
        p: int = 0,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        layer = [
            nn.Conv2d(
                inp,
                outp,
                k,
                s,
                p,
                groups=g,
                padding_mode="reflect",
                bias=False,
            ),
        ]
        layer += [nn.GroupNorm(1, outp)]  # layernorm
        if act:
            layer += [nn.LeakyReLU(0.2, inplace=True)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class GraphConv(Module):
    def __init__(self, in_features, n_channel, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = n_channel
        self.weight = nn.Linear(in_features, n_channel)

    def forward(self, channel_conn, img_feature):
        # feature * weight
        support = self.weight(img_feature)
        output = torch.matmul(channel_conn, support)

        return output


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


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
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


class AdjacencyMatrix(nn.Module):
    def __init__(self):
        super(AdjacencyMatrix, self).__init__()

    def forward(self, patch_features):
        patch_features = F.normalize(patch_features)
        norm = patch_features.norm()

        dot = patch_features @ patch_features.transpose(1, 2)
        b, c, r = dot.size()
        dot /= norm
        dot = dot + torch.eye(c)
        return dot
