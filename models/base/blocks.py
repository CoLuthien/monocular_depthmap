
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
import einops
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
        return output


class GCNet(nn.Module):
    def __init__(self, nfeat: int, nhid: int, n_out: int):
        super(GCNet, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, n_out)
        self.dropout = nn.Dropout()

    def forward(self, adj, x):
        x = F.relu(self.gc1(adj, x))
        x = self.dropout(x)
        x = self.gc2(adj, x)
        return x


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


class AdjacencyMatrix(nn.Module):
    def __init__(self, n_class: int):
        super(AdjacencyMatrix, self).__init__()
        self.n_class = n_class

    def normalize(self, x: torch.Tensor):
        rinv = x.sum(dim=0).float_power(-1)
        rinv.nan_to_num(0, 0, 0)
        rinv = rinv.diag().to(dtype=torch.float32)
        x = torch.einsum('bij,i->bij', x, rinv)
        return x.to(dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        # Bin the values of the max pooled feature map
        b, _, _ = x.size()
        bin = x.view(b, -1, self.n_class)
        y = bin.transpose(1, 2)
        adj = torch.einsum('bij,bjk -> bik', bin, y)

        return adj


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
                bias=False,
            ),
        ]
        layer += [nn.GroupNorm(1, outp)]  # layernorm
        if act:
            layer += [nn.ReLU(True)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int = 128) -> None:
        super().__init__()
        blocks = [
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, groups=32, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, in_c, 1, bias=False),
        ]

        self.blocks = nn.Sequential(*blocks)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)

        return self.act(out + x)


class FeatureConnection(nn.Module):
    def __init__(self, patch_per_direction: int, n_label: int, resize_to: int = 256) -> None:
        super().__init__()
        patch_size = resize_to // patch_per_direction

        # p ** 2 patch, f feature per patch => (p**2) * f object
        self.block = nn.Sequential(
            Resize(resize_to),
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(3 * patch_size ** 2, n_label),
            Rearrange('b h w c -> b (h w) c'),
            nn.Softmax(dim=1),
            AdjacencyMatrix(n_label)
        )

    def forward(self, batched_img: torch.Tensor) -> torch.Tensor:
        """
        input: (b, c, h, w) batched tensor image
        output: (b, 1, patch_count**2 * feature_count,patch_count**2 * feature_count)
        """

        return self.block(batched_img)
