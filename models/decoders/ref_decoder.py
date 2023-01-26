# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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

p = 0.7
nclass = 1


def normalize(x: torch.Tensor):
    rinv = x.sum(dim=1).float_power(-1)
    rinv.nan_to_num(0, 0, 0)
    rinv = rinv.diag().to(dtype=torch.float32)
    x = rinv @ x
    return x.to(dtype=torch.float32)


# def normalize(x: torch.Tensor):
    #rinv = x.sum(dim=0).float_power(-1)
    #rinv.nan_to_num(0, 0, 0)
    #rinv = rinv.diag().to(dtype=torch.float32)
    #x = torch.einsum('bij,i->bij', x, rinv)
    # return x.to(dtype=torch.float32)


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc: List[int]):
        super(DepthDecoder, self).__init__()

        bottleneck = 256
        stage = 4
        self.do = nn.Dropout(p=0.2)

        self.nfeat = bottleneck // 4
        self.nhid = self.nfeat * 4
        self.nhid2 = self.nhid * 4
        nclass = 1
        p = 0.3

        #adj = nx.adjacency_matrix(nx.gnp_random_graph(self.nhid, p))
        #xx = np.identity(self.nhid, dtype=np.float32)
        #adj = adj + xx
        #adj = torch.from_numpy(adj).to(dtype=torch.float32)
        # self.adj = nn.Parameter(
        # normalize(adj), requires_grad=False).to(dtype=torch.float32)

        self.reduce4 = Conv1x1(num_ch_enc[4], 512, bias=False)
        self.iconv4 = ConvBlock(512, bottleneck, 3, 1, 1, 1)

        self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck, bias=False)
        self.iconv3 = ConvBlock(bottleneck*2+1, bottleneck, 3, 1, 1, 1)

        self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck, bias=False)
        self.iconv2 = ConvBlock(bottleneck*2+1, bottleneck, 3, 1, 1, 1)

        self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck, bias=False)
        self.iconv1 = ConvBlock(bottleneck*2+1, bottleneck, 3, 1, 1, 1)

        self.crp4 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp3 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp2 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp1 = self._make_crp(bottleneck, bottleneck, stage)

        self.merge4 = Conv3x3(bottleneck, bottleneck)
        self.merge3 = Conv3x3(bottleneck, bottleneck)
        self.merge2 = Conv3x3(bottleneck, bottleneck)
        self.merge1 = Conv3x3(bottleneck, bottleneck)

        # disp
        self.disp4 = nn.Sequential(
            ResBlock(bottleneck),
            nn.PixelShuffle(2),
            ConvBlock(bottleneck // 4, 1, 4, 2, 1)
        )
        self.disp3 = nn.Sequential(
            ResBlock(bottleneck),
            nn.PixelShuffle(2),
            ConvBlock(bottleneck // 4, 1, 4, 2, 1)
        )
        self.disp2 = nn.Sequential(
            ResBlock(bottleneck),
            nn.PixelShuffle(2),
            ConvBlock(bottleneck // 4, 1, 4, 2, 1)
        )
        self.disp1 = nn.Sequential(
            ResBlock(bottleneck),
            nn.PixelShuffle(2),
            ConvBlock(bottleneck // 4, 32, 3, 1, 1),
            nn.PixelShuffle(2),
            ResBlock(8),
            ConvBlock(8, 1, 3, 1, 1, act=False),
            nn.Sigmoid()
        )

        # GCN
        self.gcn1 = GCN(self.nfeat, self.nhid, 1, 1, 0.3)
        self.gcn2 = GCN(self.nfeat, self.nhid2, 1, 1, 0.3)
        to_adj = [
            nn.Conv2d(64, 64, 3, 2, 1, bias=False, groups=8),
            nn.PReLU(64),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False, groups=8),
            nn.PReLU(64)
        ]
        self.to_adj = nn.Sequential(*to_adj)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, input_features, frame_id):
        l0, l1, l2, l3, l4 = input_features
        adj = self.to_adj(l0).view(256, 256)
        l4 = self.do(l4)
        l3 = self.do(l3)
        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)

        x4 = F.leaky_relu(x4)
        x4 = self.crp4(x4)
        x4 = self.merge4(x4)
        x4 = F.leaky_relu(x4)
        y4 = x4.view(self.nhid, -1)
        y3 = self.gcn1(y4, adj)
        y4 = y3.view(1, 1, self.nfeat // 4, self.nfeat // 4)
        y4 = self.do(y4)
        disp4 = y4
        x4 = upsample(x4)

        z3 = torch.transpose(y3, 0, 1)
        yy = torch.matmul(y3, z3)
        yy = normalize(yy)
        yy = yy.view(1, 1, self.nhid, self.nhid)
        yy = F.interpolate(yy, scale_factor=4, mode="nearest")
        yy = yy.view(self.nhid2, -1)

        x3 = self.reduce3(l3)
        x3 = torch.cat((x3, x4, disp4), 1)
        x3 = self.iconv3(x3)
        x3 = F.leaky_relu(x3)
        x3 = self.crp3(x3)
        x3 = self.merge3(x3)
        x3 = F.leaky_relu(x3)
        y5 = x3.view(self.nhid * 4, -1)
        y5 = self.gcn2(y5, yy)
        y5 = y5.view(1, 1, self.nfeat // 2, self.nfeat // 2)
        y5 = self.do(y5)
        disp3 = y5
        x3 = upsample(x3)

        x2 = self.reduce2(l2)
        x2 = torch.cat((x2, x3, disp3), 1)
        x2 = self.iconv2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.crp2(x2)
        x2 = self.merge2(x2)
        x2 = F.leaky_relu(x2)
        x2 = upsample(x2)
        disp2 = self.disp2(x2)

        x1 = self.reduce1(l1)
        x1 = torch.cat((x1, x2, disp2), 1)
        x1 = self.iconv1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.crp1(x1)
        x1 = self.merge1(x1)
        x1 = F.leaky_relu(x1)
        disp1 = self.disp1(x1)

        return disp1


class PoseDecoder(nn.Module):
    def __init__(self, stride=1):
        super(PoseDecoder, self).__init__()
        self.reduce = nn.Conv2d(1024, 256, 1)
        self.conv1 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv3 = nn.Conv2d(256, 6, 1)

        self.relu = nn.ReLU()

    def forward(self, input_features):
        f = input_features
        out = self.relu(self.reduce(f))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation


class PoseDecoder2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        squeezer = [
            nn.LazyConv2d(128, 3),
            nn.PReLU(128)
        ]

        convs = [
            ResBlock(256),
            ResBlock(256),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.PReLU(1)
        ]
        linear = [
            nn.LazyLinear(256, bias=False),
            nn.ReLU(True),
            nn.LazyLinear(16, bias=False),
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
