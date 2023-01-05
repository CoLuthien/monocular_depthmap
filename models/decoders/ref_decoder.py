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

p = 0.7
nclass = 1


def normalize(x):
    rowsum = np.array(x.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    x = r_mat_inv.dot(x)
    return x


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc: List[int]):
        super(DepthDecoder, self).__init__()

        bottleneck = 256
        stage = 4
        self.do = nn.Dropout(p=0.5)

        self.nfeat = bottleneck // 4
        self.nhid = self.nfeat * 4
        self.nhid2 = self.nhid * 4
        nclass = 1
        p = 0.8

        adj = nx.adjacency_matrix(nx.gnp_random_graph(self.nhid, p))
        xx = np.identity(self.nhid, dtype=np.float32)
        adj = normalize(adj + xx)
        self.adj = nn.Parameter(torch.from_numpy(adj).float())

        self.reduce4 = Conv1x1(num_ch_enc[4], 512, bias=False)
        self.iconv4 = Conv3x3(512, bottleneck)

        self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck, bias=False)
        self.iconv3 = Conv3x3(bottleneck*2+1, bottleneck)

        self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck, bias=False)
        self.iconv2 = Conv3x3(bottleneck*2+1, bottleneck)

        self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck, bias=False)
        self.iconv1 = Conv3x3(bottleneck*2+1, bottleneck)

        self.crp4 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp3 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp2 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp1 = self._make_crp(bottleneck, bottleneck, stage)

        self.merge4 = Conv3x3(bottleneck, bottleneck)
        self.merge3 = Conv3x3(bottleneck, bottleneck)
        self.merge2 = Conv3x3(bottleneck, bottleneck)
        self.merge1 = Conv3x3(bottleneck, bottleneck)

        # disp
        self.disp4 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp3 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp1 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())

        # GCN
        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        self.gc2 = GraphConvolution(self.nhid, nclass)
        self.gc3 = GraphConvolution(self.nfeat, self.nhid2)
        self.gc4 = GraphConvolution(self.nhid2, nclass)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, input_features, frame_id):
        outputs = {}
        l0, l1, l2, l3, l4 = input_features
        l4 = self.do(l4)
        l3 = self.do(l3)
        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)

        x4 = F.leaky_relu(x4)
        x4 = self.crp4(x4)
        x4 = self.merge4(x4)
        x4 = F.leaky_relu(x4)
        y4 = x4.view(self.nhid, -1)
        y4 = self.gc1(y4, self.adj)  # adj * y4 * weight
        y3 = self.gc2(y4, self.adj)
        y4 = y3.view(1, 1, self.nfeat // 4, self.nfeat // 4)
        y4 = self.do(y4)
        disp4 = y4
        x4 = upsample(x4)

        z3 = torch.transpose(y3, 0, 1)
        yy = torch.matmul(y3, z3)
        yy = yy.cpu()
        yy = yy.detach().numpy()
        yy = normalize(yy)
        yy = torch.from_numpy(yy).float().cuda()
        yy = yy.view(1, 1, self.nhid, self.nhid)
        yy = F.interpolate(yy, scale_factor=4, mode="nearest")
        yy = yy.view(self.nhid * 4, -1)

        x3 = self.reduce3(l3)
        x3 = torch.cat((x3, x4, disp4), 1)
        x3 = self.iconv3(x3)
        x3 = F.leaky_relu(x3)
        x3 = self.crp3(x3)
        x3 = self.merge3(x3)
        x3 = F.leaky_relu(x3)
        y5 = x3.view(self.nhid * 4, -1)
        y5 = self.gc3(y5, yy)
        y5 = self.gc4(y5, yy)
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
        x1 = upsample(x1)
        disp1 = self.disp1(x1)

        outputs[("disp", frame_id, 3)] = disp4
        outputs[("disp", frame_id, 2)] = disp3
        outputs[("disp", frame_id, 1)] = disp2
        outputs[("disp", frame_id, 0)] = disp1

        return outputs


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, stride=1):
        super(PoseDecoder, self).__init__()

        self.reduce = nn.Conv2d(num_ch_enc[-1], 256, 1)
        self.conv1 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv3 = nn.Conv2d(256, 6, 1)

        self.relu = nn.ReLU()

    def forward(self, input_features):
        f = input_features[-1]
        out = self.relu(self.reduce(f))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation
