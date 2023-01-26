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


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(
            out_channels), kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out






class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv5x5(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv5x5, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(2)
        else:
            self.pad = nn.ZeroPad2d(2)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 5)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nclassa, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.softmax(x, dim=1)


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'pointwise'),
                    Conv1x1(in_planes if (i == 0) else out_planes, out_planes, False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'pointwise'))(top)
            x = top + x
        return x


class Backproject(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Backproject, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(
            range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.from_numpy(self.id_coords)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = torch.cat([self.pix_coords, self.ones], 1)

    def forward(self, depth: torch.Tensor, inv_K) -> torch.Tensor:
        cam_points = torch.matmul(
            inv_K[:, :3, :3], self.pix_coords.to(device=depth.device))
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat(
            [cam_points, self.ones.to(device=depth.device)], dim=1)
        return cam_points


class Project(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / \
            (cam_points[:, 3, :].unsqueeze(1) + self.eps)
        pix_coords = cam_points[:, :2, :]
        pix_coords = pix_coords.view(
            self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        coef = 1 / 127
        pix_coords[..., 0] *= coef
        pix_coords[..., 1] *= coef

        return pix_coords
