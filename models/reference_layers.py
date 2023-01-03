
from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class BackprojectDepth(nn.Module):
    """
    Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size: int, width: int):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.width = width

        pix, ones = self.create_pixel_coords(batch_size, width)

        self.pix_coords = nn.Parameter(pix, requires_grad=False)
        self.ones = nn.Parameter(ones, requires_grad=False)

    def create_pixel_coords(self, batch: int, width: int):
        import itertools

        item = list(itertools.product([i for i in range(0, width)], repeat=2))

        item = torch.Tensor(item)

        item = list(torch.split(item, 1, 1))

        item = torch.stack(item, 0)
        item = item.reshape(2, width, width).flip(0)
        ones = torch.ones_like(item[0][:]).repeat([1, 1, 1])
        item = torch.cat([item, ones], dim=0)

        ones = ones.repeat([batch, 1, 1, 1])
        item = item.repeat([batch, 1, 1, 1])

        return (item.view(batch, 3, width, width).flatten(2, 3),
                ones.view(batch, 1, width * width))

    def forward(self, depth: torch.Tensor, inv_K: torch.Tensor):
        """
        depth: N x 1 x W x H
        inv_k : N x 4 x 4

        output : N x 3 x W * H
        """
        n, c, w, h = depth.size()
        cam_points = torch.matmul(inv_K[:n, :3, :3], self.pix_coords[:n, ...])
        cam_points = depth.view(n, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones[:n, ...]], dim=1)

        return cam_points


class Disp2Depth(nn.Module):
    """
    Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """

    def __init__(self, min_: float, max_: float) -> None:
        super().__init__()
        self.min_disp = nn.Parameter(
            torch.Tensor([1. / max_]), requires_grad=False)
        self.max_disp = nn.Parameter(
            torch.Tensor([1. / min_]), requires_grad=False)

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        disp = disp.clamp(0, 1)
        scaled_disp = self.min_disp + (self.max_disp - self.min_disp) * disp
        depth = 1. / scaled_disp
        return depth


class Project3D(nn.Module):
    """
    Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size: int, width: int, eps=1e-7):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.coef = 1 / (width // 2 - 1)
        self.eps = eps

    def forward(self, points: torch.Tensor, K: torch.Tensor, T: torch.Tensor):
        """
        points: B x 4 x W * W
        K : B x 4 x 4
        T : B x 4 x 4
        """
        n, c, h = points.size()
        K = K.view(n, 4, 4)
        T = T.view(n, 4, 4)
        P = torch.matmul(K, T)
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :3, ...]
        pix_coords = pix_coords.view(
            n, 3, self.width, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        pix_coords *= self.coef

        return pix_coords
