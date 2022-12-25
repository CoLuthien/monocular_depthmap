
from typing import Tuple, Dict, ByteString
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, img_chan: 3, dim: int = 16) -> None:
        super().__init__()
        # input (N, 3, 192, 192)
        self.e0 = nn.Sequential(*[
            ConvBlock(img_chan, dim, 3, 1, 1),  # maintain
            ResBlock(dim),
        ])
        self.e1 = nn.Sequential(*[
            ConvBlock(dim, dim * 2, 4, 2, 1),  # /2
            ResBlock(dim * 2),
        ])
        self.e2 = nn.Sequential(*[
            ConvBlock(dim * 2, dim * 8, 8, 4, 2),  # /4
            ResBlock(dim * 8),
        ])
        self.e3 = nn.Sequential(*[
            ConvBlock(dim * 8, dim * 16, 4, 2, 1),  # / 2
            ResBlock(dim * 16),
        ])
        self.e4 = nn.Sequential(*[
            ConvBlock(dim * 16, dim * 16, 3, 1, 1),
            ResBlock(dim * 16)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.e0(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        return {
            's0': s0,
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
        }


class DepthDecoder(nn.Module):
    def __init__(self, in_: int) -> None:
        super().__init__()
        # (N, in_ * 16, W / 16, H / 16) : input
        dim = in_
        self.d0 = nn.Sequential(*[
            ConvBlock(dim * 2, 3, 3, 1, 1),  # maintain
            ResBlock(3),
        ])
        self.d1 = nn.Sequential(*[
            nn.PixelShuffle(2),
            AttentionLayer(),
            ConvBlock(dim, dim, 3, 1, 1),  # /2
            ResBlock(dim),
        ])
        self.d2 = nn.Sequential(*[
            nn.PixelShuffle(4),
            AttentionLayer(),
            ConvBlock(dim, dim * 2, 3, 1, 1),  # /4
            ResBlock(dim * 2),
        ])
        self.d3 = nn.Sequential(*[
            nn.PixelShuffle(2),
            AttentionLayer(),
            ConvBlock(dim * 8, dim * 8, 3, 1, 1),  # / 2
            ResBlock(dim * 8),
        ])
        self.d4 = nn.Sequential(*[
            ConvBlock(dim * 16, dim * 16, 3, 1, 1),
            ResBlock(dim * 16)
        ])

    def forward(self, x) -> torch.Tensor:
        s0 = x['s0']
        s1 = x['s1']
        s2 = x['s2']
        s3 = x['s3']
        s4 = x['s4']
        u3 = self.d4(s4)
        u2 = self.d3(torch.cat([s3, u3], 1))
        u1 = self.d2(torch.cat([s2, u2], 1))
        u0 = self.d1(torch.cat([s1, u1], 1))
        o0 = self.d0(torch.cat([s0, u0], 1))

        return o0


class AttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.LazyLinear(16)
        self.key = nn.LazyLinear(16)
        self.softmax = nn.Softmax2d()

    def forward(self, x: torch.Tensor):
        _, _, w, h = x.size()
        x = x.flatten(2, 3)
        q = self.query(x)
        k = self.key(x)

        key_query = torch.matmul(q, k.transpose(1, 2))
        key_query = self.softmax(key_query).unsqueeze(1)

        x = x.unflatten(2, (w, h))
        for idx, proj in enumerate(key_query.split(1, 0)):
            proj = proj.permute([3, 2, 1, 0])
            with torch.no_grad():
                x[idx, :, :, :] = F.conv2d(
                    x[idx, :, :, :], proj, bias=None, stride=1, padding=0)

        return x


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


class DepthProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(*[
            ConvBlock(4, 3, 1, 1, 0)
        ])

    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(x[:, 1, :, :][:, None, :], requires_grad=False)
        x = torch.cat([x, ones], dim=1)

        for idx, proj in enumerate(pose.split(1, 0)):
            with torch.no_grad():
                proj = proj.permute([3, 2, 1, 0])
                x[idx, :, :, :] = F.conv2d(
                    x[idx, :, :, :], proj, bias=None, stride=1, padding=0)

        # 1 x 1 with no bias convolution
        """
        channel wise multiplication with pose matrix 
        4x4 * [x, y, z, w]
        """
        x = self.block(x)
        return x


class PoseDecoder(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        # (N, dim * 8 * 2, 4, 4)
        module = [
            ConvBlock(dim * 16, 3, 3, 1, 1),
            ResBlock(3),
            ConvBlock(3, 1, 3, 1, 1),
            nn.AdaptiveAvgPool2d((4, 4))
        ]

        self.block = nn.Sequential(*module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x['s4'])


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = 1,
                 activation: bool = True,
                 ) -> None:
        super().__init__()
        layer = [
            nn.Conv2d(
                in_channel, out_channel, kernel_size, stride, padding, groups, bias=False),
            nn.GroupNorm(1, out_channel)  # Equivalent to LayerNorm
        ]
        if (activation):
            layer += [nn.LeakyReLU(0.2, inplace=True)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        layer = [ConvBlock(in_channel, in_channel, 3, 1, 1, activation=False)]
        layer += [nn.GroupNorm(1, in_channel)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        return skip + self.block(x)
