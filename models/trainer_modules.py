
from typing import Callable, List, Tuple, Union, Any

import torch
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
import torch

from .encoder_blocks import EncoderBlock, DepthDecoder, PoseDecoder, ImageDecoder, DepthProjection


class Model(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        dim = 32
        # in -> N x 3 x W x H, out -> feature map
        self.encoder = EncoderBlock(3, dim)
        # in feature map-> , out -> N x 3 x W x H
        self.depth_decoder = DepthDecoder(dim)
        # in feature map -> , out -> N x 1 x 4 x 4
        self.pose_decoder = PoseDecoder(dim)
        self.depth_projection = DepthProjection()
        # in -> N x 3 x W x H -> N x 3 x W x H
        self.image_decoder = ImageDecoder(3)

        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.my_logger = TensorBoardLogger("./")

    def forward(self, x: Tuple):
        r0, r1 = x

        f0, f1 = self.encoder(r0), self.encoder(r1)

        p0, p1 = self.pose_decoder(f0), self.pose_decoder(f1)

        d0, d1 = self.depth_decoder(f0), self.depth_decoder(f1)

        dp0, dp1 = self.depth_projection(d0, p1), self.depth_projection(d1, p0)

        i0, i1 = self.image_decoder(d0, p0), self.image_decoder(d1, p1)

        return {
            'image': (i0, i1),
            'depth': (d0, d1),
            'proj_depth': (dp0, dp1),
        }

    def training_step(self, batch, batch_idx):
        i0s, i1s = batch
        results = self(batch)

        i0_gen, i1_gen = results['image']
        d0_gen, d1_gen = results['depth']
        dp0_gen, dp1_gen = results['proj_depth']

        bce_i0_loss = self.l1_loss(i0s, i0_gen)
        bce_i1_loss = self.l1_loss(i1s, i1_gen)

        d0_dp1_loss = self.l1_loss(d0_gen, dp1_gen)
        d1_dp0_loss = self.l1_loss(d1_gen, dp0_gen)

        # loss calculation
        return bce_i0_loss + bce_i1_loss + d0_dp1_loss + d1_dp0_loss

    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adamax(self.parameters(), lr=1e-4)
        return optim
