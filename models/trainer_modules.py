
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

from models.encoder_blocks import *
from models.decoder_blocks import *
from models.reference_layers import *
from losses.image import *
from functools import reduce


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
        self.depth_projection = BackprojectDepth(4, 256)
        self.to_depth = Disp2Depth(0.5, 100)
        self.to_image = Project3D(4, 256)
        # in -> N x 3 x W x H -> N x 3 x W x H

        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.projection_loss = ReprojectionLoss()
        self.smooth_loss = SmoothingLoss()

    def forward(self, batch):

        imgs, cam, cam_inv = batch
        features = list(map(lambda img: self.encoder(img), imgs))

        disps = list(map(lambda f: self.depth_decoder(f), features))
        depths = list(map(lambda disp: self.to_depth(disp), disps))

        poses = self.pose_decoder([feature[-1] for feature in features])

        clouds = list(map(
            lambda depth: self.depth_projection(depth, cam_inv),
            depths))

        pixels = list(
            map(lambda cloud, pose: self.to_image(cloud, cam, pose),
                clouds, poses))

        return {
            'pixel_sample': pixels,
            'disp': disps,
            'depth': depths,
            'cloud': clouds,
        }

    def training_step(self, batch, batch_idx):
        imgs, _, _ = batch
        results = self(batch)

        pixels = results['pixel_sample']
        pixs =[]
        projected_depth = []
        for p in pixels:
            pix, z = torch.split(p, [2, 1], dim=3)
            pixs.append(pix)
            projected_depth.append(z)
        depth = results['depth']
        sampled_images = [F.grid_sample(img, pixel)
                          for img, pixel in zip(imgs, pixs)]

        losses = [self.projection_loss(pred, ref)
                  for pred, ref in zip(sampled_images, imgs)]
        bce_losses = [self.bce(pred, ref)
                      for pred, ref in zip(sampled_images, imgs)]
        depth_loss = [self.l1(proj, pred)
                      for proj, pred in zip(projected_depth, depth)]

        smooth_loss = []
        disps = results['disp']
        for disp, target in zip(disps, imgs):
            mean = disp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            norm = disp / (mean + 1e-7)
            smooth_loss.append(self.smooth_loss(norm, target))

        losses = losses

        loss = reduce(lambda acc, value: acc + value.mean(), losses, 0)
        loss += reduce(lambda acc, value: acc + value.mean(), bce_losses, 0)
        loss += reduce(lambda acc, value: acc + value.mean(), depth_loss, 0) * 1e-4
        loss += reduce(lambda acc, value: acc +
                       value.mean(), smooth_loss, 0) * 1e-3

        idx = (batch_idx) * (self.current_epoch + 1)
        if (idx % 100 == 0):
            tb = self.logger.experiment
            self.logger.experiment.add_image("reference",
                                             torch.cat(imgs, dim=2),
                                             idx,
                                             dataformats="NCHW")
            self.logger.experiment.add_image("disp",
                                             torch.cat(disps, dim=2),
                                             idx,
                                             dataformats="NCHW")
            self.logger.experiment.add_image("sampled",
                                             torch.cat(sampled_images, dim=2),
                                             idx,
                                             dataformats="NCHW")
            self.logger.experiment.add_image("depth",
                                             torch.cat(depth, dim=2),
                                             idx,
                                             dataformats="NCHW")

        return loss

    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adamax(self.parameters(), lr=1e-4)
        return optim
