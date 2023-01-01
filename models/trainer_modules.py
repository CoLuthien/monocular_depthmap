
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
        self.projection_loss = ReprojectionLoss()
        self.smooth_loss = SmoothingLoss()

    def forward(self, batch):

        imgs, cam, cam_inv = batch
        features = list(map(lambda img: self.encoder(img), imgs))

        disps = list(map(lambda f: self.depth_decoder(f), features))
        depths = list(map(lambda disp: self.to_depth(disp), disps))

        pose = self.pose_decoder([feature[-1] for feature in features])

        clouds = list(map(
            lambda depth: self.depth_projection(depth, cam_inv),
            depths))

        pixels = list(
            map(lambda cloud: self.to_image(cloud, cam, pose),
                clouds))

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
        sampled_images = [F.grid_sample(img, pixel)
                          for img, pixel in zip(imgs, pixels)]

        losses = [self.projection_loss(pred, imgs[0])
                  for pred in sampled_images]

        smooth_loss = []
        disps = results['disp']
        for disp, target in zip(disps, imgs):
            mean = disp.mean(2, True).mean(3, True)
            norm = disp / (mean + 1e-7)
            smooth_loss.append(self.smooth_loss(norm, target))

        losses = losses + smooth_loss

        loss = torch.empty_like(losses[0])

        for l in losses:
            loss += l
        idx = (batch_idx) * (self.current_epoch + 1) 
        if (idx % 100 == 0):
            tb = self.logger.experiment
            self.logger.experiment.add_image("reference",
                                             imgs[0],
                                             idx,
                                             dataformats="NCHW")
            self.logger.experiment.add_image("disp",
                                             disps[0],
                                             idx,
                                             dataformats="NCHW")
            self.logger.experiment.add_image("sampled",
                                             sampled_images[0],
                                             idx,
                                             dataformats="NCHW")

        return l

    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adamax(self.parameters(), lr=1e-4)
        return optim
