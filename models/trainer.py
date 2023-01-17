

from typing import Callable, List, Tuple, Union, Any

import torch
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as FT
import torch.utils.data as data
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
import torch

from kornia import filters as KF

from losses.image import *
from models.decoders.ref_decoder import *
from models.encoders.ref_encoder import *
from models.base.basic_blocks import *

from functools import reduce


class Model(pl.LightningModule):
    def __init__(self, batch, height, width) -> None:
        super().__init__()

        self.depth_encoder = DepthEncoder(18, True)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)

        self.pose_decoder = PoseDecoder2()

        self.batch = 1
        self.width = 256
        self.height = 256

        self.ssim = SSIM()
        self.backproject = Backproject(self.batch, 256, 256)
        self.project = Project(self.batch, self.height, self.width)
        self.pad = nn.ReflectionPad2d(1)

    def predict_cloud_and_rasterize(self, depth, kmat, kmat_inv, projection):
        cam_points = self.backproject(depth, kmat_inv)
        rasterized_points = self.project(
            cam_points,
            kmat,
            projection)
        return rasterized_points, cam_points.view(1, 4, 256, 256)[:, :3, ...]

    def sample_image(self, orig: torch.Tensor, sample_point: torch.Tensor) -> torch.Tensor:
        sampled_image = F.grid_sample(
            orig,
            sample_point,
            padding_mode='border'
        )
        return sampled_image

    def predict_disparity(self, feature: torch.Tensor) -> torch.Tensor:
        disp = self.depth_decoder(feature, 0)
        return disp

    def forward(self, batch: Tuple[List[torch.Tensor]]):
        imgs, K, Kinv, mask = batch
        prev_frame = imgs[0]
        cur_frame = imgs[1]
        next_frame = imgs[2]

        features = [self.depth_encoder(img) for img in imgs]

        disp = self.predict_disparity(features[1])
        depth = self.predict_depth(disp, 0.1, 100.)

        left_pose = self.predict_pose(
            [features[0][-1], features[1][-1]])
        right_pose = self.predict_pose(
            [features[1][-1], features[2][-1]])

        pixel_left, point_left = self.predict_cloud_and_rasterize(
            depth, K, Kinv, left_pose)

        pixel_right, point_right = self.predict_cloud_and_rasterize(
            depth, K, Kinv, right_pose)

        sample_left = self.sample_image(prev_frame, pixel_left)
        sample_right = self.sample_image(next_frame, pixel_right)

        return sample_left, sample_right, disp, depth, point_left, point_right

    def laplace(self, x: torch.Tensor) -> torch.Tensor:
        pad = nn.ReflectionPad2d(1)
        points = pad(x)
        ddu = points[..., :-2, 1:-1] - 2 * \
            points[..., 1:-1, 1:-1] + points[..., 2:, 1:-1]
        ddv = points[..., 1:-1, :-2] - 2 * \
            points[..., 1:-1, 1:-1] + points[..., 1:-1, 2:]

        return F.normalize(ddu + ddv)

    def training_step(self, inputs, batch_idx):
        imgs, _, _, mask = inputs
        p_frame = imgs[0]
        c_frame = imgs[1]
        n_frame = imgs[2]
        sample_left, sample_right, disp, depth, pl, pr = self(inputs)

        disp = disp.masked_fill(mask, 0)
        depth = depth.masked_fill(mask, 0)

        left_loss = self.compute_reprojection_loss(sample_left, c_frame)
        right_loss = self.compute_reprojection_loss(sample_right, c_frame)

        ident_left_loss = self.compute_reprojection_loss(
            sample_left, p_frame) + torch.randn_like(left_loss) * 1e-3
        ident_right_loss = self.compute_reprojection_loss(
            sample_right, n_frame) + torch.randn_like(left_loss) * 1e-3

        repr_loss = [left_loss, right_loss, ident_left_loss, ident_right_loss]
        #repr_loss = [l.masked_fill(mask, 0) for l in repr_loss]
        repr_loss = torch.cat(repr_loss, dim=1)
        repr_loss = repr_loss.mean()

        mean_disp = depth.mean(2, True).mean(3, True)
        norm_disp = depth / (mean_disp + 1e-7)

        smooth_loss = self.get_smooth_loss(norm_disp, c_frame) * 1e-3
        smooth_loss = smooth_loss.mean()

        d_loss = self.correlation_loss(pl[:, 2, :, :], depth)
        d_loss += self.correlation_loss(depth, pr[:, 2, :, :])

        ssim_loss = self.ssim(depth, FT.rgb_to_grayscale(c_frame)).mean() * 1e-3

        d_loss *= 1e-4

        step = self.global_step
        if step % 100 == 0:
            logger = self.logger.experiment
            with torch.no_grad():
                self.log_dict(
                    {
                        'reprojection_loss': repr_loss,
                        'point_cloud_correlation': d_loss,
                        'ssim_depth' : ssim_loss
                    }
                )
                logger.add_image('ref', make_grid(torch.cat(imgs)).cpu(),
                                 step, dataformats='CHW')
                logger.add_image('disp', disp.clone().cpu(),
                                 step, dataformats='NCHW')
                logger.add_image('depth', depth.clone().cpu(),
                                 step, dataformats='NCHW')
                logger.add_image('left', sample_left.clone().cpu(),
                                 step, dataformats='NCHW')
                logger.add_image('right', sample_right.clone().cpu(),
                                 step, dataformats='NCHW')

        return repr_loss + d_loss + ssim_loss

    def predict_depth(self, disp, min_depth, max_depth):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def gradient(self, D):
        dy = D[:, :, 1:] - D[:, :, :-1]
        dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return dx, dy

    def predict_pose(self, feature: torch.Tensor) -> torch.Tensor:
        return self.pose_decoder(feature)

    def robust_l1(self, pred, target):
        # return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)
        return F.l1_loss(pred, target)

    def compute_reprojection_loss(self, pred, target) -> torch.Tensor:
        photometric_loss = self.robust_l1(pred, target)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
            torch.mean(disp_dy.abs() * torch.exp(-a1 *
                       img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
            torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
            torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
            torch.mean(disp_dyy.abs() * torch.exp(-a2 *
                       img_dyy.abs().mean(1, True)))

        return (smooth1+smooth2)

    def configure_optimizers(self) -> Any:
        #optim = torch.optim.Rprop(self.parameters(), lr=1e-4)
        #optim = torch.optim.RMSprop(self.parameters(), lr=1e-4)
        #optim = torch.optim.Adadelta(self.parameters(), lr=1e-3)
        #optim = torch.optim.Adamax(self.parameters(), lr=1e-4)
        #optim = torch.optim.Adagrad(self.parameters(), lr=1e-4)
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        #optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
        #optim = torch.optim.LBFGS(self.parameters(), lr=1e-4)
        return optim

    def correlation_loss(self, y_pred, y_true):
        x = y_pred.clone()
        y = y_true.clone()
        x = F.normalize(x)
        y = F.normalize(y)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) *
                      torch.sqrt(torch.sum(vy ** 2)) + 1e-12)

        return 1 - corr
