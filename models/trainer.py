

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
from torchvision.datasets import MNIST
import torch

from losses.image import *
from models.decoders.ref_decoder import *
from models.encoders.ref_encoder import *
from models.base.basic_blocks import *

from functools import reduce


class Model(pl.LightningModule):
    def __init__(self, batch, height, width) -> None:
        super().__init__()

        self.depth_encoder = DepthEncoder(18, False)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)

        self.pose_decoder = PoseDecoder()

        self.batch = 1
        self.width = 256
        self.height = 256

        self.ssim = SSIM()
        self.backproject = Backproject(
            self.batch, 256, 256)
        self.project = Project(self.batch,
                               self.height, self.width)

    def predict_pose(self, feature: torch.Tensor, invert: bool) -> torch.Tensor:
        angle, position = self.pose_decoder(feature)

        return self.transformation_from_parameters(angle[:, 0], position[:, 0], invert)

    def predict_cloud_and_rasterize(self, depth, kmat, kmat_inv, projection):
        points_in_camera_volume = self.backproject(depth, kmat_inv)
        rasterized_points = self.project(
            points_in_camera_volume, kmat, projection)
        return rasterized_points

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

    def predict_all(self, imgs: List[torch.Tensor], K, Kinv):
        prev_frame = imgs[0]
        cur_frame = imgs[1]
        next_frame = imgs[2]

        features = [self.depth_encoder(img) for img in imgs]

        disp = self.predict_disparity(features[1])
        depth = self.predict_depth(disp, 0.1, 100.)

        prev_cur = torch.cat([features[0][-1], features[1][-1]], dim=1)
        cur_next = torch.cat([features[1][-1], features[2][-1]], dim=1)
        left_pose = self.predict_pose(prev_cur, True)
        right_pose = self.predict_pose(cur_next, False)

        pixel_left = self.predict_cloud_and_rasterize(
            depth, K, Kinv, left_pose)

        pixel_right = self.predict_cloud_and_rasterize(
            depth, K, Kinv, right_pose)

        sample_left = self.sample_image(prev_frame, pixel_left)
        sample_right = self.sample_image(next_frame, pixel_right)

        return sample_left, sample_right, disp, depth

    def forward(self, batch: Tuple[List[torch.Tensor]]):
        imgs, camera_info, camera_info_inv, _ = batch

        return self.predict_all(imgs, camera_info, camera_info_inv)

    def training_step(self, inputs, batch_idx):
        imgs, _, _, mask = inputs
        p_frame = imgs[0]
        c_frame = imgs[1]
        n_frame = imgs[2]
        sample_left, sample_right, disp, depth = self(inputs)

        left_loss = self.compute_reprojection_loss(sample_left, c_frame)
        right_loss = self.compute_reprojection_loss(sample_right, c_frame)

        ident_left_loss = self.compute_reprojection_loss(sample_left, p_frame)
        ident_right_loss = self.compute_reprojection_loss(
            sample_right, n_frame)

        repr_loss = [left_loss, right_loss, ident_left_loss, ident_right_loss]
        repr_loss = [l.masked_fill(mask, 0) for l in repr_loss]
        repr_loss = torch.cat(repr_loss, dim=1)
        repr_loss = repr_loss.mean()

        disp = disp.masked_fill(mask, 0)
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)

        smooth_loss = self.get_smooth_loss(
            norm_disp, FT.rgb_to_grayscale(c_frame)) * 1e-4
        smooth_loss = smooth_loss.mean()

        depth_loss = self.get_smooth_loss(
            depth, FT.rgb_to_grayscale(c_frame)) * 1e-4
            
        smooth_loss += depth_loss.mean()
        step = (self.current_epoch + 1) * batch_idx
        if step % 100 == 0:
            depth = depth.masked_fill(mask, 0)
            logger = self.logger.experiment
            logger.add_image('left', sample_left, step, dataformats='NCHW')
            logger.add_image('right', sample_right,
                             step, dataformats='NCHW')
            logger.add_image('disp', disp, step, dataformats='NCHW')
            logger.add_image('depth', depth, step, dataformats='NCHW')
            logger.add_image('ref', c_frame, step, dataformats='NCHW')

        return repr_loss + smooth_loss.mean()

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

    def get_translation_matrix(self, translation_vector):
        T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
        t = translation_vector.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    def rot_from_axisangle(self, vec):
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1
        return rot

    def transformation_from_parameters(self, axisangle, translation, invert=False):
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = self.get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)
        return M

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_reprojection_loss(self, pred, target) -> torch.Tensor:
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
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
        #optim = torch.optim.Adadelta(self.parameters(), lr=1e-4)
        #optim = torch.optim.Adamax(self.parameters(), lr=1e-4)
        #optim = torch.optim.Adagrad(self.parameters(), lr=1e-4)
        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optim
