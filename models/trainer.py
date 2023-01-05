

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

from losses.image import *
from models.decoders.ref_decoder import *
from models.encoders.ref_encoder import *
from models.base.basic_blocks import *

from functools import reduce


frame_ids = [-1, 0, 1]


class Model(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.depth_encoder = DepthEncoder(18, False)

        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)
        self.pose_encoder = PoseEncoder(18)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc)

        self.batch = 4
        self.width = 256
        self.height = 256

        self.ssim = SSIM()
        self.backproject = Backproject(
            self.batch, 256, 256)
        self.project = Project(self.batch,
                               self.height, self.width)

    def make_pose(self, inputs, outputs):
        pose_inputs = [inputs[('color_aug', i, 0)] for i in frame_ids]
        pose_features = [
            self.pose_encoder(torch.cat(pose_inputs[0:2:1], dim=1)),
            self.pose_encoder(torch.cat(pose_inputs[1:3:1], dim=1))
        ]

        pose_outputs = [
            self.pose_decoder(torch.cat(feature), dim=1) for feature in pose_features
        ]

        for idx in [-1, 1]:
            cur = pose_outputs.pop(0)
            axis, translation = cur
            outputs[('axisangle', 0, idx)] = axis
            outputs[('translation', 0, idx)] = translation
            outputs[('cam_T_cam', 0, idx)] = self.transformation_from_parameters(
                axis, translation, idx < 0
            )

        return outputs

    def forward(self, inputs):
        depth_feat = self.depth_encoder(inputs[('color_aug', 0, 0)])
        outputs = self.depth_decoder(depth_feat, 0)
        outputs = self.make_pose(inputs, outputs)

        for i in range(4):
            self.reproject_image(inputs, outputs, i)

        return outputs

    def disp_to_depth(self, disp, min_depth, max_depth):
        min_disp = 1 / max_depth  # 0.01
        max_disp = 1 / min_depth  # 10
        scaled_disp = min_disp + (max_disp - min_disp) * \
            disp  # (10-0.01)*disp+0.01
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def reproject_image(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(
            disp, [self.height, self.width], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, 0.5, 30)

        for i, frame_id in enumerate([-1, 1]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs[("inv_K")])
            pix_coords = self.project(cam_points, inputs[("K")], T)
            img = inputs[("color", frame_id, 0)]
            outputs[("depth", frame_id, scale)] = depth
            outputs[("color", frame_id, scale)] = F.grid_sample(
                img,
                pix_coords,
                padding_mode="border")
        return outputs

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

    def compute_perceptional_loss(self, tgt_f, src_f):
        loss = self.robust_l1(tgt_f, src_f).mean(1, True)
        return loss

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def training_step(self, inputs, batch_idx):
        outputs = self(inputs)
        loss_dict = torch.empty().cuda()
        for scale in [0, 1, 2, 3]:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]
            target = inputs[("color", 0, 0)]

            reprojection_losses = []

            """
            reconstruction
            """
            # print(outputs)
            outputs = self.reproject_image(inputs, outputs, scale)
            """
            automask
            """
            for frame_id in [-1, 1]:
                pred = inputs[("color", frame_id, 0)]
                identity_reprojection_loss = self.compute_reprojection_loss(
                    pred, target)
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 1e-5
                reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in [-1, 1]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            min_reconstruct_loss, outputs[("min_index", scale)] = torch.min(
                reprojection_loss, dim=1)

            """
            disp mean normalization
            """
            mean_disp = disp.mean(2, True).mean(3, True)
            disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = self.get_smooth_loss(disp, target)

            smooth_loss = self.opt.smoothness_weight * \
                smooth_loss / (2 ** scale)/len(self.opt.scales)
            reprojection_loss = torch.cat(reprojection_losses, 1)
            reconstruction_loss = min_reconstruct_loss.mean()/len(self.opt.scales)

            loss_dict += (smooth_loss + reprojection_loss +
                          reconstruction_loss)

        idx = (batch_idx) * (self.current_epoch + 1)

        return loss_dict

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

        return smooth1+smooth2

    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adamax(self.parameters(), lr=1e-4)
        return optim
