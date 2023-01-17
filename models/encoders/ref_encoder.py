from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from .resnet import ResNet, BasicBlock, resnet18, resnet34, resnet50, resnet101, Bottleneck
from torch.nn import BatchNorm2d as bn


class DepthEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self,
                 num_layers: int,
                 pretrained: bool = True):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}
        resnets_weights = {18:  models.ResNet18_Weights,
                           34:  models.ResNet34_Weights,
                           50:  models.ResNet50_Weights,
                           101: models.ResNet101_Weights,
                           152: models.ResNet152_Weights, }

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](
            pretrained, resnets_weights[num_layers].DEFAULT)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        features = []
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        features.append(self.activation(x))
        features.append(
            self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features
