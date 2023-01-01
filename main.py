
from pathlib import Path
import numpy as np
import pyarrow as pa
import cv2 as cv

from datas.dataset import CycleDepthDataset
from torch.utils import data
from models.reference_layers import *
from models.decoder_blocks import *
from models.encoder_blocks import *

from models.trainer_modules import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.nn as nn
import torch.nn.functional as F


from PIL import Image


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(
        device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def create_pixel_coords(batch: int, width: int):
    import itertools

    item = list(itertools.product([i for i in range(0, width)], repeat=2))

    item = torch.Tensor(item)

    item = list(torch.split(item, 1, 1))

    item = torch.stack(item, 0)
    item = item.reshape(2, width, width).flip(0)
    ones = torch.ones_like(item[0][:]).view(1, width, width)

    return torch.cat([item, ones], dim=0).view(batch, 3, width, width).flatten(2, 3)


if __name__ == '__main__':

    # a = CycleDepthDataset('./', ['name'], ['image'])

    dataset = CycleDepthDataset('./', ['index'], ['image'])
    loader = data.DataLoader(dataset, batch_size=2, num_workers=2)
    device = torch.device('cuda:0')
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=TensorBoardLogger('./'))
    model = Model()
    trainer.fit(model, loader)

