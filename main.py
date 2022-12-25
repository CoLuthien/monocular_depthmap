
from pathlib import Path
import numpy as np
import pyarrow as pa
import cv2 as cv

from datasets.dataset import CycleDepthDataset
from torch.utils import data
from models.encoder_blocks import  EncoderBlock, PoseDecoder, DepthDecoder, DepthProjection, AttentionLayer
from models.trainer_modules import Model

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


from PIL import Image


def create_pixel_coords(batch: int, width: int):
    import itertools

    item = list(itertools.product([i for i in range(0, width)], repeat=2))

    item = torch.Tensor(item)

    item = list(torch.split(item, 1, 1))

    item = torch.stack(item, 0)
    item = item.reshape(2, width, width).flip(0)
    ones = torch.ones_like(item[0][:]).view(1, width, width)

    print(ones.size())

    return torch.cat([item, ones, ones], 0).view(1, 4, width, width)

if __name__ == '__main__':

    # a = CycleDepthDataset('./', ['name'], ['image'])

    dataset = CycleDepthDataset('E:/parquetized_fish', ['index'], ['image'])
    loader = data.DataLoader(dataset, 4, num_workers=4)
    device =torch.device('cuda:0') 
    trainer = pl.Trainer(accelerator='gpu', devices=1)
    model = Model()

    trainer.fit(model, train_dataloaders=loader)



