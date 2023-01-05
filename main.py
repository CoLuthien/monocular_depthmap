
from pathlib import Path
import numpy as np
import pyarrow as pa
import cv2 as cv

from datas.dataset import CycleDepthDataset
from datas.rtv_pq_dataset import *
from torch.utils import data

from models.encoders.ref_encoder import *
from models.decoders.ref_decoder import *
from models.trainer import Model

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        help='train config file path')
    parser.add_argument('--work_dir',
                        help='the dir to save logs and models')
    parser.add_argument('--gpus',
                        default='0',
                        type=str,
                        help='number of gpus to use '
                             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # a = CycleDepthDataset('./', ['name'], ['image'])

    dataset = RTVParquetDataset('E:/parquets', ['index'], ['image'], 4)
    loader = data.DataLoader(dataset, batch_size=1, num_workers=4)
    device = torch.device('cuda:0')
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=TensorBoardLogger('./'))
    model = Model()
    trainer.fit(model, loader)

