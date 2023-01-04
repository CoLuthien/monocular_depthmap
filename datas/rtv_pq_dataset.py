
from typing import Callable, List, Tuple, Union, Dict, Callable
from dataclasses import dataclass

import torch
from .pq_dataset import BaseParquetDataset

import numpy as np
import io

import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


@dataclass
class ImageAttributes():
    height: int
    width: int
    scales: int
    frame_idxs: List[int]
    brightness: List[float]
    contrast:  List[float]
    satuation: List[float]
    hue:  List[float]
    camera_matrix: torch.Tensor  # 4 x 4


default_attr = ImageAttributes(
    256,
    256, 2, [0, 1, -1], [0.8, 1.2], [0.8, 1.2], [.8, 1.2], [-.1, .1],
    torch.from_numpy(np.array([[1, 0, 0.5, 0],
                               [0, 1., 0.6, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)))


class RTVParquetDataset(BaseParquetDataset):
    def __init__(self,
                 parquet_path: str,
                 idx_column: List[str],
                 use_column: List[str],
                 num_scales: int,
                 frame_idxs: List[int] = [-1, 0, 1],
                 height: int = 256,
                 width: int = 256,
                 num_cache: int = 1, 
                 is_train = True):
        super().__init__(parquet_path, idx_column, use_column, num_cache)
        self.interp = Image.ANTIALIAS

        k_mat = np.array([[1, 0, 0.5, 0],
                          [0, 1., 0.6, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        k_mat_inv = np.linalg.pinv(k_mat)
        self.k_mat = torch.from_numpy(k_mat)
        self.k_mat_inv = torch.from_numpy(k_mat_inv)
        self.is_train = is_train

        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.frame_idxs = frame_idxs

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def preprocess(self, inputs: Dict, color_aug: Callable):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = F.to_tensor(f)
                inputs[(n + "_aug", im, i)] = F.to_tensor(f)

        return 

    def decode_image(self, image) -> torch.Tensor:
        image = Image.open(io.BytesIO(image))
        image = image.resize((self.height, self.width))
        return image

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        do_color_aug = self.is_train and random.random() > 0.5

        inputs = {}

        for i in self.frame_idxs:
            item = super().__getitem__(index)
            inputs[("color", i, -1)] = self.decode_image(item)

        for scale in range(self.num_scales):
            K = self.k_mat.clone()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = K.inverse()

            inputs[("K", scale)] = K
            inputs[("inv_K", scale)] = inv_K

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        return inputs
