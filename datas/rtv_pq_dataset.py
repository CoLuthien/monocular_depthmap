
from typing import Callable, List, Tuple, Union, Dict, Callable
from dataclasses import dataclass

import torch
import random
from .pq_dataset import BaseParquetDataset

import numpy as np
import io

import random
from PIL import Image
import cv2 as cv
from torchvision import transforms
from torchvision.transforms import functional as F
from kornia.augmentation import RandomGaussianNoise


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
                 is_train=True):
        super().__init__(parquet_path, idx_column, use_column, num_cache)
        self.interp = Image.ANTIALIAS
        fx = 324.25157867 / 960
        fy = 325.94358134 / 960
        cx = 477.57375239 / 960
        cy = 483.6168017 / 960
        k_mat = np.array([[fx, 0, cx, 0],
                          [0, fy, cy, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        k_mat_inv = np.linalg.pinv(k_mat)
        self.kmat = torch.from_numpy(k_mat)
        self.kmat_inv = torch.from_numpy(k_mat_inv)
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
        self.mask = torch.from_numpy(
            self.create_circular_mask(self.height, self.width)
        )

    def augment(self, img: torch.Tensor, aug: Tuple, angle: float):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = aug

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        img = F.rotate(img, angle)
        return img

    def create_circular_mask(self, h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center >= radius
        return mask

    def decode_image(self, image) -> Image.Image:
        image = Image.open(io.BytesIO(image)).resize((self.height, self.width))
        return image

    def __getitem__(self, index):
        imgs = [super(__class__, self).__getitem__(idx)
                for idx in self.frame_idxs]
        imgs = [self.decode_image(img) for img in imgs]

        #def rand_angle(x): return (random.random() * 2 - 1) * x

        #if random.random() > 0.5:
            #aug = transforms.ColorJitter.get_params(
                #self.brightness, self.contrast, self.saturation, self.hue)
            #base = rand_angle(180)
            #diff = rand_angle(10)
            #angles = [base - diff, base, base + diff]

            #imgs = [F.equalize(img) for img in imgs]
            #imgs = [self.augment(img, aug, angle)
                    #for img, angle in zip(imgs, angles)]

        imgs = [F.to_tensor(img) for img in imgs]

        return imgs, self.kmat, self.kmat_inv, self.mask
