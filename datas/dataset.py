
from typing import Callable, List, Tuple, Union
from dataclasses import dataclass

import torch
from .pq_dataset import BaseParquetDataset

import cv2
import numpy as np
import io

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


class CycleDepthDataset(BaseParquetDataset):
    def __init__(self,
                 parquet_path: str,
                 idx_column: List[str],
                 use_column: List[str],
                 in_attributes: ImageAttributes = default_attr,
                 num_cache: int = 1):
        super().__init__(parquet_path, idx_column, use_column, num_cache)
        self.attributes = in_attributes

        attr = self.attributes
        self.k_mat = attr.camera_matrix
        self.k_mat_inv = attr.camera_matrix.inverse()
        self.adjust = transforms.ColorJitter(
            attr.brightness, attr.contrast, attr.satuation, attr.hue)
        self.mask = torch.from_numpy(
            self.create_circular_mask(256, 256)
        )
        self.mask = self.mask.repeat([3, 1, 1])

    def create_circular_mask(self, h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center >= radius
        return mask

    def preprocess(self, inputs: List[torch.Tensor]):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # output = map(lambda img: self.adjust(img), inputs)
        output = map(lambda img: F.to_tensor(img), inputs)

        return list(output)

    def decode_image(self, image) -> torch.Tensor:
        attr = self.attributes
        image = Image.open(io.BytesIO(image))
        image = image.resize((attr.height, attr.width))
        return image

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        i0 = super().__getitem__(index)
        i1 = super().__getitem__(index)

        r0 = self.decode_image(i0)
        r1 = self.decode_image(i1)
        return self.preprocess([r0, r1]), self.k_mat, self.k_mat_inv, self.mask
