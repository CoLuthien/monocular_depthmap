
from typing import Callable, List, Tuple, Union

import torch
from .pq_dataset import BaseParquetDataset

import cv2
import numpy as np
import io
from PIL import Image
from torchvision import transforms


class CycleDepthDataset(BaseParquetDataset):
    def __init__(self,
                 parquet_path: str,
                 idx_column: List[str],
                 use_column: List[str],
                 num_cache: int = 1):
        super().__init__(parquet_path, idx_column, use_column, num_cache)
        self.to_tensor = transforms.ToTensor()

    def decode_image(self, image) -> torch.Tensor:
        image = Image.open(io.BytesIO(image)).resize((256, 256))
        image = self.to_tensor(image)
        return image

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        i0 = super().__getitem__(index)
        i1 = super().__getitem__(index)

        r0 = self.decode_image(i0['image'])
        r1 = self.decode_image(i1['image'])
        return r0, r1
