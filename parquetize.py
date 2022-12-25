
from pathlib import Path

from parquetize_options import ParquetizeOption

from multiprocessing import Pool

import pyarrow as pa
from pyarrow import parquet as pq
import torch
import torch.utils.data as data

import pandas as pd
import cv2

"""
we assume that directory structure will be
".../root/$(scene_name)/$(data_group)/*.jpg"
and all the image files have a ordered index
"""


def read_bin(path: Path):
    with open(path.as_posix(), 'rb') as f:
        return f.read()


def make_name(path: Path):
    return path.stem



if __name__ == "__main__":
    pool = Pool(8)
    datas = Path("E:/rtv_fisheye/rtv_fisheye/Original")
    scenes = [i for i in datas.iterdir()]
    idx = 0

    for set in scenes:
        for p in set.iterdir():
            img_paths = [i for i in p.glob('*.png')]
            print(p)
            idx += 1

            imgs = pool.map(read_bin, img_paths)
            print("bindone")
            names = pool.map(make_name, img_paths)
            df = pd.DataFrame(
                {
                    'image': imgs,
                    'name': names,
                    'index': range(len(img_paths))
                }
            )
            df.to_parquet(Path("E:/parquetized_fish/" +
                          str(idx) + '.parquet', engine='pyarrow'))
