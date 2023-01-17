
import time
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import dataset as pas
import pyarrow.parquet as pq
import pyarrow
from pathlib import Path
from torch.utils import data

from functools import reduce


class BaseParquetDataset(data.Dataset):
    def __init__(self,
                 parquet_path: str,
                 idx_column: List[str],
                 use_column: List[str],
                 num_cache: int = 1,
                 ):
        super().__init__()
        # self.parquet_list = [i.absolute()
        # for i in Path(parquet_path).glob('*.parquet')]
        self.parquet_paths = [
            i for i in Path(parquet_path).iterdir() if i.is_dir()
        ]
        self.parquet_list = [
            pas.dataset(path) for path in self.parquet_paths
        ]
        self.idx_col = idx_column
        self.local_cache_index = 0  # index for current cache
        self.num_cache = num_cache
        self.use_column = use_column

        self.local_parquet_index = 0  # file index to use next
        self.local_parquet_list = None
        self.local_parquet_length = 0
        self.local_parquet_data = []  # Dataset columns to pylist
        self.local_parquet_batche = [].__iter__()

    def total_length(self):
        length = [i.count_rows(columns=['name']) for i in self.parquet_list]
        length = reduce(lambda x, y: x + y, length, 0)
        return length

    def fetch_local_parquet_list(self) -> List[pas.Dataset]:
        info = data.get_worker_info()
        if info is None:
            return
        total = len(self.parquet_list)
        amount = total // info.num_workers
        left = total - amount * info.num_workers

        offset = 0 if (left // (info.id + 1)) == 0 else left

        start = info.id * amount + offset
        end = start + amount
        assert start < len(self.parquet_list)

        if end > len(self.parquet_list):
            end = len(self.parquet_list)

        item = self.parquet_paths[start:end]
        self.local_parquet_length = len(item)
        return item

    def load_parquet(self):
        next = self.local_parquet_index
        if next >= self.local_parquet_length:
            next = 0
            self.local_parquet_index = 0
        path = self.local_parquet_list[next]
        dset = pas.dataset(path)
        self.local_parquet_index += 1

        batches = dset.to_batches(
            fragment_readahead=2, batch_size=8, batch_readahead=8)

        return batches

    def __len__(self):
        return self.total_length()

    def __getitem__(self, index):
        # check current file end
        if len(self.local_parquet_data) <= 3:
            # if we use all parquet data in current cache
            if self.local_parquet_list is None:
                self.local_parquet_list = self.fetch_local_parquet_list()

            batch = next(self.local_parquet_batche, None)
            if batch is None:
                self.local_parquet_batche = self.load_parquet()
                batch = next(self.local_parquet_batche, None)
            batch = batch.column('image').to_pylist()
            self.local_parquet_data = batch

        return self.local_parquet_data.pop(0)
