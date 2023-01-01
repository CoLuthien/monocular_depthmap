
import time
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from torch.utils import data


class BaseParquetDataset(data.Dataset):
    def __init__(self,
                 parquet_path: str,
                 idx_column: List[str],
                 use_column: List[str],
                 num_cache: int = 1,
                 ):
        super().__init__()
        self.idx_col = idx_column
        self.parquet_list = [i.absolute()
                             for i in Path(parquet_path).glob('*.parquet')]
        self.local_cache_index = 0  # index for current cache
        self.num_cache = num_cache
        self.use_column = use_column

        self.local_parquet_index = 0  # file index to use next
        self.local_parquet_list = None
        self.local_parquet_length = 0
        self.local_parquet_data = []  # Dataset columns to pylist

    def total_length(self):
        pfs = pq.ParquetDataset(self.parquet_list)
        idx_col = pfs.read(self.idx_col)
        return len(idx_col)

    def fetch_local_parquet_list(self):
        info = data.get_worker_info()
        amount = round(len(self.parquet_list) / info.num_workers)
        amount = int(amount)
        start = info.id * amount
        end = start + amount
        if end > len(self.parquet_list):
            end = len(self.parquet_list)

        item = self.parquet_list[start:end]
        self.local_parquet_length = len(item)
        return item

    def load_parquet(self):
        next = self.local_parquet_index
        if next >= self.local_parquet_length:
            next = 0
            self.local_parquet_index = 0

        table = pq.read_table(
            self.local_parquet_list[next], columns=self.use_column).to_pylist()
        self.local_parquet_index += 1
        return table

    def __len__(self):
        return self.total_length()

    def __getitem__(self, index) -> None:
        # check current file end
        if len(self.local_parquet_data) <= 1:
            # if we use all parquet data in current cache
            if self.local_parquet_index > self.local_parquet_length:
                self.local_parquet_index = 0
            if self.local_parquet_list is None:
                self.local_parquet_list = self.fetch_local_parquet_list()

            self.local_parquet_data = self.load_parquet()

        return self.local_parquet_data.pop()
