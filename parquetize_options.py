
import os
import argparse


class ParquetizeOption:
    """
    a parquetizer's options to parse root/dataset/group/*.jpg
    and make into one file
    """

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="directory options")

        parser.add_argument("--root-dir",
                            type=str,
                            required=True
                            )
        parser.add_argument("--output-dir",
                            type=str,
                            required=True)
        parser.add_argument("--output-name",
                            type=str,
                            required=False,
                            default="result.parquet"
                            )

        parser.add_argument("--file-ext",
                            type=str,
                            required=False,
                            default="png"
                            )
