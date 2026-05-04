"""AG News dataset utilities."""

import os
import shutil
import requests
from typing import List, Optional
from dl_eng.data.base import BaseDataModule
from dl_eng.data.sharder import DataSharder

class AGNewsDataModule(BaseDataModule):
    """
    DataModule for AG News dataset.
    Supports downloading and sharding for distributed training.

    Usage:
        from dl_eng.data.datasets.ag_news import AGNewsDataModule
        dm = AGNewsDataModule(data_dir="./artifacts/data/ag_news", n_shards=4, format="csv")
        dm.prepare_data(force=True)
        dm.setup(stage="train", rank=0, world_size=2)
        dm.setup(stage=None, rank=0, world_size=2)
    """
    
    URLS = {
        "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    }

    def __init__(self, data_dir: str = "./artifacts/data/ag_news", n_shards: int = 4, format: str = "csv"):
        super().__init__(data_dir)
        self.n_shards = n_shards
        self.format = format
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.shard_dir = os.path.join(self.data_dir, "shards")

    def prepare_data(self, force: bool = False) -> None:
        """Download the raw CSV files and shard them for training."""
        os.makedirs(self.raw_dir, exist_ok=True)

        for split, url in self.URLS.items():
            raw_file = os.path.join(self.raw_dir, f"{split}.csv")
            split_shard_dir = os.path.join(self.shard_dir, split)

            if force:
                if os.path.exists(raw_file):
                    os.remove(raw_file)
                if os.path.exists(split_shard_dir):
                    shutil.rmtree(split_shard_dir)

            if not os.path.exists(raw_file):
                self.logger.info(f"Downloading {split} split...")
                response = requests.get(url)
                with open(raw_file, "wb") as f:
                    f.write(response.content)

            # Shard the data
            DataSharder.split_csv(
                input_csv=raw_file,
                output_dir=split_shard_dir,
                n_shards=self.n_shards,
                prefix=split,
                fmt=self.format
            )

    def _resolve_splits(self, stage: Optional[str]) -> List[str]:
        if stage is None:
            return list(self.URLS.keys())
        if stage not in self.URLS:
            raise ValueError(
                f"Unsupported stage '{stage}'. Expected one of {sorted(self.URLS.keys())}."
            )
        return [stage]

    def setup(self, stage: Optional[str] = None, rank: int = 0, world_size: int = 1) -> None:
        """Select the shards assigned to a simulated distributed rank."""
        splits = self._resolve_splits(stage)

        for split in splits:
            split_shard_dir = os.path.join(self.shard_dir, split)

            if not os.path.exists(split_shard_dir):
                self.logger.error(
                    f"Shard directory {split_shard_dir} does not exist. Run prepare_data first."
                )
                continue

            all_shards = sorted(
                f for f in os.listdir(split_shard_dir) if f.endswith(f".{self.format}")
            )
            my_shards = all_shards[rank::world_size]
            self.logger.info(
                f"Stage '{split}': rank {rank}/{world_size} loading "
                f"{self.format.upper()} shards: {my_shards}"
            )
