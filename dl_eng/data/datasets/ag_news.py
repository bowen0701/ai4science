import os
import requests
from typing import Optional
from dl_eng.data.base import BaseDataModule
from dl_eng.data.sharder import DataSharder

class AGNewsDataModule(BaseDataModule):
    """
    DataModule for AG News dataset.
    Supports downloading and sharding for distributed training.
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

    def prepare_data(self) -> None:
        """Download raw CSVs and shard them."""
        os.makedirs(self.raw_dir, exist_ok=True)
        
        for split, url in self.URLS.items():
            raw_file = os.path.join(self.raw_dir, f"{split}.csv")
            if not os.path.exists(raw_file):
                self.logger.info(f"Downloading {split} split...")
                response = requests.get(url)
                with open(raw_file, "wb") as f:
                    f.write(response.content)
            
            # Shard the data
            split_shard_dir = os.path.join(self.shard_dir, split)
            DataSharder.split_csv(
                input_csv=raw_file,
                output_dir=split_shard_dir,
                n_shards=self.n_shards,
                prefix=split,
                fmt=self.format
            )

    def setup(self, stage: Optional[str] = None, rank: int = 0, world_size: int = 1) -> None:
        """Logic to load shards based on rank."""
        split = stage or "train"
        split_shard_dir = os.path.join(self.shard_dir, split)
        
        if not os.path.exists(split_shard_dir):
            self.logger.error(f"Shard directory {split_shard_dir} does not exist. Run prepare_data first.")
            return

        all_shards = sorted([f for f in os.listdir(split_shard_dir) if f.endswith(f".{self.format}")])
        my_shards = all_shards[rank::world_size]
        self.logger.info(f"Rank {rank}/{world_size} loading {self.format.upper()} shards: {my_shards}")
