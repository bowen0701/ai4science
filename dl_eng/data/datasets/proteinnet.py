"""ProteinNet dataset utilities."""

import os
import shutil
import tarfile
import requests
from typing import List, Optional
from dl_eng.data.base import BaseDataModule

class ProteinNetDataModule(BaseDataModule):
    """
    DataModule for ProteinNet CASP11 dataset.
    Handles downloading, extraction, and sharding.

    Usage:
        from dl_eng.data.datasets.proteinnet import ProteinNetDataModule
        dm = ProteinNetDataModule(data_dir="./artifacts/data/proteinnet", n_shards=4)
        dm.prepare_data(force=True)
        dm.setup(stage="casp11/training_50", rank=0, world_size=2)
        dm.setup(stage=None, rank=0, world_size=2)
    """
    
    CASP11_URL = "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp11.tar.gz"

    def __init__(self, data_dir: str = "./artifacts/data/proteinnet", n_shards: int = 4, format: str = "csv"):
        super().__init__(data_dir)
        self.n_shards = n_shards
        self.format = format
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.tar_path = os.path.join(self.raw_dir, "casp11.tar.gz")
        self.extracted_dir = os.path.join(self.data_dir, "extracted")

    def prepare_data(self, force: bool = False) -> None:
        """Download and extract ProteinNet CASP11 data."""
        os.makedirs(self.raw_dir, exist_ok=True)

        if force:
            if os.path.exists(self.tar_path):
                os.remove(self.tar_path)
            if os.path.exists(self.extracted_dir):
                shutil.rmtree(self.extracted_dir)

        # 1. Download
        if not os.path.exists(self.tar_path):
            self.logger.info("Downloading ProteinNet CASP11 (this may take a while)...")
            response = requests.get(self.CASP11_URL, stream=True)
            response.raise_for_status()
            with open(self.tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # 2. Extract
        if not os.path.exists(self.extracted_dir):
            self.logger.info("Extracting CASP11...")
            with tarfile.open(self.tar_path, "r:gz") as tar:
                tar.extractall(self.extracted_dir)
                
        self.logger.info(f"ProteinNet preparation complete in {self.extracted_dir}")

    def _available_stages(self) -> List[str]:
        if not os.path.exists(self.extracted_dir):
            return []

        stages = []
        for root, _, files in os.walk(self.extracted_dir):
            for filename in files:
                relative_path = os.path.relpath(os.path.join(root, filename), self.extracted_dir)
                stages.append(relative_path)

        return sorted(stages)

    def _resolve_stages(self, stage: Optional[str]) -> List[str]:
        available_stages = self._available_stages()
        if not available_stages:
            return []

        if stage is None:
            return available_stages

        if stage not in available_stages:
            raise ValueError(
                f"Unsupported stage '{stage}'. Expected one of {available_stages}."
            )
        return [stage]

    def setup(self, stage: Optional[str] = None, rank: int = 0, world_size: int = 1) -> None:
        """Assign extracted files to a simulated distributed rank."""
        if not os.path.exists(self.extracted_dir):
            self.logger.error("Data not prepared. Run prepare_data first.")
            return

        stages = self._resolve_stages(stage)
        if not stages:
            self.logger.error("No ProteinNet stages found. Run prepare_data first.")
            return

        for split in stages:
            split_path = os.path.join(self.extracted_dir, split)
            my_files = [split_path][rank::world_size]

            self.logger.info(
                f"Stage '{split}': rank {rank}/{world_size} assigned {len(my_files)} files."
            )
            if my_files:
                self.logger.info(f"Stage '{split}': first assigned file: {my_files[0]}")
