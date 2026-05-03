import os
import tarfile
import requests
from typing import Optional
from dl_eng.data.base import BaseDataModule

class ProteinNetDataModule(BaseDataModule):
    """
    DataModule for ProteinNet CASP11 dataset.
    Handles downloading, extraction, and sharding.
    """
    
    CASP11_URL = "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp11.tar.gz"

    def __init__(self, data_dir: str = "./artifacts/data/proteinnet", n_shards: int = 4, format: str = "csv"):
        super().__init__(data_dir)
        self.n_shards = n_shards
        self.format = format
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.tar_path = os.path.join(self.raw_dir, "casp11.tar.gz")
        self.extracted_dir = os.path.join(self.data_dir, "extracted")

    def prepare_data(self) -> None:
        """Download and extract ProteinNet data."""
        os.makedirs(self.raw_dir, exist_ok=True)
        
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

    def setup(self, stage: Optional[str] = None, rank: int = 0, world_size: int = 1) -> None:
        """Assign files/shards to ranks."""
        if not os.path.exists(self.extracted_dir):
            self.logger.error("Data not prepared. Run prepare_data first.")
            return

        # Find all files in the extracted directory
        all_files = []
        for root, _, files in os.walk(self.extracted_dir):
            for f in files:
                all_files.append(os.path.join(root, f))
        
        all_files = sorted(all_files)
        my_files = all_files[rank::world_size]
        
        self.logger.info(f"Rank {rank}/{world_size} assigned {len(my_files)} files.")
        if len(my_files) > 0:
            self.logger.info(f"First assigned file: {my_files[0]}")
