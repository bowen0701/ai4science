from abc import ABC, abstractmethod
from typing import Any, Optional

from ai4science.utils.logging import setup_logger

class BaseDataModule(ABC):
    """
    Abstract base class for all data modules in ai4science.
    
    Inspired by PyTorch Lightning, this class separates:
    - prepare_data: Download, sharded, and save to disk (run once).
    - setup: Load data into memory/datasets based on rank/world_size.

    Usage:
        class MyDataModule(BaseDataModule):
            def prepare_data(self):
                # Download/shard logic
            def setup(self, rank, world_size):
                # Load shards based on rank
        
        dm = MyDataModule()
        dm.prepare_data()
        dm.setup(rank=0, world_size=1)
    """
    
    def __init__(self, data_dir: str = "./artifacts/data"):
        self.data_dir = data_dir
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def prepare_data(self, force: bool = False) -> None:
        """Download and shard data. Set force=True to rebuild existing artifacts."""
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None, rank: int = 0, world_size: int = 1) -> None:
        """Load data based on rank and world_size for distributed training."""
        pass
