import os
import numpy as np
import pandas as pd
from typing import List
from ai4science.infra.logger import setup_logger

logger = setup_logger("DataSharder")

class DataSharder:
    """Utility to split large datasets into shards for DDP/FSDP."""
    
    @staticmethod
    def shard_dataframe(
        df: pd.DataFrame, 
        output_dir: str, 
        n_shards: int, 
        prefix: str = "shard",
        fmt: str = "csv"
    ) -> List[str]:
        """
        Splits a DataFrame into N shards of the specified format.
        """
        os.makedirs(output_dir, exist_ok=True)
        total_rows = len(df)
        rows_per_shard = int(np.ceil(total_rows / n_shards))
        
        shard_paths = []
        for i in range(n_shards):
            start_idx = i * rows_per_shard
            end_idx = min((i + 1) * rows_per_shard, total_rows)
            
            if start_idx >= total_rows:
                break
                
            shard_df = df.iloc[start_idx:end_idx]
            shard_path = os.path.join(output_dir, f"{prefix}_{i:03d}.{fmt}")
            
            if fmt == "parquet":
                shard_df.to_parquet(shard_path, index=False)
            else:
                shard_df.to_csv(shard_path, index=False)
                
            shard_paths.append(shard_path)
            
        logger.info(f"Split {total_rows} rows into {len(shard_paths)} {fmt.upper()} shards in {output_dir}")
        return shard_paths

    @staticmethod
    def split_csv(
        input_csv: str, 
        output_dir: str, 
        n_shards: int, 
        prefix: str = "shard",
        fmt: str = "csv"
    ) -> List[str]:
        """Reads a CSV and shards it."""
        logger.info(f"Reading {input_csv} for sharding...")
        df = pd.read_csv(input_csv)
        return DataSharder.shard_dataframe(df, output_dir, n_shards, prefix, fmt)
