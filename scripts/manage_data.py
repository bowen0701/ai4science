"""Dataset management CLI.

AG News usage:
    python scripts/manage_data.py prepare_data --dataset ag_news --shards 4 --format csv --force
    python scripts/manage_data.py setup --dataset ag_news --stage train --rank 0 --world_size 2

ProteinNet usage:
    python scripts/manage_data.py prepare_data --dataset proteinnet --force
    python scripts/manage_data.py setup --dataset proteinnet --stage casp11/training_50 --rank 0 --world_size 2
"""

import argparse
from dl_eng.data.datasets.ag_news import AGNewsDataModule
from dl_eng.data.datasets.proteinnet import ProteinNetDataModule

def main():
    """Run the dataset management CLI."""
    parser = argparse.ArgumentParser(description="DL-Eng Data Management CLI")
    parser.add_argument("action", choices=["prepare_data", "setup"], help="Action to perform")
    parser.add_argument("--dataset", choices=["ag_news", "proteinnet"], default="ag_news", help="Dataset name")
    parser.add_argument("--shards", type=int, default=4, help="Number of shards to create")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Data format (default: csv)")
    parser.add_argument("--force", action="store_true", help="Rebuild dataset artifacts from scratch during prepare_data")
    parser.add_argument("--stage", default=None, help="Dataset split/stage to load during setup")
    parser.add_argument("--rank", type=int, default=0, help="Simulated rank for setup")
    parser.add_argument("--world_size", type=int, default=1, help="Simulated world_size for setup")

    args = parser.parse_args()

    if args.dataset == "ag_news":
        dm = AGNewsDataModule(n_shards=args.shards, format=args.format)
    elif args.dataset == "proteinnet":
        dm = ProteinNetDataModule(n_shards=args.shards, format=args.format)
    else:
        print(f"Dataset {args.dataset} not implemented.")
        return

    if args.action == "prepare_data":
        dm.prepare_data(force=args.force)
    elif args.action == "setup":
        dm.setup(stage=args.stage, rank=args.rank, world_size=args.world_size)

if __name__ == "__main__":
    main()
