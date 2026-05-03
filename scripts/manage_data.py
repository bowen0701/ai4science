import argparse
from dl_eng.data.datasets.ag_news import AGNewsDataModule
from dl_eng.data.datasets.proteinnet import ProteinNetDataModule

def main():
    parser = argparse.ArgumentParser(description="DL-Eng Data Management CLI")
    parser.add_argument("action", choices=["prepare", "test_setup"], help="Action to perform")
    parser.add_argument("--dataset", choices=["ag_news", "proteinnet"], default="ag_news", help="Dataset name")
    parser.add_argument("--shards", type=int, default=4, help="Number of shards to create")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Data format (default: csv)")
    parser.add_argument("--rank", type=int, default=0, help="Simulated rank for test_setup")
    parser.add_argument("--world_size", type=int, default=1, help="Simulated world_size for test_setup")

    args = parser.parse_args()

    if args.dataset == "ag_news":
        dm = AGNewsDataModule(n_shards=args.shards, format=args.format)
    elif args.dataset == "proteinnet":
        dm = ProteinNetDataModule(n_shards=args.shards)
    else:
        print(f"Dataset {args.dataset} not implemented.")
        return

    if args.action == "prepare":
        dm.prepare_data()
    elif args.action == "test_setup":
        dm.setup(rank=args.rank, world_size=args.world_size)

if __name__ == "__main__":
    main()
