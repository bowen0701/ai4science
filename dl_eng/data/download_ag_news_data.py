import os
import shutil

import requests
from bento import fwdproxy  # Bento required


DATA_DIR = "./ag_news_data"

AG_NEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

os.makedirs(DATA_DIR, exist_ok=True)


def _split_csv_into_chunks(csv_path, out_dir, lines_per_file=5000):
    """Split a CSV file into multiple files, each with up to lines_per_file lines."""
    part = 0
    lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            lines.append(line)
            if i % lines_per_file == 0:
                part += 1
                out_file = os.path.join(
                    out_dir,
                    f"{os.path.splitext(os.path.basename(csv_path))[0]}-part-{part:05d}.csv",
                )
                with open(out_file, "w", encoding="utf-8") as out_f:
                    out_f.writelines(lines)
                lines = []
        # Last chunk
        if lines:
            part += 1
            out_file = os.path.join(
                out_dir,
                f"{os.path.splitext(os.path.basename(csv_path))[0]}-part-{part:05d}.csv",
            )
            with open(out_file, "w", encoding="utf-8") as out_f:
                out_f.writelines(lines)
    print(f"[{os.path.basename(csv_path)}] Split into {part} files in {out_dir}")


# pyre-ignore
def download_ag_news(base_dir=DATA_DIR, overwrite=False):
    """Download AG News dataset from GitHub."""
    if overwrite:
        shutil.rmtree(base_dir, ignore_errors=True)

    for split, url in AG_NEWS_URLS.items():
        split_path = os.path.join(base_dir, split)
        os.makedirs(split_path, exist_ok=True)

        out_file = os.path.join(split_path, f"{split}.csv")
        if not os.path.exists(out_file):
            print(f"[{split}] Downloading from {url} ...")
            with fwdproxy():
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(out_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"[{split}] Saved to {out_file}")
        else:
            print(f"[{split}] File already exists: {out_file}")

        # Split into 10,000-line chunks
        _split_csv_into_chunks(out_file, split_path, lines_per_file=10_000)

        # Remove original large CSV
        print(f"[{split}] Removing original file: {out_file}")
        os.remove(out_file)


# Run in Jupyter
# download_ag_news()