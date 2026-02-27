from pathlib import Path
from gpt_lib.utils.default import CACHE_DIR, RANDOM_SEED
import argparse, random
from gpt_lib.data.download import download_parquets
from gpt_lib.utils.schemas import DownloadConfig

random.seed(RANDOM_SEED)

DATA_PATH = CACHE_DIR / ".data"
DATA_PATH.mkdir(parents=True, exist_ok=True)
BASE_DATASET = "HuggingFaceFW/fineweb-edu"
BASE_URL = f"https://huggingface.co/datasets/{BASE_DATASET}/resolve/main"
MAX_SHARD = 1000  # Example maximum number of shards available

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download parquet shards for dataset.")
    parser.add_argument("--num_shards", type=int, default=-1, help="Number of parquet shards to download. Default is -1 (download all).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel download workers. Default is 4.")
    args = parser.parse_args()

    num_shards = min(args.num_shards, MAX_SHARD) if args.num_shards > 0 else MAX_SHARD
    num_workers = args.num_workers

    download_config = DownloadConfig(
        max_shards=num_shards,
        num_workers=num_workers
    )
    ds_names = ["HuggingFaceFW/fineweb-edu", "HuggingFaceTB/finemath"]
    for ds_name in ds_names:
        download_parquets(ds_name, download_config)
    
    