import argparse, requests, time, random
from multiprocessing import Pool
from datasets import load_dataset_builder
from typing import Tuple
from pathlib import Path

from gpt_lib.utils.default import CACHE_DIR, DATA_DIR, RANDOM_SEED
from gpt_lib.utils.schemas import DownloadConfig


make_url = lambda file_rep, file_name: f"https://huggingface.co/datasets/{file_rep}/resolve/main/{file_name}"

def make_file_config(address) -> Tuple[str]:
    parts = address.split("/")
    ds_proprietor = parts[3]
    ds_name = parts[4].split("@")[0]
    file_base = f"{ds_proprietor}/{ds_name}"
    file_name = "/".join(parts[5:])
    return file_base, file_name


def download_single_parquet(file_address: str, config: DownloadConfig) -> bool:
    file_base, file_name = make_file_config(file_address)
    file_path: Path = config.data_dir / file_base / file_name
    print(f"Downloading {file_name} to {file_path}...")
    if file_path.exists():
        print(f"{file_name} already exists. Skipping download.")
        return True
    
    url = make_url(file_base, file_name)

    max_attempts = config.max_retries
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024**2):
                    f.write(chunk)
            print(f"Downloaded {file_name} successfully.")
            return True
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                if file_path.exists():
                    file_path.unlink()
                print(f"Failed to download {file_name} after {max_attempts} attempts.")
                return False
            else:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                print("Retrying...")
    return False

def download_parquets(ds_name: str, config: DownloadConfig):
    ds_builder = load_dataset_builder(ds_name, cache_dir=config.data_dir)

    shards = list(set(ds_builder.config.data_files.get("train", [])))
    max_shards = min(len(shards), config.max_shards) if len(shards) > 0 else config.max_shards

    if max_shards < len(shards):
        shards = random.sample(shards, k=max_shards)

    # TODO: move download single parquet outside function to avoid redefinition error
    with Pool(processes=config.num_workers) as pool:
        results = pool.starmap(download_single_parquet, [(shard, config) for shard in shards])
    
    successful_downloads = sum(1 for res in results if res)
    print(f"Downloaded {successful_downloads} out of {config.max_shards} shards successfully.")