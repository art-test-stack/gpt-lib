import torch
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from bisect import bisect_right, insort
from collections import defaultdict

from tqdm import tqdm

from gpt_lib.utils.distributed import get_dist_info
from gpt_lib.data.loader import list_parquet_files


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No dataset parquet files found."
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1

    if resume_state_dict and "world_size" in resume_state_dict:
        assert resume_state_dict["world_size"] == ddp_world_size, \
            "DDP world size changed between runs."

    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0

        while pq_idx < len(parquet_paths):
            pf = pq.ParquetFile(parquet_paths[pq_idx])

            if first_pass and resume_rg_idx is not None and pq_idx == resume_pq_idx:
                base_idx = resume_rg_idx // ddp_world_size + 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank

            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()

                for i in range(0, len(texts), tokenizer_batch_size):
                    yield texts[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)

                rg_idx += ddp_world_size

            pq_idx += 1

        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    buffer_size=1000,
    min_buffer_threshold=None,
):
    assert split in ["train", "val"]

    device_obj = torch.device(device)
    use_cuda = device_obj.type == "cuda"

    row_capacity = T + 1
    min_buffer_threshold = min_buffer_threshold or buffer_size // 2

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.bos_token_id

    # -----------------------------------------------------------------------
    # Token buffer: length-bucketed for O(log N) best-fit
    # -----------------------------------------------------------------------

    length_buckets = defaultdict(list)
    sorted_lengths = []

    pq_idx = rg_idx = epoch = 0

    def add_doc_tensor(doc_tensor):
        length = doc_tensor.size(0)
        if length not in length_buckets:
            insort(sorted_lengths, length)
        length_buckets[length].append(doc_tensor)

    def pop_best_fit(remaining):
        idx = bisect_right(sorted_lengths, remaining) - 1
        if idx < 0:
            return None
        length = sorted_lengths[idx]
        doc = length_buckets[length].pop()
        if not length_buckets[length]:
            del length_buckets[length]
            sorted_lengths.pop(idx)
        return doc

    def pop_min_overflow_crop(remaining):
        # choose doc minimizing overflow
        best_len = None
        best_overflow = None
        for length in sorted_lengths:
            overflow = length - remaining
            if overflow >= 0:
                if best_overflow is None or overflow < best_overflow:
                    best_overflow = overflow
                    best_len = length
        if best_len is None:
            # fallback: longest available
            best_len = sorted_lengths[-1]

        doc = length_buckets[best_len].pop()
        if not length_buckets[best_len]:
            del length_buckets[best_len]
            sorted_lengths.remove(best_len)

        return doc[:remaining]

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        texts, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(
            texts,
            prepend=bos_token,
            num_threads=tokenizer_threads
        )
        for tokens in token_lists:
            add_doc_tensor(torch.tensor(tokens, dtype=torch.long))

    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device_obj)

    cpu_inputs = cpu_buffer[:B*T].view(B, T)
    cpu_targets = cpu_buffer[B*T:].view(B, T)

    inputs = gpu_buffer[:B*T].view(B, T)
    targets = gpu_buffer[B*T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                if len(sorted_lengths) < min_buffer_threshold:
                    refill_buffer()
                remaining = row_capacity - pos
                doc = pop_best_fit(remaining)

                if doc is None:
                    doc = pop_min_overflow_crop(remaining)

                doc_len = doc.size(0)
                if doc_len > 1:
                    end = pos + doc_len - 1
                    cpu_inputs[row_idx, pos:end] = doc[:-1]
                    cpu_targets[row_idx, pos:end] = doc[1:]
                    pos += doc_len
                else:
                    pos += doc_len

        state_dict = {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "epoch": epoch,
            "world_size": get_dist_info()[3],
        }

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    for inputs, targets, _ in \
        tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


def pretokenize_split(
    tokenizer,
    split,
    output_dir,
    tokenizer_threads=8,
    dtype=np.uint32
):  
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    bin_path = output_dir / f"{split}.bin"
    idx_path = output_dir / f"{split}.idx"

    offsets = []
    total_tokens = 0

    bos_token = tokenizer.bos_token_id

    with open(bin_path, "wb") as bin_file:
        for path in tqdm(parquet_paths, desc=f"Pretokenizing {split}"):

            pf = pq.ParquetFile(path)

            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
                token_lists = tokenizer.encode(
                    texts,
                    prepend=bos_token,
                    num_threads=tokenizer_threads
                )
                for tokens in token_lists:
                    arr = np.asarray(tokens, dtype=dtype)
                    offsets.append(total_tokens)
                    bin_file.write(arr.tobytes())
                    total_tokens += len(arr)

    # write index file
    offsets = np.asarray(offsets, dtype=np.uint64)
    offsets.tofile(idx_path)
    print(f"Done. {total_tokens:,} tokens written.")

class PretokenizedDataset:
    def __init__(self, data_dir, split):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        bin_path = data_dir / f"{split}.bin"
        idx_path = data_dir / f"{split}.idx"

        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.offsets = np.fromfile(idx_path, dtype=np.uint64)

        self.num_docs = len(self.offsets)

    def __len__(self):
        return self.num_docs

    def get_doc(self, idx):
        start = self.offsets[idx]
        if idx + 1 < self.num_docs:
            end = self.offsets[idx + 1]
        else:
            end = len(self.tokens)
        return torch.from_numpy(self.tokens[start:end].astype(np.int64))


class DistributedDocIterator:
    def __init__(self, dataset: PretokenizedDataset):
        _, rank, _, world_size = get_dist_info()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.ptr = rank

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= len(self.dataset):
            self.ptr = self.rank
        doc = self.dataset.get_doc(self.ptr)
        self.ptr += self.world_size
        return doc
