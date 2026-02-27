import time
from gpt_lib.data.pretraining_loader import tokenizing_distributed_data_loader_bos_bestfit
from gpt_lib.utils.schemas import TokenizerConfig
from gpt_lib.tokenizer.tokenizer import Tokenizer
from gpt_lib.utils.report import get_report
from gpt_lib.utils.common import get_banner, print0
from gpt_lib.utils.default import DATA_DIR, DEVICE_NAME

# init tokenizer
tokenizer = Tokenizer.from_disk("ic1-32k")
loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer=tokenizer,
    B=4,
    T=1024,
    split="train",
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device=DEVICE_NAME,
)

if __name__ == "__main__":
    print(get_banner())

    print0("Testing tokenizing distributed-data loader from disk...")
    t0 = time.time()
    for i, (inputs, targets) in enumerate(loader):
        if i >= 5:
            break
        print(f"Batch {i}: inputs shape {inputs.shape}, targets shape {targets.shape}")
    t1 = time.time()
    print0(f"Time taken for 5 batches: {t1 - t0:.2f} seconds")