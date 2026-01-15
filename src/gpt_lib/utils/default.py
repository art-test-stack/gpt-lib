import torch
import os
from pathlib import Path
from gpt_lib.utils.special_tokens import SpecialTokens

# ----------- PROCESSOR -----------

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(0.)
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

NUM_THREADS = os.cpu_count() # 16

# ------------- DATA -------------

IS_TIKTOKEN = False # TODO: parse as arg

SPECIAL_TOKENS = SpecialTokens()
FORCED_TOKENS = []

# CACHE_DIR = Path.home() / ".gpt_lib"
CACHE_DIR = Path(".gpt_lib")
DATA_FOLDER = CACHE_DIR / ".data"
MIN_DOCUMENT_SIZE = 0
OUTPUT_FOLDER = CACHE_DIR / ".output"
MODELS_FOLDER = CACHE_DIR / ".models"
VOCAB_SIZE = 32_000
VOCAB_FILE = DATA_FOLDER / "vocab.json"
MAX_TOKEN_LENGTH = 32

# TOKEN_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # GPT 4 SPLIT

PAT_STR_GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_STR_GPT4 = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# ------------- DRIVE -------------

SAVE_ON_DRIVE = True
DRIVE_FILE = ""
SAVE_ON_WANDB = True

# ------------- MODEL -------------

VOCAB_SIZE = 32_000
MAX_CONTEXT = 64

NUM_HEADS = 2
NUM_LAYERS = 2

DIM_MODEL = 128
DIM_FFN = 4 * DIM_MODEL

DIM_HEAD = DIM_MODEL // NUM_HEADS
# DIM_KEY = DIM_MODEL // NUM_HEADS
# DIM_VALUE = DIM_MODEL // NUM_HEADS

DROPOUT = .1

MASK_VALUE = -1e9
LINEAR_BIAS = False
FLASH_ATTENTION = False # TODO: Not implemented

# ------------- TRAIN -------------

BATCH_SIZE = 128
PRETRAINING_VAL_RATIO = 1e-3

MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_ITERS = 2_000

WEIGHT_DECAY = .1
DECAY_ITERS = 100_000

BETA_1 = .9
BETA_2 = .95

EPSILON = 1e-8

VALIDATION_STEP = 50

RANDOM_SEED = 42

adamw_opt_params = {
    "weight_decay": WEIGHT_DECAY,
    "beta_1": BETA_1,
    "beta_2": BETA_2,
    "epsilon": EPSILON
}
muon_opt_params = {
    "lr_min": MIN_LEARNING_RATE,
    "lr_max": MAX_LEARNING_RATE,
    "warmup_steps": WARMUP_ITERS,
    "decay_steps": DECAY_ITERS,
    "weight_decay": WEIGHT_DECAY,
    "beta1": BETA_1,
    "beta2": BETA_2,
    "epsilon": EPSILON
}

opt_params = dict(
    adamw=adamw_opt_params,
    muon=muon_opt_params
)
