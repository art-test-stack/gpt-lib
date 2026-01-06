import torch
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, List, Callable, Literal
from torch import nn
from pathlib import Path
import json
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

from gpt_lib.utils.default import (
    DEVICE,
    MODELS_FOLDER, 
    VOCAB_SIZE, 
    MAX_CONTEXT, 
    NUM_HEADS, 
    NUM_LAYERS, 
    DIM_MODEL, 
    DIM_FFN, 
    DIM_HEAD, 
    DROPOUT, 
    BATCH_SIZE, 
    MAX_LEARNING_RATE, 
    MIN_LEARNING_RATE, 
    WARMUP_ITERS, 
    VALIDATION_STEP, 
    PRETRAINING_VAL_RATIO,
    PAT_STR_GPT2,
    PAT_STR_GPT4
)
from gpt_lib.utils.special_tokens import SpecialTokens

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class TokenizerConfig(BaseModel):
    name: str = "yc_tok1"
    dirname: Path = MODELS_FOLDER
    vocab_size: int | None = VOCAB_SIZE
    max_context: int | None = MAX_CONTEXT
    pat_str: str | None = PAT_STR_GPT2
    special_tokens: SpecialTokens | None = Field(default_factory=SpecialTokens)
    source: Literal["tiktoken", "bytelevelbpe", "rustbpe", "huggingface", "dummy"] = "tiktoken"

    def model_post_init(self, context: Any) -> None:
        self.dirname = self.dirname / self.name
        if not self.dirname.exists():
            self.dirname.mkdir(parents=True, exist_ok=True)

    def get_mergeable_ranks(self) -> dict:
        if not self.dirname.exists():
            raise FileNotFoundError(f"Tokenizer directory {self.dirname} does not exist.")
        mergeable_ranks_path = self.dirname / "mergeable_ranks.pkl"
        if not mergeable_ranks_path.exists():
            raise FileNotFoundError(f"Mergeable ranks file {mergeable_ranks_path} does not exist.")
        with open(mergeable_ranks_path, "rb") as f:
            mergeable_ranks = pickle.load(f)
        assert len(mergeable_ranks) == self.vocab_size, "Mergeable ranks size does not match vocab size."
        return mergeable_ranks

class TrainingTokenizerConfig(TokenizerConfig):
    max_chars: int = 10_000_000_000
    chars_per_doc: int = 10_000
    merges_per_pass: int = 512

class TransformerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    vocab_size: int = VOCAB_SIZE
    max_context: int = MAX_CONTEXT
    pad_id: int = -100

    positional_encoding: Literal["positional", "rope"] = "rope" # Options: "positional", "rope"

    d_model: int = DIM_MODEL
    d_ffn: int = DIM_FFN  # 4 * dim_model
    n_heads: int = NUM_HEADS
    n_layers: int = NUM_LAYERS
    d_head: int = DIM_HEAD  # dim_model // num_heads

    dropout: float = DROPOUT
    norm_before_attn: bool = True

    attn_type: Literal["sdpa", "flash", "torch"] = "sdpa"  # Options: "sdpa", "flash", "torch"
    enable_gqa: bool = False

    softcap: float = 18.0
    
    def model_post_init(self, context: Any) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.d_head != self.d_model // self.n_heads:
            warnings.warn(f"d_head ({self.d_head}) is not equal to d_model/n_heads ({self.d_model // self.n_heads}). This may lead to unexpected behavior in attention mechanisms.")
        if not self.norm_before_attn:
            warnings.warn("Using post-attention normalization (norm_before_attn=False) may lead to training instability.")
        if self.attn_type == "flash":
            try:
                import flash_attn
            except ImportError:
                warnings.warn("FlashAttention is not installed. Falling back to standard attention.")
                self.attn_type = "sdpa"

class DenseTransformerConfig(TransformerConfig):
    tf_type: Literal["dense"] = "dense"

class MoETransformerConfig(TransformerConfig):
    tf_type: Literal["moe"] = "moe"
    nb_experts: int = 16
    expert_capacity_factor: float = 1.0

class ObjectiveConfig(BaseModel):
    objective_fn: Literal["cross_entropy", "kl_divergence"] = "cross_entropy"
    kwargs: dict = Field(default_factory=dict)
    ignore_index: int = -100
    reduction: Literal["none", "mean", "sum"] = "none"

class GenerationConfig(BaseModel):
    max_length: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1
    stream: bool = False

class GPTConfig(BaseModel):
    """
    # GPTConfig
    GPTConfig is the configuration class for GPT models. It encapsulates all the necessary settings for
    defining the architecture, tokenizer, and training objectives of a GPT model. It provides methods 
    to save and load configurations. It derives from Pydantic's BaseModel for easy serialization and validation.

    Args:
        name (str): The name of the model.
        tokenizer (TokenizerConfig): Configuration for the tokenizer.
        dir (str | Path): Directory to save/load the model.
        model (TransformerConfig): Configuration for the transformer model.
        objective (ObjectiveConfig): Configuration for the training objective.

    ## Methods:
        to_file(mode="json" | "pickle"): Save the configuration to a file in the specified format.
        from_file(model_name: str, model_dir: str | Path): Load the
    """
    model_config = ConfigDict(
        json_encoders={Path: str}
    )
    name: str = "yc1"
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    dirname: str | Path = MODELS_FOLDER
    model: TransformerConfig = Field(default_factory=TransformerConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: Literal["cpu", "cuda", "mps"] = DEVICE

    def model_post_init(self, context: Any) -> None:
        if isinstance(self.dirname, str):
            self.dirname = Path(self.dirname)
        self.dirname = self.dirname / self.name
        if not self.dirname.exists():
            self.dirname.mkdir(parents=True, exist_ok=True)

        if not hasattr(self.model, "vocab_size"):
            raise ValueError("Model configuration must have a vocab_size attribute.")
        if not hasattr(self.tokenizer, "vocab_size"):
            raise ValueError("Tokenizer configuration must have a vocab_size attribute.")
        if hasattr(self.model, "vocab_size") and hasattr(self.tokenizer, "vocab_size"):
            if self.model.vocab_size != self.tokenizer.vocab_size:
                raise ValueError(f"Model vocab_size ({self.model.vocab_size}) does not match tokenizer vocab_size ({self.tokenizer.vocab_size})")
            
        if not hasattr(self.model, "max_context"):
            raise ValueError("Model configuration must have a max_context attribute.")
        if not hasattr(self.tokenizer, "max_context"):
            raise ValueError("Tokenizer configuration must have a max_context attribute.")
        if hasattr(self.model, "max_context") and hasattr(self.tokenizer, "max_context"):
            if self.model.max_context != self.tokenizer.max_context:
                raise ValueError(f"Model max_context ({self.model.max_context}) does not match tokenizer max_context ({self.tokenizer.max_context})")
            
        

        self.dtype = getattr(torch, self.dtype)
        self.device = torch.device(self.device)

    def __eq__(self, other: "GPTConfig") -> bool:
        if not isinstance(other, GPTConfig):
            return False
        return self.__dict__ == other.__dict__

    def to_file(self, mode="json") -> None:
        suffix_ = "pickle" if mode == "pickle" else "json"
        if isinstance(self.dirname, str):
            self.dirname = Path(self.dirname)
        path = self.dirname  / f"config.{suffix_}"
        if mode not in ["json", "python", "pickle"]:
            raise ValueError(f"Unsupported mode: {mode}")
        
        with open(str(path), "wb") as f:
            if mode == "pickle":
                pickle.dump(self, f)
            else:
                json.dump(self.model_dump(mode=mode), f, indent=4)
        # self.dirname = Path(self.dirname)

    @classmethod
    def from_file(cls, model_name: str, model_dir: str | Path = MODELS_FOLDER) -> "GPTConfig":
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        config_path_json = model_dir / model_name / "config.json"
        config_path_pickle = model_dir / model_name / "config.pickle"
        if config_path_json.exists():
            with open(config_path_json, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            return cls.model_validate(config_dict)
        elif config_path_pickle.exists():
            with open(config_path_pickle, "rb") as f:
                config: GPTConfig = pickle.load(f)
            return config
        else:
            raise FileNotFoundError(f"No configuration file found for model {model_name} in {model_dir}")
        
    @classmethod
    def from_yaml(cls, path: str | Path) -> "GPTConfig":
        import yaml
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)
    
class TransformerOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: torch.Tensor
    attentions: List[torch.Tensor] | None = None
    hidden_states: List[torch.Tensor] | None = None
    kv_cache: dict | None = None

class ModelOutput(TransformerOutput):
    loss: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None

class ModelCompletionOutput(ModelOutput):
    completions: List[str] | None = None
    done: bool = False

class TrainingConfig(BaseModel):
    batch_size: int = BATCH_SIZE
    steps: int = 100000
    accumulation_steps: int = 100

    max_learning_rate: float = MAX_LEARNING_RATE
    min_learning_rate: float = MIN_LEARNING_RATE
    warmup_iters: int = WARMUP_ITERS

    optimizer: Literal["adamw", "sgd", "adam", "muon"] = "adamw"
    optimizer_params: dict = Field(default_factory=dict)

    validation_step: int = VALIDATION_STEP
    pretraining_val_ratio: float = PRETRAINING_VAL_RATIO

class TrainingState(BaseModel):
    step: int = 0
    best_val_loss: float = float("inf")
    early_stopping_counter: int = 0
    train_losses: List[float] = Field(default_factory=list)
    val_losses: List[float] = Field(default_factory=list)

class TrainingResults(BaseModel):
    train_loss: List[float] = Field(default_factory=list)
    val_loss: List[float] = Field(default_factory=list)
    steps: List[int] = Field(default_factory=list)

def get_config_from_huggingface(model_name: str) -> TransformerConfig:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return TransformerConfig(
        tokenizer=tokenizer.encode,
        pad_id=pad_id,
        vocab_size=vocab_size
    )