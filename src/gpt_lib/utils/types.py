from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, Literal

import torch

TParams = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

# Type Literals
Devices = Literal["cpu", "cuda", "mps"]
TokenizerSources = Literal["tiktoken", "bytelevelbpe", "rustbpe", "huggingface", "dummy"]
TokenizerTensors = Literal["pt", "np", "tf", "jax"]
NormalizationTypes = Literal["rms", "layer"]
AttnImplTypes = Literal["sdpa", "flash_attention", "impl"]
PositionalEncodingTypes = Literal["positional", "rope"]
TfTypes = Literal["dense", "moe"]
Dtypes = Literal["float32", "float16", "bfloat16"]
OptimizerNames = Literal[ "adam", "adamw", "muon", "shampoo", "adahessian"]
LossTypes = Literal["cross_entropy", "kl_divergence"]
LossReductionTypes = Literal["none", "mean", "sum"]
TpModes = Literal["row", "column"]