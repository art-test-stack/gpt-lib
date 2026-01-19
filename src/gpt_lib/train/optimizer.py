from gpt_lib.utils.default import *

import torch
from torch import Tensor, optim
from typing import Iterable, Dict, Any, Tuple

class AdamW(optim.AdamW):
    def __init__(
            self,
            params: Iterable[Tensor] | Iterable[Dict[str, Any]],
            lr: float | Tensor = MAX_LEARNING_RATE,
            eps: float = EPSILON,
            weight_decay: float = WEIGHT_DECAY,
            fused: bool | None = None,
            **kwargs
        ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=(BETA_1, BETA_2),
            weight_decay=weight_decay,
            eps=eps,
            fused=fused,
            **kwargs
        )

    def update_learning_rate(self, iter):
        if iter < WARMUP_ITERS:
            self.learning_rate = MAX_LEARNING_RATE * iter / WARMUP_ITERS
		
        elif iter < WARMUP_ITERS + DECAY_ITERS:
            ratio = (iter - WARMUP_ITERS) / (DECAY_ITERS - WARMUP_ITERS)
            assert 0 <= ratio <+ 1
            coeff = 0.5 * (1.0 + torch.cos(torch.pi * ratio))
            self.learning_rate = MIN_LEARNING_RATE + coeff * (MAX_LEARNING_RATE - MIN_LEARNING_RATE) 
		
        else:
            self.learning_rate = MIN_LEARNING_RATE
        
        for g in self.param_groups:
            g['lr'] = self.learning_rate

class MuON(optim.Optimizer):
    def __init__(
            self,
            params: Iterable[Tensor] | Iterable[Dict[str, Any]],
            lr_min: float = MIN_LEARNING_RATE,
            lr_max: float = MAX_LEARNING_RATE,
            warmup_steps: int = WARMUP_ITERS,
            decay_steps: int = DECAY_ITERS,
            weight_decay: float = WEIGHT_DECAY,
            betas: float = BETA_1,
            beta2: float = BETA_2,
            epsilon: float = EPSILON,
            **kwargs
        ) -> None:
        defaults = dict(
            lr_min=lr_min,
            lr_max=lr_max,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon
        )
        super().__init__(params, defaults)
        self.iter = 0

    def step(self, closure=None):
        # Implementation of MuON optimizer step
        pass