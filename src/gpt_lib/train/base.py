from gpt_lib.model.model import GPTModel
from gpt_lib.utils.schemas import TrainingConfig, TrainingState, TrainingMetrics
from gpt_lib.train.optimizer import AdamW


from gpt_lib.utils.default import DEVICE

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import time, pickle
# import time, pickle, wandb
from typing import Callable, Iterable, Literal
from pathlib import Path


class BaseTrainer:
    def __init__(
            self,
            model: GPTModel,
            train_dataset: Iterable,
            val_dataset: Iterable,
            test_dataset: Iterable,
            config: TrainingConfig,
            board_type: Literal["wandb", "tensorboard", "none"] = "none",
            device: torch.device = DEVICE,
            dtype: torch.dtype = torch.float32,
        ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device
        self.dtype = dtype
        

    def init_board(self, board_type: Literal["wandb", "tensorboard", "none"] = "none") -> None:
        pass
    
    def evaluate(self, dataset: Iterable) -> TrainingMetrics:
        raise NotImplementedError
    
    def get_lr_multiplier(self, current_step: int) -> float:
        warmup

    def fit(self):
        raise NotImplementedError
    
    def save_model(self, path: Path) -> None:
        raise NotImplementedError
    
    def load_model(self, path: Path) -> None:
        raise NotImplementedError
    
    def save_metrics(self) -> None:
        raise NotImplementedError
    
    def load_metrics(self, path: Path) -> None:
        raise NotImplementedError
