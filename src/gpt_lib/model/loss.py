from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F

from gpt_lib.utils.schemas import ModelOutput, ObjectiveConfig 

from cut_cross_entropy import linear_cross_entropy

class Objective(ABC):
    @abstractmethod
    def __call__(
        self,
        output: ModelOutput,
        labels: torch.Tensor,
        ref_output: Optional[ModelOutput] = None,
    ) -> torch.Tensor:
        ...


class CrossEntropyObjective(Objective):
    def __init__(self, ignore_index=-100, reduction="mean"):
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, output: ModelOutput, labels: torch.Tensor, ref_output=None):
        logits = output.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss

class CutCrossEntropyObjective(Objective):
    def __init__(self, ignore_index=-100, reduction="mean"):
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, output: ModelOutput, labels: torch.Tensor, ref_output=None):
        raise NotImplementedError("CutCrossEntropyObjective is not yet supported.")
        logits = output.logits

        loss = linear_cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            impl="torch_compile"
        )
        return loss


class KLDivObjective(Objective):
    def __init__(self, epsilon: float = 0.1, ignore_index: int = -100):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, output: ModelOutput, labels: torch.Tensor, ref_output=None):
        log_probs = output.log_probs
        vocab_size = log_probs.size(-1)

        mask = labels != self.ignore_index

        with torch.no_grad():
            target_dist = torch.zeros_like(log_probs)
            target_dist.fill_(self.epsilon / (vocab_size - 1))
            target_dist.scatter_(
                -1, labels.unsqueeze(-1), 1.0 - self.epsilon
            )

        kl = F.kl_div(
            log_probs[mask],
            target_dist[mask],
            reduction="batchmean",
            log_target=False,
        )
        return kl


def build_objective(config: ObjectiveConfig) -> "Objective":
    loss_type = config.objective_fn
    loss_kwargs = config.kwargs
    if loss_type == "cross_entropy":
        return CrossEntropyObjective(
            ignore_index=loss_kwargs.get("ignore_index", -100),
            reduction=loss_kwargs.get("reduction", "mean"),
        )
    elif loss_type == "kl_divergence":
        return KLDivObjective(
            epsilon=loss_kwargs.get("epsilon", 0.1),
            ignore_index=loss_kwargs.get("ignore_index", -100),
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    