from gpt_lib.optimizers.adamw import AdamW
from gpt_lib.optimizers.muon import Muon
from gpt_lib.optimizers.shampoo import Shampoo
from gpt_lib.optimizers.adam import Adam
from gpt_lib.optimizers.adahessian import Adahessian

from typing import Literal

opt_class = {
    "adamw": AdamW,
    "muon": Muon,
    "shampoo": Shampoo,
    "adam": Adam,
    "adahessian": Adahessian,
}

OptimizerNames = Literal[tuple(opt_class.keys())]