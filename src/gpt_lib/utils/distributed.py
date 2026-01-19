from gpt_lib.utils.schemas import ParallelismConfig
from gpt_lib.utils.log import logger
import torch
import torch.distributed as dist
import os

def is_ddp_initialized() -> bool:
    return dist.is_initialized() and dist.is_available()
 

def init_process_groups(world_size, tp_size):
    assert world_size % tp_size == 0
    dp_size = world_size // tp_size

    rank = dist.get_rank()

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    tp_group = dist.new_group(
        ranks=[dp_rank * tp_size + i for i in range(tp_size)]
    )

    dp_group = dist.new_group(
        ranks=[i * tp_size + tp_rank for i in range(dp_size)]
    )

    return dp_group, tp_group, dp_rank, tp_rank, dp_size

def get_dist_info():
    is_initialized = is_ddp_initialized()
    if is_initialized:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        tp_size = int(os.getenv("TP_SIZE", "1"))
    else:
        world_size = 1
        rank = 0
        tp_size = 1

    return is_initialized, world_size, rank, tp_size

def choose_parallelism(world_size: int, tp_size: int) -> ParallelismConfig:
    # TODO: expand this function for more complex parallelism strategies
    is_initialized, world_size, rank, tp_size = get_dist_info()
    if not is_initialized:
        return "single"
    config = dict(
        world_size=world_size,
        tp_size=tp_size,
        dp_size=world_size // tp_size,
        
    )
    if tp_size > 1 and world_size // tp_size > 1:
        return "dp_tp"
    elif tp_size > 1:
        return "tp"
    elif world_size > 1:
        return "dp"
    else:
        return "single"
    

# hardcoded BF16 peak flops for various GPUs
# adapted from: https://github.com/karpathy/nanochat/blob/master/nanochat/common.py
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # --- NVIDIA Blackwell ---
    if "gb200" in name or "grace blackwell" in name:
        return 2.5e15
    if "b200" in name:
        return 2.25e15
    if "b100" in name:
        return 1.8e15

    # --- NVIDIA Hopper (H100/H200/H800) ---
    if "h200" in name:
        if "nvl" in name or "pcie" in name:
            return 836e12
        return 989e12  # H200 SXM
    if "h100" in name:
        if "nvl" in name:
            return 835e12
        if "pcie" in name:
            return 756e12
        return 989e12  # H100 SXM
    if "h800" in name:
        if "nvl" in name:
            return 989e12
        return 756e12  # H800 PCIe

    # --- NVIDIA Ampere data center ---
    if "a100" in name or "a800" in name:
        return 312e12
    if "a40" in name:
        return 149.7e12
    if "a30" in name:
        return 165e12

    # --- NVIDIA Ada data center ---
    if "l40s" in name or "l40-s" in name or "l40 s" in name:
        return 362e12
    if "l4" in name:
        return 121e12

    # --- AMD CDNA accelerators ---
    if "mi355" in name:
        return 2.5e15
    if "mi325" in name or "mi300x" in name:
        return 1.3074e15
    if "mi300a" in name:
        return 980.6e12
    if "mi250x" in name:
        return 383e12
    if "mi250" in name:
        return 362.1e12

    # --- Intel ---
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # --- Consumer RTX (for hobbyists) ---
    if "5090" in name:
        return 209.5e12
    if "4090" in name:
        return 165.2e12
    if "3090" in name:
        return 71e12

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')


base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}

base_model_pp_plan = {
    "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    "norm": (["hidden_states"], ["hidden_states"]),
}