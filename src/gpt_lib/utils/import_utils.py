from packaging import version as pkg_version
from functools import lru_cache


@lru_cache()
def is_flash_attention_installed() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False
    
@lru_cache()
def is_flash_attention3_installed() -> bool:
    try:
        import flash_attn3  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache()
def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

@lru_cache()
def is_torch_cuda_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    return torch.cuda.is_available()

@lru_cache()
def is_torch_mps_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    return torch.backends.mps.is_available()

@lru_cache()
def is_torch_xpu_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    return hasattr(torch, "xpu") and torch.xpu.is_available()

@lru_cache() 
def is_torch_greater_or_equal(version: str) -> bool:
    if not is_torch_available():
        return False
    import torch
    return pkg_version(torch.__version__) >= pkg_version(version)

@lru_cache()
def is_torch_bf16_gpu_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    if is_torch_xpu_available():
        return torch.xpu.is_bf16_supported()
    if is_torch_mps_available():
        return torch.backends.mps.is_macos_or_newer(14, 0)
    return False


@lru_cache()
def is_flash_attn3_available_from_kernel() -> bool:
    if not is_torch_available():
        return False
    try: 
        from kernels import get_kernels
        flash_attn3 = get_kernels('varunneal/flash-attention-3').flash_attn_interface
        return True
    except:
        return False