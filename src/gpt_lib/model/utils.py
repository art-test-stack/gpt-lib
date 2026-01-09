import torch
import warnings

class SelfAttentionMask:
    def __init__(self, pad_idx: int = -100, max_context: int = 512) -> None:
        self.pad_idx = pad_idx
        self.max_context = max_context
        self.base_mask = torch.tril(torch.ones((max_context, max_context), dtype=torch.bool), diagonal=0) # S x S # [ i >= j ]
        self.base_mask.requires_grad = False
    
    def get(self, seq: torch.Tensor, mask_pad_token: bool = True, to_bool: bool = True, is_causal: bool = True) -> torch.Tensor:
        if seq.dim() == 1:
            warnings.warn("Input sequence tensor has dimension 1. Unsqueezing to dimension 2.")
            seq = seq.unsqueeze(0)
        assert seq.dim() == 2, f"Input sequence tensor must be of dimension 2 (B, S). Got {seq.shape}."
        B, S = seq.shape
        assert S <= self.max_context, f"Sequence length {S} exceeds max_context {self.max_context}. If you want to process longer sequences, please update the max_context using `update_max_context` method."

        device = seq.device
        if mask_pad_token:
            pad_mask = (seq != self.pad_idx) # B x S
        else:
            pad_mask = torch.ones_like(seq, dtype=torch.bool, device=device)
        
        if is_causal:
            causal_mask = self.base_mask[:S, :S].to(device)
            print("causal_mask", causal_mask)
        else:
            warnings.warn("Non-causal attention mask is not yet optimized for large sequences.", UserWarning)
            causal_mask = torch.ones((S, S), dtype=torch.bool, device=device)

        attn_mask = (
            pad_mask.unsqueeze(1).unsqueeze(2) &   # keys
            causal_mask.unsqueeze(0).unsqueeze(1)
        ) # B x 1 x S x S

        if not to_bool:
            attn_mask = (~attn_mask).type(torch.float32)
            attn_mask = attn_mask.masked_fill_(attn_mask.bool(), float("-inf"))

        return attn_mask
    
    def __call__(self, seq: torch.Tensor, mask_pad_token: bool = True, to_bool: bool = True, is_causal: bool = True) -> torch.Tensor:
        return self.get(seq, mask_pad_token=mask_pad_token, to_bool=to_bool, is_causal=is_causal)
    
    def update_max_context(self, max_context: int) -> None:
        if max_context > self.max_context:
            self.base_mask = torch.tril(torch.ones((max_context, max_context), dtype=torch.bool))
            self.base_mask.requires_grad = False
        self.max_context = max_context

class RowState:
    def __init__(self) -> None:
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False

class KVState:
    def __init__(self, batch_size: int, n_layers: int, n_heads: int, d_head: int, seq_len: int) -> None:
        self.kv_shape = (n_layers, 2, batch_size, n_heads, seq_len, d_head)
        self.kv_cache = None
        self.current_pos = 0

    def reset(self):
        self.current_pos = 0

    def init(self, cache):
        assert self.kv_cache is None, "Cache is already initialized"
        assert cache.kv_cache is not None, "Provided cache is empty"

        pass

    def insert(self, layer, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(self.kv_shape, device=k.device, dtype=k.dtype)
        
        self.kv_cache[layer, 0, :, :, self.current_pos, :] = k
        self.kv_cache[layer, 1, :, :, self.current_pos, :] = v
        self.current_pos += 1