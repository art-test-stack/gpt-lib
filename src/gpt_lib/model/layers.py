from gpt_lib.utils.schemas import (
    ATTN_IMPL_TYPES,
    NORMALIZATION_TYPES,
    TransformerConfig
)
from gpt_lib.utils.default import DEVICE, DEVICE_NAME
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Literal, Tuple
import warnings

# -------------- Check for optional dependencies -------------- #
try:
    import flash_attn
    _flash_attention_installed = True
except ImportError:
    flash_attn = None
    _flash_attention_installed = False
try: 
    from kernels import get_kernels
    flash_attn3 = get_kernels('varunneal/flash-attention-3').flash_attn_interface
    _kernels_installed = True
except ImportError:
    kernels = None
    flash_attn3 = None
    _kernels_installed = False

# -------------- Positional Encoding utilities -------------- #

def precompute_rope(seq_len: int, d_head: int, base: int = 10000, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    if not device:
        warnings.warn(f"Device not specified for RoPE precomputation. Using default device {DEVICE_NAME}.")
        device = DEVICE
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    channel_range = torch.arange(0, d_head, 2.0, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (channel_range / d_head))
    pos_seq = torch.arange(0, seq_len, dtype=dtype, device=device)

    sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
    rope_cache = torch.stack((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=-1)
    return rope_cache # seq_len x (d_head/2) x 2

def precompute_positional_encoding(n_pos: int, d_model: int, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    if not device:
        warnings.warn("Device not specified for positional encoding precomputation. Using default device.")
        device = torch.device(DEVICE)
    pos = torch.arange(n_pos, dtype=dtype, device=device)
    i = torch.arange(d_model, dtype=dtype, device=device)

    pos_enc = torch.ger(pos, 1e4 ** (- 2 * (i//2) / d_model))

    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2]) 
    return pos_enc

def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor, pairwise_split: bool = True) -> torch.Tensor:
    assert x.dim() == 4, "Input tensor x must be of shape (batch_size, n_heads, seq_len, d_head)"
    sin, cos = rope_cache[..., 0], rope_cache[..., 1]
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.device)
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.device)
    if pairwise_split:
        x1, x2 = x[..., ::2], x[..., 1::2]
    else:
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
    batch, n_heads, seq_len, d_head = x.shape
    assert x1.shape == x2.shape == (batch, n_heads, seq_len, d_head // 2), f"Unexpected shapes: x1 {x1.shape}, x2 {x2.shape}"

    _x1 = x1 * cos - x2 * sin
    _x2 = - x1 * sin + x2 * cos
    x = torch.stack([_x1, _x2], dim=-1).reshape_as(x)
    # x_rotated = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    # x = x * cos + x_rotated * sin
    return x


def apply_positional_encoding(x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
    return pos_enc[:,:x.size(-1)]

def apply_rms_norm(x: torch.Tensor, eps: float = 1e-8, torch_impl: bool = True) -> torch.Tensor:
    if torch_impl:
        return torch.rms_norm(x, normalized_shape=(x.size(-1),), eps=eps)
    else:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return x / rms

def apply_layer_norm(x: torch.Tensor, eps: float = 1e-5, torch_impl: bool = True) -> torch.Tensor:
    if torch_impl:
        return torch.layer_norm(x, normalized_shape=(x.size(-1),), eps=eps)
    else:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, unbiased=False, keepdim=True)
        return (x - mean) / (std + eps)

# -------------- Utility layers definitions -------------- #

class Module(nn.Module):
    def nb_parameters(self) -> int:
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])

    def nb_trainable_parameters(self) -> int:
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])

    def nb_non_trainable_parameters(self) -> int:
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])

    def summary(self) -> None:
        print(f'Number of parameters: {self.nb_parameters():,}')
        print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
        print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')

    def clean_nan(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)

    def clip_gradient(self, max_norm: float) -> None:
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    # def init_weights(self) -> None:
    #     '''Initialize the module weights'''
    #     for module in self.modules():
    #         if hasattr(module, 'init_weights'):
    #             module.init_weights()

class Embedding(Module):
    '''Embedding layer'''
    def __init__(self, config: TransformerConfig = TransformerConfig(), dtype=torch.float32, device=torch.device(DEVICE)) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.d_model, 
            padding_idx=config.pad_id, 
            # max_norm=config.max_norm, 
            # norm_type=config.norm_type, 
            # scale_grad_by_freq=config.scale_grad_by_freq, 
            # sparse=config.sparse or True, 
            # device=config.device,
            # dtype=config.dtype
        )
        # self.embedding = nn.Parameter(
        #     data=torch.randn(config.vocab_size, config.d_model),
        #     requires_grad=True
        # )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class Linear(nn.Linear, Module):
    '''Linear layer'''
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: torch.device = DEVICE, dtype=None) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        # TODO: Reparametrize weights and bias here

    def init_weights(self, std: float = 0.01, method: str ="uniform") -> None:
        assert method in ["uniform", "normal", "zero"], "Method must be 'uniform', 'normal' or 'zero'"
        if method == "uniform":
            nn.init.uniform_(self.weight, -std, std)
        elif method == "normal":
            nn.init.normal_(self.weight, 0, std)
        elif method == "zero":
            nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class TPLinear(Linear):
    '''Tensor Parallel Linear layer'''
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 device: torch.device = DEVICE, dtype=None, tp_spec=None, parallel_axis=None) -> None:
        tp_size = tp_spec.size if tp_spec else None
        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        self.tp_spec = tp_spec
        self.axis = parallel_axis
        if tp_spec is None:
            super().__init__(
                in_features=in_features, 
                out_features=out_features, bias=bias, device=device, dtype=dtype
            )
        
        assert self.axis in ["row", "column"], "parallel_axis must be 'row' or 'column'"

        if parallel_axis == "row":
            assert out_features % tp_size == 0, f"out_features must be divisible by tp_size for row parallelism. Got out_features={out_features}, tp_size={tp_size}."
            local_out = out_features // tp_size
            super().__init__(
                in_features=in_features // tp_size, 
                out_features=out_features, bias=bias, device=device, dtype=dtype
            )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if (self.tp_spec is None) or (self.axis == "column"):
            return super().forward(x)
        
        out = super().forward(x)
        dist.all_reduce(out, group=self.tp_spec.tp_group)
        return out

# --------------      Attention utilities      -------------- #

def scaled_dot_product_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: torch.Tensor | None = None, 
        dropout_p: float = 0.0,
        is_causal: bool = False, 
        scale: float | None = None, 
        enable_gqa: bool = False,
        return_attn_weights: bool = False,
        device: torch.device | str | None = None
    ) -> torch.Tensor:
    if device is None:
        device = query.device
    if isinstance(device, str):
        device = torch.device(device)
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, "Query, Key and Value must be 4D tensors."
    assert query.size(-1) == key.size(-1) == value.size(-1), "Last dimension of Query, Key and Value must be the same"
    assert query.device == key.device == value.device, f"Query, Key and Value must be on the same device. Got Query device: {query.device}, Key device: {key.device}, Value device: {value.device}."
    assert query.device == device, f"Q, K, V devices and specified device must be the same. Got Query device: {query.device}, Key device: {key.device}, Value device: {value.device}, specified device: {device}."

    assert (not is_causal) or (attn_mask is None), "`is_causal` cannot be True when `attn_mask` is provided. This behavior is given as a imitation of PyTorch's scaled_dot_product_attention."

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1) # B x 1 x 1 x S
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # B x 1 x L x S
        else:
            assert attn_mask.size(0) == query.size(0) and attn_mask.size(0) == key.size(0), f"Attention mask batch size must match Query and Key batch size. Got attn_mask size: {attn_mask.size()} (B,...), Query size: {query.size()} (B,...), Key size: {key.size()} (B,...)."
            assert attn_mask.size(-2) == query.size(-2), f"Attention mask size must match the sequence length of Query. Got attn_mask size: {attn_mask.size()} (B,...,L,S), Query size: {query.size()} (B,...,Hq,L,E), Key size: {key.size()} (B,...,H,S,E)."
            assert attn_mask.size(-1) == key.size(-2), f"Attention mask size must match the batch size and sequence lengths of Query and Key. Got attn_mask size: {attn_mask.size()} (B,...,L,S), Query size: {query.size()} (B,...,Hq,L,E), Key size: {key.size()} (B,...,H,S,E)."

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_bias = torch.zeros((1, 1, L, S), device=device)

    if is_causal:
        causal = torch.tril(torch.ones(L, S, device=device))
        attn_bias = attn_bias.masked_fill(causal == 0, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_mask

    if enable_gqa:
        if query.size(-3) != key.size(-3):
            raise ValueError("For GQA, the number of query heads must match the number of key heads.")
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = F.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p, training=True)
    return attn_weight @ value, attn_weight if return_attn_weights else None

class FusedQKVProjection(Module):
    """
    Purpose: project input tensor to Q, K, V in a single linear layer for efficiency. 
    Use case: Should be used for training, or no KV cache inference, or no paging.
    """
    def __init__(self, d_model, n_heads, d_head, bias=False) -> None:
        super().__init__()

        self.w_qkv = Linear(d_model, 3 * n_heads * d_head, bias=bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _bs, _sl, _ = x.size()
        qkv = self.w_qkv(x)  # B x S x (3*n_heads*d_head)
        return torch.chunk(qkv, 3, dim=-1)
    
class QKVProjection(Module):
    """
    Purpose: project input tensor to Q, K, V in separate linear layers.
    Use case: Should be used for inference with KV cache and paging.
    """
    def __init__(self, d_model, n_heads_q, d_heads_kv, d_head, bias=False) -> None:
        super().__init__()

        self.w_q = Linear(d_model, n_heads_q * d_head, bias=bias)
        self.w_k = Linear(d_model, n_heads_q * d_head, bias=bias)
        self.w_v = Linear(d_model, n_heads_q * d_head, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _bs, _sl, _ = x.size()
        q = self.w_q(x)  # B x S x (n_heads_q*d_head)
        k = self.w_k(x)  # B x S x (n_heads_kv*d_head)
        v = self.w_v(x)  # B x S x (n_heads_kv*d_head)
        return q, k, v

def build_qkv_projection(d_model: int, n_heads_q: int, n_heads_kv: int, d_head: int, bias: bool = False
                         ) -> Literal[QKVProjection, FusedQKVProjection]:
    if n_heads_q == n_heads_kv:
        return FusedQKVProjection(d_model, n_heads_q, d_head, bias=bias)
    else:
        return QKVProjection(d_model, n_heads_q, n_heads_kv, d_head, bias=bias)

class CausalSelfAttention(Module):
    def __init__(self, dim_model: int, n_heads: int, d_head: int, 
                 norm_before_attn: bool, enable_gqa: bool, dropout: float = .0, 
                 normalization: NORMALIZATION_TYPES = "rms",
                 attn_impl: ATTN_IMPL_TYPES = "sdpa", layer_idx: int = 0
        ) -> None:
        super().__init__()
        assert dim_model == d_head * n_heads, f"Dimensions are not correct. dim_model must be equal to d_head * n_heads. Got dim_model={dim_model}, d_head={d_head}, n_heads={n_heads}"
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.d_head = d_head
        self.norm_before_attn = norm_before_attn
        self.dropout_rate = dropout

        if normalization == "rms":
            self.norm = apply_rms_norm
        elif normalization == "layer":
            self.norm = apply_layer_norm
        else:
            raise ValueError(f"Unknown normalization: {normalization}. Supported normalizations are 'rms' and 'layer'.")
        
        if enable_gqa:
            assert n_heads % 2 == 0, "Number of heads must be even for GQA."
            self.n_heads = n_heads // 2
            d_k = d_head
        else:
            d_k = d_head * n_heads
        
        if not _flash_attention_installed and attn_impl == "flash_attention":
            warnings.warn("FlashAttention is not installed. Falling back to torch implementation.")
            attn_impl = "sdpa"
        if _flash_attention_installed and not attn_impl == "flash_attention":
            warnings.warn("FlashAttention is installed. It is recommended to use 'flash_attention' implementation for better performance.")

        self.attn_impl = attn_impl

        # self.w_q = Linear(dim_model, d_head * n_heads, bias=False) 
        # self.w_k = Linear(dim_model, d_head * n_heads, bias=False) 
        # self.w_v = Linear(dim_model, d_head * n_heads, bias=False)

        self.w_qkv = Linear(dim_model, 3 * d_head * n_heads, bias=False)
        self.w_o = Linear(dim_model, dim_model, bias=False)
    
    def init_weights(self) -> None:
        dim_model = self.n_heads * self.d_head
        std = math.sqrt(3.0 / dim_model)
        self.w_qkv.init_weights(std)
        # self.w_q.init_weights(std)
        # self.w_k.init_weights(std)
        # self.w_v.init_weights(std)
        self.w_o.init_weights(method="zero")

    def forward(self, x: torch.Tensor, rope_cache=None, attn_mask=None, kv_cache=None, return_attn_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        n_heads = self.n_heads
        d_head = self.d_head

        bs = x.size(0)
        len_x = x.size(1)

        # q = self.w_q(x) # b X l X (n_heads*d_head)
        # k = self.w_k(x)
        # v = self.w_v(x)
        _qkv = self.w_qkv(x)
        q, k, v = torch.chunk(_qkv, 3, dim=-1)

        q = q.view(bs, len_x, n_heads, d_head)
        k = k.view(bs, len_x, n_heads, d_head)
        v = v.view(bs, len_x, n_heads, d_head)

        if rope_cache is not None:
            q = apply_rope(q, rope_cache)
            k = apply_rope(k, rope_cache)
        if self.norm_before_attn:
            q = self.norm(q, eps=1e-8, torch_impl=True)
            k = self.norm(k, eps=1e-8, torch_impl=True)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.attn_impl == "flash_attention" and _flash_attention_installed and not return_attn_weights:
            try:
                output = flash_attn.flash_attn_func(
                    q, k, v,
                    attn_bias=attn_mask,
                    causal=True,
                    dropout_p=self.dropout_rate if self.training else 0.0,
                    return_attn_weights=return_attn_weights
                )
                output = output.reshape(bs, n_heads, len_x, d_head) 
                return output.transpose(1, 2).contiguous().view(bs, len_x, -1), None
            except ImportError:
                warnings.warn("FlashAttention is not installed. Falling back to torch attention.")
                self.attn_impl = "sdpa"
            
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # b X n_heads X l X d_head

        if kv_cache is not None:
            k, v = kv_cache.insert(k, v)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)  
        # TODO: Support different attention implementations
        # TODO: Use context manager to select attention implementation
        # TODO: Detect context manager and use the corresponding attention implementation
        if self.attn_impl == "sdpa" and not return_attn_weights:
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_rate if self.training else .0, 
                is_causal=attn_mask is None
            )
            attn_weights = None
        elif self.attn_impl == "impl" or return_attn_weights:
            output, attn_weights = scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_rate if self.training else .0, 
                is_causal=attn_mask is None, return_attn_weights=return_attn_weights
            )
        else:
            raise ValueError(f"Unknown attention implementation: {self.attn_impl}. Supported implementations are 'sdpa', 'flash_attention', and 'impl'.")
    
        output = output.transpose(1, 2).contiguous().view(bs, len_x, -1)
        output = self.w_o(output)
        return output, attn_weights
    
# -------------- Transformer layers definitions -------------- #
    
class FeedForward(Module):
    '''Position-Wise Feed Forward Network'''
    def __init__(self, d_in: int, d_latent: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = Linear(d_in, d_latent)
        self.w_2 = Linear(d_latent, d_in)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w_2(F.relu(self.w_1(x)))

        if self.training:
            h = F.dropout(h, p=self.dropout_rate, training=True)
                
        h += x
        output = apply_rms_norm(h)
        return output
    
class SwigLUFeedForward(Module):
    '''Position-Wise Feed Forward Network with SwiGLU activation'''
    def __init__(self, d_in: int, d_latent: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = Linear(d_in, d_latent * 2)
        self.w_2 = Linear(d_latent, d_in)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.w_1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        h = self.w_2(F.silu(x1) * x2)

        if self.training:
            h = F.dropout(h, p=self.dropout_rate, training=True)
                
        h += x
        output = apply_rms_norm(h)
        return output
    
class DecoderLayer(Module):
    '''Decoder layer'''
    def __init__(
            self, 
            dim_model: int, 
            dim_ffn: int, 
            n_heads: int, 
            d_head: int, 
            dropout: float,
            attn_impl: ATTN_IMPL_TYPES = "sdpa",
            normalization: NORMALIZATION_TYPES = "rms",
            enable_gqa: bool = False,
            norm_before_attn: bool = True,
            layer_idx: int = 0
        ) -> None:
        super().__init__()
        if not norm_before_attn:
            warnings.warn("Using 'norm_before_attn=False' is not recommended and may lead to training instability.", UserWarning)
        self.attention = CausalSelfAttention(
            dim_model=dim_model, 
            n_heads=n_heads, 
            d_head=d_head, 
            enable_gqa=enable_gqa,
            normalization=normalization,
            attn_impl=attn_impl,
            layer_idx=layer_idx, 
            norm_before_attn=norm_before_attn   
        )
        
        self.ffn = FeedForward(d_in=dim_model, d_latent=dim_ffn, dropout=dropout)
        self.dropout_rate = dropout

        if normalization == "rms":
            self.norm = apply_rms_norm
        elif normalization == "layer":
            self.norm = apply_layer_norm
        else:
            raise ValueError(f"Unknown normalization: {normalization}. Supported normalizations are 'rms' and 'layer'.")

    def forward(self, x, attn_mask=None, kv_cache=None, rope_cache=None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        h, self_attn = self.attention(x, attn_mask=attn_mask, kv_cache=kv_cache, rope_cache=rope_cache)

        if self.training:
            h = F.dropout(h, p=self.dropout_rate, training=True)
        h += x
        h = self.norm(h, eps=1e-8, torch_impl=True)
        output = h + self.ffn(h)

        return output, self_attn

class MixtureOfExpertsLayer(Module):
    '''Mixture of Experts layer. Note: This is a simplified version without load balancing or capacity constraints. '''
    def __init__(
            self, 
            dim_model: int, 
            dim_ffn: int, 
            n_experts: int, 
            dropout: float
        ) -> None:
        super().__init__()
        warnings.warn("This is a simplified Mixture of Experts layer without load balancing or capacity constraints. Use with caution.", UserWarning)
        self.experts = nn.ModuleList([
            FeedForward(d_in=dim_model, d_latent=dim_ffn, dropout=dropout)
            for _ in range(n_experts)
        ])
        self.router = Linear(dim_model, n_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_scores = F.softmax(self.router(x), dim=-1)  
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  
        router_scores = router_scores.unsqueeze(2)  
        output = torch.sum(expert_outputs * router_scores, dim=-1)  
        return output