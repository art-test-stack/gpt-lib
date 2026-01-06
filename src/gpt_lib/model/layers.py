from gpt_lib.utils.schemas import TransformerConfig
from gpt_lib.utils.default import DEVICE, DEVICE_NAME, MAX_CONTEXT
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings


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

    x1 = x1 * cos - x2 * sin
    x2 = - x1 * sin + x2 * cos
    x = torch.stack([x1, x2], dim=-1).reshape_as(x)
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

def scaled_dot_product_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: torch.Tensor | None = None, 
        dropout_p: float = 0.0,
        is_causal: bool = False, 
        scale: float | None = None, 
        enable_gqa: bool = False,
        device: torch.device | str | None = None
    ) -> torch.Tensor:
    if device is None:
        device = query.device
    if isinstance(device, str):
        device = torch.device(device)
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, "Query, Key and Value must be 4D tensors."
    assert query.size(-1) == key.size(-1) == value.size(-1), "Last dimension of Query, Key and Value must be the same"
    assert query.device == key.device == value.device, "Query, Key and Value must be on the same device"
    assert query.device == device, "Query device and specified device must be the same"
    print("query shape:", query.shape)
    print("key shape:", key.shape)
    print("value shape:", value.shape)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        # if attn_mask.dim() == 2:
        #     attn_mask = ...
        #     print("unsqueezed attn_mask shape:", attn_mask.shape)
        # assert attn_mask.size(1) == query.size(-2) and attn_mask.size(2) == key.size(-2), f"Attention mask size must match the sequence lengths of Query and Key. Got attn_mask size: {attn_mask.size()} (B,...,L,S), Query size: {query.size()} (B,...,Hq,L,E), Key size: {key.size()} (B,...,H,S,E)."

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    causal_mask = torch.ones(0, dtype=torch.bool, device=device)
    if is_causal:
        assert attn_mask is not None, "'attn_mask' cannot be None when 'is_causal' is True."
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            print("causal_mask shape:", causal_mask.shape)
            print("attn_mask shape:", attn_mask.shape)
            attn_bias = causal_mask & attn_mask
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            causal_mask.masked_fill_(causal_mask.logical_not(), float("-inf"))
            attn_bias = attn_mask + causal_mask

    if enable_gqa:
        if query.size(-3) != key.size(-3):
            raise ValueError("For GQA, the number of query heads must match the number of key heads.")
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = F.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p, training=True)
    return attn_weight @ value

class Module(nn.Module):
    '''class Module'''
    def nb_parameters(self) -> int:
        '''Give the number of parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])

    def nb_trainable_parameters(self) -> int:
        '''Give the number of trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])

    def nb_non_trainable_parameters(self) -> int:
        '''Give the number of non-trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])

    def summary(self) -> None:
        '''Summarize the module'''
        print(f'Number of parameters: {self.nb_parameters():,}')
        print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
        print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')

    def clean_nan(self) -> None:
        '''Remove NaNs from the module gradients'''
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)

    def clip_gradient(self, max_norm: float) -> None:
        '''Clip the module gradients'''
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
    

class CausalSelfAttention(Module):
    def __init__(self, dim_model: int, n_heads: int, d_head: int, norm_before_attn: bool, enable_gqa: bool, layer_idx: int = 0) -> None:
        super().__init__()
        assert dim_model == d_head * n_heads, f"Dimensions are not correct. dim_model must be equal to d_head * n_heads. Got dim_model={dim_model}, d_head={d_head}, n_heads={n_heads}"
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.d_head = d_head
        self.norm_before_attn = norm_before_attn

        if enable_gqa:
            assert n_heads % 2 == 0, "Number of heads must be even for GQA."
            self.n_heads = n_heads // 2
            d_k = d_head
        else:
            d_k = d_head * n_heads

        self.w_q = Linear(dim_model, d_head * n_heads, bias=False) 
        self.w_k = Linear(dim_model, d_head * n_heads, bias=False) 
        self.w_v = Linear(dim_model, d_head * n_heads, bias=False)
        self.w_o = Linear(dim_model, dim_model, bias=False)
    
    def init_weights(self) -> None:
        dim_model = self.n_heads * self.d_head
        std = math.sqrt(3.0 / dim_model)
        self.w_q.init_weights(std)
        self.w_k.init_weights(std)
        self.w_v.init_weights(std)
        self.w_o.init_weights(methods="zero")

    def forward(self, x: torch.Tensor, rope_cache=None, mask=None, kv_cache=None) -> torch.Tensor:
        n_heads = self.n_heads
        d_head = self.d_head

        bs = x.size(0)
        len_x = x.size(1)

        q = self.w_q(x) # b X l X (n_heads*d_head)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(bs, len_x, n_heads, d_head)
        k = k.view(bs, len_x, n_heads, d_head)
        v = v.view(bs, len_x, n_heads, d_head)

        if rope_cache is not None:
            q = apply_rope(q, rope_cache)
            k = apply_rope(k, rope_cache)
        if self.norm_before_attn:
            q = apply_rms_norm(q, eps=1e-8, torch_impl=True)
            k = apply_rms_norm(k, eps=1e-8, torch_impl=True)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # b X n_heads X l X d_head

        if kv_cache is not None:
            k, v = kv_cache.insert(k, v)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)  
        # TODO: Support different attention implementations
        # TODO: Use context manager to select attention implementation
        # TODO: Detect context manager and use the corresponding attention implementation
        output = scaled_dot_product_attention(q, k, v, scale=d_head, attn_mask=mask, is_causal=True)
    
        output = output.transpose(1, 2).contiguous().view(bs, len_x, -1)
        output = self.w_o(output)
        return output, None

    
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
    

class DecoderLayer(Module):
    '''Decoder layer'''
    def __init__(
            self, 
            dim_model: int, 
            dim_ffn: int, 
            n_heads: int, 
            d_head: int, 
            dropout: float,
            enable_gqa: bool = False,
            layer_idx: int = 0,
            norm_before_attn: bool = True
        ) -> None:
        super().__init__()
        if not norm_before_attn:
            warnings.warn("Using 'norm_before_attn=False' is not recommended and may lead to training instability.", UserWarning)
        self.attention = CausalSelfAttention(
            dim_model=dim_model, 
            n_heads=n_heads, 
            d_head=d_head, 
            enable_gqa=enable_gqa,
            layer_idx=layer_idx, 
            norm_before_attn=norm_before_attn   
        )
        
        self.ffn = FeedForward(d_in=dim_model, d_latent=dim_ffn, dropout=dropout)
        self.dropout_rate = dropout

    def forward(self, x, mask=None,):
        h, self_attention = self.attention(x, mask=mask)

        if self.training:
            h = F.dropout(h, p=self.dropout_rate, training=True)
        h += x
        h = apply_rms_norm(h)
        output = h + self.ffn(h)

        return output, self_attention, None

class MixtureOfExpertsLayer(Module):
    '''Mixture of Experts layer'''
    def __init__(
            self, 
            dim_model: int, 
            dim_ffn: int, 
            n_experts: int, 
            dropout: float
        ) -> None:
        super().__init__()
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