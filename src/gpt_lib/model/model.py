from gpt_lib.model.layers import (
    Module, 
    DecoderLayer, 
    Linear, 
    apply_rms_norm,
    precompute_rope, 
    precompute_positional_encoding
)
from gpt_lib.model.loss import build_objective
from gpt_lib.tokenizer.tokenizer import build_tokenizer
from gpt_lib.utils.schemas import (
    get_default_device,
    GenerationConfig,
    GPTConfig, 
    ModelOutput, 
    ModelCompletionOutput,
    TransformerConfig, 
    TransformerOutput, 
)
from gpt_lib.utils.default import MODELS_FOLDER, DEVICE

from typing import List, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

class Transformer(Module):
    def __init__(
            self,
            args: TransformerConfig = TransformerConfig(),
            device: str | torch.device = DEVICE,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.config = args
        self.vocab_size = args.vocab_size
        self.padding_idx = args.pad_id
        self.device = device
        self.dtype = dtype
        if self.config.positional_encoding == "rope":
            rope_cache = precompute_rope(
                seq_len=args.max_context,
                d_head=args.d_head,
                base=10_000,
                dtype=dtype,
                device=device,
            )
            self.register_buffer("rope_cache", rope_cache, persistent=False)
        elif self.config.positional_encoding == "positional":
            pos_enc = precompute_positional_encoding(args.max_context, args.d_model, dtype=dtype, device=device)
            self.register_buffer("pos_enc", pos_enc, persistent=False)
        elif self.config.positional_encoding == "alibi":
            raise NotImplementedError("ALiBi positional encoding is not yet implemented.")
        else:
            raise ValueError(f"Unknown positional encoding: {self.config.positional_encoding}")

        # TODO: Will change to customed embedding
        embedding = nn.Embedding(
            num_embeddings = args.vocab_size, 
            embedding_dim = args.d_model, 
            padding_idx = args.pad_id,
            sparse=False,
            device=device,
            dtype=dtype
        )

        self.layers = nn.ModuleDict(dict(
            emb=embedding,
            blocks=nn.ModuleList([
                DecoderLayer(
                    dim_model=args.d_model,
                    dim_ffn=args.d_ffn, 
                    n_heads=args.n_heads, 
                    d_head=args.d_head, 
                    dropout=args.dropout,
                    layer_idx=layer_idx,
                    norm_before_attn=args.norm_before_attn,
                    enable_gqa=args.enable_gqa,
                ) 
                for layer_idx in range(args.n_layers)
            ])
        ))
        
        self.model_head = Linear(args.d_model, args.vocab_size, bias=False)

                
    def forward(
            self, 
            input_ids: torch.Tensor,
            kv_cache: dict | None = None,
            return_attentions: bool = False,
            return_hidden_states: bool = False,
        ):
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        assert input_ids.shape[-1] <= self.config.max_context, f"Input sequence length {input_ids.shape[-1]} exceeds max context {self.config.max_context}"
        assert input_ids.dim() == 2, "Input ids should be of shape (batch_size, seq_len)"
        mask = self.get_pad_mask(input_ids)
        x = self.layers.emb(input_ids)

        if self.config.positional_encoding == "positional_encoding":
            x = x + self.pos_enc[:x.size(2)]

        x = apply_rms_norm(x, eps=1e-8, torch_impl=True)
        attentions = []
        for i, decoder_layer in enumerate(self.layers.blocks):
            return_attn = return_attentions and (i == len(self.layers.blocks) - 1)
            x, attn, _ = decoder_layer(
                x=x, 
                mask=mask, 
                # kv_cache=kv_cache, 
                # return_attentions=return_attn
            )
            assert x.nansum() != 0, f"NaN detected in layer {i} output."
            if return_attn:
                attentions.append(attn)

        x = apply_rms_norm(x, eps=1e-8, torch_impl=True)

        if return_hidden_states:
            hidden_states = x

        softcap = 18
        logits = self.model_head(x)
        logits = torch.clamp(logits, min=-softcap, max=softcap)
        
        return TransformerOutput(
            logits=logits,
            attentions=attn if return_attentions else None,
            hidden_states=hidden_states if return_hidden_states else None,
            kv_cache=kv_cache
        )
    
    def get_pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        pad_idx = self.padding_idx
        pad_mask = (seq != pad_idx)
        return pad_mask
        # _, len_s = seq.size()
        # subsequent_mask = (1 - torch.triu(
        #     torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        # return pad_mask & subsequent_mask

class GPTModel:
    def __init__(
            self,
            model: torch.nn.Module | None = None,
            config: GPTConfig = GPTConfig()
        ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = build_tokenizer(config.tokenizer)
        # TODO: Add support for loading Dense vs MoE vs Hybrid models
        self.model = model
        self.loss_fn = build_objective(config.objective)
        self.model = self.model.to(DEVICE)

        assert self.model is not None, "Model must be provided"
        assert all(hasattr(self.model, attr) for attr in ["config", "padding_idx", "vocab_size"]), "Model must have config, padding_idx and vocab_size attributes"
        assert self.model.padding_idx == self.tokenizer.pad_token_id, "Tokenizer pad token id must match model pad id"
        assert self.model.vocab_size == self.tokenizer.vocab_size, "Model vocab size must match tokenizer vocab size"
        assert self.model.padding_idx == self.config.objective.ignore_index, "Objective ignore index must match model pad id"
    
    def __call__(self, input_ids, labels=None, *args, **kwargs) -> ModelOutput:
        input_ids = input_ids.to(self.config.device)
        logits = self.model(input_ids, *args, **kwargs).logits
        loss = None
        if labels is not None:
            labels = labels.to(self.config.device)
            loss = self.loss_fn(logits, labels)
        return ModelOutput(logits=logits, loss=loss)

    def __repr__(self) -> str:
        return f"GPTModel(config={self.config}, model={self.model})"
    
    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def forward(
            self, 
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            attentions: bool = False,
            kv_cache: dict | None = None,
            log_prob: bool = False,
            temperature: float = 1.0,
            **kwargs
        ) -> ModelOutput:
        input_ids = input_ids.to(self.config.device)
        labels = labels.to(self.config.device) if labels is not None else None
        output: TransformerOutput = self.model(
            input_ids, 
            return_attentions=attentions, 
            kv_cache=kv_cache
        )
        if temperature > 0:
            logits = output.logits / temperature
        else:
            logits = output.logits
        
        loss = None
        output = ModelOutput(
            logits=logits,
            loss=loss,
            attentions=output.attentions if attentions else None,
            log_probs=F.log_softmax(logits, dim=-1) if log_prob else None,
            hidden_states=output.hidden_states,
            kv_cache=kv_cache,
        )
        
        if labels is not None:
            assert output.logits.device == labels.device, f"Logits and labels must be on the same device. Got {output.logits.device} and {labels.device}"
            loss = self.loss_fn(output, labels)
            output.loss = loss

        return output
    
    @torch.inference_mode()
    def generate(
            self,
            text: str | List[str],
            ground_truth: str | List[str] | None = None,
            generation_config: GenerationConfig | None = None,
        ) -> ModelCompletionOutput | Iterator[ModelCompletionOutput]:
        if generation_config is None:
            generation_config = GenerationConfig()
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        self.model.eval()
        if isinstance(text, str):
            text = [text]
        input_ids = self.tokenizer.batch_encode(text, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, device=next(self.model.parameters()).device)

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            label_ids = self.tokenizer.encode(ground_truth, add_special_tokens=True)
            label_ids = torch.tensor(label_ids, device=next(self.model.parameters()).device)
        else:
            label_ids = None

        generated = input_ids

        for _ in range(generation_config.max_length):
            inputs = generated[:, -generation_config.max_length:]
            output: ModelOutput = self(
                inputs, 
                label_ids=label_ids,
                temperature=generation_config.temperature,
                kv_cache=None,
                attentions=False
            )
            next_token_logits = output.logits[:, -1, :] / generation_config.temperature
            filtered_logits = self.top_k_top_p_filtering(
                next_token_logits, top_k=generation_config.top_k, top_p=generation_config.top_p
            )

            probabilities = nn.functional.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if generation_config.stream:
                yield ModelCompletionOutput(
                    logits=output.logits,
                    loss=output.loss,
                    log_probs=output.log_probs,
                    completions=[
                        self.tokenizer.decode(generated[i].tolist()) for i in range(generated.size(0))
                    ],
                    kv_cache=output.kv_cache,
                    done=False
                )

        output = ModelCompletionOutput(
            logits=output.logits,
            loss=output.loss,
            log_probs=output.log_probs,
            completions=[
                self.tokenizer.decode(generated[i].tolist()) for i in range(generated.size(0))
            ],
            kv_cache=output.kv_cache,
            done=True
        )
        return output
    
    def number_of_parameters(self) -> int:
        try:
            return self.model.nb_parameters()
        except AttributeError:
            return sum([p.numel() for p in self.model.parameters()])

    def init_weights(self) -> None:
        try:
            for name, layer in self.model.layers.items():
                if hasattr(layer, "init_weights"):
                    layer.init_weights()
                else:
                    for p in layer.parameters():
                        if p.dim() > 1:
                            nn.init.uniform_(p)
        except:
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def get_optimzer(self) -> torch.optim.Optimizer:
        # TODO: return optimizer based on config
        pass
        
    
    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            model_dir: str | None = None,
        ) -> "GPTModel":
        return cls.load(
            model_name=model_name,
            checkpoint_version="latest",
            model_dir=model_dir
        )
    
    @classmethod
    def load(
            cls,
            model_name: str,
            ckpt_version: str,
            model_dir: str | None = None,
            device: str | None = None,
        ) -> "GPTModel":
        if model_dir is None:
            model_dir = MODELS_FOLDER
        if not ckpt_version.endswith(".pth"):
            ckpt_version += ".pth"
        config = GPTConfig.from_file(model_name=model_name, model_dir=model_dir)
        model = Transformer(args=config.model, device=config.device, dtype=config.dtype)
        model_path = config.dirname / ckpt_version
        if not device:
            device = config.device
        if not config.device:
            device = get_default_device()
            config.device = device
        model.load_state_dict(torch.load(model_path, map_location=device))
        return cls(model=model, config=config)
    
    @classmethod
    def from_scratch(
            cls,
            config: GPTConfig = GPTConfig()
        ) -> "GPTModel":
        config.to_file(mode="pickle")
        model = Transformer(args=config.model, device=config.device, dtype=config.dtype)
        gpt = cls(model=model, config=config)
        gpt.init_weights()
        return gpt
    
    @classmethod
    def from_yaml(
            cls,
            yaml_path: str | Path,
        ) -> "GPTModel":
        config = GPTConfig.from_yaml(yaml_path)
        model = Transformer(args=config.model, device=config.device, dtype=config.dtype)
        gpt = cls(model=model, config=config)
        gpt.init_weights()
        return gpt
    
    def save_checkpoint(
            self,
            ckpt_version: str | None = None,
            keep_vars: bool = True,
        ) -> None:
        if ckpt_version is None:
            ckpt_version = "checkpoint.pth"
        if not ckpt_version.endswith(".pth"):
            ckpt_version += ".pth"
        model_path = self.config.dirname / ckpt_version
        torch.save(self.model.state_dict(keep_vars=keep_vars), model_path)
    
