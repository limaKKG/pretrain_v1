from typing import Any, Dict, Optional, Tuple
from transformers import AutoTokenizer
import torch
from torch import nn
from torch.nn import functional as F
import os
from torch.utils.checkpoint import checkpoint
from config.training_config import LLaMAConfig
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).to(x.dtype)

    
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0, 
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = float(rope_theta)
        self.dtype = dtype
        self.device = device
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._seq_len_cached = 0

    @torch.no_grad()
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._seq_len_cached and self.cos_cached.device == device:
            return
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1) 
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self.cos_cached = cos.view(seq_len, 1, 1, self.dim).to(device)
        self.sin_cached = sin.view(seq_len, 1, 1, self.dim).to(device)
        self._seq_len_cached = seq_len

    def forward(
        self,
        q: torch.Tensor, 
        k: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        seq_len_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q.dim() == 4 and k.dim() == 4, "q,k must be [B,S,H,D]"
        bq, s, hq, d = q.shape
        bk, sk, hk, dk = k.shape
        assert bq == bk and s == sk and d == dk, "q,k must match in [B,S,D]"

        device = q.device
        dtype = q.dtype if q.dtype.is_floating_point else self.dtype

        if position_ids is None:
            max_pos = seq_len_offset + s
            self._update_cos_sin_cache(max_pos, device, dtype)
            cos = self.cos_cached[seq_len_offset : seq_len_offset + s].transpose(0, 1)
            sin = self.sin_cached[seq_len_offset : seq_len_offset + s].transpose(0, 1)
        else:
            max_pos = int(position_ids.max().item()) + 1
            self._update_cos_sin_cache(max_pos, device, dtype)
            cos = self.cos_cached.index_select(0, position_ids.reshape(-1)).view(bq, s, 1, d)
            sin = self.sin_cached.index_select(0, position_ids.reshape(-1)).view(bq, s, 1, d)

        q_out = (q * cos) + (rotate_half(q) * sin)
        k_out = (k * cos) + (rotate_half(k) * sin)
        return q_out, k_out

class Attention(nn.Module):
    def __init__(self, num_heads: int = 32, 
                       n_kv_heads: Optional[int] = None,
                       dim: int = 4096,
                       rope: RotaryEmbedding = None
     ):
        super().__init__()
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.dim = dim
        self.rope = rope
        self.head_dim = self.dim // self.num_heads
        self.attn_dropout = 0.0
        self.q_proj = nn.Linear(
            self.dim,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.dim,
            bias=False
        )
    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        B, S, H, D = x.shape
        return x[:, :, :, None, :].expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)

    def forward(
        self,
        x: torch.Tensor,                   
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,  
        is_causal: bool = True,                      
        seq_len_offset: int = 0,       
    ) -> torch.Tensor:
        B, S, C = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        q, k = self.rope(q, k, position_ids=position_ids, seq_len_offset=seq_len_offset)
        if self.n_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)   
            v = self._repeat_kv(v, n_rep)
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)

        # Handle attention mask for F.scaled_dot_product_attention
        # attention_mask is expected to be [B, S] with 1 for tokens and 0 for padding.
        # We keep causal masking ON and add a padding mask for keys.
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, S] -> [B, 1, 1, S]
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() != 4:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")
            # Convert to additive mask: 0 for valid, -inf for masked
            attention_mask = attention_mask.to(dtype=q.dtype)
            attn_mask = (1.0 - attention_mask) * torch.finfo(q.dtype).min

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, C)
        return self.o_proj(attn_out) 


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        layer_id: int, 
        num_heads: int = 32, 
        n_kv_heads: Optional[int] = None, 
        dim: int = 4096, 
        intermediate_size: int = 11008,
        norm_eps: float = 1e-6, 
        rope: RotaryEmbedding = None
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.dim = dim
        self.norm_eps = norm_eps
        self.head_dim = self.dim // self.num_heads
        
        self.attention = Attention(self.num_heads, self.n_kv_heads, self.dim, rope)
        self.feed_forward = SwiGLU(self.dim, intermediate_size)
        self.attention_norm = RMSNorm(self.dim, eps=self.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=self.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seq_len_offset: int = 0,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_causal=True,
            seq_len_offset=seq_len_offset
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size
        )
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                layer_id=i,
                num_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                dim=config.hidden_size,
                intermediate_size=config.intermediate_size,
                norm_eps=config.rms_norm_eps,
                rope=self.rope,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, position_ids=None, seq_len_offset=0):
        h = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(
                    layer,
                    h,
                    attention_mask,
                    position_ids,
                    seq_len_offset,
                    use_reentrant=False,
                )
            else:
                h = layer(h, attention_mask=attention_mask, position_ids=position_ids, seq_len_offset=seq_len_offset)

        return self.norm(h)


class LLaMAForCausalLM(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.model = LLaMAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {"loss": loss, "logits": logits} 


class LLaMAModelWrapper:    
    def __init__(self, config: LLaMAConfig, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model: Optional[LLaMAForCausalLM] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self) -> LLaMAForCausalLM:
        self.model = LLaMAForCausalLM(self.config)
        self.model.to(self.device)
        return self.model
