"""Compact modern decoder-only Transformer used by the training harness.

The default path is token embedding -> pre-norm residual blocks -> final norm ->
LM head. An input projection is available only as an explicit legacy ablation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _round_up(value: float, multiple: int) -> int:
    if multiple < 1:
        raise ValueError("multiple must be positive")
    return int(math.ceil(value / multiple) * multiple)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        is_causal: bool = True,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        qk_norm: bool = True,
        max_seq_len: Optional[int] = None,
        norm_eps: float = 1e-5,
        bias: bool = False,
        dropout: float = 0.0,
        fused_qkv: bool = True,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.is_causal = bool(is_causal)
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.max_seq_len = max_seq_len
        self.dropout = float(dropout)
        self.fused_qkv = bool(fused_qkv)
        if self.use_rope and self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")

        if self.fused_qkv:
            self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
            self.to_q = self.to_k = self.to_v = None
        else:
            self.to_qkv = None
            self.to_q = nn.Linear(dim, dim, bias=bias)
            self.to_k = nn.Linear(dim, dim, bias=bias)
            self.to_v = nn.Linear(dim, dim, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        rope_cos = rope_sin = None
        if self.use_rope and self.max_seq_len is not None:
            half = self.head_dim // 2
            frequencies = torch.arange(half, dtype=torch.float32)
            inverse = self.rope_theta ** (-frequencies / half)
            positions = torch.arange(self.max_seq_len, dtype=torch.float32)
            angles = torch.outer(positions, inverse)
            rope_cos = angles.cos()[None, None]
            rope_sin = angles.sin()[None, None]
        self.register_buffer("_rope_cos", rope_cos, persistent=False)
        self.register_buffer("_rope_sin", rope_sin, persistent=False)

    def _rope_tables(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        rebuild = (
            self._rope_cos is None
            or self._rope_cos.device != device
            or self._rope_cos.shape[-2] < seq_len
        )
        if rebuild:
            build_len = max(seq_len, self.max_seq_len or 0)
            half = self.head_dim // 2
            with torch.amp.autocast(device_type=device.type, enabled=False):
                frequencies = torch.arange(half, device=device, dtype=torch.float32)
                inverse = self.rope_theta ** (-frequencies / half)
                positions = torch.arange(build_len, device=device, dtype=torch.float32)
                angles = torch.outer(positions, inverse)
                self._rope_cos = angles.cos()[None, None]
                self._rope_sin = angles.sin()[None, None]
        return (
            self._rope_cos[..., :seq_len, :],
            self._rope_sin[..., :seq_len, :],
        )

    def _apply_rope(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        if not self.use_rope:
            return query, key
        half = query.shape[-1] // 2
        cosine, sine = self._rope_tables(query.shape[-2], query.device)
        cosine = cosine.to(query.dtype)
        sine = sine.to(query.dtype)

        def rotate(value: Tensor) -> Tensor:
            first, second = value[..., :half], value[..., half:]
            return torch.cat(
                (first * cosine - second * sine, first * sine + second * cosine),
                dim=-1,
            )

        return rotate(query), rotate(key)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, sequence, _ = hidden.shape
        if self.to_qkv is not None:
            query, key, value = self.to_qkv(hidden).chunk(3, dim=-1)
        else:
            query = self.to_q(hidden)
            key = self.to_k(hidden)
            value = self.to_v(hidden)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self._apply_rope(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, self.dim)
        return self.to_out(attended)


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        *,
        multiplier: float = 8.0 / 3.0,
        align_multiple: int = 64,
        bias: bool = False,
        fused: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = _round_up(dim * multiplier, align_multiple)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.fused = bool(fused)
        if self.fused:
            self.proj_in = nn.Linear(dim, 2 * self.hidden_dim, bias=bias)
            self.up_proj = self.gate_proj = None
        else:
            self.proj_in = None
            self.up_proj = nn.Linear(dim, self.hidden_dim, bias=bias)
            self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.proj_out = nn.Linear(self.hidden_dim, dim, bias=bias)

    def forward(self, hidden: Tensor) -> Tensor:
        if self.proj_in is not None:
            value, gate = self.proj_in(hidden).chunk(2, dim=-1)
        else:
            value = self.up_proj(hidden)
            gate = self.gate_proj(hidden)
        return self.proj_out(value * F.silu(gate))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_hidden_dim: Optional[int] = None,
        *,
        ff_multiplier: float = 8.0 / 3.0,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        qk_norm: bool = True,
        max_seq_len: Optional[int] = None,
        norm_eps: float = 1e-5,
        bias: bool = False,
        dropout: float = 0.0,
        fused_qkv: bool = True,
        fused_swiglu: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(
            dim,
            heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
            bias=bias,
            dropout=dropout,
            fused_qkv=fused_qkv,
        )
        self.ff_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ff = SwiGLU(
            dim,
            ff_hidden_dim,
            multiplier=ff_multiplier,
            bias=bias,
            fused=fused_swiglu,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden))
        hidden = hidden + self.ff(self.ff_norm(hidden))
        return hidden


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        ff_mult: float,
        vocab_size: int,
        max_seq_len: int,
        *,
        ff_hidden_dim: Optional[int] = None,
        gradient_checkpointing: bool = False,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
        depth_scaled_residual_init: bool = True,
        tie_embeddings: bool = True,
        input_projection: bool = False,
        bias: bool = False,
        dropout: float = 0.0,
        fused_qkv: bool = True,
        fused_swiglu: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1 or dim < 1 or vocab_size < 2 or max_seq_len < 2:
            raise ValueError("depth/dim must be positive and vocab/max_seq_len must exceed one")
        self.dim = int(dim)
        self.depth = int(depth)
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.use_rope = bool(use_rope)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.tie_embeddings = bool(tie_embeddings)

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = None if use_rope else nn.Embedding(max_seq_len, dim)
        # Reproduce the imported architecture only when explicitly requested.
        self.input_norm = nn.RMSNorm(dim, eps=norm_eps) if input_projection else nn.Identity()
        self.input_projection = (
            nn.Linear(dim, dim, bias=bias) if input_projection else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    ff_hidden_dim,
                    ff_multiplier=ff_mult,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                    qk_norm=qk_norm,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    bias=bias,
                    dropout=dropout,
                    fused_qkv=fused_qkv,
                    fused_swiglu=fused_swiglu,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self._initialize(depth_scaled_residual_init)
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize(self, depth_scaled_residual_init: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        if depth_scaled_residual_init:
            residual_std = 0.02 / math.sqrt(2.0 * self.depth)
            for block in self.blocks:
                nn.init.normal_(block.attn.to_out.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.ff.proj_out.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
        return_logits: bool = False,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        sequence = input_ids.shape[1]
        if sequence > self.max_seq_len:
            raise ValueError(
                f"sequence length {sequence} exceeds max_seq_len {self.max_seq_len}"
            )
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(sequence, device=input_ids.device)
            hidden = hidden + self.position_embedding(positions)
        hidden = self.input_projection(self.input_norm(hidden))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    block,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return (loss, logits) if return_logits else loss

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        if new_size < 2:
            raise ValueError("new_size must exceed one")
        old_embedding = self.token_embedding
        new_embedding = nn.Embedding(
            new_size,
            self.dim,
            device=old_embedding.weight.device,
            dtype=old_embedding.weight.dtype,
        )
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
        copied = min(old_embedding.num_embeddings, new_size)
        with torch.no_grad():
            new_embedding.weight[:copied].copy_(old_embedding.weight[:copied])
        old_head = self.lm_head
        self.token_embedding = new_embedding
        self.lm_head = nn.Linear(
            self.dim,
            new_size,
            bias=False,
            device=new_embedding.weight.device,
            dtype=new_embedding.weight.dtype,
        )
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.lm_head.weight[:copied].copy_(old_head.weight[:copied])
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.vocab_size = int(new_size)
        return self.token_embedding


__all__ = ["Attention", "SwiGLU", "TransformerBlock", "Transformer"]
