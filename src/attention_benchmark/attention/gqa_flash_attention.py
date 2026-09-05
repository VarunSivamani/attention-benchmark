"""
Flash Attention with Grouped Query Attention (GQA).

Uses PyTorch's native SDPA (scaled_dot_product_attention) with enable_gqa=True,
requires PyTorch >= 2.5. Runs on CPU/MPS/CUDA.

Explicit flash backend: prefers FLASH_ATTENTION -> EFFICIENT_ATTENTION via
torch.nn.attention.sdpa_kernel; disables math fallback. On T4 Turing
(flash_sdp_enabled=False) it auto-falls to mem_efficient, still O(N) fused.
Configure via torch.backends.cuda.enable_flash_sdp / enable_mem_efficient_sdp.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer flash/mem_efficient on CUDA; keep math enabled for CPU/MPS fallback
try:
    torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
    torch.backends.cuda.enable_math_sdp(True)  # type: ignore[attr-defined]  # CPU fallback needs math
except Exception:
    pass


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding."""

    def __init__(self, d_model: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply RoPE to query or key embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, n_head, d_head)
            seq_len: Sequence length (defaults to x.shape[1])

        Returns:
            Tensor of same shape with RoPE applied
        """
        batch, seq_len, n_head, d_head = x.shape
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        x1 = x[..., : d_head // 2]
        x2 = x[..., d_head // 2 :]
        return torch.cat([
            x1 * cos[None, :seq_len, None, : d_head // 2] - x2 * sin[None, :seq_len, None, d_head // 2 :],
            x1 * sin[None, :seq_len, None, : d_head // 2] + x2 * cos[None, :seq_len, None, d_head // 2 :],
        ], dim=-1)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


class CausalSelfAttentionGQA(nn.Module):
    """
    Causal self-attention with Grouped Query Attention and Flash backend.

    Args:
        d_model: Model embedding dimension
        n_head: Number of query heads
        n_kv_head: Number of key-value heads (typically < n_head for GQA)
        max_seq_len: Maximum sequence length for RoPE
        dropout: Dropout rate
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply grouped query attention with causal masking.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attn_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Explicit flash dispatch: FLASH -> EFFICIENT -> MATH fallback (CPU needs MATH)
        def _sdpa(q, k, v):  # type: ignore[no-untyped-def]
            try:
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0, enable_gqa=True
                )
            except TypeError:
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
                )

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # On CUDA prefer FLASH/EFFICIENT, on CPU will fallback to MATH
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                attn_out = _sdpa(q, k, v)
        except Exception:
            attn_out = _sdpa(q, k, v)

        attn_out = attn_out.reshape(batch_size, seq_len, -1)  # fix GQA non-contiguous view
        return self.o_proj(attn_out)
