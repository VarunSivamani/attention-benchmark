"""
Native Sparse Attention with Grouped Query Attention (NSA+GQA).

Uses DeepSeek's native-sparse-attention Triton kernel via parallel_nsa.
Requires CUDA GPU — will assert at construction if unavailable.

Reference: https://github.com/fla-org/native-sparse-attention
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from native_sparse_attention.ops.parallel import parallel_nsa
    HAS_NSA = True
except ImportError:
    HAS_NSA = False
    parallel_nsa = None


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


class CausalSelfAttentionNSA(nn.Module):
    """
    Causal self-attention with Grouped Query Attention and Native Sparse Attention backend.

    Args:
        d_model: Model embedding dimension
        n_head: Number of query heads
        n_kv_head: Number of key-value heads (typically < n_head for GQA)
        max_seq_len: Maximum sequence length
        nsa_block_size: Block granularity for sparsity pattern
        nsa_window_size: Sliding window size for local attention
        nsa_block_counts: Number of sparse blocks to select
        dropout: Dropout rate
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int = 8192,
        nsa_block_size: int = 64,
        nsa_window_size: int = 512,
        nsa_block_counts: int = 16,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert HAS_NSA, (
            "native_sparse_attention not installed. Install via:\n"
            "  git clone https://github.com/fla-org/native-sparse-attention.git\n"
            "  cd native-sparse-attention\n"
            "  git submodule update --init --recursive\n"
            "  pip install ."
        )
        assert torch.cuda.is_available(), (
            "Native Sparse Attention requires a CUDA GPU. "
            "This module will not run on CPU or MPS. "
            "Use the Flash Attention variant (gqa_flash_attention) for CPU/MPS testing."
        )
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.nsa_block_size = nsa_block_size
        self.nsa_window_size = nsa_window_size
        self.nsa_block_counts = nsa_block_counts
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.g_slc = nn.Linear(d_model, n_head, bias=False)
        self.g_swa = nn.Linear(d_model, n_head, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Native Sparse Attention with GQA and causal masking.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attn_mask: Optional attention mask (not used by NSA kernel, kept for API compat)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        g_slc = torch.sigmoid(self.g_slc(x)).transpose(1, 2)
        g_swa = torch.sigmoid(self.g_swa(x)).transpose(1, 2)

        attn_out = parallel_nsa(
            q=q,
            k=k,
            v=v,
            g_slc=g_slc,
            g_swa=g_swa,
            block_size=self.nsa_block_size,
            window_size=self.nsa_window_size,
            block_counts=self.nsa_block_counts,
            cu_seqlens=None,
        )

        attn_out = attn_out.view(batch_size, seq_len, -1)
        return self.o_proj(attn_out)
