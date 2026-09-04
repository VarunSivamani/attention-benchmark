"""
Transformer block: layer norm + attention + layer norm + MLP (SwiGLU).
Attention core swapped based on config.attention_type.
"""


import torch
import torch.nn as nn

from src.attention_benchmark.attention.gqa_flash_attention import CausalSelfAttentionGQA
from src.attention_benchmark.attention.nsa_gqa_attention import CausalSelfAttentionNSA


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm."""
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class SwiGLU(nn.Module):
    """SwiGLU MLP: FFN(x) = (x * W + b) ⊗ SiLU(x * V + c), where ⊗ is element-wise product."""

    def __init__(self, d_model: int, hidden_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        hidden_dim = int(d_model * hidden_ratio)
        self.w = nn.Linear(d_model, hidden_dim, bias=bias)
        self.v = nn.Linear(d_model, hidden_dim, bias=bias)
        self.out = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU."""
        return self.out(self.w(x) * torch.nn.functional.silu(self.v(x)))


class TransformerBlock(nn.Module):
    """
    Single transformer block: [RMSNorm -> Attention] -> [RMSNorm -> SwiGLU MLP]

    Args:
        config: GPTConfig object
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.n_embd, eps=config.norm_eps)

        if config.attention_type == "flash_gqa":
            self.attn = CausalSelfAttentionGQA(
                d_model=config.n_embd,
                n_head=config.n_head,
                n_kv_head=config.n_kv_head,
                max_seq_len=config.block_size,
                dropout=config.dropout,
                bias=config.bias,
            )
        elif config.attention_type == "nsa_gqa":
            self.attn = CausalSelfAttentionNSA(
                d_model=config.n_embd,
                n_head=config.n_head,
                n_kv_head=config.n_kv_head,
                max_seq_len=config.block_size,
                nsa_block_size=config.nsa_block_size,
                nsa_window_size=config.nsa_window_size,
                nsa_block_counts=config.nsa_block_counts,
                dropout=config.dropout,
                bias=config.bias,
            )
        else:
            raise ValueError(f"Unknown attention_type: {config.attention_type}")

        self.mlp = SwiGLU(d_model=config.n_embd, hidden_ratio=4.0, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with pre-norm residual."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
