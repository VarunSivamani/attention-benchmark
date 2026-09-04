"""
Tests for GQA Flash Attention module.
"""

import pytest
import torch

from src.attention_benchmark.attention.gqa_flash_attention import CausalSelfAttentionGQA


@pytest.fixture
def attention_module():
    """Create a GQA Flash attention module."""
    return CausalSelfAttentionGQA(
        d_model=768,
        n_head=12,
        n_kv_head=4,
        max_seq_len=8192,
        dropout=0.1,
        bias=False,
    )


def test_attention_creation(attention_module):
    """Test attention module creation."""
    assert attention_module is not None
    assert attention_module.n_head == 12
    assert attention_module.n_kv_head == 4


def test_attention_forward_shape(attention_module):
    """Test attention forward pass output shape."""
    x = torch.randn(2, 10, 768)
    output = attention_module(x)
    assert output.shape == x.shape


def test_attention_causal_mask():
    """Test that attention respects causal masking."""
    attn = CausalSelfAttentionGQA(
        d_model=64,
        n_head=4,
        n_kv_head=2,
        dropout=0.0,
    )
    attn.eval()

    x = torch.randn(1, 5, 64)
    output1 = attn(x)

    output2 = attn(x[:, :3, :])
    assert torch.allclose(output1[:, :3, :], output2, atol=1e-5)


def test_attention_gradient_flow(attention_module):
    """Test that gradients flow through attention."""
    x = torch.randn(2, 5, 768, requires_grad=True)
    output = attention_module(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None


def test_attention_different_batch_sizes(attention_module):
    """Test attention with different batch sizes."""
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 10, 768)
        output = attention_module(x)
        assert output.shape == (batch_size, 10, 768)


def test_attention_different_seq_lengths(attention_module):
    """Test attention with different sequence lengths."""
    for seq_len in [1, 10, 100, 512]:
        x = torch.randn(2, seq_len, 768)
        output = attention_module(x)
        assert output.shape == (2, seq_len, 768)


def test_attention_gqa_head_ratio():
    """Test that GQA correctly handles different head ratios."""
    attn = CausalSelfAttentionGQA(
        d_model=768,
        n_head=12,
        n_kv_head=3,
        dropout=0.0,
    )
    x = torch.randn(2, 10, 768)
    output = attn(x)
    assert output.shape == (2, 10, 768)
