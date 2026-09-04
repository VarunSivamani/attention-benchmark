"""
Tests for Kronecker embedding wrapper.
"""

import pytest
import torch
from transformers import GPT2Tokenizer

from src.attention_benchmark.embeddings.kronecker_embedding import build_kronecker_embedding


@pytest.fixture
def tokenizer():
    """Get GPT-2 tokenizer."""
    return GPT2Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def embedding(tokenizer):
    """Create a Kronecker embedding module."""
    return build_kronecker_embedding(
        vocab_size=50257,
        d_model=768,
        tokenizer=tokenizer,
        pos_dim=16,
    )


def test_kronecker_embedding_creation(embedding):
    """Test that embedding is created correctly."""
    assert embedding is not None
    assert hasattr(embedding, "forward")


def test_kronecker_embedding_forward_shape(tokenizer):
    """Test embedding forward pass output shape."""
    emb = build_kronecker_embedding(vocab_size=50257, d_model=768, tokenizer=tokenizer, pos_dim=16)
    token_ids = torch.randint(0, 50257, (2, 10))
    output = emb(token_ids)
    assert output.shape == (2, 10, 768)


def test_kronecker_embedding_dtype(tokenizer):
    """Test that output dtype matches model dtype."""
    emb = build_kronecker_embedding(vocab_size=50257, d_model=768, tokenizer=tokenizer, pos_dim=16)
    token_ids = torch.randint(0, 50257, (1, 5))
    output = emb(token_ids)
    assert output.dtype == torch.float32


def test_kronecker_embedding_device(tokenizer):
    """Test that output is on same device as input."""
    emb = build_kronecker_embedding(vocab_size=50257, d_model=768, tokenizer=tokenizer, pos_dim=16)
    token_ids = torch.randint(0, 50257, (1, 5))
    output = emb(token_ids)
    assert output.device == token_ids.device


def test_kronecker_embedding_gradient_flow(tokenizer):
    """Test that gradients flow through embedding."""
    emb = build_kronecker_embedding(vocab_size=50257, d_model=768, tokenizer=tokenizer, pos_dim=16)
    token_ids = torch.randint(0, 50257, (2, 5))
    output = emb(token_ids)
    loss = output.sum()
    loss.backward()
    assert any(p.grad is not None for p in emb.parameters())
