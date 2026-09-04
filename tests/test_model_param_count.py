"""
Tests for model parameter counting and architecture.
"""

import pytest
import torch

from src.attention_benchmark.model.config import GPTConfig
from src.attention_benchmark.model.gpt import GPTModel


@pytest.fixture
def config():
    """Create a small config for testing."""
    return GPTConfig(
        vocab_size=50257,
        n_layer=2,
        n_embd=256,
        n_head=8,
        n_kv_head=2,
        block_size=512,
        pos_dim=16,
        attention_type="flash_gqa",
        dropout=0.1,
        bias=False,
    )


def test_model_creation(config):
    """Test model creation."""
    model = GPTModel(config)
    assert model is not None
    assert len(model.blocks) == config.n_layer


def test_model_forward_shape(config):
    """Test model forward pass output shape."""
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(input_ids)
    assert logits.shape == (2, 10, config.vocab_size)
    assert loss is None


def test_model_forward_with_targets(config):
    """Test model forward pass with loss computation."""
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    targets = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 10, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_model_parameter_count(config):
    """Test model has expected approximate parameter count."""
    model = GPTModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    assert total_params < 100_000_000


def test_model_gradient_flow(config):
    """Test gradients flow through model."""
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    targets = torch.randint(0, config.vocab_size, (2, 5))
    logits, loss = model(input_ids, targets)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_model_nsa_variant_creation():
    """Test NSA variant model creation fails gracefully if not available."""
    config = GPTConfig(
        vocab_size=50257,
        n_layer=1,
        n_embd=128,
        n_head=4,
        n_kv_head=2,
        attention_type="nsa_gqa",
    )
    with pytest.raises(AssertionError):
        GPTModel(config)


def test_model_generation(config):
    """Test model generation."""
    model = GPTModel(config)
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    assert generated.shape == (1, 15)
