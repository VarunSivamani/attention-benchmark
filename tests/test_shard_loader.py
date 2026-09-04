"""
Tests for shard loader.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.attention_benchmark.dataset.shard_loader import ShardLoader


@pytest.fixture
def temp_shards():
    """Create temporary shard files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        np.arange(1000, dtype=np.uint16).tofile(
            Path(tmpdir) / "shard_00000_train_tokens.bin"
        )
        np.arange(1000, 1500, dtype=np.uint16).tofile(
            Path(tmpdir) / "shard_00001_train_tokens.bin"
        )
        np.arange(2000, 2100, dtype=np.uint16).tofile(
            Path(tmpdir) / "shard_00000_val_tokens.bin"
        )
        yield tmpdir


def test_shard_loader_creation(temp_shards):
    """Test shard loader creation."""
    device = "cpu"
    loader = ShardLoader(temp_shards, split="train", device=device)
    assert loader is not None
    assert loader.total_tokens == 1500


def test_shard_loader_batch_shape(temp_shards):
    """Test shard loader batch shapes."""
    device = "cpu"
    loader = ShardLoader(temp_shards, split="train", device=device)
    input_ids, targets = loader.get_batch(batch_size=4, seq_len=10)
    assert input_ids.shape == (4, 10)
    assert targets.shape == (4, 10)


def test_shard_loader_dtype(temp_shards):
    """Test shard loader returns correct dtype."""
    device = "cpu"
    loader = ShardLoader(temp_shards, split="train", device=device)
    input_ids, targets = loader.get_batch(batch_size=2, seq_len=5)
    assert input_ids.dtype in [torch.uint16, torch.int32, torch.int64]
    assert targets.dtype in [torch.uint16, torch.int32, torch.int64]


def test_shard_loader_device(temp_shards):
    """Test shard loader places tensors on correct device."""
    device = "cpu"
    loader = ShardLoader(temp_shards, split="train", device=device)
    input_ids, targets = loader.get_batch(batch_size=2, seq_len=5)
    assert input_ids.device.type == device
    assert targets.device.type == device


def test_shard_loader_multiple_batches(temp_shards):
    """Test shard loader can produce multiple batches."""
    device = "cpu"
    loader = ShardLoader(temp_shards, split="train", device=device)
    batches = [loader.get_batch(batch_size=2, seq_len=10) for _ in range(5)]
    assert len(batches) == 5
    for input_ids, targets in batches:
        assert input_ids.shape == (2, 10)
        assert targets.shape == (2, 10)


def test_shard_loader_val_split(temp_shards):
    """Test shard loader loads val split correctly."""
    device = "cpu"
    loader = ShardLoader(temp_shards, split="val", device=device)
    assert loader.total_tokens == 100
    input_ids, targets = loader.get_batch(batch_size=1, seq_len=10)
    assert input_ids.shape == (1, 10)
