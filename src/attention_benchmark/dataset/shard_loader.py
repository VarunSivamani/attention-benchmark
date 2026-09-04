"""
Memmap-based token shard loader for training.

Reads uint16 .bin files (GPT-2 BPE tokenized), supports re-chunking per curriculum stage.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class ShardLoader:
    """
    Loads token shards from .bin files (uint16 format).

    Args:
        shard_dir: Directory containing shard files
        split: "train" or "val"
        device: torch device for tensors
    """

    def __init__(
        self,
        shard_dir: str,
        split: str = "train",
        device: str = "cuda",
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.device = device

        pattern = f"*_{split}_*.bin"
        self.shard_paths = sorted(self.shard_dir.glob(pattern))

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shard files found in {self.shard_dir} matching pattern '{pattern}'. "
                f"Run src/attention_benchmark/dataset/prepare_fineweb.py or "
                f"`python -m src.attention_benchmark.dataset.prepare_dataset` first."
            )

        self.mmaps = []
        self.total_tokens = 0
        self._load_mmaps()

    def _load_mmaps(self) -> None:
        """Load all shard files as memmaps."""
        for path in self.shard_paths:
            mmap = np.memmap(path, dtype=np.uint16, mode="r")
            self.mmaps.append(mmap)
            self.total_tokens += len(mmap)
        print(f"Loaded {len(self.mmaps)} {self.split} shards, {self.total_tokens:,} tokens total")

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        start_token: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of token sequences.

        Args:
            batch_size: Number of sequences per batch
            seq_len: Sequence length
            start_token: Starting token index (for curriculum resumption)

        Returns:
            Tuple of (input_ids, targets), both shape (batch_size, seq_len)
        """
        input_ids = []
        targets = []

        for _ in range(batch_size):
            idx = (start_token + np.random.randint(0, max(1, self.total_tokens - seq_len))) % (
                self.total_tokens - seq_len
            )
            tokens = self._get_tokens(idx, seq_len + 1)
            input_ids.append(tokens[:-1])
            targets.append(tokens[1:])

        return (
            torch.from_numpy(np.stack(input_ids)).to(self.device).long(),
            torch.from_numpy(np.stack(targets)).to(self.device).long(),
        )

    def _get_tokens(self, start_idx: int, length: int) -> np.ndarray:
        """
        Get tokens starting at start_idx, wrapping around shards if needed.

        Args:
            start_idx: Global token index
            length: Number of tokens to fetch

        Returns:
            Array of token IDs
        """
        tokens = []
        current_idx = start_idx

        while len(tokens) < length:
            shard_idx, local_idx = self._global_to_shard(current_idx)
            mmap = self.mmaps[shard_idx]
            remaining_in_shard = len(mmap) - local_idx
            to_take = min(remaining_in_shard, length - len(tokens))
            tokens.extend(mmap[local_idx : local_idx + to_take])
            current_idx += to_take

        return np.array(tokens[:length], dtype=np.uint16)

    def _global_to_shard(self, global_idx: int) -> Tuple[int, int]:
        """Convert global token index to (shard_idx, local_idx)."""
        global_idx = global_idx % self.total_tokens
        pos = 0
        for shard_idx, mmap in enumerate(self.mmaps):
            if pos + len(mmap) > global_idx:
                return shard_idx, global_idx - pos
            pos += len(mmap)
        return len(self.mmaps) - 1, len(self.mmaps[-1]) - 1
