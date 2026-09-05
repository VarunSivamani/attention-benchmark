"""
Simple DDP (Distributed Data Parallel) utilities.

Auto-detects multi-GPU via WORLD_SIZE/RANK env vars set by `torchrun` (or `torch.distributed.launch`).
No FSDP or model parallelism — data parallelism only.

Usage:
    torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_flash_gqa.yaml
    # or debug: configs/model_flash_gqa_debug.yaml

Notes:
- `wrap_ddp()` uses `broadcast_buffers=False` — Kronecker embeddings have int16 `byte_buffer`/`codebook`
  buffers that NCCL cannot broadcast (would error `ncclUnhandledCudaError`). Skipping broadcast is safe for this model.
- `init_distributed()` is a no-op when WORLD_SIZE=1 (single-GPU/CPU local baseline).
"""

import os
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn


def get_world_size() -> int:
    """Get world size from environment."""
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """Get process rank from environment."""
    return int(os.environ.get("RANK", 0))


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return get_world_size() > 1


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    """
    Initialize distributed process group.

    Args:
        backend: DDP backend ("nccl" for GPU, "gloo" for CPU)
    """
    if not is_distributed():
        return

    dist.init_process_group(backend=backend)
    torch.cuda.set_device(get_rank())
    print(f"[Rank {get_rank()}/{get_world_size()}] Initialized DDP")


def cleanup_distributed() -> None:
    """Cleanup distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module) -> nn.Module:
    """
    Wrap model in DDP if WORLD_SIZE>1 (torchrun).

    Args:
        model: Model to wrap (optionally already `torch.compile`-ed).

    Returns:
        DDP-wrapped model if distributed, else original model.
        Uses `broadcast_buffers=False` to avoid NCCL error on Kronecker int16 buffers.
    """
    if not is_distributed():
        return model

    # broadcast_buffers is deprecated (FutureWarning) but required for Kronecker
    # int16 buffers; suppress warning until init_sync_buffers is available
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_rank()],
            output_device=get_rank(),
            find_unused_parameters=False,
            broadcast_buffers=False,  # Kronecker has int16 buffers NCCL can't broadcast
        )


def get_data_loader_kwargs() -> dict:
    """Get DataLoader kwargs accounting for DDP rank/world size."""
    world_size = get_world_size()
    rank = get_rank()

    return {
        "num_replicas": world_size,
        "rank": rank,
        "shuffle": True,
    }
