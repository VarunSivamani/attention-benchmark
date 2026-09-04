"""
Simple DDP (Distributed Data Parallel) utilities.

Auto-detects multi-GPU setup via WORLD_SIZE env var.
No FSDP or model parallelism — data parallelism only.
"""

import os

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
    Wrap model in DDP if distributed.

    Args:
        model: Model to wrap

    Returns:
        DDP-wrapped model if distributed, else original model
    """
    if not is_distributed():
        return model

    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[get_rank()],
        output_device=get_rank(),
        find_unused_parameters=False,
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
