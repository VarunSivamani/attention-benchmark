"""
Training metrics logging and tracking.

Per-step metrics: loss, ppl, tokens/sec, lr, grad_norm, GPU mem, epochs_completed.
Logged to JSONL files for later visualization.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class TrainMetrics:
    """Single training step metrics."""
    step: int
    stage: int
    seq_len: int
    loss: float
    ppl: float
    lr: float
    grad_norm: float
    tokens_seen: int
    tokens_per_sec: float
    gpu_mem_gb: float
    epochs_completed: float
    wall_clock_s: float


@dataclass
class EvalMetrics:
    """Single validation step metrics."""
    step: int
    stage: int
    seq_len: int
    val_loss: float
    val_ppl: float
    wall_clock_s: float


class MetricsLogger:
    """Log training metrics to JSONL file."""

    def __init__(self, metrics_file: str, eval_file: str) -> None:
        """
        Args:
            metrics_file: Path to training metrics JSONL
            eval_file: Path to eval metrics JSONL
        """
        self.metrics_file = Path(metrics_file)
        self.eval_file = Path(eval_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.eval_file.parent.mkdir(parents=True, exist_ok=True)

    def log_train(self, metrics: TrainMetrics) -> None:
        """Append training metrics to JSONL."""
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

    def log_eval(self, metrics: EvalMetrics) -> None:
        """Append eval metrics to JSONL."""
        with open(self.eval_file, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")


class ThroughputTracker:
    """Track tokens/sec over a rolling window."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.times = []
        self.token_counts = []

    def update(self, tokens: int, current_time: float) -> Optional[float]:
        """
        Update with token count at current time.

        Args:
            tokens: Number of tokens processed
            current_time: Current wall-clock time

        Returns:
            Tokens/sec over window (or None if window not full)
        """
        self.times.append(current_time)
        self.token_counts.append(tokens)

        if len(self.times) > self.window_size:
            self.times.pop(0)
            self.token_counts.pop(0)

        if len(self.times) < 2:
            return None

        elapsed = self.times[-1] - self.times[0]
        total_tokens = sum(self.token_counts)
        return total_tokens / elapsed if elapsed > 0 else None


def get_grad_norm(model: torch.nn.Module) -> float:
    """Compute gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def get_gpu_memory_gb() -> float:
    """Get peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0
