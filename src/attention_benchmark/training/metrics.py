"""
Training metrics logging and tracking.

Per-step metrics: loss, ppl, tokens/sec, lr, grad_norm, GPU mem, epochs_completed,
plus extended metrics: param_norm, MFU/TFLOPs, step/data timing, scaler state,
memory reserved, stage progress. Logged to JSONL for later visualization.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Dataclasses — all new fields have defaults for backwards compat
# ---------------------------------------------------------------------------

@dataclass
class TrainMetrics:
    """Single training step metrics — extended suite."""

    # Core (original)
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

    # Optimizer / stability
    param_norm: float = 0.0
    scaler_scale: float = 0.0
    scaler_enabled: bool = False
    weight_decay: float = 0.0
    grad_norm_raw: float = 0.0  # alias, same as grad_norm after clip (kept for clarity)

    # Throughput / efficiency
    tflops: float = 0.0  # 6*N*tok/s / 1e12
    mfu: float = 0.0  # tflops / peak_flops (T4=65, A100=312)
    step_time_ms: float = 0.0
    data_time_ms: float = 0.0
    peak_tokens_per_sec: float = 0.0

    # Memory
    gpu_mem_reserved_gb: float = 0.0
    gpu_mem_alloc_gb: float = 0.0  # same as gpu_mem_gb (alias for clarity)

    # Curriculum / data
    tokens_in_stage: int = 0
    stage_progress: float = 0.0  # tokens_in_stage / stage_budget
    num_params: int = 0
    batch_size_items: int = 0
    world_size: int = 1
    rank: int = 0


@dataclass
class EvalMetrics:
    """Single validation step metrics — extended."""

    step: int
    stage: int
    seq_len: int
    val_loss: float
    val_ppl: float
    wall_clock_s: float

    # Extended
    tokens_seen: int = 0
    epochs_completed: float = 0.0
    gpu_mem_gb: float = 0.0
    gpu_mem_reserved_gb: float = 0.0
    eval_time_s: float = 0.0
    num_batches: int = 0
    batch_size_items: int = 0


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
        self.peak_tps = 0.0

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
        tps = total_tokens / elapsed if elapsed > 0 else 0.0
        if tps > self.peak_tps:
            self.peak_tps = tps
        return tps

    @property
    def peak(self) -> float:
        return self.peak_tps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_grad_norm(model: torch.nn.Module) -> float:
    """Compute gradient norm across all parameters (post-clip if called after clip)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def get_param_norm(model: torch.nn.Module) -> float:
    """L2 norm of all parameters (weight magnitude, for stability tracking)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm += p.data.norm(2).item() ** 2
    return total_norm ** 0.5


def get_gpu_memory_gb() -> float:
    """Peak allocated GPU memory in GB (max_memory_allocated)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def get_gpu_memory_reserved_gb() -> float:
    """Peak reserved GPU memory in GB (max_memory_reserved, includes fragmentation)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / (1024**3)
    return 0.0


def get_gpu_memory_alloc_gb() -> float:
    """Current allocated GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_scaler_scale(scaler) -> tuple[float, bool]:
    """Return (scale, enabled) for GradScaler, handling None / disabled."""
    if scaler is None:
        return 1.0, False
    try:
        return float(scaler.get_scale()), True
    except Exception:
        return 1.0, False


def estimate_tflops(tokens_per_sec: float, num_params: int, device: str = "cuda") -> tuple[float, float]:
    """
    Estimate TFLOPs and MFU.

    Uses PaLM/Chinchilla approximation: FLOPs/token ≈ 6*N (forward+backward).
    TFLOPs = 6*N * tok/s / 1e12,  MFU = TFLOPs / peak.

    Peaks (FP16/BF16): T4 ~65 TFLOPs, A100 ~312 TFLOPs, else 100 TFLOPs fallback.
    Selects T4 peak when device is cuda and torch.cuda.get_device_name contains T4.

    Returns:
        (tflops, mfu) — mfu in [0,1]
    """
    if tokens_per_sec is None or tokens_per_sec <= 0 or num_params <= 0:
        return 0.0, 0.0
    flops_per_token = 6.0 * num_params
    tflops = flops_per_token * tokens_per_sec / 1e12
    # peak heuristic
    peak = 65.0  # T4 default (this repo targets T4 debug)
    try:
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            name = torch.cuda.get_device_name(0).lower()
            if "a100" in name:
                peak = 312.0
            elif "h100" in name:
                peak = 989.0
            elif "v100" in name:
                peak = 125.0
            elif "t4" in name:
                peak = 65.0
            elif "l4" in name:
                peak = 121.0
            else:
                peak = 100.0
        elif str(device) == "cpu":
            peak = 10.0
    except Exception:
        pass
    mfu = tflops / peak if peak > 0 else 0.0
    return tflops, mfu
