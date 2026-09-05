"""
Main training entrypoint with DDP, AMP, and curriculum learning.

Usage:
    # Single GPU / CPU (local 2L baseline, HF auth disabled, shards in ./data)
    python -m src.attention_benchmark.training.train --config configs/model_flash_gqa.yaml
    python -m src.attention_benchmark.training.train --config configs/model_flash_gqa_debug.yaml  # 6×512, 1024 seq, 50M tokens

    # Multi-GPU DDP (Kaggle 2×T4, auto-detected via WORLD_SIZE/RANK):
    torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_flash_gqa.yaml
    torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_flash_gqa_debug.yaml

Features:
- DDP: auto-detected via `ddp_utils.is_distributed()` (WORLD_SIZE env from torchrun), wraps model with
  `DistributedDataParallel(..., broadcast_buffers=False)` to skip Kronecker int16 buffers (NCCL).
- AMP: `torch.amp.GradScaler` + `torch.autocast(dtype=float16)` on CUDA (T4) for 1.5-2× speed.
- Simple speed-ups: TF32 (`allow_tf32=True`, `set_float32_matmul_precision('high')`), `torch.compile(reduce-overhead)`,
  fused AdamW (`fused=True` on CUDA) — no extra infra.
- Curriculum: stages from `config.curriculum["stages"]` (e.g. [2048,4096,8192] or debug [1024]), per-stage token budgets.
- Checkpoints: per-stage `stage_{N}.pt` + per-epoch `epoch_{N}.pt` (saved after each full pass over dataset, `epochs_completed = tokens_seen / total_tokens`).
- Logging: every `log_interval_steps` (500) steps via `MetricsLogger`, metrics to `runs/*/metrics.jsonl`.

Note: HF Hub download disabled for local 2L baseline (`_maybe_download_from_hf` stub); shards expected in `./data`.
      TODO: re-enable with `snapshot_download` when scaling to full 10B.
"""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Tuple

# Suppress noisy Kaggle/T4 warnings not actionable (must be before torch import)
os.environ.setdefault("OMP_NUM_THREADS", "1")  # torchrun default W: OMP_NUM_THREADS
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")  # T4 40 SMs < A100, avoid autotune W

import torch
import torch.nn as nn
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
try:
    torch._inductor.config.max_autotune_gemm = False  # type: ignore[attr-defined]
except Exception:
    pass

from src.attention_benchmark.dataset.shard_loader import ShardLoader  # noqa: E402
from src.attention_benchmark.model.config import build_config  # noqa: E402
from src.attention_benchmark.model.gpt import GPTModel  # noqa: E402
from src.attention_benchmark.training.ddp_utils import (  # noqa: E402
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    wrap_ddp,
)
from src.attention_benchmark.training.metrics import (  # noqa: E402
    EvalMetrics,
    MetricsLogger,
    ThroughputTracker,
    TrainMetrics,
    estimate_tflops,
    get_gpu_memory_alloc_gb,
    get_gpu_memory_gb,
    get_gpu_memory_reserved_gb,
    get_grad_norm,
    get_param_norm,
    get_scaler_scale,
)


# ---------------------------------------------------------------------------
# HF Hub auth — disabled for local baseline, keep stub for later
# ---------------------------------------------------------------------------
def _maybe_download_from_hf(config) -> None:  # type: ignore[no-untyped-def]
    """No-op stub. Later: snapshot_download from HF_DATASET_REPO with HF_TOKEN.

    Example (re-enable):
        from huggingface_hub import snapshot_download
        if getattr(config, "dataset_repo", None):
            snapshot_download(
                repo_id=config.dataset_repo,
                repo_type="dataset",
                local_dir="./data",
                token=os.getenv("HF_TOKEN"),
            )
    """
    return


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler=None,
) -> float:
    """Single training step with AMP (float16 on T4)."""
    model.train()
    optimizer.zero_grad()
    use_amp = scaler is not None and str(device).startswith("cuda")
    if use_amp:
        # T4 Turing -> float16, Ampere+ can use bfloat16
        dtype = torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, loss = model(input_ids, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return loss.item()


def eval_step(
    model: nn.Module,
    data_loader: ShardLoader,
    num_batches: int,
    seq_len: int,
    batch_size: int,
    device: str,
) -> Tuple[float, float]:
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_batches):
            input_ids, targets = data_loader.get_batch(batch_size, seq_len)
            if str(device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(input_ids, targets)
            else:
                _, loss = model(input_ids, targets)
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def train(
    config,
    device: str = "cuda",
) -> None:
    """Main training loop with curriculum, AMP, and per-epoch checkpoints.

    - Curriculum: iterates `config.curriculum["stages"]` (e.g. [2048,4096,8192] or debug [1024])
    - Logging: every `config.log_interval_steps` (500) steps
    - Checkpoints: per-stage `stage_{N}.pt` + per-epoch `epoch_{N}.pt` (via `epochs_completed = tokens_seen / total_tokens`)
    - Speed-ups: TF32, torch.compile, fused AdamW, AMP GradScaler on CUDA
    """
    init_distributed()

    # Simple speed-ups beyond DDP (no extra infra)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training {config.attention_type} variant")
        print(f"{'='*60}\n")

    model = GPTModel(config).to(device)
    # torch.compile — simple 10-20% speedup on T4/A100, no code change
    if str(device).startswith("cuda"):
        try:
            if hasattr(torch, "compile"):
                model = torch.compile(model, mode="reduce-overhead")
                if is_main_process():
                    print("  torch.compile enabled (reduce-overhead)")
        except Exception as e:
            if is_main_process():
                print(f"  torch.compile skipped: {e}")
    model = wrap_ddp(model)

    # fused AdamW is ~10% faster on CUDA (if available)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            fused=str(device).startswith("cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

    scaler = torch.amp.GradScaler("cuda") if str(device).startswith("cuda") else None
    metrics_logger = MetricsLogger(config.metrics_file, config.eval_file)
    throughput = ThroughputTracker()

    # HF auth disabled for now — local shards only. See _maybe_download_from_hf() stub for later.
    # _maybe_download_from_hf(config)  # TODO: uncomment when re-enabling HF Hub
    data_loader_train = ShardLoader("./data", split="train", device=device)
    data_loader_val = ShardLoader("./data", split="val", device=device)

    # Pre-compute total trainable params for MFU / TFLOPs (6*N*tok/s)
    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        num_params = 0
    world_size = get_world_size()
    rank = get_rank()
    if is_main_process():
        print(f"  Params: {num_params:,} | world_size={world_size} | peak_flop heuristic: T4=65/A100=312 TFLOPs")

    total_tokens_seen = 0
    prev_epoch = 0
    wall_start = time.time()
    log_interval_tokens = config.log_interval_steps * config.batch_size_tokens

    # curriculum stages normalized to ints in GPTConfig.__post_init__
    for stage_idx, stage in enumerate(config.curriculum["stages"]):
        seq_len = stage["seq_len"] if isinstance(stage, dict) else int(stage)
        stage_tokens_budget = config.tokens_per_stage[stage_idx]
        tokens_in_stage = 0

        if is_main_process():
            print(f"\n[Stage {stage_idx + 1}] seq_len={seq_len}, budget={stage_tokens_budget:,} tokens")

        while tokens_in_stage < stage_tokens_budget:
            batch_size_items = max(1, config.batch_size_tokens // seq_len)

            # --- timing: data vs compute ---
            data_start = time.time()
            input_ids, targets = data_loader_train.get_batch(batch_size_items, seq_len)
            data_time_ms = (time.time() - data_start) * 1000.0

            step_start = time.time()
            loss = train_step(model, input_ids, targets, optimizer, device, scaler)
            step_time_ms = (time.time() - step_start) * 1000.0

            batch_tokens = input_ids.numel()
            tokens_in_stage += batch_tokens
            total_tokens_seen += batch_tokens

            lr = optimizer.param_groups[0]["lr"]
            grad_norm = get_grad_norm(model)
            param_norm = get_param_norm(model)
            gpu_mem = get_gpu_memory_gb()
            gpu_reserved = get_gpu_memory_reserved_gb()
            gpu_alloc = get_gpu_memory_alloc_gb()
            scaler_scale, scaler_enabled = get_scaler_scale(scaler)
            ppl = torch.exp(torch.tensor(loss)).item()
            # guard against empty dataset (total_tokens could be 0 in debug/no shards)
            try:
                epochs_completed = total_tokens_seen / data_loader_train.total_tokens if data_loader_train.total_tokens > 0 else 0.0
            except Exception:
                epochs_completed = 0.0
            stage_progress = tokens_in_stage / stage_tokens_budget if stage_tokens_budget > 0 else 0.0

            current_time = time.time()
            tokens_per_sec = throughput.update(batch_tokens, current_time)
            wall_clock = current_time - wall_start
            # throughput returns None for first step (<2 samples); still compute tflops/mfu as 0 then
            tps_for_mfu = tokens_per_sec if tokens_per_sec is not None else 0.0
            tflops, mfu = estimate_tflops(tps_for_mfu, num_params, device)

            # Log every log_interval (e.g. 500 steps) — but also ensure we log at least every stage boundary
            # Use metrics even when tokens_per_sec is None (first step) to avoid missing early loss
            should_log = (total_tokens_seen % log_interval_tokens == 0) or (tokens_in_stage >= stage_tokens_budget)
            # For live console, only print when we have a stable tps window
            if is_main_process():
                # Build metrics object every log interval (cheap — only on main)
                if should_log:
                    metrics = TrainMetrics(
                        step=total_tokens_seen // config.batch_size_tokens,
                        stage=stage_idx + 1,
                        seq_len=seq_len,
                        loss=loss,
                        ppl=ppl,
                        lr=lr,
                        grad_norm=grad_norm,
                        tokens_seen=total_tokens_seen,
                        tokens_per_sec=tps_for_mfu,
                        gpu_mem_gb=gpu_mem,
                        epochs_completed=epochs_completed,
                        wall_clock_s=wall_clock,
                        param_norm=param_norm,
                        scaler_scale=scaler_scale,
                        scaler_enabled=scaler_enabled,
                        weight_decay=config.weight_decay,
                        grad_norm_raw=grad_norm,
                        tflops=tflops,
                        mfu=mfu,
                        step_time_ms=step_time_ms,
                        data_time_ms=data_time_ms,
                        peak_tokens_per_sec=throughput.peak,
                        gpu_mem_reserved_gb=gpu_reserved,
                        gpu_mem_alloc_gb=gpu_alloc,
                        tokens_in_stage=tokens_in_stage,
                        stage_progress=stage_progress,
                        num_params=num_params,
                        batch_size_items=batch_size_items,
                        world_size=world_size,
                        rank=rank,
                    )
                    print(
                        f"  Step {metrics.step} [stage {metrics.stage} seq={metrics.seq_len}]: "
                        f"loss={loss:.4f}, ppl={ppl:.2f}, tok/s={tps_for_mfu:.0f} (peak {throughput.peak:.0f}), "
                        f"tflops={tflops:.1f}, mfu={mfu:.1%}, lr={lr:.2e}, grad={grad_norm:.2f} param={param_norm:.1f}, "
                        f"step={step_time_ms:.0f}ms data={data_time_ms:.0f}ms, "
                        f"gpu={gpu_mem:.2f}/{gpu_reserved:.2f}GB, tokens={total_tokens_seen:,}, epochs={epochs_completed:.2f}, wall={wall_clock:.0f}s"
                    )
                    metrics_logger.log_train(metrics)
                elif tokens_per_sec is not None and total_tokens_seen % (10 * config.batch_size_tokens) == 0:
                    # lightweight console heartbeat every 10 steps without JSONL spam
                    pass

            # per-epoch checkpoint — save after each full pass over dataset
            if is_main_process():
                curr_epoch_int = int(epochs_completed)
                if curr_epoch_int > prev_epoch:
                    prev_epoch = curr_epoch_int
                    epoch_ckpt = Path(config.checkpoint_dir) / f"epoch_{curr_epoch_int}.pt"
                    epoch_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state, epoch_ckpt)
                    print(f"    Saved epoch checkpoint: {epoch_ckpt} (epoch {curr_epoch_int}, tokens {total_tokens_seen:,})")

        if is_main_process():
            print(f"  [Stage {stage_idx + 1} complete] Running validation...")
            eval_start = time.time()
            val_loss, val_ppl = eval_step(
                model,
                data_loader_val,
                num_batches=10,
                seq_len=seq_len,
                batch_size=batch_size_items,
                device=device,
            )
            eval_time_s = time.time() - eval_start
            eval_metrics = EvalMetrics(
                step=total_tokens_seen // config.batch_size_tokens,
                stage=stage_idx + 1,
                seq_len=seq_len,
                val_loss=val_loss,
                val_ppl=val_ppl,
                wall_clock_s=time.time() - wall_start,
                tokens_seen=total_tokens_seen,
                epochs_completed=epochs_completed,
                gpu_mem_gb=get_gpu_memory_gb(),
                gpu_mem_reserved_gb=get_gpu_memory_reserved_gb(),
                eval_time_s=eval_time_s,
                num_batches=10,
                batch_size_items=batch_size_items,
            )
            print(
                f"    Val [stage {eval_metrics.stage} seq={eval_metrics.seq_len}] "
                f"loss: {val_loss:.4f}, ppl: {val_ppl:.2f}, eval_time={eval_time_s:.1f}s, wall={eval_metrics.wall_clock_s:.0f}s"
            )
            metrics_logger.log_eval(eval_metrics)

            ckpt_path = Path(config.checkpoint_dir) / f"stage_{stage_idx + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            # Save unwrapped state for clean reload (mirrors epoch ckpt logic)
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, ckpt_path)
            print(f"    Saved checkpoint: {ckpt_path}")

    cleanup_distributed()
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training complete! Total tokens: {total_tokens_seen:,}")
        print(f"Metrics logged to: {config.metrics_file}")
        print(f"Checkpoints saved to: {config.checkpoint_dir}")
        print(f"{'='*60}\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--tokens-per-stage", help="Comma-separated token budgets per stage")
    args = parser.parse_args()

    load_dotenv()
    config = build_config(args.config, cli_tokens_per_stage=args.tokens_per_stage)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(config, device=device)


if __name__ == "__main__":
    main()
