"""
Main training entrypoint with DDP support and curriculum learning.

Usage:
    python -m src.attention_benchmark.training.train --config configs/model_flash_gqa.yaml
    torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_nsa_gqa.yaml

Note: HF Hub authentication is intentionally disabled for local 2L baseline.
      Training uses local shards from ./data via ShardLoader directly.
      TODO (later): re-enable HF Hub download with snapshot_download + HF_TOKEN
      when scaling to full 10B or multi-node. See stub _maybe_download_from_hf() below.
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from dotenv import load_dotenv

from src.attention_benchmark.dataset.shard_loader import ShardLoader
from src.attention_benchmark.model.config import build_config
from src.attention_benchmark.model.gpt import GPTModel
from src.attention_benchmark.training.ddp_utils import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    wrap_ddp,
)
from src.attention_benchmark.training.metrics import (
    EvalMetrics,
    MetricsLogger,
    ThroughputTracker,
    TrainMetrics,
    get_gpu_memory_gb,
    get_grad_norm,
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
) -> float:
    """Single training step."""
    model.train()
    optimizer.zero_grad()

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
    """Main training loop with curriculum learning."""
    init_distributed()

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training {config.attention_type} variant")
        print(f"{'='*60}\n")

    model = GPTModel(config).to(device)
    model = wrap_ddp(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    metrics_logger = MetricsLogger(config.metrics_file, config.eval_file)
    throughput = ThroughputTracker()

    # HF auth disabled for now — local shards only. See _maybe_download_from_hf() stub for later.
    # _maybe_download_from_hf(config)  # TODO: uncomment when re-enabling HF Hub
    data_loader_train = ShardLoader("./data", split="train", device=device)
    data_loader_val = ShardLoader("./data", split="val", device=device)

    total_tokens_seen = 0
    wall_start = time.time()

    # curriculum stages normalized to ints in GPTConfig.__post_init__
    for stage_idx, stage in enumerate(config.curriculum["stages"]):
        seq_len = stage["seq_len"] if isinstance(stage, dict) else int(stage)
        stage_tokens_budget = config.tokens_per_stage[stage_idx]
        tokens_in_stage = 0

        if is_main_process():
            print(f"\n[Stage {stage_idx + 1}] seq_len={seq_len}, budget={stage_tokens_budget:,} tokens")

        while tokens_in_stage < stage_tokens_budget:
            batch_size_items = max(1, config.batch_size_tokens // seq_len)

            input_ids, targets = data_loader_train.get_batch(batch_size_items, seq_len)
            loss = train_step(model, input_ids, targets, optimizer, device)

            batch_tokens = input_ids.numel()
            tokens_in_stage += batch_tokens
            total_tokens_seen += batch_tokens

            lr = optimizer.param_groups[0]["lr"]
            grad_norm = get_grad_norm(model)
            gpu_mem = get_gpu_memory_gb()
            ppl = torch.exp(torch.tensor(loss)).item()
            epochs_completed = total_tokens_seen / data_loader_train.total_tokens

            current_time = time.time()
            tokens_per_sec = throughput.update(batch_tokens, current_time)
            wall_clock = current_time - wall_start

            if is_main_process() and tokens_per_sec is not None:
                metrics = TrainMetrics(
                    step=total_tokens_seen // config.batch_size_tokens,
                    stage=stage_idx + 1,
                    seq_len=seq_len,
                    loss=loss,
                    ppl=ppl,
                    lr=lr,
                    grad_norm=grad_norm,
                    tokens_seen=total_tokens_seen,
                    tokens_per_sec=tokens_per_sec,
                    gpu_mem_gb=gpu_mem,
                    epochs_completed=epochs_completed,
                    wall_clock_s=wall_clock,
                )

                if total_tokens_seen % (config.log_interval_steps * config.batch_size_tokens) == 0:
                    print(
                        f"  Step {metrics.step}: loss={loss:.4f}, ppl={ppl:.2f}, "
                        f"tok/s={tokens_per_sec:.0f}, lr={lr:.2e}, "
                        f"grad_norm={grad_norm:.4f}, epochs={epochs_completed:.2f}"
                    )
                    metrics_logger.log_train(metrics)

        if is_main_process():
            print(f"  [Stage {stage_idx + 1} complete] Running validation...")
            val_loss, val_ppl = eval_step(
                model,
                data_loader_val,
                num_batches=10,
                seq_len=seq_len,
                batch_size=batch_size_items,
                device=device,
            )
            eval_metrics = EvalMetrics(
                step=total_tokens_seen // config.batch_size_tokens,
                stage=stage_idx + 1,
                seq_len=seq_len,
                val_loss=val_loss,
                val_ppl=val_ppl,
                wall_clock_s=time.time() - wall_start,
            )
            print(f"    Val loss: {val_loss:.4f}, Val ppl: {val_ppl:.2f}")
            metrics_logger.log_eval(eval_metrics)

            ckpt_path = Path(config.checkpoint_dir) / f"stage_{stage_idx + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
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
