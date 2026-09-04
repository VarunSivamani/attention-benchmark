"""
Model configuration with environment/CLI/YAML override resolution.

Precedence: CLI args > environment variables > YAML config > built-in defaults
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class GPTConfig:
    """GPT-style model configuration."""
    vocab_size: int = 50257
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    block_size: int = 8192
    pos_dim: int = 16
    attention_type: str = "flash_gqa"
    dropout: float = 0.1
    bias: bool = False
    norm_eps: float = 1e-6

    batch_size_tokens: int = 524288
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_tokens: int = 2_000_000_000
    gradient_accumulation_steps: int = 1

    tokens_per_stage: List[int] = None
    log_interval_steps: int = 50
    eval_interval_steps: int = 500
    save_interval_tokens: int = 1_000_000_000
    checkpoint_dir: str = "runs/flash_gqa"
    metrics_file: str = "runs/flash_gqa/metrics.jsonl"
    eval_file: str = "runs/flash_gqa/eval.jsonl"

    num_workers: int = 4
    prefetch_factor: int = 2

    nsa_block_size: int = 64
    nsa_window_size: int = 512
    nsa_block_counts: int = 16

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.n_embd // self.n_head

    def __post_init__(self) -> None:
        """Set default tokens_per_stage if not provided."""
        if self.tokens_per_stage is None:
            self.tokens_per_stage = [2_000_000_000, 3_000_000_000, 5_000_000_000]


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load YAML config file."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    return raw or {}


def resolve_tokens_per_stage(
    cli_value: Optional[str] = None,
    yaml_value: Optional[List[int]] = None,
) -> List[int]:
    """
    Resolve tokens_per_stage from CLI, env, YAML, or default.

    Precedence: CLI > env > YAML > default
    """
    default = [2_000_000_000, 3_000_000_000, 5_000_000_000]

    if cli_value:
        return [int(float(x)) for x in cli_value.split(",")]

    env_value = os.environ.get("TOKENS_PER_STAGE")
    if env_value:
        return [int(float(x)) for x in env_value.split(",")]

    if yaml_value is not None:
        return yaml_value

    return default


def build_config(
    config_path: str,
    cli_tokens_per_stage: Optional[str] = None,
) -> GPTConfig:
    """
    Build GPTConfig from YAML + env/CLI overrides.

    Args:
        config_path: Path to YAML config file
        cli_tokens_per_stage: CLI override for tokens_per_stage (comma-separated)

    Returns:
        Fully resolved GPTConfig
    """
    raw = load_config_from_yaml(config_path)

    model_cfg = raw.get("model", {})
    training_cfg = raw.get("training", {})
    logging_cfg = raw.get("logging", {})
    data_cfg = raw.get("data", {})

    tokens_per_stage = resolve_tokens_per_stage(
        cli_value=cli_tokens_per_stage,
        yaml_value=raw.get("tokens_per_stage"),
    )

    def _f(v, d):  # float-safe
        try:
            return float(v) if v is not None else d
        except Exception:
            return d

    def _i(v, d):  # int-safe
        try:
            return int(float(v)) if v is not None else d
        except Exception:
            return d

    cfg = GPTConfig(
        vocab_size=_i(model_cfg.get("vocab_size", 50257), 50257),
        n_layer=_i(model_cfg.get("n_layer", 12), 12),
        n_embd=_i(model_cfg.get("n_embd", 768), 768),
        n_head=_i(model_cfg.get("n_head", 12), 12),
        n_kv_head=_i(model_cfg.get("n_kv_head", 4), 4),
        block_size=_i(model_cfg.get("block_size", 8192), 8192),
        pos_dim=_i(model_cfg.get("pos_dim", 16), 16),
        attention_type=model_cfg.get("attention_type", "flash_gqa"),
        dropout=_f(model_cfg.get("dropout", 0.1), 0.1),
        bias=model_cfg.get("bias", False),
        norm_eps=_f(model_cfg.get("norm_eps", 1e-6), 1e-6),
        batch_size_tokens=_i(training_cfg.get("batch_size_tokens", 524288), 524288),
        learning_rate=_f(training_cfg.get("learning_rate", 6e-4), 6e-4),
        weight_decay=_f(training_cfg.get("weight_decay", 0.1), 0.1),
        max_grad_norm=_f(training_cfg.get("max_grad_norm", 1.0), 1.0),
        warmup_tokens=_i(training_cfg.get("warmup_tokens", 2_000_000_000), 2_000_000_000),
        gradient_accumulation_steps=_i(training_cfg.get("gradient_accumulation_steps", 1), 1),
        tokens_per_stage=tokens_per_stage,
        log_interval_steps=logging_cfg.get("log_interval_steps", 50),
        eval_interval_steps=logging_cfg.get("eval_interval_steps", 500),
        save_interval_tokens=logging_cfg.get("save_interval_tokens", 1_000_000_000),
        checkpoint_dir=logging_cfg.get("checkpoint_dir", "runs/flash_gqa"),
        metrics_file=logging_cfg.get("metrics_file", "runs/flash_gqa/metrics.jsonl"),
        eval_file=logging_cfg.get("eval_file", "runs/flash_gqa/eval.jsonl"),
        num_workers=data_cfg.get("num_workers", 4),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        nsa_block_size=model_cfg.get("nsa_block_size", 64),
        nsa_window_size=model_cfg.get("nsa_window_size", 512),
        nsa_block_counts=model_cfg.get("nsa_block_counts", 16),
    )
    return cfg
