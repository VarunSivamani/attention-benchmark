"""
Prepare FineWeb-Edu dataset: download, tokenize, write shards, upload to HF Hub.

Shards are stored as uint16 .bin files (~100M tokens each), uploaded to a shared
HF dataset repo to ensure both training runs use byte-identical data.

Baseline: 200k docs (2 Lakh, ~150-200M tokens) by default for fast iteration.
Pass --max-docs 0 or max_docs=None for full split.

Env fallback: --dataset-repo / --hf-token default to HF_DATASET_REPO / HF_TOKEN
from .env via load_dotenv() if not passed explicitly.

Faster encoding (default: Rust): HF fast tokenizer via transformers enabled by
default for 3-5x speedup; fallback to tiktoken. Disable with --no-use-hf-tokenizer.

Usage:
    python -m src.attention_benchmark.dataset.prepare_fineweb \
      --dataset-repo your_username/fineweb-edu-10bt-gpt2-shards \
      --split sample-10BT --max-docs 200000   # 0 = full (sharding only, no upload if token missing)
    # Or rely on .env:
    #   HF_TOKEN=hf_...  HF_DATASET_REPO=user/repo in .env
    #   python -m src.attention_benchmark.dataset.prepare_fineweb --split sample-10BT
"""

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


def prepare_fineweb(
    output_dir: str = "./data",
    dataset_repo: str = None,
    split: str = "sample-10BT",
    hf_token: str = None,
    max_docs: int = 200_000,
    use_hf_tokenizer: bool = True,
) -> None:
    """
    Tokenize FineWeb-Edu and save shards.

    Args:
        output_dir: Directory to save .bin shard files
        dataset_repo: HF dataset repo (e.g., "username/fineweb-edu-shards").
                      If None, falls back to HF_DATASET_REPO from .env / env.
                      If still None, shards saved locally and upload skipped.
        split: FineWeb config (e.g., "sample-10BT")
        hf_token: HF API token. If None, falls back to HF_TOKEN from .env / env
                  (load_dotenv() is called). Required only when uploading.
        max_docs: Max docs to tokenize (default: 200_000 = 2L baseline; 0/None = full).
        use_hf_tokenizer: If True (default), use HF fast tokenizer (transformers,
                          Rust, 3-5x faster). False uses tiktoken (byte-identical).

    Env fallback: HF_TOKEN and HF_DATASET_REPO auto-loaded from .env via load_dotenv()
                  if not passed explicitly. Precedence: explicit arg > env var > .env file.

    Faster encoding (default: Rust): HF fast tokenizer enabled by default; falls
                                     back to tiktoken if transformers not available.
    """
    # Load .env for fallback (HF_TOKEN, HF_DATASET_REPO) if not passed via CLI
    load_dotenv(override=False)

    # Fallback: dataset_repo from .env / env if not passed
    if not dataset_repo:
        env_repo = os.environ.get("HF_DATASET_REPO")
        if env_repo and env_repo.strip():
            dataset_repo = env_repo.strip()
            print(f"ℹ️  Using HF_DATASET_REPO from .env/env: {dataset_repo}")

    # Fallback: hf_token from .env / env if not passed
    if not hf_token:
        env_token = os.environ.get("HF_TOKEN")
        if env_token and env_token.strip():
            hf_token = env_token.strip()
            print("ℹ️  Using HF_TOKEN from .env/env")

    # Sharding proceeds without HF auth (auth removed for now)
    if dataset_repo:
        print(f"ℹ️  Sharding mode: shards saved locally, upload to '{dataset_repo}' later (HF auth disabled for now)")
    else:
        print("ℹ️  Sharding mode: local-only — HF auth disabled, shards saved to ./data")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Faster encoding library check: HF fast tokenizer optional
    if use_hf_tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            tokenizer_has_batch = hasattr(tokenizer, "batch_encode_plus") or hasattr(tokenizer, "__call__")
            is_hf = True
            print("🚀 Using HF fast tokenizer (transformers, Rust) for faster encoding")
        except Exception as e:
            print(f"⚠️  HF fast tokenizer unavailable ({e}), falling back to tiktoken")
            tokenizer = tiktoken.get_encoding("gpt2")
            is_hf = False
    else:
        tokenizer = tiktoken.get_encoding("gpt2")
        is_hf = False

    print(f"Loading FineWeb-Edu ({split})...")
    if max_docs and max_docs > 0:
        split_slice = f"train[:{max_docs}]"
        print(f"  Limiting to first {max_docs:,} docs (2L baseline)")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=split,
            split=split_slice,
            streaming=False,
        )
    else:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=split,
            split="train",
            streaming=False,
        )

    print(f"Tokenizing {len(dataset)} documents...")
    shard_size = 100_000_000
    shard_count = 0
    current_shard = []
    total_tokens = 0

    for doc_idx, doc in enumerate(dataset):
        if is_hf:
            # HF fast: encode without special tokens, then append eos (50256)
            tokens = tokenizer.encode(doc["text"], add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256)
        else:
            tokens = tokenizer.encode(doc["text"]) + [tokenizer.eot_token]
        current_shard.extend(tokens)
        total_tokens += len(tokens)

        if len(current_shard) >= shard_size:
            _save_shard(
                current_shard[:shard_size],
                output_dir,
                shard_count,
                "train",
            )
            current_shard = current_shard[shard_size:]
            shard_count += 1

        if (doc_idx + 1) % 10000 == 0:
            print(f"  Processed {doc_idx + 1} docs, {total_tokens:,} tokens")

    if current_shard:
        _save_shard(current_shard, output_dir, shard_count, "train")
        shard_count += 1

    print(f"Saved {shard_count} shards, {total_tokens:,} tokens total")

    if dataset_repo:
        if not hf_token:
            print(f"\n⚠️  Upload skipped: HF_TOKEN not set for '{dataset_repo}'.")
            print(f"   Shards remain local in {output_dir}. Set HF_TOKEN in .env or pass --hf-token to upload later.")
        else:
            print(f"\n📤 Uploading shards to {dataset_repo}...")
            _upload_shards(output_dir, dataset_repo, hf_token)
    else:
        print(f"\nℹ️  Local-only complete. Shards in {output_dir}. Set HF_DATASET_REPO + HF_TOKEN to upload later.")


def _save_shard(tokens: list, output_dir: str, shard_idx: int, split: str) -> None:
    """Save token shard as uint16 .bin file."""
    arr = np.array(tokens, dtype=np.uint16)
    path = Path(output_dir) / f"shard_{shard_idx:05d}_{split}_tokens.bin"
    arr.tofile(path)
    print(f"  Saved {path.name} ({len(arr):,} tokens)")


def _upload_shards(output_dir: str, dataset_repo: str, hf_token: str) -> None:
    """Upload .bin shards to HuggingFace Hub."""
    api = HfApi()

    try:
        api.repo_info(dataset_repo, token=hf_token, repo_type="dataset")
        print(f"  Repo {dataset_repo} already exists")
    except Exception:
        print(f"  Creating repo {dataset_repo}...")
        create_repo(dataset_repo, repo_type="dataset", token=hf_token, private=True)

    shard_files = sorted(Path(output_dir).glob("shard_*.bin"))
    for shard_file in shard_files:
        print(f"  Uploading {shard_file.name}...")
        api.upload_file(
            path_or_fileobj=str(shard_file),
            path_in_repo=shard_file.name,
            repo_id=dataset_repo,
            repo_type="dataset",
            token=hf_token,
        )

    print(f"Upload complete to {dataset_repo}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data", help="Output directory for shards")
    parser.add_argument(
        "--dataset-repo",
        default=None,
        help="HF dataset repo for upload (e.g., username/fineweb-edu-shards). "
        "If omitted, falls back to HF_DATASET_REPO from .env / env. If still not set, upload skipped.",
    )
    parser.add_argument("--split", default="sample-10BT", help="FineWeb split")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token (default: HF_TOKEN from .env / env via load_dotenv if not passed)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=200_000,
        help="Max docs (default: 200000 = 2L baseline; 0 = full)",
    )
    parser.add_argument(
        "--use-hf-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HF fast tokenizer (transformers, Rust) for faster encoding (default: True, 3-5x). Use --no-use-hf-tokenizer for tiktoken.",
    )
    args = parser.parse_args()

    prepare_fineweb(
        output_dir=args.output_dir,
        dataset_repo=args.dataset_repo,
        split=args.split,
        hf_token=args.hf_token,
        max_docs=args.max_docs if args.max_docs != 0 else None,
        use_hf_tokenizer=args.use_hf_tokenizer,
    )
