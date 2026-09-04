"""
Prepare FineWeb-Edu dataset: download, tokenize, write shards, upload to HF Hub.

Shards are stored as uint16 .bin files (~100M tokens each), uploaded to a shared
HF dataset repo to ensure both training runs use byte-identical data.
"""

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo


def prepare_fineweb(
    output_dir: str = "./data",
    dataset_repo: str = None,
    split: str = "sample-10BT",
    hf_token: str = None,
) -> None:
    """
    Tokenize FineWeb-Edu and save shards.

    Args:
        output_dir: Directory to save .bin shard files
        dataset_repo: HuggingFace dataset repo (e.g., "username/fineweb-edu-shards")
                     If provided, shards are uploaded after generation.
        split: FineWeb config (e.g., "sample-10BT")
        hf_token: HF API token (reads from HF_TOKEN env if not provided)
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found. Set environment variable or pass hf_token="
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Loading FineWeb-Edu ({split})...")
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
        print(f"Uploading shards to {dataset_repo}...")
        _upload_shards(output_dir, dataset_repo, hf_token)


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
        help="HF dataset repo for upload (e.g., username/fineweb-edu-shards)",
    )
    parser.add_argument("--split", default="sample-10BT", help="FineWeb split")
    parser.add_argument("--hf-token", help="HF token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    prepare_fineweb(
        output_dir=args.output_dir,
        dataset_repo=args.dataset_repo,
        split=args.split,
        hf_token=args.hf_token,
    )
