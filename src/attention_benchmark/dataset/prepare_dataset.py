#!/usr/bin/env python3
"""
Optimized parallel dataset preparation for FineWeb-Edu.

Tokenizes FineWeb-Edu with multiprocessing (parallel CPU tokenization),
saves as uint16 .bin shards (~100M tokens each), and uploads to HuggingFace Hub.

Usage:
    python prepare_dataset.py \
      --dataset-repo your_username/fineweb-edu-10bt-gpt2-shards \
      --split sample-10BT \
      --num-workers 8

Features:
- Parallel tokenization across N CPU cores (3-5x faster, default: auto-detect)
- Batch processing to reduce Python loop overhead
- Resume-safe uploads (already-uploaded shards skip re-upload)
- Progress tracking every 1280 documents
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo


def _tokenize_doc(doc_text: str, tokenizer_name: str = "gpt2") -> List[int]:
    """Tokenize a single document. Runs in worker process."""
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    tokens = tokenizer.encode(doc_text)
    tokens.append(tokenizer.eot_token)
    return tokens


def _tokenize_batch_parallel(
    texts: List[str],
    num_workers: int,
    tokenizer_name: str = "gpt2",
) -> List[List[int]]:
    """Tokenize batch of documents in parallel using multiprocessing."""
    with Pool(processes=num_workers) as pool:
        token_lists = pool.starmap(
            _tokenize_doc,
            [(text, tokenizer_name) for text in texts],
        )
    return token_lists


def _save_shard(tokens: np.ndarray, output_dir: str, shard_idx: int, split: str) -> None:
    """Save token shard as uint16 .bin file."""
    arr = np.asarray(tokens, dtype=np.uint16)
    path = Path(output_dir) / f"shard_{shard_idx:05d}_{split}_tokens.bin"
    arr.tofile(path)
    print(f"  Saved {path.name} ({len(arr):,} tokens)")


def _upload_shards(output_dir: str, dataset_repo: str, hf_token: str) -> None:
    """Upload .bin shards to HuggingFace Hub, skipping already-uploaded files."""
    api = HfApi()

    try:
        repo_info = api.repo_info(dataset_repo, token=hf_token, repo_type="dataset")
        print(f"  Repo {dataset_repo} already exists")
        existing_files = {f.rfilename for f in repo_info.siblings}
    except Exception:
        print(f"  Creating repo {dataset_repo}...")
        create_repo(
            dataset_repo,
            repo_type="dataset",
            token=hf_token,
            private=True,
            exist_ok=True,
        )
        existing_files = set()

    shard_files = sorted(Path(output_dir).glob("shard_*.bin"))
    uploaded_count = 0
    skipped_count = 0

    for shard_file in shard_files:
        if shard_file.name in existing_files:
            print(f"  Skipping {shard_file.name} (already uploaded)")
            skipped_count += 1
            continue

        print(f"  Uploading {shard_file.name}...")
        api.upload_file(
            path_or_fileobj=str(shard_file),
            path_in_repo=shard_file.name,
            repo_id=dataset_repo,
            repo_type="dataset",
            token=hf_token,
        )
        uploaded_count += 1

    print(f"\n✅ Upload complete: {uploaded_count} new, {skipped_count} skipped")


def prepare_fineweb(
    output_dir: str = "./data",
    dataset_repo: str = None,
    split: str = "sample-10BT",
    hf_token: str = None,
    num_workers: int = None,
    batch_size: int = 128,
) -> None:
    """
    Tokenize FineWeb-Edu with parallel processing.

    Args:
        output_dir: Directory to save .bin shard files
        dataset_repo: HuggingFace dataset repo (e.g., "username/fineweb-edu-shards")
                     If provided, shards are uploaded after generation.
        split: FineWeb config (e.g., "sample-10BT", "sample-100BT")
        hf_token: HF API token (reads from HF_TOKEN env if not provided)
        num_workers: Number of CPU workers for parallel tokenization (default: auto-detect)
        batch_size: Batch size for tokenization (default: 128)
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found. Set HF_TOKEN environment variable or pass --hf-token"
            )

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading FineWeb-Edu ({split})...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=split,
        split="train",
        streaming=False,
    )

    print(f"Tokenizing {len(dataset):,} documents with {num_workers} workers...")
    shard_size = 100_000_000
    shard_count = 0
    current_shard = []
    total_tokens = 0

    texts_batch = []
    for doc_idx, doc in enumerate(dataset):
        texts_batch.append(doc["text"])

        # Tokenize when batch is full
        if len(texts_batch) >= batch_size or doc_idx == len(dataset) - 1:
            token_lists = _tokenize_batch_parallel(texts_batch, num_workers)

            for tokens in token_lists:
                current_shard.extend(tokens)
                total_tokens += len(tokens)

                # Save shard when threshold hit
                while len(current_shard) >= shard_size:
                    _save_shard(
                        np.array(current_shard[:shard_size], dtype=np.uint16),
                        output_dir,
                        shard_count,
                        "train",
                    )
                    current_shard = current_shard[shard_size:]
                    shard_count += 1

            texts_batch = []

            # Progress update every 1280 docs (128 batch * 10)
            if (doc_idx + 1) % (batch_size * 10) == 0:
                print(f"  Processed {doc_idx + 1:,} docs, {total_tokens:,} tokens")

    # Save remaining tokens
    if current_shard:
        _save_shard(
            np.array(current_shard, dtype=np.uint16),
            output_dir,
            shard_count,
            "train",
        )
        shard_count += 1

    print(f"\n✅ Saved {shard_count} shards, {total_tokens:,} tokens total")

    if dataset_repo:
        print(f"\n📤 Uploading shards to {dataset_repo}...")
        _upload_shards(output_dir, dataset_repo, hf_token)
        print(f"✅ Done! Use HF_DATASET_REPO={dataset_repo} in Kaggle training")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu dataset with parallel tokenization"
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Output directory for shards (default: ./data)",
    )
    parser.add_argument(
        "--dataset-repo",
        required=True,
        help="HF dataset repo for upload (e.g., your_username/fineweb-edu-10bt-gpt2-shards)",
    )
    parser.add_argument(
        "--split",
        default="sample-10BT",
        help="FineWeb split (default: sample-10BT)",
    )
    parser.add_argument(
        "--hf-token",
        help="HF token (default: HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of CPU workers (default: auto-detect as cpu_count() - 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for parallel tokenization (default: 128)",
    )
    args = parser.parse_args()

    try:
        prepare_fineweb(
            output_dir=args.output_dir,
            dataset_repo=args.dataset_repo,
            split=args.split,
            hf_token=args.hf_token,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Preparation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
