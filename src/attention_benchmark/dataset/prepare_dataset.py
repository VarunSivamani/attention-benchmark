#!/usr/bin/env python3
"""
Optimized parallel dataset preparation for FineWeb-Edu.

Tokenizes FineWeb-Edu with multiprocessing (parallel CPU tokenization),
saves as uint16 .bin shards (~100M tokens each), and uploads to HuggingFace Hub.

Default baseline: 200k docs (2 Lakh, ~150-200M tokens, 2-3 shards) for fast CPU
iteration. Use --max-docs 0 for full split (10B tokens, ~14M docs).

Focus: sharding only — HF auth removed for now, upload handled later.

Usage:
    python -m src.attention_benchmark.dataset.prepare_dataset \
      --dataset-repo your_username/fineweb-edu-10bt-gpt2-shards \
      --split sample-10BT \
      --num-workers 8 \
      --max-docs 200000   # 2L baseline; 0 = full

Features:
- Parallel tokenization across N CPU cores (3-5x faster, default: auto-detect)
- Batch processing to reduce Python loop overhead
- Resume-safe uploads (already-uploaded shards skip re-upload)
- Progress tracking every 1280 documents
- Baseline cap: 200k docs by default; slice at load time to cut download/tokenization time
- Env fallback: --dataset-repo/--hf-token fall back to HF_DATASET_REPO/HF_TOKEN from .env via load_dotenv()
- Encoding (default: tiktoken): byte-identical GPT-2 BPE via tiktoken (cached per worker); optional --use-hf-tokenizer for HF fast (Rust, 3-5x) if needed
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


# ---------------------------------------------------------------------------
# Tokenization — tiktoken (default, byte-identical) + optional HF fast tokenizer
# ---------------------------------------------------------------------------
# tiktoken is Rust-backed and cached per worker; HF fast tokenizer
# `AutoTokenizer.from_pretrained("gpt2", use_fast=True)` is 3-5x faster
# but can produce slightly longer sequences (different BPE). Default is
# tiktoken for stable sequence lengths; use --use-hf-tokenizer to enable HF fast.
# ---------------------------------------------------------------------------
_ENCODER = None  # tiktoken.Encoding, initialized per worker
_HF_TOKENIZER = None  # transformers PreTrainedTokenizerFast
_USE_HF_FAST = False


def _init_worker(tokenizer_name: str = "gpt2", use_hf_fast: bool = False) -> None:
    """Pool initializer - cache encoder per worker process to avoid re-load per doc."""
    global _ENCODER, _HF_TOKENIZER, _USE_HF_FAST
    _USE_HF_FAST = use_hf_fast
    if use_hf_fast:
        try:
            from transformers import AutoTokenizer

            _HF_TOKENIZER = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            # Fix: default model_max_length=1024 causes spam warning "Token indices sequence length is longer than..."
            # for every >1024 tok doc (common in FineWeb). Remove clamp.
            _HF_TOKENIZER.model_max_length = int(1e9)
            try:
                _HF_TOKENIZER.deprecation_warnings = {}
            except Exception:
                pass
            _ENCODER = None
        except Exception as e:
            print(f"⚠️  HF fast tokenizer unavailable ({e}), falling back to tiktoken")
            _ENCODER = tiktoken.get_encoding(tokenizer_name)
            _USE_HF_FAST = False
    else:
        _ENCODER = tiktoken.get_encoding(tokenizer_name)
        _HF_TOKENIZER = None


def _tokenize_doc(doc_text: str, tokenizer_name: str = "gpt2") -> List[int]:  # kept for compat
    """Tokenize a single document. Runs in worker process (legacy path)."""
    # Prefer cached encoder if available (via initializer), else create
    enc = globals().get("_ENCODER")
    if enc is not None:
        tokens = enc.encode(doc_text)
        tokens.append(enc.eot_token)
        return tokens
    # fallback - should not happen when using Pool(initializer=...)
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    tokens = tokenizer.encode(doc_text)
    tokens.append(tokenizer.eot_token)
    return tokens


def _tokenize_doc_cached(doc_text: str) -> List[int]:
    """Fast path used by Pool with initializer - uses cached per-worker encoder."""
    if _USE_HF_FAST and _HF_TOKENIZER is not None:
        # HF fast tokenizer returns list[int]; eos_id is 50256 for gpt2
        ids = _HF_TOKENIZER.encode(doc_text, add_special_tokens=False)
        ids.append(_HF_TOKENIZER.eos_token_id if _HF_TOKENIZER.eos_token_id is not None else 50256)
        return ids
    # tiktoken path
    assert _ENCODER is not None, "Encoder not initialized - call _init_worker"
    tokens = _ENCODER.encode(doc_text)
    tokens.append(_ENCODER.eot_token)
    return tokens


def _tokenize_batch_parallel(
    texts: List[str],
    num_workers: int,
    tokenizer_name: str = "gpt2",
    use_hf_fast: bool = False,
) -> List[List[int]]:
    """Tokenize batch of documents in parallel using multiprocessing.

    Args:
        texts: List of document strings
        num_workers: Number of pool workers
        tokenizer_name: tiktoken encoding name (default gpt2)
        use_hf_fast: If True, use HuggingFace fast tokenizer (Rust batch) - faster
                     but verify BPE parity vs tiktoken if needed. Falls back to tiktoken.

    Faster library: HF `tokenizers` via transformers is Rust-parallel and avoids
    Python GIL; benchmark 200k docs: tiktoken+Pool ~4m vs hf-fast+Pool ~1.5m on 8 cores.
    For single-process fallback, hf-fast `batch_encode` is still faster than loop.
    """
    # Use initializer to cache encoder per worker (biggest win)
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_name, use_hf_fast),
    ) as pool:
        # Use map on cached function (no per-doc tokenizer creation)
        token_lists = pool.map(_tokenize_doc_cached, texts)
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
    max_docs: int = 200_000,
    use_hf_tokenizer: bool = False,
) -> None:
    """
    Tokenize FineWeb-Edu with parallel processing.

    Args:
        output_dir: Directory to save .bin shard files
        dataset_repo: HuggingFace dataset repo (e.g., "username/fineweb-edu-shards").
                      If None, falls back to HF_DATASET_REPO from .env / env.
                      If still None, shards are saved locally and upload is skipped.
        split: FineWeb config (e.g., "sample-10BT", "sample-100BT")
        hf_token: HF API token. If None, falls back to HF_TOKEN from .env / env
                  (load_dotenv() is called). Required only when uploading.
        num_workers: Number of CPU workers for parallel tokenization (default: auto-detect)
        batch_size: Batch size for tokenization (default: 128)
        max_docs: Max documents to tokenize (default: 200_000 = 2L baseline).
                  Set to 0 or None to use full split. Limits download/time.
        use_hf_tokenizer: If True, use HuggingFace fast tokenizer (transformers
                          AutoTokenizer, Rust - 3-5x faster but can cause longer
                          sequence lengths vs tiktoken). Default False keeps
                          tiktoken GPT-2 BPE (byte-identical, stable).

    Env fallback:
        HF_TOKEN and HF_DATASET_REPO are auto-loaded from .env via load_dotenv()
        if not passed explicitly. Precedence: explicit arg > env var > .env file.

    Encoding (default: tiktoken):
        tiktoken (Rust) cached per worker via Pool(initializer=_init_worker) is
        default for parity. Use --use-hf-tokenizer for HF fast if needed.
    """
    # Load .env for fallback (HF_TOKEN, HF_DATASET_REPO) if not passed via CLI
    # HF auth is deferred to upload phase — sharding proceeds without token
    load_dotenv(override=False)

    # Fallback: dataset_repo from .env / env if not passed
    if not dataset_repo:
        env_repo = os.environ.get("HF_DATASET_REPO")
        if env_repo and env_repo.strip():
            dataset_repo = env_repo.strip()
            print(f"ℹ️  Using HF_DATASET_REPO from .env/env: {dataset_repo}")

    # Fallback: hf_token from .env / env if not passed (deferred, not required for sharding)
    if not hf_token:
        env_token = os.environ.get("HF_TOKEN")
        if env_token and env_token.strip():
            hf_token = env_token.strip()
            print("ℹ️  Using HF_TOKEN from .env/env (deferred to upload)")

    # HF auth deferred — focus on sharding now, validate only at upload
    if dataset_repo:
        print(f"ℹ️  Sharding mode: HF auth deferred — shards will be saved locally, upload to '{dataset_repo}' handled later")
    else:
        print("ℹ️  Sharding mode: local-only (no HF_DATASET_REPO) — HF auth skipped, shards saved to ./data")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading FineWeb-Edu ({split})...")
    if max_docs and max_docs > 0:
        # Slice at load time to avoid downloading full split when possible
        split_slice = f"train[:{max_docs}]"
        print(f"  Limiting to first {max_docs:,} docs (2L baseline) -> split='{split_slice}'")
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

    print(f"Tokenizing {len(dataset):,} documents with {num_workers} workers... (single Pool reused)")
    shard_size = 100_000_000
    shard_count = 0
    current_shard = []
    total_tokens = 0

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=("gpt2", use_hf_tokenizer),
    ) as pool:
        texts_batch = []
        for doc_idx, doc in enumerate(dataset):
            texts_batch.append(doc["text"])

            # Tokenize when batch is full (use cached encoder)
            if len(texts_batch) >= batch_size or doc_idx == len(dataset) - 1:
                token_lists = pool.map(_tokenize_doc_cached, texts_batch)

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

        # Save remaining tokens (inside Pool context still ok, no more pool.map needed)
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
        if not hf_token:
            print(f"\n⚠️  Upload skipped: HF_TOKEN not set for '{dataset_repo}'.")
            print(f"   Shards remain local in {output_dir}. Set HF_TOKEN in .env or pass --hf-token to upload later.")
        else:
            print(f"\n📤 Uploading shards to {dataset_repo}...")
            _upload_shards(output_dir, dataset_repo, hf_token)
            print(f"✅ Done! Use HF_DATASET_REPO={dataset_repo} in Kaggle training")
    else:
        print(f"\nℹ️  Local-only complete. Shards in {output_dir}. Set HF_DATASET_REPO + HF_TOKEN to upload later (HF auth removed for now).")


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
        default=None,
        help="HF dataset repo for upload (e.g., your_username/fineweb-edu-10bt-gpt2-shards). "
        "If omitted, falls back to HF_DATASET_REPO from .env / env. If still not set, upload is skipped (local-only).",
    )
    parser.add_argument(
        "--split",
        default="sample-10BT",
        help="FineWeb split (default: sample-10BT)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token. If omitted, falls back to HF_TOKEN from .env / env (via load_dotenv).",
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
    parser.add_argument(
        "--max-docs",
        type=int,
        default=200_000,
        help="Max docs to tokenize (default: 200000 = 2L baseline; 0 = full split)",
    )
    parser.add_argument(
        "--use-hf-tokenizer",
        action="store_true",
        help="Use HuggingFace fast tokenizer (transformers AutoTokenizer, Rust) for faster encoding "
        "(3-5x faster but may change sequence lengths). Default: tiktoken GPT-2 BPE (byte-identical).",
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
            max_docs=args.max_docs if args.max_docs != 0 else None,
            use_hf_tokenizer=args.use_hf_tokenizer,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Preparation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
