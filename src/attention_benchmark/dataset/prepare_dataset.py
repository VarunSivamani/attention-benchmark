#!/usr/bin/env python3
"""
Optimized parallel dataset preparation for FineWeb-Edu.

Tokenizes FineWeb-Edu with multiprocessing (parallel CPU tokenization),
saves as uint16 .bin shards (~100M tokens each), and uploads to HuggingFace Hub.

Default baseline: 200k docs (2 Lakh, ~150-200M tokens, 2-3 shards) for fast CPU
iteration. Use --max-docs 0 for full split (10B tokens, ~14M docs).

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
- HF auth validation via whoami-v2 before download; clear errors for invalid/expired tokens
- Env fallback: --dataset-repo/--hf-token fall back to HF_DATASET_REPO/HF_TOKEN from .env via load_dotenv()
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
from huggingface_hub.utils import HfHubHTTPError


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


def _validate_hf_token(hf_token: str, dataset_repo: str | None = None) -> dict:
    """
    Validate HF token via HuggingFace `whoami` auth endpoint.

    Uses HfApi.whoami() (GET https://huggingface.co/api/whoami-v2).
    Fails fast with clear guidance if token is missing/invalid/expired.

    Args:
        hf_token: HuggingFace access token (hf_...)
        dataset_repo: Optional repo to check write access via auth_check.

    Returns:
        whoami dict on success.

    Raises:
        ValueError with actionable message on auth failure.
    """
    token = hf_token.strip() if hf_token else ""
    if not token:
        raise ValueError(
            "HF_TOKEN is empty. Set HF_TOKEN env var or pass --hf-token. "
            "Create a token at https://huggingface.co/settings/tokens (role: Write)."
        )
    if not token.startswith("hf_"):
        print("⚠️  HF token does not start with 'hf_' - may be invalid or legacy format.")

    api = HfApi()
    try:
        # Use whoami endpoint for validation - primary auth check
        info = api.whoami(token=token)
        user = info.get("name", "unknown")
        utype = info.get("type", "user")
        print(f"✅ HF token validated for '{user}' (type: {utype}) via whoami-v2")
        # Optional: check write access to target repo if provided
        if dataset_repo:
            try:
                api.auth_check(dataset_repo, repo_type="dataset", token=token)  # read check
                # If repo exists, also verify write by attempting to probe create? Light check:
                print(f"  Repo access check: read OK for '{dataset_repo}'")
            except HfHubHTTPError as e:
                status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                if status == 404:
                    # Repo does not exist yet - will be created, whoami success is enough
                    print(f"  Repo '{dataset_repo}' not found (will be created) - token OK")
                else:
                    print(f"⚠️  Repo access check failed ({e}), but token is valid - upload may still fail")
            except Exception as e:
                print(f"⚠️  Repo access probe skipped: {e}")
        return info
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None) if hasattr(e, "response") and e.response is not None else None
        msg = str(e)
        if status == 401 or "Invalid user token" in msg or "Invalid credentials" in msg:
            raise ValueError(
                "❌ HF token validation failed (401 Unauthorized - Invalid user token).\n"
                "   • Check HF_TOKEN is correct and not expired.\n"
                "   • Create a new token at https://huggingface.co/settings/tokens\n"
                "   • Token needs 'Write' role for dataset repo creation/upload.\n"
                "   Original: " + msg
            ) from e
        elif status == 403:
            raise ValueError(f"❌ HF token forbidden (403): {msg}") from e
        else:
            raise ValueError(f"❌ HF token validation failed (HTTP {status}): {msg}") from e
    except Exception as e:
        # Network or other error - warn but don't block strictly? Fail fast with guidance
        raise ValueError(f"❌ HF token validation error: {e}. Check network and token.") from e


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

    Env fallback:
        HF_TOKEN and HF_DATASET_REPO are auto-loaded from .env via load_dotenv()
        if not passed explicitly. Precedence: explicit arg > env var > .env file.
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

    # Decide if HF auth is needed (upload path)
    needs_upload = bool(dataset_repo and dataset_repo.strip())
    if needs_upload:
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found. Upload requires HF_TOKEN.\n"
                "  • Pass --hf-token, or set HF_TOKEN in .env / env.\n"
                "  • Create at https://huggingface.co/settings/tokens (Write role required).\n"
                f"  • Target repo: {dataset_repo}"
            )
        # 🔐 Validate token via HF auth endpoint before heavy download/tokenization
        print("🔐 Validating HF token via whoami-v2...")
        _validate_hf_token(hf_token, dataset_repo=dataset_repo)
    elif hf_token:
        # Token provided but no repo - still validate for correctness, non-blocking
        print("🔐 Validating HF token via whoami-v2 (no upload repo specified)...")
        try:
            _validate_hf_token(hf_token, dataset_repo=None)
        except ValueError as e:
            print(f"⚠️  Token validation warning: {e}")
    else:
        print("ℹ️  No HF_DATASET_REPO / HF_TOKEN - local-only mode, skipping HF auth validation")

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
        )
    except KeyboardInterrupt:
        print("\n⚠️  Preparation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
