"""
Prepare FineWeb-Edu dataset: download, tokenize, write shards, upload to HF Hub.

Shards are stored as uint16 .bin files (~100M tokens each), uploaded to a shared
HF dataset repo to ensure both training runs use byte-identical data.

Baseline: 200k docs (2 Lakh, ~150-200M tokens) by default for fast iteration.
Pass --max-docs 0 or max_docs=None for full split.

Env fallback: --dataset-repo / --hf-token default to HF_DATASET_REPO / HF_TOKEN
from .env via load_dotenv() if not passed explicitly.

HF auth is validated via HfApi.whoami() (whoami-v2) before heavy work.

Usage:
    python -m src.attention_benchmark.dataset.prepare_fineweb \
      --dataset-repo your_username/fineweb-edu-10bt-gpt2-shards \
      --split sample-10BT --max-docs 200000   # 0 = full
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
from huggingface_hub.utils import HfHubHTTPError


def _validate_hf_token(hf_token: str, dataset_repo: str | None = None) -> dict:
    """
    Validate HF token via HuggingFace `whoami` auth endpoint.

    Uses HfApi.whoami() (GET https://huggingface.co/api/whoami-v2).
    Fails fast with clear guidance if token is missing/invalid/expired.

    Args:
        hf_token: HuggingFace access token (hf_...)
        dataset_repo: Optional repo to check read access via auth_check.

    Returns:
        whoami dict on success.

    Raises:
        ValueError with actionable message on auth failure.
    """
    token = hf_token.strip() if hf_token else ""
    if not token:
        raise ValueError(
            "HF_TOKEN is empty. Set HF_TOKEN env var or pass --hf-token. "
            "Create at https://huggingface.co/settings/tokens (role: Write)."
        )
    if not token.startswith("hf_"):
        print("⚠️  HF token does not start with 'hf_' - may be invalid or legacy.")

    api = HfApi()
    try:
        info = api.whoami(token=token)
        user = info.get("name", "unknown")
        utype = info.get("type", "user")
        print(f"✅ HF token validated for '{user}' (type: {utype}) via whoami-v2")
        if dataset_repo:
            try:
                api.auth_check(dataset_repo, repo_type="dataset", token=token)
                print(f"  Repo access check: read OK for '{dataset_repo}'")
            except HfHubHTTPError as e:
                status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                if status == 404:
                    print(f"  Repo '{dataset_repo}' not found (will be created) - token OK")
                else:
                    print(f"⚠️  Repo access check failed ({e}), but token valid")
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
                "   • Create new token at https://huggingface.co/settings/tokens (Write role)\n"
                "   Original: " + msg
            ) from e
        elif status == 403:
            raise ValueError(f"❌ HF token forbidden (403): {msg}") from e
        else:
            raise ValueError(f"❌ HF token validation failed (HTTP {status}): {msg}") from e
    except Exception as e:
        raise ValueError(f"❌ HF token validation error: {e}. Check network/token.") from e


def prepare_fineweb(
    output_dir: str = "./data",
    dataset_repo: str = None,
    split: str = "sample-10BT",
    hf_token: str = None,
    max_docs: int = 200_000,
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

    HF token is validated via HfApi.whoami() (whoami-v2) before download when needed.
    Env fallback: HF_TOKEN and HF_DATASET_REPO auto-loaded from .env via load_dotenv()
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
                "  • Create at https://huggingface.co/settings/tokens (Write role)\n"
                f"  • Target repo: {dataset_repo}"
            )
        print("🔐 Validating HF token via whoami-v2...")
        _validate_hf_token(hf_token, dataset_repo=dataset_repo)
    elif hf_token:
        print("🔐 Validating HF token via whoami-v2 (no upload repo)...")
        try:
            _validate_hf_token(hf_token, dataset_repo=None)
        except ValueError as e:
            print(f"⚠️  Token validation warning: {e}")
    else:
        print("ℹ️  No HF_DATASET_REPO / HF_TOKEN - local-only mode, skipping HF auth validation")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")

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
    args = parser.parse_args()

    prepare_fineweb(
        output_dir=args.output_dir,
        dataset_repo=args.dataset_repo,
        split=args.split,
        hf_token=args.hf_token,
        max_docs=args.max_docs if args.max_docs != 0 else None,
    )
