# GPT-style LM: Flash Attention vs Native Sparse Attention

Comparison study of two decoder-only language models using identical architectures and training data, differing only in their attention mechanism:

- **Variant A**: Kronecker embeddings + Flash Attention + Grouped Query Attention (GQA) — `Attention type: flash_gqa` (`src/attention_benchmark/attention/gqa_flash_attention.py:135` `sdpa_kernel [FLASH,EFFICIENT,MATH]`)
- **Variant B**: Kronecker embeddings + Native Sparse Attention (NSA) + GQA

Both trained on FineWeb-Edu with a staged context-length curriculum (2k → 4k → 8k tokens) and logged for side-by-side performance comparison.

**Model sizes:** Full `124M` (`145,571,328` params `12×768 n_head12 n_kv4 block8192` `configs/model_flash_gqa.yaml`) | Debug `50.6M` (`50,641,920` trainable: Embedding `2,097,152` + Blocks `22,812,672` + LM Head `25,731,584` `6×512 n_head8 n_kv2 block2048/8192` `configs/model_flash_gqa_debug.yaml`) — `T4 14.5GB` uses debug `AMP float16` `torch.compile`.

## Setup

### Prerequisites
- Python 3.11+
- CUDA GPU (recommended for full training; Flash variant runs on CPU/MPS for testing)
- `uv` package manager
- HuggingFace account + API token (for dataset upload)

### Installation

1. Clone and navigate to this repository:
```bash
cd /path
```

2. Create `.env` from `.env.example`:
```bash
cp .env.example .env
# Edit .env and fill in credentials and data
```

3. Install dependencies:
```bash
uv sync
```

4. For NSA variant only (GPU box), install native-sparse-attention from source:
```bash
bash scripts/run_nsa_gqa.sh --install-only
```

### Prepare Dataset

Before training, prepare FineWeb-Edu shards with parallel tokenization:

```bash
# From repo root, run the optimized dataset preparation
# Default baseline: 200k docs (2 Lakh, ~150-200M tokens, 2-3 shards) for fast CPU iteration
# HF_TOKEN / HF_DATASET_REPO fall back to .env if not passed via CLI (load_dotenv)
uv run python -m src.attention_benchmark.dataset.prepare_dataset \
  --dataset-repo your_username/fineweb-edu-10bt-gpt2-shards \
  --split sample-10BT \
  --num-workers 8 \
  --max-docs 200000   # 0 = full 10B split

# Or rely on .env (recommended):
#   echo "HF_TOKEN=hf_...\nHF_DATASET_REPO=user/repo" > .env
#   uv run python -m src.attention_benchmark.dataset.prepare_dataset --split sample-10BT

# Focus: sharding only — HF auth removed for now, upload deferred
# Features:
# - Parallel tokenization (3-5x faster) + per-worker cached tiktoken (default, byte-identical); optional --use-hf-tokenizer for HF fast (Rust, 3-5x)
# - Auto-detects CPU cores if --num-workers not specified
# - Resume-safe uploads (skips already-uploaded shards) — upload requires HF_TOKEN/HF_DATASET_REPO, sharding does not
# - Default split: sample-10BT (~10B tokens); default cap: 200k docs (2L baseline) to cut download/tokenization time
# - For full 10B run: pass --max-docs 0
# - Env fallback: --dataset-repo/--hf-token default to HF_DATASET_REPO/HF_TOKEN from .env (only needed for upload)
# - Encoding (default: tiktoken): byte-identical GPT-2 BPE; optional --use-hf-tokenizer for faster Rust encoding (may cause longer sequences)
```

Then set `HF_DATASET_REPO=your_username/fineweb-edu-10bt-gpt2-shards` in `.env` before training (or pass `--dataset-repo` explicitly; `.env` is auto-loaded).

## Quick Start

### Local Testing (no GPU required for Flash variant)
```bash
# Run tests (CPU-compatible; NSA tests skipped without CUDA)
uv run pytest tests/ -v

# Lint & type-check
ruff check --fix . && ty check .
```

### Training on GPU (Kaggle/Colab)

#### Flash Attention + GQA (2k → 4k → 8k curriculum)
```bash
bash scripts/run_flash_gqa.sh
```

#### Native Sparse Attention + GQA (requires native-sparse-attention installed)
```bash
bash scripts/run_nsa_gqa.sh
```

Both scripts:
- Read config from `configs/model_*.yaml`
- Log metrics to `runs/<variant>/metrics.jsonl`
- Save checkpoints to `runs/<variant>/ckpt_*.pt`
- Print per-stage and per-step stats to stdout

#### 2k → 4k → 8k Curriculum — Detailed Steps (Kaggle 2×T4 example)
Full 124M (`12×768`, `block_size 8192`) needs `A100`; for `T4` use debug `50M` (`6×512`, `block_size 8192`) — same curriculum, fits `14.5GB` with `AMP float16` + `torch.compile` + `SDPA flash/mem_efficient` (`src/attention_benchmark/attention/gqa_flash_attention.py:15` prefers `FLASH->EFFICIENT`, `T4` falls to `mem_efficient`).

**1. Prepare shards (5L docs ~517M tokens, 6 shards):**
```bash
uv run python -m src.attention_benchmark.dataset.prepare_dataset --split sample-10BT --max-docs 500000 --num-workers 8
# creates ./data/shard_*.bin (~191M each); split into train/val if needed:
# !mv ./data/shard_00004_train_tokens.bin ./data/shard_00000_val_tokens.bin
```

**2. Create / verify curriculum config:**
```yaml
# configs/model_flash_gqa.yaml or debug copy
model: {n_layer: 6, n_embd: 512, n_head: 8, n_kv_head: 2, block_size: 8192}
training: {batch_size_tokens: 8192}  # 8192/seq_len seqs: 2k=4, 4k=2, 8k=1
curriculum: {stages: [{seq_len: 2048}, {seq_len: 4096}, {seq_len: 8192}]}
tokens_per_stage: [300000000, 300000000, 300000000]  # 1 epoch/stage for 3 shards 300M (400M if 4 shards)
```
Debug helper:
```bash
python -c "import pathlib,yaml; p=pathlib.Path('configs/model_flash_gqa.yaml'); y=yaml.safe_load(p.read_text()); y['model'].update(n_layer=6,n_embd=512,n_head=8,n_kv_head=2,block_size=8192); y['training']['batch_size_tokens']=8192; y['curriculum']['stages']=[{'seq_len':2048},{'seq_len':4096},{'seq_len':8192}]; y['tokens_per_stage']=[300000000,300000000,300000000]; y['logging']['checkpoint_dir']='runs/flash_gqa_3ep'; pathlib.Path('configs/model_flash_gqa_3ep.yaml').write_text(yaml.dump(y))"
```

**3. Run 3-epoch curriculum (1 epoch per stage, ~7.3hr on 2×T4 `3×300M 109k steps`, `3×400M 146k steps`):**
```bash
# single-GPU
TOKENS_PER_STAGE=300000000,300000000,300000000 uv run python -m src.attention_benchmark.training.train --config configs/model_flash_gqa_3ep.yaml
# DDP 2×T4 (recommended)
TOKENS_PER_STAGE=300000000,300000000,300000000 uv run torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_flash_gqa_3ep.yaml
# tweak budget via CLI without editing yaml:
# uv run torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_flash_gqa.yaml --tokens-per-stage 300000000,300000000,300000000
```
Same for NSA: `configs/model_nsa_gqa_3ep.yaml` (`attention_type: nsa_gqa`, same `stages`).

**4. Short ablations:**
- `1 epoch 1024 single stage 300M 2.4hr`: `curriculum.stages=[{seq_len:1024}] tokens_per_stage=[300000000]`
- `Chinchilla 50M optimal 1B (~3.3 epochs) 7-8hr`: `tokens_per_stage=[1000000000]` or `[300M,300M,400M]`

Warnings `wrapt`/`OMP_NUM_THREADS`/`socket.cpp`/`broadcast_buffers`/`max_autotune_gemm` are now suppressed (`pyproject.toml:14 wrapt`, `train.py:34 OMP/TORCHINDUCTOR`, `ddp_utils.py:12 FutureWarning`); `flash` uses explicit `sdpa_kernel([FLASH,EFFICIENT])` so `T4 mem_efficient` is intentional.

### Generate Comparison Report
After both runs complete:
```bash
uv run python -m src.attention_benchmark.report.build_html_report
```
Generates `comparison_report.html` with side-by-side loss curves, throughput, perplexity, and memory usage.

## Architecture

### File Structure
```
.
├── plan.md                          # Detailed implementation plan
├── README.md                        # This file
├── QUICK_START.md                   # TL;DR guide (5 steps to training)
├── PREPARE_DATASET.md               # Dataset preparation walkthrough
├── pyproject.toml                   # uv-managed dependencies
├── .env.example                     # Environment variable template
├── .gitignore
├── configs/
│   ├── model_flash_gqa.yaml         # 124M full (145.6M with LM head 38.6M)
│   ├── model_flash_gqa_debug.yaml   # 50.6M debug (2.09M Embed + 22.8M Blocks + 25.7M LM Head)
│   └── model_nsa_gqa.yaml           # 124M, Kronecker + NSA + GQA
├── src/
│   └── attention_benchmark/
│       ├── __init__.py
│       ├── embeddings/
│       │   └── kronecker_embedding.py
│       ├── attention/
│       │   ├── gqa_flash_attention.py
│       │   └── nsa_gqa_attention.py
│       ├── model/
│       │   ├── config.py
│       │   ├── block.py
│       │   └── gpt.py
│       ├── dataset/
│       │   ├── prepare_dataset.py   # Optimized parallel tokenization script
│       │   ├── prepare_fineweb.py   # Dataset preparation function
│       │   └── shard_loader.py
│       ├── training/
│       │   ├── train.py
│       │   ├── ddp_utils.py
│       │   └── metrics.py
│       └── report/
│           └── build_html_report.py
├── scripts/
│   ├── run_flash_gqa.sh
│   └── run_nsa_gqa.sh
└── tests/
    ├── test_kronecker_embedding.py
    ├── test_gqa_flash_attention.py
    ├── test_model_param_count.py
    └── test_shard_loader.py
```

### Key Components

- **Kronecker Embeddings**: Byte-level codec input embedding (input-side only), Full `38.6M` LM Head + `3.1M` Embed vs Debug `25.7M` + `2.09M` (Blocks `103.8M`→`22.8M`).
- **Flash Attention + GQA**: PyTorch native SDPA `enable_gqa=True` `src/attention_benchmark/attention/gqa_flash_attention.py:135` with explicit `sdpa_kernel([FLASH,EFFICIENT,MATH])` — `T4` `mem_efficient` fallback, `torch 2.5+`.
- **Native Sparse Attention + GQA**: Triton-based sparse attention (requires CUDA), gated compression + sliding-window fusion.
- **Grouped Query Attention**: Single query head count / multiple KV head counts, natively expressed as different head dimensions.
- **Parallel Dataset Preparation**: Multiprocessing tokenization with `prepare_dataset.py` (3-5× faster, tiktoken default per-worker cached), baseline capped at 200k docs (2L, ~150M tokens) via `--max-docs`; slice at load time to avoid full download.
- **FineWeb-Edu Shards**: GPT-2 BPE (tiktoken byte-identical by default, optional HF fast Rust), split across uint16 `.bin` files (~100M tokens/shard), uploaded to HF Hub for reproducibility. Full `sample-10BT` = ~10B tokens (~14M docs); baseline = 200k docs (~2-3 shards). HF auth removed for now — sharding local-only, upload later.
- **Staged Curriculum**: Train at 2k context first, then 4k, then 8k — same shards reused at each stage.
- **Metrics Logging**: JSONL `src/attention_benchmark/training/metrics.py:17` `TrainMetrics` `(step, stage, seq_len, loss, ppl, lr, grad_norm, tokens_seen, tokens_per_sec, gpu_mem_gb, epochs_completed, wall_clock_s)` every `500` steps, plus `EvalMetrics`.

## Metrics & Reporting

All training runs produce:
1. **metrics.jsonl** — per-step training metrics
2. **eval.jsonl** — periodic validation metrics at stage boundaries
3. **Checkpoints** — model states after each curriculum stage

The HTML report generator combines both variants' logs into a single interactive comparison showing:
- Training loss convergence curves (by stage)
- Tokens/second throughput (wall-clock efficiency)
- Validation perplexity (per stage)
- GPU memory usage (peak and average)

## Implementation Notes

### Why Kronecker Embeddings?
Parameter efficiency: input embeddings are typically large (vocab×d_model). Kronecker's byte-level codec + single projection is ~90% smaller while maintaining performance.

### Why Staged Curriculum?
Longer context windows are compute-expensive. Starting at 2k allows quick training signal and gradient flow, then progressively increasing context (4k, 8k) lets the model adapt without early divergence.

### Why GQA in Both Variants?
Reduces KV cache size during inference (smaller n_kv_head than n_head). Both attention mechanisms natively support it, making the comparison fair — the *only* difference is how q/k/v are combined (Flash vs NSA).

### Hardware Requirements

| Variant | CPU | MPS (Mac GPU) | CUDA GPU |
|---------|-----|---------------|----------|
| Flash+GQA | ✓ (slow) | ✓ (slow) | ✓ |
| NSA+GQA | ✗ | ✗ | ✓ |

For training: CUDA GPU strongly recommended (both variants). For local testing: Flash variant works on CPU/MPS.

## Troubleshooting

### HF_TOKEN not found
Sharding (`prepare_dataset`) now runs without HF_TOKEN (HF auth removed for now, local-only). Upload (`--dataset-repo`) will require `HF_TOKEN`; set in `.env` or pass `--hf-token` when uploading. Training (`train.py`) is also local-only via `./data` (HF Hub download disabled, see `_maybe_download_from_hf` stub).

### NSA import fails
NSA is not on PyPI. Install via:
```bash
git clone https://github.com/fla-org/native-sparse-attention.git
cd native-sparse-attention
git submodule update --init --recursive
pip install .
```

Or run the install script included in `scripts/run_nsa_gqa.sh`.

### Shard cache not found
`train.py` currently uses local `./data` shards only (HF Hub download disabled for 2L baseline). For local baseline, ensure shards exist via `prepare_dataset` (default 200k docs). HF Hub download will be re-enabled later via `_maybe_download_from_hf()` in `train.py:18`.

### DDP (multi-GPU) — torchrun
`train.py` auto-detects DDP via `WORLD_SIZE`/`RANK` env vars set by `torchrun` (`ddp_utils.py:15-32`). No code change needed.

```bash
# Single GPU / CPU (local 2L baseline)
uv run python -m src.attention_benchmark.training.train --config configs/model_flash_gqa.yaml
uv run python -m src.attention_benchmark.training.train --config configs/model_flash_gqa_debug.yaml  # debug: 6×512, 1024 seq, 50M

# Multi-GPU DDP (Kaggle 2×T4) — also enables AMP + torch.compile + fused AdamW automatically
torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_flash_gqa.yaml
torchrun --nproc_per_node=2 -m src.attention_benchmark.training.train --config configs/model_flash_gqa_debug.yaml

# DDP details: wrap_ddp() uses broadcast_buffers=False (Kronecker int16 buffers would NCCL-fail), init is no-op when WORLD_SIZE=1
```

### DDP issues (troubleshooting)

## License

Apache 2.0 (follows kronecker-embeddings) + MIT (follows native-sparse-attention).

## Citation

If you use this comparison, cite:
- Kronecker Embeddings: [theschoolofai/kronecker-embeddings](https://github.com/theschoolofai/kronecker-embeddings)
- Native Sparse Attention: [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention)
- FineWeb-Edu: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
