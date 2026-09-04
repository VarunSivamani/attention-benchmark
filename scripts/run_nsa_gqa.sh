#!/bin/bash
set -e

INSTALL_ONLY="${1:-}"

if [[ "$INSTALL_ONLY" == "--install-only" ]]; then
    echo "Installing native-sparse-attention from source..."
    if [[ ! -d "native-sparse-attention" ]]; then
        git clone https://github.com/fla-org/native-sparse-attention.git
    fi
    cd native-sparse-attention
    git submodule update --init --recursive
    pip install .
    cd ..
    echo "Installation complete!"
    exit 0
fi

echo "=========================================="
echo "Training: Native Sparse Attention + GQA"
echo "=========================================="
echo "Note: Ensure native-sparse-attention is installed:"
echo "  bash scripts/run_nsa_gqa.sh --install-only"
echo ""

CONFIG="configs/model_nsa_gqa.yaml"
TOKENS_PER_STAGE="${1:-}"

if [[ -n "$TOKENS_PER_STAGE" ]]; then
    echo "Using custom tokens-per-stage: $TOKENS_PER_STAGE"
    uv run python -m src.attention_benchmark.training.train \
        --config "$CONFIG" \
        --tokens-per-stage "$TOKENS_PER_STAGE"
else
    uv run python -m src.attention_benchmark.training.train --config "$CONFIG"
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "Metrics saved to: runs/nsa_gqa/metrics.jsonl"
echo "=========================================="
