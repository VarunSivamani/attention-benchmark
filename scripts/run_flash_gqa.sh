#!/bin/bash
set -e

echo "=========================================="
echo "Training: Flash Attention + GQA Variant"
echo "=========================================="

CONFIG="configs/model_flash_gqa.yaml"
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
echo "Metrics saved to: runs/flash_gqa/metrics.jsonl"
echo "=========================================="
