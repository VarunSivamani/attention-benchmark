"""
Kronecker embedding wrapper: byte-level codec input embedding with trainable projection.

Reference: https://github.com/theschoolofai/kronecker-embeddings
"""

from typing import Optional

import torch.nn as nn
from kronecker_embeddings import KroneckerEmbedding as KroneckerEmbeddingCore
from transformers import GPT2Tokenizer


def build_kronecker_embedding(
    vocab_size: int,
    d_model: int,
    tokenizer: Optional[object] = None,
    pos_dim: int = 16,
    mode: str = "dynamic",
) -> nn.Module:
    """
    Build a Kronecker embedding module (deterministic byte-level codec + trainable projection).

    Args:
        vocab_size: Vocabulary size (not directly used by Kronecker codec, but kept for API compat)
        d_model: Model embedding dimension
        tokenizer: Optional HuggingFace tokenizer. Defaults to GPT-2 tokenizer.
        pos_dim: Position dimension for Kronecker codec
        mode: Mode for Kronecker codec ("dynamic" or "static")

    Returns:
        nn.Module wrapping KroneckerEmbeddingCore for token → embedding conversion
    """
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return KroneckerEmbeddingCore(
        vocab_size=vocab_size,
        d_model=d_model,
        tokenizer=tokenizer,
        pos_dim=pos_dim,
        mode=mode,
    )
