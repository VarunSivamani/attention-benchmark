"""
Full GPT-style decoder-only model with Kronecker embeddings.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.attention_benchmark.embeddings.kronecker_embedding import build_kronecker_embedding
from src.attention_benchmark.model.block import RMSNorm, TransformerBlock


class GPTModel(nn.Module):
    """
    GPT-style decoder-only language model with Kronecker embeddings.

    Args:
        config: GPTConfig object
        tokenizer: Optional tokenizer for Kronecker embeddings (e.g., tiktoken)
    """

    def __init__(self, config, tokenizer: Optional[object] = None) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.embedding = build_kronecker_embedding(
            vocab_size=config.vocab_size,
            d_model=config.n_embd,
            tokenizer=tokenizer,
            pos_dim=config.pos_dim,
            mode="dynamic",
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self._init_weights()
        self._print_param_count()

    def _init_weights(self) -> None:
        """Initialize weights using standard GPT initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_param_count(self) -> None:
        """Print total and layer-wise parameter counts."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model initialized with {total:,} trainable parameters")
        print(f"  Embedding: {sum(p.numel() for p in self.embedding.parameters() if p.requires_grad):,}")
        print(f"  Blocks: {sum(p.numel() for p in self.blocks.parameters() if p.requires_grad):,}")
        print(f"  LM Head: {sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad):,}")
        print(f"Attention type: {self.config.attention_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass, optionally computing loss.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            targets: Target token IDs for loss computation, shape (batch, seq_len)

        Returns:
            Tuple of (logits, loss). Loss is None if targets not provided.
        """
        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs, shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering

        Returns:
            Generated token IDs including prompt, shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.block_size:
                input_ids = input_ids[:, -self.config.block_size :]

            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
