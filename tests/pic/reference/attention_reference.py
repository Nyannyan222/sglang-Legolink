from __future__ import annotations

import math

import torch


def logical_causal_mask(
    query_positions: torch.Tensor, kv_positions: torch.Tensor
) -> torch.Tensor:
    return kv_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    visible: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Small explicit GQA attention reference, shaped [token, head, dim]."""

    query_tokens, query_heads, head_dim = query.shape
    kv_tokens, kv_heads, value_dim = value.shape
    if key.shape != (kv_tokens, kv_heads, head_dim):
        raise ValueError("K/V shape mismatch")
    if visible.shape != (query_tokens, kv_tokens):
        raise ValueError("mask shape mismatch")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group_size = query_heads // kv_heads
    outputs = []
    factor = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    for query_index in range(query_tokens):
        head_outputs = []
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            scores = torch.matmul(
                key[:, kv_head].to(torch.float64),
                query[query_index, query_head].to(torch.float64),
            ) * factor
            scores = scores.masked_fill(~visible[query_index], -torch.inf)
            probabilities = torch.softmax(scores, dim=0)
            head_outputs.append(
                torch.matmul(
                    probabilities,
                    value[:, kv_head].to(torch.float64),
                )
            )
        outputs.append(torch.stack(head_outputs))
    return torch.stack(outputs).to(query.dtype).reshape(
        query_tokens, query_heads, value_dim
    )

