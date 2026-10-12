from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def apply_rows(
    rows: Sequence[torch.Tensor], operation: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    return torch.cat([operation(row) for row in rows], dim=0)


def write_repaired_kv(
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    destination_slots: torch.Tensor,
    repaired_key: torch.Tensor,
    repaired_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result_key = key_buffer.clone()
    result_value = value_buffer.clone()
    result_key.index_copy_(0, destination_slots, repaired_key)
    result_value.index_copy_(0, destination_slots, repaired_value)
    return result_key, result_value


def final_row_logits(
    grouped_hidden: torch.Tensor,
    row_lengths: Sequence[int],
    lm_head: torch.Tensor,
) -> torch.Tensor:
    if sum(row_lengths) != grouped_hidden.shape[0]:
        raise ValueError("row lengths do not cover grouped hidden states")
    final_hidden = grouped_hidden[-1:]
    return final_hidden @ lm_head

