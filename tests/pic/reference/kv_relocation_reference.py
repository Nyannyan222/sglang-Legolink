from __future__ import annotations

import torch

from .rope_reference import remap_neox


def relocate_kv(
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    local_positions: torch.Tensor,
    destination_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_key = key_buffer.clone()
    expected_value = value_buffer.clone()
    source_key = key_buffer.index_select(0, source_slots).clone()
    source_value = value_buffer.index_select(0, source_slots).clone()
    remapped = remap_neox(source_key, local_positions, destination_positions)
    expected_key.index_copy_(0, destination_slots, remapped)
    expected_value.index_copy_(0, destination_slots, source_value)
    return expected_key, expected_value

