from __future__ import annotations

import torch


def scrambled_slot_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    logical_positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    physical_slots = torch.tensor([100, 5, 80, 12], dtype=torch.long)
    return logical_positions, physical_slots


def k88_failure_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """A minimal guard showing that truncating repair work can change output."""

    reference = torch.zeros(96, dtype=torch.float32)
    aggressive = reference.clone()
    aggressive[90] = 0.25
    return reference, aggressive
