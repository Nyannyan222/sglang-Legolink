from __future__ import annotations

import pytest
import torch

from .reference.attention_reference import logical_causal_mask
from .reference.fixtures import scrambled_slot_fixture
from .support import load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def _visible_slots(indices: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
    return [
        indices[mask[:, query_index, 0]].tolist()
        for query_index in range(mask.shape[1])
    ]


def test_physical_slots_never_define_logical_causal_order():
    module = load_srt_module("flashinfer_sparse_repair", "flashinfer_sparse_repair.py")
    logical, physical = scrambled_slot_fixture()
    _, indices, mask = module.build_explicit_causal_masked_pattern(
        physical, repair_start=2, repair_end=4, pool_size=128
    )
    expected_mask = logical_causal_mask(torch.tensor([2, 3]), logical)

    assert _visible_slots(indices, mask) == [
        physical[expected_mask[row]].tolist() for row in range(2)
    ]
    assert _visible_slots(indices, mask) == [[100, 5, 80], [100, 5, 80, 12]]


def test_mask_preserves_visibility_order_and_multiplicity():
    module = load_srt_module("flashinfer_sparse_repair", "flashinfer_sparse_repair.py")
    physical = torch.tensor([9, 3, 9, 17, 2], dtype=torch.int32)
    _, indices, mask = module.build_explicit_causal_masked_pattern(
        physical, repair_start=3, repair_end=5, pool_size=32
    )

    assert _visible_slots(indices, mask) == [[9, 3, 9, 17], [9, 3, 9, 17, 2]]


def test_grouped_masks_keep_request_rows_isolated():
    module = load_srt_module("flashinfer_sparse_repair", "flashinfer_sparse_repair.py")
    rows = [torch.tensor([8, 2, 6, 1]), torch.tensor([15, 4, 19, 3, 7, 5])]
    indptr, indices, mask = module.build_grouped_explicit_causal_masked_pattern(
        rows, repair_ranges=[(2, 4), (4, 6)], pool_size=32
    )

    assert indptr.tolist() == [0, 4, 10]
    assert _visible_slots(indices[:4], mask[:4]) == [[8, 2, 6], [8, 2, 6, 1]]
    assert _visible_slots(indices[4:], mask[4:]) == [
        [15, 4, 19, 3, 7],
        [15, 4, 19, 3, 7, 5],
    ]

