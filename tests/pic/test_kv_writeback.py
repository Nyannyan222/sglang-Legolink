from __future__ import annotations

import pytest
import torch

from .reference.repair_reference import write_repaired_kv
from .support import fixed_generator

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_repaired_kv_updates_only_private_destination_slots():
    generator = fixed_generator()
    key = torch.randn(20, 2, 4, generator=generator)
    value = torch.randn(20, 2, 4, generator=generator)
    source = torch.tensor([2, 7, 2, 9])
    destination = torch.tensor([11, 15, 13, 18])
    repaired_key = torch.randn(4, 2, 4, generator=generator)
    repaired_value = torch.randn(4, 2, 4, generator=generator)
    before_key = key.clone()
    before_value = value.clone()

    actual_key = key.clone()
    actual_value = value.clone()
    actual_key.index_copy_(0, destination, repaired_key)
    actual_value.index_copy_(0, destination, repaired_value)
    expected_key, expected_value = write_repaired_kv(
        key, value, destination, repaired_key, repaired_value
    )

    assert torch.equal(actual_key, expected_key)
    assert torch.equal(actual_value, expected_value)
    assert torch.equal(actual_key[source], before_key[source])
    assert torch.equal(actual_value[source], before_value[source])
    target = set(destination.tolist())
    untouched = torch.tensor([slot for slot in range(20) if slot not in target])
    assert torch.equal(actual_key[untouched], before_key[untouched])
    assert torch.equal(actual_value[untouched], before_value[untouched])


def test_multiple_repair_rows_write_non_contiguous_slots_in_row_order():
    key = torch.zeros(12, 1, 2)
    value = torch.zeros_like(key)
    row_slots = [torch.tensor([8, 2]), torch.tensor([10]), torch.tensor([5, 7])]
    destinations = torch.cat(row_slots)
    repaired_key = torch.arange(10, dtype=torch.float32).view(5, 1, 2)
    repaired_value = repaired_key + 100

    result_key, result_value = write_repaired_kv(
        key, value, destinations, repaired_key, repaired_value
    )

    assert torch.equal(result_key[destinations], repaired_key)
    assert torch.equal(result_value[destinations], repaired_value)
    assert torch.equal(result_key[torch.tensor([0, 1, 3, 4, 6, 9, 11])], torch.zeros(7, 1, 2))

