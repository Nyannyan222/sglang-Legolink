from __future__ import annotations

import pytest
import torch

from .reference.attention_reference import gqa_attention, logical_causal_mask
from .reference.repair_reference import write_repaired_kv
from .reference.rope_reference import rotate_neox
from .support import FakeRotary, fixed_generator, load_srt_module, make_pic_runtime

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.integration]


def test_small_position_independent_link_and_repair_matches_full_causal_reference():
    """Compile A/B locally, link B->A, then compare with one direct causal view."""

    generator = fixed_generator()
    raw_a_key = torch.randn(3, 1, 8, generator=generator)
    raw_b_key = torch.randn(2, 1, 8, generator=generator)
    value_a = torch.randn(3, 1, 8, generator=generator)
    value_b = torch.randn(2, 1, 8, generator=generator)
    compiled_a = rotate_neox(raw_a_key, torch.arange(3))
    compiled_b = rotate_neox(raw_b_key, torch.arange(2))

    compiled_link_order = torch.cat((compiled_b, compiled_a))
    raw_link_order = torch.cat((raw_b_key, raw_a_key))
    value_link_order = torch.cat((value_b, value_a))
    local_positions = torch.tensor([0, 1, 0, 1, 2])
    global_positions = torch.arange(5)
    physical_slots = torch.tensor([17, 3, 24, 8, 20], dtype=torch.int32)

    _, runtime = make_pic_runtime(FakeRotary(max_position=16, rotary_dim=8))
    relocated_key = runtime._remap_key_positions(
        0,
        compiled_link_order,
        local_positions,
        global_positions,
    )
    direct_key = rotate_neox(raw_link_order, global_positions)
    torch.testing.assert_close(relocated_key, direct_key, atol=2e-5, rtol=2e-5)

    key_pool = torch.zeros(32, 1, 8)
    value_pool = torch.zeros(32, 1, 8)
    key_pool[physical_slots.long()] = relocated_key
    value_pool[physical_slots.long()] = value_link_order
    sparse_module = load_srt_module(
        "flashinfer_sparse_repair", "flashinfer_sparse_repair.py"
    )
    _, indices, mask_data = sparse_module.build_explicit_causal_masked_pattern(
        physical_slots,
        repair_start=3,
        repair_end=5,
        pool_size=32,
    )
    query = torch.randn(2, 2, 8, generator=generator)

    actual_hidden = gqa_attention(
        query,
        key_pool[indices.long()],
        value_pool[indices.long()],
        mask_data[:, :, 0].T,
    )
    reference_hidden = gqa_attention(
        query,
        direct_key,
        value_link_order,
        logical_causal_mask(torch.tensor([3, 4]), global_positions),
    )
    torch.testing.assert_close(actual_hidden, reference_hidden)

    repaired_slots = physical_slots[-2:].long()
    repaired_key = direct_key[-2:] + 0.01 * actual_hidden[:, :1]
    repaired_value = value_link_order[-2:] + 0.01 * actual_hidden[:, :1]
    actual_key_pool, actual_value_pool = write_repaired_kv(
        key_pool,
        value_pool,
        repaired_slots,
        repaired_key,
        repaired_value,
    )
    expected_key_pool = key_pool.clone()
    expected_value_pool = value_pool.clone()
    expected_key_pool[repaired_slots] = repaired_key
    expected_value_pool[repaired_slots] = repaired_value
    assert torch.equal(actual_key_pool, expected_key_pool)
    assert torch.equal(actual_value_pool, expected_value_pool)

    lm_head = torch.randn(16, 9, generator=generator)
    actual_logits = actual_hidden[-1:].flatten(1) @ lm_head
    reference_logits = reference_hidden[-1:].flatten(1) @ lm_head
    torch.testing.assert_close(actual_logits, reference_logits)

