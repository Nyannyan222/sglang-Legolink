from __future__ import annotations

import pytest
import torch

from .reference.kv_relocation_reference import relocate_kv
from .reference.rope_reference import rotate_neox
from .support import FakeRotary, fixed_generator, load_srt_module, make_pic_runtime

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_kv_relocation_uses_source_addressing_and_preserves_other_slots():
    generator = fixed_generator()
    key = torch.randn(16, 2, 8, generator=generator)
    value = torch.randn(16, 2, 8, generator=generator)
    source = torch.tensor([2, 7, 4, 10])
    destination = torch.tensor([11, 12, 13, 14])
    local = torch.tensor([0, 1, 2, 3])
    global_positions = torch.tensor([4, 5, 6, 7])
    key[source] = rotate_neox(key[source], local)
    original_key = key.clone()
    original_value = value.clone()

    expected_key, expected_value = relocate_kv(
        key, value, source, destination, local, global_positions
    )
    rotary = FakeRotary(max_position=16, rotary_dim=8)
    _, runtime = make_pic_runtime(rotary)
    actual_key = key.clone()
    actual_value = value.clone()
    remapped = runtime._remap_key_positions(
        0, actual_key[source].clone(), local, global_positions
    )
    actual_key.index_copy_(0, destination, remapped)
    actual_value.index_copy_(0, destination, actual_value[source].clone())

    torch.testing.assert_close(actual_key, expected_key, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_value, expected_value)
    assert torch.equal(actual_key[source], original_key[source])
    assert torch.equal(actual_value[source], original_value[source])
    untouched = torch.tensor([0, 1, 3, 5, 6, 8, 9, 15])
    assert torch.equal(actual_key[untouched], original_key[untouched])
    assert torch.equal(actual_value[untouched], original_value[untouched])


@pytest.mark.cuda
@pytest.mark.triton
@pytest.mark.integration
def test_triton_kv_relocation_matches_cpu_reference_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    triton_module = load_srt_module("legolink_triton", "legolink_triton.py")
    if not triton_module.triton_legolink_available():
        pytest.skip("Triton LegoLink backend is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(20260824)
    key = torch.randn(
        32, 2, 8, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        32, 2, 8, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    source = torch.tensor([2, 7, 4, 10], device="cuda")
    destination = torch.tensor([21, 22, 23, 24], device="cuda")
    local = torch.tensor([0, 1, 2, 3], device="cuda")
    global_positions = torch.tensor([4, 5, 6, 7], device="cuda")
    rotary = FakeRotary(max_position=16, rotary_dim=8)
    cache = rotary.cos_sin_cache.cuda()
    raw_source = key[source].clone()
    cos, sin = cache[local].chunk(2, dim=-1)
    first, second = raw_source.chunk(2, dim=-1)
    key[source] = torch.cat(
        (
            first * cos[:, None] - second * sin[:, None],
            second * cos[:, None] + first * sin[:, None],
        ),
        dim=-1,
    ).to(key.dtype)
    expected_key, expected_value = relocate_kv(
        key.cpu(),
        value.cpu(),
        source.cpu(),
        destination.cpu(),
        local.cpu(),
        global_positions.cpu(),
    )

    triton_module.fused_kv_link_inplace(
        key_buffer=key,
        value_buffer=value,
        source_slots=source,
        destination_slots=destination,
        local_positions=local,
        global_positions=global_positions,
        cos_sin_cache=cache,
        prefix_token_count=0,
        rotary_dim=8,
        is_neox_style=True,
    )

    torch.testing.assert_close(key.cpu(), expected_key, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(value.cpu(), expected_value)
