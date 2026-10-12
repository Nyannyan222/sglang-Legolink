from __future__ import annotations

import pytest
import torch

from .reference.rope_reference import remap_neox, rotate_neox
from .support import FakeRotary, fixed_generator, make_pic_runtime

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("local", "destination"),
    [
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [4, 5, 6, 7]),
        ([4, 5, 6, 7], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [7, 4, 9, 5]),
        ([0, 1, 0, 1], [2, 3, 8, 9]),
    ],
)
def test_rope_position_remap_matches_explicit_reference(local, destination):
    rotary = FakeRotary(max_position=32, rotary_dim=8)
    _, runtime = make_pic_runtime(rotary)
    local_positions = torch.tensor(local)
    destination_positions = torch.tensor(destination)
    raw_key = torch.randn(4, 2, 8, generator=fixed_generator())
    local_key = rotate_neox(raw_key, local_positions)

    actual = runtime._remap_key_positions(
        0, local_key, local_positions, destination_positions
    )
    expected = remap_neox(local_key, local_positions, destination_positions)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_rope_remap_does_not_modify_value_tensor():
    rotary = FakeRotary(max_position=16, rotary_dim=8)
    _, runtime = make_pic_runtime(rotary)
    generator = fixed_generator()
    key = torch.randn(4, 2, 8, generator=generator)
    value = torch.randn(4, 2, 8, generator=generator)
    before = value.clone()

    runtime._remap_key_positions(
        0, key, torch.arange(4), torch.tensor([4, 6, 8, 10])
    )

    assert torch.equal(value, before)


def test_bf16_rope_remap_preserves_single_delta_rounding_regression():
    rotary = FakeRotary(max_position=9000, rotary_dim=8, dtype=torch.float32)
    _, runtime = make_pic_runtime(rotary)
    key = torch.zeros(1, 1, 8, dtype=torch.bfloat16)
    key[0, 0, 0] = 1.6953125
    key[0, 0, 4] = -0.53125
    local = torch.tensor([814])
    destination = torch.tensor([8621])

    actual = runtime._remap_key_positions(0, key, local, destination)
    local_cos, local_sin = rotary.cos_sin_cache[local].chunk(2, dim=-1)
    global_cos, global_sin = rotary.cos_sin_cache[destination].chunk(2, dim=-1)
    delta_cos = global_cos * local_cos + global_sin * local_sin
    delta_sin = global_sin * local_cos - global_cos * local_sin
    delta_cos = delta_cos.to(torch.bfloat16)
    delta_sin = delta_sin.to(torch.bfloat16)
    first, second = key.chunk(2, dim=-1)
    expected = torch.cat(
        (
            first * delta_cos.unsqueeze(1) - second * delta_sin.unsqueeze(1),
            second * delta_cos.unsqueeze(1) + first * delta_sin.unsqueeze(1),
        ),
        dim=-1,
    ).to(torch.bfloat16)

    assert torch.equal(actual, expected)
