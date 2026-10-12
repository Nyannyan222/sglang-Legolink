from __future__ import annotations

import torch


def rope_angles(
    positions: torch.Tensor,
    rotary_dim: int,
    *,
    theta: float = 10_000.0,
) -> torch.Tensor:
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float64)
            / rotary_dim
        )
    )
    return positions.to(torch.float64).unsqueeze(1) * frequencies.unsqueeze(0)


def rotate_neox(
    value: torch.Tensor,
    positions: torch.Tensor,
    *,
    inverse: bool = False,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """Apply an explicit 2x2 rotation to each Neox half pair."""

    rotary_dim = value.shape[-1]
    angles = rope_angles(positions, rotary_dim, theta=theta)
    if inverse:
        angles = -angles
    cos = angles.cos().to(value.dtype).unsqueeze(-2)
    sin = angles.sin().to(value.dtype).unsqueeze(-2)
    first, second = value.chunk(2, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )


def remap_neox(
    local_rope_key: torch.Tensor,
    local_positions: torch.Tensor,
    destination_positions: torch.Tensor,
    *,
    theta: float = 10_000.0,
) -> torch.Tensor:
    unrotated = rotate_neox(
        local_rope_key, local_positions, inverse=True, theta=theta
    )
    return rotate_neox(
        unrotated, destination_positions, inverse=False, theta=theta
    )

