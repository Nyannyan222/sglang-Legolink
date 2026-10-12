from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .reference.attention_reference import gqa_attention
from .reference.repair_reference import apply_rows
from .support import fixed_generator, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def _row_attention(row: torch.Tensor) -> torch.Tensor:
    key = row.view(row.shape[0], 1, row.shape[1])
    query = key.repeat(1, 2, 1)
    visible = torch.tril(torch.ones(row.shape[0], row.shape[0], dtype=torch.bool))
    return gqa_attention(query, key, key, visible).flatten(1)


def test_sequential_and_grouped_attention_preserve_row_boundaries():
    rows = [
        torch.randn(2, 4, generator=fixed_generator(1)),
        torch.randn(2, 4, generator=fixed_generator(2)),
        torch.randn(1, 4, generator=fixed_generator(3)),
    ]
    sequential = apply_rows(rows, _row_attention)
    grouped = torch.cat([_row_attention(row) for row in rows], dim=0)

    torch.testing.assert_close(grouped, sequential)


def test_grouped_attention_rowwise_projection_matches_sequential(monkeypatch):
    module = load_srt_module("sparse_repair_compute", "sparse_repair_compute.py")
    rows = [
        torch.randn(2, 8, generator=fixed_generator(4), dtype=torch.bfloat16),
        torch.randn(3, 8, generator=fixed_generator(5), dtype=torch.bfloat16),
    ]
    weight = torch.randn(8, 6, generator=fixed_generator(6), dtype=torch.bfloat16)
    metadata = {
        "batch_execution_unit_size": 2,
        "repair_ranges": [{"start": 10, "end": 12}, {"start": 20, "end": 23}],
    }
    batch = SimpleNamespace(pic_sparse_repair_metadata=metadata)
    monkeypatch.setenv("SGLANG_PIC_SPARSE_REPAIR_ROWWISE_OUTPUT_PROJECTION", "1")
    grouped = torch.cat(rows)
    slices = module.sparse_repair_rowwise_output_projection_slices(
        batch, grouped.shape[0]
    )

    sequential = torch.cat([row @ weight for row in rows])
    corrected = torch.cat([grouped[row_slice] @ weight for row_slice in slices])

    assert slices == (slice(0, 2), slice(2, 5))
    assert torch.equal(corrected, sequential)
    assert metadata["rowwise_output_projection_observed"] is True


def test_grouped_attention_rowwise_ffn_matches_sequential(monkeypatch):
    module = load_srt_module("sparse_repair_compute", "sparse_repair_compute.py")
    rows = [
        torch.randn(2, 8, generator=fixed_generator(7), dtype=torch.bfloat16),
        torch.randn(3, 8, generator=fixed_generator(8), dtype=torch.bfloat16),
    ]
    gate = torch.randn(8, 12, generator=fixed_generator(9), dtype=torch.bfloat16)
    down = torch.randn(12, 8, generator=fixed_generator(10), dtype=torch.bfloat16)

    def ffn(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(value @ gate) @ down

    metadata = {
        "batch_execution_unit_size": 2,
        "repair_ranges": [{"start": 10, "end": 12}, {"start": 20, "end": 23}],
    }
    batch = SimpleNamespace(pic_sparse_repair_metadata=metadata)
    monkeypatch.setenv("SGLANG_PIC_SPARSE_REPAIR_ROWWISE_FFN", "1")
    grouped = torch.cat(rows)
    slices = module.sparse_repair_rowwise_ffn_slices(batch, grouped.shape[0])

    sequential = torch.cat([ffn(row) for row in rows])
    corrected = torch.cat([ffn(grouped[row_slice]) for row_slice in slices])

    assert slices == (slice(0, 2), slice(2, 5))
    assert torch.equal(corrected, sequential)
    assert metadata["rowwise_ffn_observed"] is True

