from __future__ import annotations

import pytest

from .support import load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_execution_plan_groups_compatible_rows_and_isolates_final_row():
    module = load_srt_module("sparse_repair_compute", "sparse_repair_compute.py")
    plan = module.plan_sparse_repair_batches([(0, 4), (10, 14), (20, 24)])

    assert module.build_sparse_repair_execution_units(plan) == ((0, 1), (2,))


def test_execution_plan_keeps_incompatible_lengths_separate():
    module = load_srt_module("sparse_repair_compute", "sparse_repair_compute.py")
    plan = module.plan_sparse_repair_batches([(0, 4), (10, 15), (20, 24)])

    assert module.build_sparse_repair_execution_units(plan) == ((0,), (1,), (2,))


def test_execution_plan_rejects_overlapping_rows():
    module = load_srt_module("sparse_repair_compute", "sparse_repair_compute.py")

    with pytest.raises(module.SparseRepairComputeError, match="ordered and disjoint"):
        module.plan_sparse_repair_batches([(0, 4), (3, 6)])


def test_all_row_mode_requires_external_compatibility_gate():
    module = load_srt_module("sparse_repair_compute", "sparse_repair_compute.py")
    plan = module.plan_sparse_repair_batches([(0, 4), (10, 14), (20, 24)])
    native_units = module.build_sparse_repair_execution_units(plan)
    all_row_units = (tuple(range(len(plan.row_to_group))),)

    assert native_units == ((0, 1), (2,))
    assert all_row_units == ((0, 1, 2),)
    assert all(group.repair_lengths[0] == 4 for group in plan.groups)


def test_flashinfer_paged_rows_split_long_ranges_by_workspace_cap():
    module = load_srt_module("pic_runtime", "pic_runtime.py")
    ranges = [(0, 152), (867, 6991)]

    split = module._split_ranges_by_max_repair_tokens(ranges, max_tokens=2048)
    units = module._build_flashinfer_paged_execution_units(split, max_tokens=2048)

    assert split == [
        (0, 152),
        (867, 2915),
        (2915, 4963),
        (4963, 6991),
    ]
    assert units == ((0,), (1,), (2,), (3,))
    assert all(
        sum(split[index][1] - split[index][0] for index in unit) <= 2048
        for unit in units
    )


def test_flashinfer_paged_rows_keep_small_ranges_grouped_under_cap():
    module = load_srt_module("pic_runtime", "pic_runtime.py")
    ranges = [(0, 128), (256, 384), (512, 640)]

    split = module._split_ranges_by_max_repair_tokens(ranges, max_tokens=512)
    units = module._build_flashinfer_paged_execution_units(split, max_tokens=512)

    assert split == ranges
    assert units == ((0, 1, 2),)
