from __future__ import annotations

import ast
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.pic.support import SRT_ROOT
import torch

from .reference.fixtures import k88_failure_fixture
from .support import PROJECT_ROOT, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_k88_is_not_unconditionally_promotable():
    reference, aggressive_k88 = k88_failure_fixture()

    assert not torch.allclose(reference, aggressive_k88, atol=0.01, rtol=0.01)


def test_pic_sparse_repair_disables_incompatible_cuda_graph():
    path = (
        SRT_ROOT
        / "model_executor"
        / "piecewise_cuda_graph_runner.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    can_run = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "can_run"
    )
    source = ast.unparse(can_run)

    assert "pic_sparse_repair_metadata" in source
    assert "return False" in source


def test_forced_flashinfer_backend_fails_closed_without_backend():
    module = load_srt_module("flashinfer_sparse_repair", "flashinfer_sparse_repair.py")
    batch = SimpleNamespace(input_ids=torch.tensor([1]))

    with (
        mock.patch.dict(
            os.environ, {module.SPARSE_REPAIR_BACKEND_ENV: "flashinfer"}, clear=False
        ),
        mock.patch.object(module, "_flashinfer_available", return_value=(False, "missing")),
        pytest.raises(module.SparseRepairAttentionError, match="missing"),
    ):
        module.make_sparse_repair_metadata(
            batch, start=0, end=1, block_count=1, block_size=1
        )


def test_auto_fallback_is_explicitly_recorded():
    module = load_srt_module("flashinfer_sparse_repair", "flashinfer_sparse_repair.py")
    batch = SimpleNamespace(input_ids=torch.tensor([1]))

    with (
        mock.patch.dict(
            os.environ, {module.SPARSE_REPAIR_BACKEND_ENV: "auto"}, clear=False
        ),
        mock.patch.object(module, "_flashinfer_available", return_value=(False, "missing")),
    ):
        metadata = module.make_sparse_repair_metadata(
            batch, start=0, end=1, block_count=1, block_size=1
        )

    assert metadata["actual_backend"] == "native"
    assert metadata["fallback_reason"] == "missing"
    assert metadata["compatibility_status"] == "precheck_failed"

