"""FlashInfer workspace capacity handling.

Regression cover for the 2026-08-27 formal run, where a paged-prefill plan
exceeded the configured FlashInfer workspace, the fail-closed
``SparseRepairAttentionError`` propagated out of ``ModelRunner.forward`` under
``SGLANG_PIC_LEGOLINK_ONLY=1``, and the SGLang scheduler process died. Every
later agent call then failed against an unreachable backend and the 30-trial
matrix was lost.

Two independent defects produced that outcome and both are covered here:

1. Repair-row splitting was gated on full row batching, while the attention
   dispatcher selects paged prefill from environment alone. A request that
   overrode row batching off still reached paged prefill with unsplit rows.
2. A capacity failure was treated like a correctness failure. Native
   sparse-repair attention computes the same logical mask, so capacity can be
   answered by degrading the operator for one request instead of failing the
   server.
"""

from __future__ import annotations

import ast
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.pic.support import SRT_ROOT
import torch

from .support import PROJECT_ROOT, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

# The exact allocator message FlashInfer produced in the failed formal run.
FLASHINFER_OVERFLOW_MESSAGE = (
    "Error in function 'aligned_alloc' at "
    "/workspace/include/flashinfer/allocator.h:49: Buffer overflow when "
    "allocating memory for batch_prefill_tmp_v with size 2921332736 and "
    "alignment 16, but only 2684354560 bytes available in AlignedAllocator. "
    "Increase the workspace buffer size."
)


def flashinfer_module():
    return load_srt_module(
        "flashinfer_sparse_repair", "flashinfer_sparse_repair.py"
    )


def pic_module():
    return load_srt_module("pic_runtime", "pic_runtime.py")


def paged_env(**overrides: str) -> dict[str, str]:
    module = flashinfer_module()
    env = {
        module.SPARSE_REPAIR_BACKEND_ENV: "flashinfer",
        module.SPARSE_REPAIR_GROUPED_ATTENTION_MODE_ENV: "paged_batched",
        module.SPARSE_REPAIR_GROUPED_ROWS_ENV: "1",
    }
    env.update(overrides)
    return env


def capacity_key(module, **overrides):
    fields = {
        "device": "cuda:0",
        "workspace_bytes": 2684354560,
        "qo_heads": 16,
        "kv_heads": 2,
        "head_dim": 128,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.bfloat16,
        "fixed_split_size": 1024,
    }
    fields.update(overrides)
    return module.paged_batch_capacity_key(**fields)


# --------------------------------------------------------------------------
# Classification: capacity is not correctness
# --------------------------------------------------------------------------


def test_flashinfer_allocator_overflow_is_classified_as_capacity():
    module = flashinfer_module()

    assert module._is_capacity_exhaustion(RuntimeError(FLASHINFER_OVERFLOW_MESSAGE))


def test_cuda_out_of_memory_is_classified_as_capacity():
    module = flashinfer_module()

    assert module._is_capacity_exhaustion(RuntimeError("CUDA out of memory."))


def test_compatibility_failure_is_not_classified_as_capacity():
    module = flashinfer_module()

    assert not module._is_capacity_exhaustion(
        RuntimeError("head_dim 96 is not supported by this kernel")
    )


def test_capacity_error_is_a_sparse_repair_error_subclass():
    module = flashinfer_module()

    # Existing handlers catch SparseRepairAttentionError; the new subclass must
    # stay inside that contract rather than escaping as an unrelated type.
    assert issubclass(
        module.SparseRepairCapacityError, module.SparseRepairAttentionError
    )


# --------------------------------------------------------------------------
# Blast radius: a too-large request degrades, it does not kill the server
# --------------------------------------------------------------------------


def dispatch_failure(module, error):
    metadata = {"requested_backend": "flashinfer"}
    batch = SimpleNamespace(pic_sparse_repair_metadata=metadata, batch_size=1)
    patch = mock.patch.object(
        module, "_dispatch_flashinfer_sparse_attention", side_effect=error
    )
    return metadata, batch, patch


def test_capacity_failure_degrades_to_native_instead_of_raising():
    module = flashinfer_module()
    metadata, batch, patch = dispatch_failure(
        module, module.SparseRepairCapacityError(FLASHINFER_OVERFLOW_MESSAGE)
    )

    with mock.patch.dict(os.environ, paged_env(), clear=False), patch:
        output = module.try_flashinfer_sparse_attention(
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
            SimpleNamespace(), batch, True,
        )

    # None routes RadixAttention to its native backend for this layer.
    assert output is None
    assert metadata["actual_backend"] == "native"
    assert metadata["capacity_degraded"] is True
    assert metadata["decision_reason"] == "flashinfer_workspace_capacity_exhausted"


def test_capacity_degradation_is_explicitly_recorded_not_silent():
    module = flashinfer_module()
    metadata, batch, patch = dispatch_failure(
        module, module.SparseRepairCapacityError(FLASHINFER_OVERFLOW_MESSAGE)
    )

    with mock.patch.dict(os.environ, paged_env(), clear=False), patch:
        module.try_flashinfer_sparse_attention(
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
            SimpleNamespace(), batch, True,
        )

    # The project's no-silent-fallback rule: the reason must survive into the
    # trace so a run can be audited after the fact.
    assert "Buffer overflow" in metadata["capacity_degraded_reason"]


def test_correctness_failure_still_fails_closed():
    module = flashinfer_module()
    metadata, batch, patch = dispatch_failure(
        module,
        module.SparseRepairAttentionError("physical slot pattern mismatch"),
    )

    with (
        mock.patch.dict(os.environ, paged_env(), clear=False),
        patch,
        pytest.raises(module.SparseRepairAttentionError, match="slot pattern"),
    ):
        module.try_flashinfer_sparse_attention(
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
            SimpleNamespace(), batch, True,
        )

    assert metadata.get("capacity_degraded") is None


def test_strict_capacity_policy_still_fails_closed():
    module = flashinfer_module()
    metadata, batch, patch = dispatch_failure(
        module, module.SparseRepairCapacityError(FLASHINFER_OVERFLOW_MESSAGE)
    )
    env = paged_env(**{module.SPARSE_REPAIR_CAPACITY_POLICY_ENV: "strict"})

    with (
        mock.patch.dict(os.environ, env, clear=False),
        patch,
        pytest.raises(module.SparseRepairCapacityError),
    ):
        module.try_flashinfer_sparse_attention(
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
            SimpleNamespace(), batch, True,
        )

    # Controlled experiments that must not change operator mid-run keep the old
    # behaviour behind an explicit policy.
    assert metadata.get("capacity_degraded") is None


def test_capacity_policy_rejects_unknown_values():
    module = flashinfer_module()
    env = {module.SPARSE_REPAIR_CAPACITY_POLICY_ENV: "sometimes"}

    with (
        mock.patch.dict(os.environ, env, clear=False),
        pytest.raises(module.SparseRepairAttentionError, match="adaptive or strict"),
    ):
        module.sparse_repair_capacity_policy()


def test_degraded_request_keeps_every_later_layer_on_native():
    module = flashinfer_module()
    metadata = {"requested_backend": "flashinfer", "actual_backend": "native"}
    batch = SimpleNamespace(pic_sparse_repair_metadata=metadata, batch_size=1)

    output = module._try_flashinfer_paged_rows(
        torch.zeros(1), torch.zeros(1), torch.zeros(1),
        SimpleNamespace(), batch, True, metadata,
    )

    # One request must never mix operators across layers.
    assert output is None


# --------------------------------------------------------------------------
# Learned ceiling: one request pays for discovery, not the whole server
# --------------------------------------------------------------------------


def test_capacity_failure_lowers_the_cap_for_later_requests():
    module = flashinfer_module()
    module.clear_paged_batch_capacity_memo()
    key = capacity_key(module)

    assert module.effective_paged_batch_q_token_cap(2048) == 2048
    module._PAGED_CAPACITY_MEMO.record_failure(key, 2048)

    assert module.effective_paged_batch_q_token_cap(2048) == 1024
    module.clear_paged_batch_capacity_memo()


def test_repeated_failures_bisect_downwards():
    module = flashinfer_module()
    module.clear_paged_batch_capacity_memo()
    key = capacity_key(module)

    assert module._PAGED_CAPACITY_MEMO.record_failure(key, 2048) == 1024
    assert module._PAGED_CAPACITY_MEMO.record_failure(key, 1024) == 512
    module.clear_paged_batch_capacity_memo()


def test_learned_ceiling_never_drops_below_a_known_good_shape():
    module = flashinfer_module()
    module.clear_paged_batch_capacity_memo()
    key = capacity_key(module)

    module._PAGED_CAPACITY_MEMO.record_success(key, 1024)
    # A transient allocation failure must not permanently cripple a shape that
    # has already been proven to plan. The degrade-to-native path is the
    # backstop for the genuinely tighter case.
    assert module._PAGED_CAPACITY_MEMO.record_failure(key, 1024) == 1024
    module.clear_paged_batch_capacity_memo()


def test_learned_ceiling_never_collapses_below_the_progress_floor():
    module = flashinfer_module()
    module.clear_paged_batch_capacity_memo()
    key = capacity_key(module)

    for _ in range(12):
        cap = module._PAGED_CAPACITY_MEMO.record_failure(key, 8)

    assert cap == module._MIN_PAGED_BATCH_Q_TOKENS
    module.clear_paged_batch_capacity_memo()


def test_effective_cap_is_the_lowest_ceiling_across_shapes():
    module = flashinfer_module()
    module.clear_paged_batch_capacity_memo()

    module._PAGED_CAPACITY_MEMO.record_failure(capacity_key(module), 2048)
    module._PAGED_CAPACITY_MEMO.record_failure(
        capacity_key(module, head_dim=64), 512
    )

    # Borrowing another shape's ceiling can only split more finely, never less,
    # so the conservative minimum is always safe.
    assert module.effective_paged_batch_q_token_cap(2048) == 256
    module.clear_paged_batch_capacity_memo()


def test_capacity_snapshot_reports_learned_state():
    module = flashinfer_module()
    module.clear_paged_batch_capacity_memo()
    module._PAGED_CAPACITY_MEMO.record_failure(capacity_key(module), 2048)

    snapshot = module.paged_batch_capacity_snapshot()

    assert snapshot["capacity_failures"] == 1
    assert list(snapshot["learned_ceilings"].values()) == [1024]
    module.clear_paged_batch_capacity_memo()


# --------------------------------------------------------------------------
# Splitting follows the dispatcher, not the row-batching gate
# --------------------------------------------------------------------------


def test_paged_path_is_reachable_from_environment_alone():
    module = flashinfer_module()

    with mock.patch.dict(os.environ, paged_env(), clear=False):
        # No row-batching flag is consulted here, which is exactly why the
        # split condition had to stop depending on one.
        assert module.paged_attention_path_reachable("flashinfer")


def test_paged_path_is_not_reachable_for_native_or_per_row():
    module = flashinfer_module()

    with mock.patch.dict(os.environ, paged_env(), clear=False):
        assert not module.paged_attention_path_reachable("native")
    with mock.patch.dict(
        os.environ,
        paged_env(**{module.SPARSE_REPAIR_GROUPED_ATTENTION_MODE_ENV: "per_row"}),
        clear=False,
    ):
        assert not module.paged_attention_path_reachable("flashinfer")


def test_long_repair_row_is_split_under_the_cap_preserving_order():
    pic = pic_module()

    # The unsplit 6,276-token repair row observed in the failed formal run.
    split = pic._split_ranges_by_max_repair_tokens([(0, 6276)], max_tokens=2048)

    assert all(end - start <= 2048 for start, end in split)
    assert split[0][0] == 0
    assert split[-1][1] == 6276
    for (_, previous_end), (next_start, _) in zip(split, split[1:]):
        assert previous_end == next_start


def test_batched_recompute_gates_the_split_on_paged_reachability():
    path = (
        SRT_ROOT
        / "pic_runtime.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_batched_recompute"
    )
    source = ast.unparse(method)

    assert "paged_attention_path_reachable" in source
    assert "effective_paged_batch_q_token_cap" in source
    # The old fixed literal cap must no longer be the only source of the bound.
    assert "'SGLANG_PIC_FLASHINFER_MAX_PAGED_BATCH_Q_TOKENS', 2048" not in source


# --------------------------------------------------------------------------
# Degradation is per request, not per execution unit
# --------------------------------------------------------------------------


def test_batched_recompute_carries_degradation_across_execution_units():
    path = (
        SRT_ROOT
        / "pic_runtime.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_batched_recompute"
    )
    source = ast.unparse(method)

    # A real RTX 5070 run against a 16 MiB workspace showed one request
    # alternating native and FlashInfer between execution units: the
    # actual_backend guard only kept layers of a single unit consistent.
    # Once any unit degrades the whole request must finish on native.
    assert "_request_capacity_degraded" in source
    assert "request_already_capacity_degraded" in source


def test_run_link_resets_the_per_request_degradation_flag():
    path = (
        SRT_ROOT
        / "pic_runtime.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_link"
    )
    source = ast.unparse(method)

    # Without a reset, one degraded request would pin every later request on
    # native for the life of the server.
    assert "self._request_capacity_degraded = False" in source
