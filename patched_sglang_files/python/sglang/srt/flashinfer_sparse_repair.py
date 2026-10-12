from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.metadata
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from typing import Any

import torch

SPARSE_REPAIR_BACKEND_ENV = "SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND"
SPARSE_REPAIR_EVENTS_ENV = "SGLANG_PIC_SPARSE_REPAIR_COMPONENT_EVENTS"
SPARSE_REPAIR_WORKSPACE_ENV = "SGLANG_PIC_FLASHINFER_WORKSPACE_BYTES"
SPARSE_REPAIR_PAGED_WRAPPER_WORKSPACE_ENV = (
    "SGLANG_PIC_FLASHINFER_PAGED_WRAPPER_WORKSPACE_BYTES"
)
SPARSE_REPAIR_PLAN_CACHE_ENTRIES_ENV = (
    "SGLANG_PIC_FLASHINFER_PLAN_CACHE_ENTRIES"
)
SPARSE_REPAIR_PLAN_CACHE_BYTES_ENV = "SGLANG_PIC_FLASHINFER_PLAN_CACHE_BYTES"
SPARSE_REPAIR_METADATA_CACHE_ENTRIES_ENV = (
    "SGLANG_PIC_FLASHINFER_METADATA_CACHE_ENTRIES"
)
SPARSE_REPAIR_METADATA_CACHE_BYTES_ENV = (
    "SGLANG_PIC_FLASHINFER_METADATA_CACHE_BYTES"
)
SPARSE_REPAIR_GROUPED_ROWS_ENV = (
    "SGLANG_PIC_ALLOW_FLASHINFER_ROW_BATCHING_EXPERIMENT"
)
SPARSE_REPAIR_GROUPED_ATTENTION_MODE_ENV = (
    "SGLANG_PIC_FLASHINFER_GROUPED_ATTENTION_MODE"
)
SPARSE_REPAIR_FIXED_SPLIT_SIZE_ENV = "SGLANG_PIC_FLASHINFER_FIXED_SPLIT_SIZE"
SPARSE_REPAIR_REUSE_PAGED_WRAPPER_ENV = (
    "SGLANG_PIC_FLASHINFER_REUSE_PAGED_WRAPPER_ACROSS_PLANS"
)
SPARSE_REPAIR_SKIP_EXACT_PAGED_METADATA_CACHE_ENV = (
    "SGLANG_PIC_FLASHINFER_SKIP_EXACT_PAGED_METADATA_CACHE"
)
SPARSE_REPAIR_PROFILE_RANGES_ENV = "SGLANG_PIC_PROFILE_RANGES_ENABLED"
SPARSE_REPAIR_CAPACITY_POLICY_ENV = "SGLANG_PIC_FLASHINFER_CAPACITY_POLICY"
SPARSE_REPAIR_MAX_PAGED_BATCH_Q_TOKENS_ENV = (
    "SGLANG_PIC_FLASHINFER_MAX_PAGED_BATCH_Q_TOKENS"
)

_DEFAULT_PLAN_CACHE_ENTRIES = 8
_DEFAULT_PLAN_CACHE_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_METADATA_CACHE_ENTRIES = 16
_DEFAULT_METADATA_CACHE_BYTES = 512 * 1024 * 1024
_DEFAULT_PAGED_WRAPPER_WORKSPACE_BYTES = 1024 * 1024 * 1024
_DEFAULT_MAX_PAGED_BATCH_Q_TOKENS = 2048
# A learned ceiling is never allowed to collapse to a value that cannot make
# progress; below this the request degrades to native attention instead.
_MIN_PAGED_BATCH_Q_TOKENS = 64


class SparseRepairAttentionError(RuntimeError):
    pass


class SparseRepairCapacityError(SparseRepairAttentionError):
    """FlashInfer could not fit this shape in its workspace.

    This is deliberately distinct from a compatibility or correctness failure.
    Native sparse-repair attention is the reference implementation for the same
    logical mask, so a capacity failure can be answered by degrading the
    operator for this request without weakening any LegoLink claim. Compare
    :class:`SparseRepairAttentionError`, which means the request must not run.
    """


_CAPACITY_ERROR_MARKERS = (
    "buffer overflow when allocating",
    "aligned_alloc",
    "alignedallocator",
    "out of memory",
    "workspace buffer size",
)


def _is_capacity_exhaustion(exc: BaseException) -> bool:
    """Recognise workspace/allocator exhaustion from FlashInfer or CUDA.

    FlashInfer raises a plain ``RuntimeError`` from its C++ ``AlignedAllocator``
    rather than a typed exception, so the message is the only reliable signal.
    ``torch.cuda.OutOfMemoryError`` is matched by type.
    """

    if isinstance(exc, SparseRepairCapacityError):
        return True
    out_of_memory = getattr(torch.cuda, "OutOfMemoryError", None)
    if out_of_memory is not None and isinstance(exc, out_of_memory):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _CAPACITY_ERROR_MARKERS)


def sparse_repair_capacity_policy() -> str:
    """``adaptive`` degrades a too-large request; ``strict`` fails it closed."""

    value = os.getenv(SPARSE_REPAIR_CAPACITY_POLICY_ENV, "adaptive").strip().lower()
    if value not in {"adaptive", "strict"}:
        raise SparseRepairAttentionError(
            f"{SPARSE_REPAIR_CAPACITY_POLICY_ENV} must be adaptive or strict; "
            f"got {value!r}."
        )
    return value


class PagedBatchCapacityMemo:
    """Learned per-shape ceiling on FlashInfer paged-prefill query tokens.

    FlashInfer sizes ``batch_prefill_tmp_v`` from a split-k decomposition whose
    exact formula is version and backend dependent, so this does not attempt to
    predict the requirement. It bisects instead: a plan failure at ``q_tokens``
    lowers the ceiling below that shape, and a success raises the known-good
    floor. Requests after the first failure pre-split under the learned ceiling,
    so one request pays for the discovery rather than the whole server.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ceiling: dict[tuple[Any, ...], int] = {}
        self._floor: dict[tuple[Any, ...], int] = {}
        self._failures: dict[tuple[Any, ...], int] = {}

    def cap_for(self, key: tuple[Any, ...], configured_cap: int) -> int:
        with self._lock:
            learned = self._ceiling.get(key)
        if learned is None:
            return configured_cap
        if configured_cap <= 0:
            return learned
        return min(configured_cap, learned)

    def record_failure(self, key: tuple[Any, ...], q_tokens: int) -> int:
        """Halve the ceiling below the shape that failed; return the new cap."""

        with self._lock:
            self._failures[key] = self._failures.get(key, 0) + 1
            candidate = max(int(q_tokens) // 2, _MIN_PAGED_BATCH_Q_TOKENS)
            floor = self._floor.get(key)
            if floor is not None:
                # Never drop below a shape already proven to plan successfully.
                candidate = max(candidate, floor)
            previous = self._ceiling.get(key)
            self._ceiling[key] = (
                candidate if previous is None else min(previous, candidate)
            )
            return self._ceiling[key]

    def record_success(self, key: tuple[Any, ...], q_tokens: int) -> None:
        with self._lock:
            previous = self._floor.get(key)
            if previous is None or q_tokens > previous:
                self._floor[key] = int(q_tokens)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "learned_ceilings": {
                    "|".join(str(part) for part in key): value
                    for key, value in self._ceiling.items()
                },
                "known_good_floors": {
                    "|".join(str(part) for part in key): value
                    for key, value in self._floor.items()
                },
                "capacity_failures": sum(self._failures.values()),
            }

    def clear(self) -> None:
        with self._lock:
            self._ceiling.clear()
            self._floor.clear()
            self._failures.clear()


_PAGED_CAPACITY_MEMO = PagedBatchCapacityMemo()


def paged_batch_capacity_key(
    *,
    device: Any,
    workspace_bytes: int,
    qo_heads: int,
    kv_heads: int,
    head_dim: int,
    q_dtype: Any,
    kv_dtype: Any,
    fixed_split_size: int,
) -> tuple[Any, ...]:
    """Shape class a learned capacity ceiling is valid for."""

    return (
        str(device),
        int(workspace_bytes),
        int(qo_heads),
        int(kv_heads),
        int(head_dim),
        str(q_dtype),
        str(kv_dtype),
        int(fixed_split_size),
    )


def configured_paged_batch_q_token_cap() -> int:
    return _positive_int_env(
        SPARSE_REPAIR_MAX_PAGED_BATCH_Q_TOKENS_ENV,
        _DEFAULT_MAX_PAGED_BATCH_Q_TOKENS,
    )


def resolve_paged_batch_q_token_cap(
    *,
    device: Any,
    workspace_bytes: int,
    qo_heads: int,
    kv_heads: int,
    head_dim: int,
    q_dtype: Any,
    kv_dtype: Any,
    fixed_split_size: int,
    configured_cap: int | None = None,
) -> int:
    """Configured paged-batch cap, lowered by anything learned for this shape."""

    key = paged_batch_capacity_key(
        device=device,
        workspace_bytes=workspace_bytes,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        fixed_split_size=fixed_split_size,
    )
    if configured_cap is None:
        configured_cap = configured_paged_batch_q_token_cap()
    return _PAGED_CAPACITY_MEMO.cap_for(key, configured_cap)


def _paged_capacity_key_for(
    layer: Any,
    q: torch.Tensor,
    key_buffer: torch.Tensor,
    workspace_bytes: int,
    fixed_split_size: int,
) -> tuple[Any, ...]:
    return paged_batch_capacity_key(
        device=q.device,
        workspace_bytes=workspace_bytes,
        qo_heads=int(layer.tp_q_head_num),
        kv_heads=int(layer.tp_k_head_num),
        head_dim=int(layer.head_dim),
        q_dtype=q.dtype,
        kv_dtype=key_buffer.dtype,
        fixed_split_size=fixed_split_size,
    )


def effective_paged_batch_q_token_cap(configured_cap: int | None = None) -> int:
    """Configured cap lowered by the smallest ceiling learned on this server.

    Callers that plan repair rows do not yet hold an attention layer, so they
    cannot build an exact shape key. A server serves one model on one device in
    every configuration this project runs, so the minimum learned ceiling is the
    right conservative answer; borrowing another shape's ceiling can only split
    more finely, never less, and never changes the computed result.
    """

    if configured_cap is None:
        configured_cap = configured_paged_batch_q_token_cap()
    learned = _PAGED_CAPACITY_MEMO.snapshot()["learned_ceilings"]
    if not learned:
        return configured_cap
    lowest = min(learned.values())
    if configured_cap <= 0:
        return lowest
    return min(configured_cap, lowest)


def paged_attention_path_reachable(requested_backend: str) -> bool:
    """Whether this configuration can route repair rows to paged prefill.

    The dispatcher selects the paged path from environment alone, so repair-row
    splitting must follow the same condition rather than the narrower
    full-row-batching gate. Otherwise a request that turns row batching off
    still reaches paged prefill, but with unsplit rows.
    """

    if requested_backend != "flashinfer":
        return False
    grouped_mode = os.getenv(
        SPARSE_REPAIR_GROUPED_ATTENTION_MODE_ENV, "per_row"
    ).strip().lower()
    if grouped_mode != "paged_batched":
        return False
    return os.getenv(SPARSE_REPAIR_GROUPED_ROWS_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def paged_batch_capacity_snapshot() -> dict[str, Any]:
    return _PAGED_CAPACITY_MEMO.snapshot()


def clear_paged_batch_capacity_memo() -> None:
    _PAGED_CAPACITY_MEMO.clear()


def requested_sparse_repair_backend() -> str:
    value = os.getenv(SPARSE_REPAIR_BACKEND_ENV, "native").strip().lower()
    if value not in {"native", "flashinfer", "auto", "hybrid"}:
        raise SparseRepairAttentionError(
            f"{SPARSE_REPAIR_BACKEND_ENV} must be native, flashinfer, auto, "
            "or hybrid; "
            f"got {value!r}."
        )
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise SparseRepairAttentionError(f"{name} must be an integer; got {raw!r}") from exc
    if value < 0:
        raise SparseRepairAttentionError(f"{name} must be non-negative; got {value}")
    return value


def _paged_wrapper_workspace_bytes(default_workspace_bytes: int) -> int:
    return _positive_int_env(
        SPARSE_REPAIR_PAGED_WRAPPER_WORKSPACE_ENV,
        min(default_workspace_bytes, _DEFAULT_PAGED_WRAPPER_WORKSPACE_BYTES),
    )


def _events_enabled() -> bool:
    return os.getenv(SPARSE_REPAIR_EVENTS_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _boolean_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise SparseRepairAttentionError(
        f"{name} must be a boolean value; got {raw!r}"
    )


def _paged_wrapper_reuse_option(forward_batch: Any) -> tuple[bool, str]:
    enabled = _boolean_env(SPARSE_REPAIR_REUSE_PAGED_WRAPPER_ENV)
    request_metadata = getattr(forward_batch, "pic_request_metadata", None)
    link_plan = (
        request_metadata[0].get("link_plan")
        if isinstance(request_metadata, list)
        and len(request_metadata) == 1
        and isinstance(request_metadata[0], dict)
        else None
    )
    key = "diagnostic_flashinfer_reuse_paged_wrapper_across_plans"
    if not isinstance(link_plan, dict) or key not in link_plan:
        return enabled, "environment"
    if not _boolean_env("SGLANG_PIC_ALLOW_REQUEST_ROW_BATCHING_OVERRIDE"):
        raise SparseRepairAttentionError(
            "Per-request FlashInfer wrapper-reuse override is disabled."
        )
    if not isinstance(link_plan[key], bool):
        raise SparseRepairAttentionError(f"pic_link.{key} must be a boolean.")
    return bool(link_plan[key]), "request_diagnostic_override"


def _skip_exact_paged_metadata_cache_option(
    forward_batch: Any,
) -> tuple[bool, str]:
    enabled = _boolean_env(SPARSE_REPAIR_SKIP_EXACT_PAGED_METADATA_CACHE_ENV)
    request_metadata = getattr(forward_batch, "pic_request_metadata", None)
    link_plan = (
        request_metadata[0].get("link_plan")
        if isinstance(request_metadata, list)
        and len(request_metadata) == 1
        and isinstance(request_metadata[0], dict)
        else None
    )
    key = "diagnostic_flashinfer_skip_exact_paged_metadata_cache"
    if not isinstance(link_plan, dict) or key not in link_plan:
        return enabled, "environment"
    if not _boolean_env("SGLANG_PIC_ALLOW_REQUEST_ROW_BATCHING_OVERRIDE"):
        raise SparseRepairAttentionError(
            "Per-request FlashInfer metadata-cache override is disabled."
        )
    if not isinstance(link_plan[key], bool):
        raise SparseRepairAttentionError(f"pic_link.{key} must be a boolean.")
    return bool(link_plan[key]), "request_diagnostic_override"


def _component_ranges_option(forward_batch: Any) -> tuple[bool, str]:
    # Keep the historical profiler-range behavior unless a formal experiment
    # explicitly disables it. The no-range path did not pass the latency gate.
    enabled = _boolean_env(SPARSE_REPAIR_PROFILE_RANGES_ENV, default=True)
    request_metadata = getattr(forward_batch, "pic_request_metadata", None)
    link_plan = (
        request_metadata[0].get("link_plan")
        if isinstance(request_metadata, list)
        and len(request_metadata) == 1
        and isinstance(request_metadata[0], dict)
        else None
    )
    key = "diagnostic_sparse_repair_record_function"
    if not isinstance(link_plan, dict) or key not in link_plan:
        return enabled, "environment"
    if not _boolean_env("SGLANG_PIC_ALLOW_REQUEST_ROW_BATCHING_OVERRIDE"):
        raise SparseRepairAttentionError(
            "Per-request sparse-repair component-range override is disabled."
        )
    if not isinstance(link_plan[key], bool):
        raise SparseRepairAttentionError(f"pic_link.{key} must be a boolean.")
    return bool(link_plan[key]), "request_diagnostic_override"


def _add_cpu_timing_ms(metadata: dict[str, Any], name: str, elapsed_s: float) -> None:
    timings = metadata.setdefault("cpu_timing_ms", {})
    timings[name] = float(timings.get(name, 0.0)) + elapsed_s * 1000.0


@contextlib.contextmanager
def _cpu_timing(metadata: dict[str, Any], name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        _add_cpu_timing_ms(metadata, name, time.perf_counter() - start)


def _flashinfer_version() -> str:
    try:
        return importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        try:
            import flashinfer

            return str(getattr(flashinfer, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - optional extension import may fail broadly
            return "unavailable"


def _flashinfer_available() -> tuple[bool, str | None]:
    try:
        import flashinfer

        wrapper = getattr(
            getattr(flashinfer, "sparse", None),
            "BlockSparseAttentionWrapper",
            None,
        )
        if wrapper is None:
            return False, "BlockSparseAttentionWrapper is unavailable"
        return True, None
    except Exception as exc:  # noqa: BLE001 - report optional extension failures
        return False, f"FlashInfer import failed: {type(exc).__name__}: {exc}"


def make_sparse_repair_metadata(
    forward_batch: Any,
    *,
    start: int,
    end: int,
    block_count: int,
    block_size: int,
    unified_view: Any | None = None,
    repair_ranges: list[tuple[int, int]] | None = None,
    record_component_ranges: bool | None = None,
    component_ranges_source: str | None = None,
) -> dict[str, Any]:
    requested = requested_sparse_repair_backend()
    available, reason = _flashinfer_available()
    actual = "native"
    compatibility = "not_requested"
    fallback_reason = None
    if requested in {"flashinfer", "auto", "hybrid"}:
        if available:
            actual = (
                "selector_pending"
                if requested == "hybrid"
                else "flashinfer_block_sparse"
            )
            compatibility = "precheck_pass"
        elif requested in {"flashinfer", "hybrid"}:
            raise SparseRepairAttentionError(reason or "FlashInfer is unavailable")
        else:
            compatibility = "precheck_failed"
            fallback_reason = reason

    normalized_ranges = repair_ranges or [(start, end)]
    if (
        not normalized_ranges
        or normalized_ranges[0][0] != start
        or normalized_ranges[-1][1] != end
        or any(row_end <= row_start for row_start, row_end in normalized_ranges)
    ):
        raise SparseRepairAttentionError(
            "repair_ranges must be non-empty and match start/end"
        )
    repair_tokens = sum(row_end - row_start for row_start, row_end in normalized_ranges)
    view_metadata = (
        unified_view.trace_metadata() if unified_view is not None else None
    )
    if record_component_ranges is None:
        record_component_ranges, component_ranges_source = _component_ranges_option(
            forward_batch
        )
    elif component_ranges_source is None:
        component_ranges_source = "caller"
    return {
        "requested_backend": requested,
        "actual_backend": actual,
        "flashinfer_version": _flashinfer_version(),
        "flashinfer_commit": os.getenv("FLASHINFER_COMMIT", "wheel-not-recorded"),
        "flashinfer_api": (
            "flashinfer.sparse.BlockSparseAttentionWrapper"
            if actual == "flashinfer_block_sparse"
            else None
        ),
        "record_component_ranges": record_component_ranges,
        "component_ranges_config_source": component_ranges_source,
        "q_tokens": repair_tokens,
        "kv_tokens": end,
        "repair_tokens": repair_tokens,
        "repair_start": start,
        "repair_end": end,
        "repair_ranges": [
            {"start": row_start, "end": row_end}
            for row_start, row_end in normalized_ranges
        ],
        "repair_row_count": len(normalized_ranges),
        "block_count": block_count,
        "block_size": block_size,
        "indptr_length": 0,
        "indices_length": 0,
        "workspace_bytes": int(
            os.getenv(SPARSE_REPAIR_WORKSPACE_ENV, str(64 * 1024 * 1024))
        ),
        "plan_time": 0.0,
        "run_time": 0.0,
        "cpu_timing_ms": {},
        "wrapper_cache_hit": False,
        "wrapper_cache_miss": False,
        "wrapper_create_ms": 0.0,
        "metadata_cache_hit": False,
        "metadata_cache_miss": False,
        "metadata_build_ms": 0.0,
        "mask_build_ms": 0.0,
        "indptr_build_ms": 0.0,
        "indices_build_ms": 0.0,
        "plan_cache_hit": False,
        "plan_cache_miss": False,
        "plan_reuse_supported": True,
        "plan_reuse_scope": "layers_and_exact_physical_slot_patterns",
        "metadata_reuse_scope": "layers_and_exact_physical_slot_patterns",
        "selected_backend": actual,
        "selector_policy": None,
        "decision_reason": None,
        "native_estimated_ms": None,
        "flashinfer_estimated_ms": None,
        "sparse_density": None,
        "pattern_hash": None,
        "layer_calls": 0,
        "plan_reuse_calls": 0,
        "fallback_reason": fallback_reason,
        "compatibility_status": compatibility,
        "execution_device": str(forward_batch.input_ids.device),
        "position_encoding_mode": "NONE_ALREADY_APPLIED",
        "mask_semantics": "bsr_single_row_physical_slots_explicit_logical_causal_mask",
        "kv_layout": "TokenToKVPool_NHD_C1",
        "unified_metadata_view": view_metadata,
        "metadata_view_hit": False,
        "_event_pairs": [],
        "_planned": False,
        "_unified_view": unified_view,
    }


class _CudaEventPool:
    def __init__(self) -> None:
        self.events: list[torch.cuda.Event] = []
        self.cursor = 0

    def acquire(self) -> torch.cuda.Event:
        if self.cursor == len(self.events):
            self.events.append(torch.cuda.Event(enable_timing=True))
        event = self.events[self.cursor]
        self.cursor += 1
        return event

    def reset(self) -> None:
        self.cursor = 0


_EVENT_POOL = _CudaEventPool()


@contextlib.contextmanager
def sparse_repair_component(forward_batch: Any, name: str) -> Iterator[None]:
    metadata = getattr(forward_batch, "pic_sparse_repair_metadata", None)
    if not metadata:
        yield
        return
    events_enabled = _events_enabled() and torch.cuda.is_available()
    ranges_enabled = bool(metadata.get("record_component_ranges", False))
    if not events_enabled and not ranges_enabled:
        yield
        return
    range_context = (
        torch.autograd.profiler.record_function(name)
        if ranges_enabled
        else contextlib.nullcontext()
    )
    if not events_enabled:
        with range_context:
            yield
        return
    start = _EVENT_POOL.acquire()
    end = _EVENT_POOL.acquire()
    with range_context:
        start.record()
        try:
            yield
        finally:
            end.record()
            metadata["_event_pairs"].append((name, start, end))


@dataclass
class _WrapperState:
    workspace: torch.Tensor
    wrapper: Any
    plan_key: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class _SparseMetadataState:
    indptr: torch.Tensor
    indices: torch.Tensor
    mask: torch.Tensor
    pattern_hash: str
    nbytes: int


@dataclass(frozen=True)
class _PagedMetadataState:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    pattern_hash: str
    nbytes: int


class FlashInferWrapperStateCache:
    """LRU of independently planned wrappers.

    ``BlockSparseAttentionWrapper.plan()`` mutates wrapper-owned execution
    state. Exact plan reuse therefore requires one wrapper state per cached
    plan; re-planning a single global wrapper would invalidate older entries.
    """

    def __init__(self) -> None:
        self._states: OrderedDict[tuple[Any, ...], _WrapperState] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def get(self, key: tuple[Any, ...]) -> _WrapperState | None:
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                self._states.move_to_end(key)
            return state

    def put(
        self,
        key: tuple[Any, ...],
        state: _WrapperState,
        *,
        workspace_bytes: int,
    ) -> None:
        max_entries = _positive_int_env(
            SPARSE_REPAIR_PLAN_CACHE_ENTRIES_ENV, _DEFAULT_PLAN_CACHE_ENTRIES
        )
        max_bytes = _positive_int_env(
            SPARSE_REPAIR_PLAN_CACHE_BYTES_ENV, _DEFAULT_PLAN_CACHE_BYTES
        )
        if max_entries == 0 or workspace_bytes > max_bytes:
            return
        with self._lock:
            previous = self._states.pop(key, None)
            if previous is not None:
                self._bytes -= int(previous.workspace.numel())
            state.plan_key = key
            self._states[key] = state
            self._bytes += workspace_bytes
            while self._states and (
                len(self._states) > max_entries or self._bytes > max_bytes
            ):
                _, evicted = self._states.popitem(last=False)
                self._bytes -= int(evicted.workspace.numel())

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
            self._bytes = 0

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._states), "workspace_bytes": self._bytes}


class SparseRepairMetadataCache:
    """LRU for exact physical-slot sparse metadata tensors."""

    def __init__(self) -> None:
        self._states: OrderedDict[
            tuple[Any, ...],
            _SparseMetadataState | _PagedMetadataState,
        ] = (
            OrderedDict()
        )
        self._bytes = 0
        self._lock = threading.RLock()

    def get(
        self, key: tuple[Any, ...]
    ) -> _SparseMetadataState | _PagedMetadataState | None:
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                self._states.move_to_end(key)
            return state

    def put(
        self,
        key: tuple[Any, ...],
        state: _SparseMetadataState | _PagedMetadataState,
    ) -> None:
        max_entries = _positive_int_env(
            SPARSE_REPAIR_METADATA_CACHE_ENTRIES_ENV,
            _DEFAULT_METADATA_CACHE_ENTRIES,
        )
        max_bytes = _positive_int_env(
            SPARSE_REPAIR_METADATA_CACHE_BYTES_ENV,
            _DEFAULT_METADATA_CACHE_BYTES,
        )
        if max_entries == 0 or state.nbytes > max_bytes:
            return
        with self._lock:
            previous = self._states.pop(key, None)
            if previous is not None:
                self._bytes -= previous.nbytes
            self._states[key] = state
            self._bytes += state.nbytes
            while self._states and (
                len(self._states) > max_entries or self._bytes > max_bytes
            ):
                _, evicted = self._states.popitem(last=False)
                self._bytes -= evicted.nbytes

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
            self._bytes = 0

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._states), "metadata_bytes": self._bytes}


_WRAPPER_STATE_CACHE = FlashInferWrapperStateCache()
_SPARSE_METADATA_CACHE = SparseRepairMetadataCache()
_PAGED_UNCACHED_KEY_COUNTER = count()


def clear_sparse_repair_caches() -> None:
    """Clear process-local caches. Intended for tests and controlled trials."""

    _WRAPPER_STATE_CACHE.clear()
    _SPARSE_METADATA_CACHE.clear()


def sparse_repair_cache_stats() -> dict[str, dict[str, int]]:
    return {
        "wrapper_plan": _WRAPPER_STATE_CACHE.snapshot(),
        "metadata": _SPARSE_METADATA_CACHE.snapshot(),
    }


def _new_wrapper_state(device: torch.device, workspace_bytes: int) -> _WrapperState:
    import flashinfer

    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    return _WrapperState(
        workspace=workspace,
        wrapper=flashinfer.sparse.BlockSparseAttentionWrapper(
            workspace, backend="auto"
        ),
    )


def _new_paged_wrapper_state(
    device: torch.device, workspace_bytes: int
) -> _WrapperState:
    import flashinfer

    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    return _WrapperState(
        workspace=workspace,
        wrapper=flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend="fa2",
        ),
    )


def _paged_wrapper_cache_key(
    q: torch.Tensor,
    key_buffer: torch.Tensor,
    layer: Any,
    *,
    workspace_bytes: int,
    fixed_split_size: int,
) -> tuple[Any, ...]:
    """Identify a paged wrapper that can be safely re-planned.

    Physical page indices intentionally do not participate in this key. They
    remain part of the exact plan key and therefore always trigger plan() when
    they change.
    """

    return (
        "paged_prefill_rows_replannable_wrapper",
        str(q.device),
        workspace_bytes,
        int(layer.tp_q_head_num),
        int(layer.tp_k_head_num),
        int(layer.head_dim),
        str(q.dtype),
        str(key_buffer.dtype),
        fixed_split_size,
    )


@dataclass(frozen=True)
class SparseRepairAttentionDecision:
    selected_backend: str
    native_estimated_ms: float
    flashinfer_estimated_ms: float
    reason: str
    policy: str = "h100_7b_piecewise_v1"


class SparseRepairAttentionSelector:
    """Explainable selector calibrated from the measured H100 repair rows.

    The first policy is intentionally conservative. FlashInfer won for the
    measured q=128 medium/long-KV rows, while native won for the q=695 row and
    for the cold short-KV row. The selector does not alter K or visibility.
    """

    _CALIBRATION = (
        (128, 2391, 0.973, 8.510, 156.688, 3.798),
        (128, 3594, 0.982, 10.101, 5.140, 3.329),
        (128, 28263, 0.998, 51.274, 15.601, 13.930),
        (128, 53277, 0.999, 93.166, 26.206, 24.160),
        (128, 54504, 0.999, 93.704, 26.702, 24.625),
        (695, 56723, 0.994, 199.541, 226.167, 216.147),
    )

    @classmethod
    def _estimate(
        cls,
        repair_tokens: int,
        kv_tokens: int,
        density: float,
        *,
        plan_cache_hit: bool,
    ) -> tuple[float, float]:
        def distance(row: tuple[float, ...]) -> float:
            q, kv, row_density, *_ = row
            return (
                2.0 * abs(math.log2(max(repair_tokens, 1) / q))
                + abs(math.log2(max(kv_tokens, 1) / kv))
                + 10.0 * abs(density - row_density)
            )

        nearest = min(cls._CALIBRATION, key=distance)
        _, _, _, native_ms, fi_total_ms, fi_run_ms = nearest
        fi_estimate = fi_run_ms if plan_cache_hit else fi_total_ms
        return float(native_ms), float(fi_estimate)

    def select(
        self,
        *,
        repair_tokens: int,
        kv_tokens: int,
        density: float,
        block_count: int,
        plan_cache_hit: bool,
        metadata_cache_hit: bool,
    ) -> SparseRepairAttentionDecision:
        del block_count  # Recorded in trace; current measured policy does not split on it.
        native_ms, flashinfer_ms = self._estimate(
            repair_tokens,
            kv_tokens,
            density,
            plan_cache_hit=plan_cache_hit,
        )
        if repair_tokens > 512:
            selected = "native"
            reason = "repair_tokens_above_measured_flashinfer_win_region"
        elif density < 0.90:
            selected = "native"
            reason = "density_outside_validated_high_density_region"
        elif plan_cache_hit:
            selected = "flashinfer_block_sparse"
            reason = "exact_plan_cache_hit_in_measured_flashinfer_win_region"
        elif metadata_cache_hit and kv_tokens >= 2391:
            selected = "flashinfer_block_sparse"
            reason = "metadata_hit_amortizes_preparation_cost"
        elif kv_tokens >= 3200:
            selected = "flashinfer_block_sparse"
            reason = "measured_q_le_512_medium_or_long_kv_flashinfer_win"
        else:
            selected = "native"
            reason = "cold_short_kv_native_total_path_win"
        return SparseRepairAttentionDecision(
            selected_backend=selected,
            native_estimated_ms=native_ms,
            flashinfer_estimated_ms=flashinfer_ms,
            reason=reason,
        )


_ATTENTION_SELECTOR = SparseRepairAttentionSelector()


def _compatibility_error(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    layer: Any,
    forward_batch: Any,
) -> str | None:
    if k is None or v is None:
        return "cross-layer shared K/V is unsupported"
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        return "Q/K/V must be CUDA tensors"
    if q.dtype not in {torch.float16, torch.bfloat16}:
        return f"unsupported Q dtype: {q.dtype}"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "Q/K/V dtypes must match"
    if k.ndim != 3 or v.ndim != 3:
        return "TokenToKVPool K/V must be rank-3 NHD tensors"
    q_shape_ok = (
        q.ndim == 2
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
    ) or (q.ndim == 3 and q.shape[-1] == layer.head_dim)
    if not q_shape_ok or k.shape[-1] != layer.head_dim:
        return "head dimension mismatch"
    if layer.tp_q_head_num % layer.tp_k_head_num != 0:
        return "GQA requires num_qo_heads divisible by num_kv_heads"
    if getattr(layer, "is_cross_attention", False):
        return "cross attention is unsupported"
    if getattr(layer, "sliding_window_size", -1) not in {-1, None}:
        return "sliding-window attention is unsupported"
    grouped_execution = getattr(
        forward_batch, "pic_sparse_repair_metadata", {}
    ).get("grouped_attention_execution")
    grouped_batched = grouped_execution in {
        "flashinfer_batched_equal_repair_rows",
        "flashinfer_paged_prefill_rows",
    }
    if forward_batch.batch_size != 1 and not grouped_batched:
        return "prototype requires batch_size=1"
    if getattr(forward_batch, "spec_info", None) is not None:
        return "speculative attention is unsupported"
    return None


def _logical_causal_density(repair_start: int, repair_end: int) -> tuple[int, int, float]:
    repair_tokens = repair_end - repair_start
    mask_elements = repair_end * repair_tokens
    true_count = (
        repair_tokens * (repair_start + 1)
        + repair_tokens * (repair_tokens - 1) // 2
    )
    density = true_count / mask_elements if mask_elements else 0.0
    return true_count, mask_elements, density


def _repair_ranges_causal_density(
    repair_ranges: list[tuple[int, int]],
) -> tuple[int, int, float]:
    rows = [_logical_causal_density(start, end) for start, end in repair_ranges]
    true_count = sum(row[0] for row in rows)
    mask_elements = sum(row[1] for row in rows)
    density = true_count / mask_elements if mask_elements else 0.0
    return true_count, mask_elements, density


def select_hybrid_request_backend(
    repair_ranges: list[tuple[int, int]],
    *,
    block_count: int,
) -> tuple[str, str]:
    """Choose one attention implementation for every repair span in a request."""
    decisions = []
    for start, end in repair_ranges:
        _, _, density = _logical_causal_density(start, end)
        decisions.append(
            _ATTENTION_SELECTOR.select(
                repair_tokens=end - start,
                kv_tokens=end,
                density=density,
                block_count=block_count,
                plan_cache_hit=False,
                metadata_cache_hit=False,
            ).selected_backend
        )
    if decisions and all(value == "flashinfer_block_sparse" for value in decisions):
        return (
            "flashinfer_block_sparse",
            "request_all_spans_in_measured_flashinfer_win_region",
        )
    if decisions and all(value == "native" for value in decisions):
        return "native", "request_all_spans_select_native"
    return "native", "mixed_backend_correctness_guard"


def _physical_slot_identity(
    forward_batch: Any, metadata: dict[str, Any]
) -> tuple[torch.Tensor, bytes, str]:
    unified_view = metadata.get("_unified_view")
    if unified_view is not None:
        if (
            int(unified_view.repair_start) != int(metadata["repair_start"])
            or int(unified_view.repair_end) != int(metadata["repair_end"])
        ):
            raise SparseRepairAttentionError(
                "unified metadata view does not match the active repair range"
            )
        metadata["metadata_view_hit"] = True
        return (
            unified_view.destination_slots,
            unified_view.ordered_slot_bytes,
            unified_view.ordered_slot_hash,
        )

    metadata["metadata_view_hit"] = False
    with (
        _cpu_timing(metadata, "slot_lookup"),
        sparse_repair_component(forward_batch, "FlashInferSlotLookup"),
    ):
        end = int(metadata["repair_end"])
        req_pool_idx = int(forward_batch.req_pool_indices[0].item())
        slots = forward_batch.req_to_token_pool.req_to_token[
            req_pool_idx, :end
        ].to(dtype=torch.int32)

    with _cpu_timing(metadata, "physical_slot_hash"):
        slot_bytes = slots.detach().cpu().contiguous().numpy().tobytes()
        pattern_hash = hashlib.sha256(slot_bytes).hexdigest()
    return slots, slot_bytes, pattern_hash


def _build_physical_slot_pattern_from_slots(
    forward_batch: Any,
    metadata: dict[str, Any],
    slots: torch.Tensor,
    pool_size: int,
) -> _SparseMetadataState:
    started = time.perf_counter()
    start = int(metadata["repair_start"])
    end = int(metadata["repair_end"])

    indptr_started = time.perf_counter()
    with (
        _cpu_timing(metadata, "indptr_indices_build"),
        sparse_repair_component(forward_batch, "FlashInferIndptrIndicesBuild"),
    ):
        indptr, indices = build_explicit_causal_pattern(
            slots,
            repair_start=start,
            repair_end=end,
            pool_size=pool_size,
        )
    indptr_elapsed_ms = (time.perf_counter() - indptr_started) * 1000.0
    metadata["indptr_build_ms"] += indptr_elapsed_ms
    metadata["indices_build_ms"] += indptr_elapsed_ms

    repair_tokens = end - start
    mask_started = time.perf_counter()
    with (
        _cpu_timing(metadata, "mask_arange_build"),
        sparse_repair_component(forward_batch, "FlashInferMaskArangeBuild"),
    ):
        column_positions = torch.arange(end, device=indices.device).view(end, 1, 1)
        query_ends = torch.arange(start, end, device=indices.device).view(
            1, repair_tokens, 1
        )

    with (
        _cpu_timing(metadata, "mask_fill_compare"),
        sparse_repair_component(forward_batch, "FlashInferMaskCompareFill"),
    ):
        mask = column_positions <= query_ends

    with (
        _cpu_timing(metadata, "mask_contiguous"),
        sparse_repair_component(forward_batch, "FlashInferMaskContiguous"),
    ):
        mask = mask.contiguous()

    metadata["mask_build_ms"] += (time.perf_counter() - mask_started) * 1000.0
    metadata["metadata_build_ms"] += (time.perf_counter() - started) * 1000.0
    pattern_hash = str(metadata["pattern_hash"])
    nbytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in (indptr, indices, mask)
    )
    return _SparseMetadataState(
        indptr=indptr,
        indices=indices,
        mask=mask,
        pattern_hash=pattern_hash,
        nbytes=nbytes,
    )


def _sparse_metadata_state(
    forward_batch: Any, metadata: dict[str, Any], pool_size: int
) -> tuple[_SparseMetadataState, tuple[Any, ...]]:
    if (
        metadata.get("grouped_attention_execution")
        == "flashinfer_batched_equal_repair_rows"
    ):
        return _grouped_sparse_metadata_state(forward_batch, metadata, pool_size)
    slots, slot_bytes, pattern_hash = _physical_slot_identity(forward_batch, metadata)
    start = int(metadata["repair_start"])
    end = int(metadata["repair_end"])
    unified_view = metadata.get("_unified_view")
    key = (
        unified_view.flashinfer_metadata_key(pool_size)
        if unified_view is not None
        else (str(slots.device), pool_size, start, end, slot_bytes)
    )
    metadata["pattern_hash"] = pattern_hash
    state = _SPARSE_METADATA_CACHE.get(key)
    if state is not None:
        metadata["metadata_cache_hit"] = True
        metadata["metadata_cache_miss"] = False
        return state, key

    metadata["metadata_cache_hit"] = False
    metadata["metadata_cache_miss"] = True
    state = _build_physical_slot_pattern_from_slots(
        forward_batch, metadata, slots, pool_size
    )
    _SPARSE_METADATA_CACHE.put(key, state)
    return state, key


def _plan_key(
    metadata_key: tuple[Any, ...],
    q: torch.Tensor,
    key_buffer: torch.Tensor,
    layer: Any,
    workspace_bytes: int,
) -> tuple[Any, ...]:
    return (
        metadata_key,
        str(q.device),
        workspace_bytes,
        int(q.shape[0]),
        int(key_buffer.shape[0]),
        int(layer.tp_q_head_num),
        int(layer.tp_k_head_num),
        int(layer.head_dim),
        str(q.dtype),
        str(key_buffer.dtype),
        float(layer.scaling),
        False,
        "NONE",
        1,
    )


def build_explicit_causal_pattern(
    slot_ids: torch.Tensor, *, repair_start: int, repair_end: int, pool_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the R=repair_tokens, C=1 BSR pattern used by the prototype.

    This pure tensor helper is also used by standalone validation so the
    benchmark and runtime exercise exactly the same visibility conversion.
    """
    if slot_ids.ndim != 1:
        raise SparseRepairAttentionError("slot_ids must be rank 1")
    if repair_start < 0 or repair_end <= repair_start:
        raise SparseRepairAttentionError("invalid repair range")
    if slot_ids.numel() < repair_end:
        raise SparseRepairAttentionError("slot_ids is shorter than repair_end")
    slots = slot_ids[:repair_end].to(dtype=torch.int32)
    if bool((slots < 0).any()) or bool((slots >= pool_size).any()):
        raise SparseRepairAttentionError("slot_ids contains invalid KV slots")
    # The whole repair span is one BSR row block. Bottom-right causal
    # alignment gives repair query i the logical prefix ending at
    # repair_start + i without constructing an O(Q x KV) custom mask.
    indptr = torch.tensor([0, repair_end], device=slots.device, dtype=torch.int32)
    return indptr, slots.contiguous()


def build_explicit_causal_masked_pattern(
    slot_ids: torch.Tensor, *, repair_start: int, repair_end: int, pool_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a physical-slot BSR pattern with an explicit logical causal mask.

    FlashInfer's implicit causal alignment is expressed in the logical BSR
    coordinate system. LegoLink indices are TokenToKVPool physical slot ids, so
    using ``causal=True`` against the full pool size can expose future repair
    tokens. This helper keeps physical-slot indexing but carries the logical
    prefix rule in mask_data.
    """
    indptr, indices = build_explicit_causal_pattern(
        slot_ids, repair_start=repair_start, repair_end=repair_end, pool_size=pool_size
    )
    repair_tokens = repair_end - repair_start
    column_positions = torch.arange(repair_end, device=indices.device).view(
        repair_end, 1, 1
    )
    query_ends = torch.arange(
        repair_start, repair_end, device=indices.device
    ).view(1, repair_tokens, 1)
    mask = column_positions <= query_ends
    return indptr, indices, mask.contiguous()


def build_grouped_explicit_causal_masked_pattern(
    row_slot_ids: list[torch.Tensor],
    *,
    repair_ranges: list[tuple[int, int]],
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one BSR plan for equal-length independent repair rows.

    Each repair row becomes one BSR row block. Physical slot IDs remain the
    column indices, and each nonzero carries its row-local logical causal mask.
    This keeps request rows isolated while allowing one FlashInfer plan/run.
    """
    if not row_slot_ids or len(row_slot_ids) != len(repair_ranges):
        raise SparseRepairAttentionError(
            "grouped slot rows and repair ranges must be non-empty and aligned"
        )
    repair_lengths = {end - start for start, end in repair_ranges}
    if len(repair_lengths) != 1 or next(iter(repair_lengths)) <= 0:
        raise SparseRepairAttentionError(
            "batched FlashInfer requires equal positive repair-row lengths"
        )

    device = row_slot_ids[0].device
    indptr_values = [0]
    index_rows = []
    mask_rows = []
    for slots, (start, end) in zip(row_slot_ids, repair_ranges, strict=True):
        if slots.ndim != 1 or slots.device != device:
            raise SparseRepairAttentionError(
                "grouped physical slot rows must be rank-1 tensors on one device"
            )
        if slots.numel() < end:
            raise SparseRepairAttentionError(
                "grouped physical slot row is shorter than repair_end"
            )
        selected = slots[:end].to(dtype=torch.int32)
        if bool((selected < 0).any()) or bool((selected >= pool_size).any()):
            raise SparseRepairAttentionError(
                "grouped physical slot row contains invalid KV slots"
            )
        index_rows.append(selected.contiguous())
        indptr_values.append(indptr_values[-1] + end)
        column_positions = torch.arange(end, device=device).view(end, 1, 1)
        query_ends = torch.arange(start, end, device=device).view(
            1, end - start, 1
        )
        mask_rows.append((column_positions <= query_ends).contiguous())

    return (
        torch.tensor(indptr_values, dtype=torch.int32, device=device),
        torch.cat(index_rows, dim=0),
        torch.cat(mask_rows, dim=0),
    )


def _grouped_sparse_metadata_state(
    forward_batch: Any,
    metadata: dict[str, Any],
    pool_size: int,
) -> tuple[_SparseMetadataState, tuple[Any, ...]]:
    ranges = _grouped_row_ranges(metadata)
    row_slots = []
    identity_parts = []
    with (
        _cpu_timing(metadata, "slot_lookup"),
        sparse_repair_component(forward_batch, "FlashInferSlotLookup"),
    ):
        for row_index, (start, end) in enumerate(ranges):
            req_pool_idx = int(forward_batch.req_pool_indices[row_index].item())
            slots = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :end
            ].to(dtype=torch.int32)
            row_slots.append(slots)
            identity_parts.extend(
                (
                    start.to_bytes(8, "little", signed=False),
                    end.to_bytes(8, "little", signed=False),
                    slots.detach().cpu().contiguous().numpy().tobytes(),
                )
            )

    slot_bytes = b"".join(identity_parts)
    pattern_hash = hashlib.sha256(slot_bytes).hexdigest()
    key = (
        "grouped_equal_rows",
        str(row_slots[0].device),
        pool_size,
        tuple(ranges),
        slot_bytes,
    )
    metadata["pattern_hash"] = pattern_hash
    state = _SPARSE_METADATA_CACHE.get(key)
    if state is not None:
        metadata["metadata_cache_hit"] = True
        metadata["metadata_cache_miss"] = False
        return state, key

    metadata["metadata_cache_hit"] = False
    metadata["metadata_cache_miss"] = True
    started = time.perf_counter()
    with (
        _cpu_timing(metadata, "indptr_indices_build"),
        sparse_repair_component(forward_batch, "FlashInferIndptrIndicesBuild"),
    ):
        indptr, indices, mask = build_grouped_explicit_causal_masked_pattern(
            row_slots,
            repair_ranges=ranges,
            pool_size=pool_size,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metadata["indptr_build_ms"] += elapsed_ms
    metadata["indices_build_ms"] += elapsed_ms
    metadata["mask_build_ms"] += elapsed_ms
    metadata["metadata_build_ms"] += elapsed_ms
    nbytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in (indptr, indices, mask)
    )
    state = _SparseMetadataState(
        indptr=indptr,
        indices=indices,
        mask=mask,
        pattern_hash=pattern_hash,
        nbytes=nbytes,
    )
    _SPARSE_METADATA_CACHE.put(key, state)
    return state, key


def _paged_sparse_metadata_state(
    forward_batch: Any,
    metadata: dict[str, Any],
    pool_size: int,
) -> tuple[_PagedMetadataState, tuple[Any, ...]]:
    ranges = _grouped_row_ranges(metadata)
    qo_indptr_values = [0]
    kv_indptr_values = [0]
    indices_rows = []
    identity_parts = []
    skip_exact_cache, cache_config_source = (
        _skip_exact_paged_metadata_cache_option(forward_batch)
    )
    metadata["exact_paged_metadata_cache_skipped"] = skip_exact_cache
    metadata["metadata_cache_config_source"] = cache_config_source
    with (
        _cpu_timing(metadata, "slot_lookup"),
        sparse_repair_component(forward_batch, "FlashInferSlotLookup"),
    ):
        for row_index, (start, end) in enumerate(ranges):
            req_pool_idx = forward_batch.req_pool_indices[row_index].to(
                dtype=torch.long
            )
            slots = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :end
            ].to(dtype=torch.int32)
            torch._assert_async(
                torch.all((slots >= 0) & (slots < pool_size)),
                "paged FlashInfer row contains invalid KV slots",
            )
            indices_rows.append(slots.contiguous())
            qo_indptr_values.append(qo_indptr_values[-1] + end - start)
            kv_indptr_values.append(kv_indptr_values[-1] + end)
            if not skip_exact_cache:
                identity_parts.extend(
                    (
                        start.to_bytes(8, "little", signed=False),
                        end.to_bytes(8, "little", signed=False),
                        slots.detach().cpu().contiguous().numpy().tobytes(),
                    )
                )

    fixed_split_size = _positive_int_env(
        SPARSE_REPAIR_FIXED_SPLIT_SIZE_ENV, 128
    )
    if skip_exact_cache:
        unique_key = next(_PAGED_UNCACHED_KEY_COUNTER)
        pattern_hash = f"request_private_uncached_{unique_key}"
        key = (
            "paged_prefill_rows_uncached",
            unique_key,
            str(indices_rows[0].device),
            pool_size,
            tuple(ranges),
            fixed_split_size,
        )
    else:
        slot_bytes = b"".join(identity_parts)
        pattern_hash = hashlib.sha256(slot_bytes).hexdigest()
        key = (
            "paged_prefill_rows",
            str(indices_rows[0].device),
            pool_size,
            tuple(ranges),
            fixed_split_size,
            slot_bytes,
        )
    metadata["pattern_hash"] = pattern_hash
    cached = None if skip_exact_cache else _SPARSE_METADATA_CACHE.get(key)
    if cached is not None:
        if not isinstance(cached, _PagedMetadataState):
            raise SparseRepairAttentionError("paged metadata cache type mismatch")
        metadata["metadata_cache_hit"] = True
        metadata["metadata_cache_miss"] = False
        return cached, key

    metadata["metadata_cache_hit"] = False
    metadata["metadata_cache_miss"] = True
    device = indices_rows[0].device
    state = _PagedMetadataState(
        qo_indptr=torch.tensor(
            qo_indptr_values, dtype=torch.int32, device=device
        ),
        kv_indptr=torch.tensor(
            kv_indptr_values, dtype=torch.int32, device=device
        ),
        indices=torch.cat(indices_rows),
        last_page_len=torch.ones(
            len(ranges), dtype=torch.int32, device=device
        ),
        pattern_hash=pattern_hash,
        nbytes=0,
    )
    state = dataclasses.replace(
        state,
        nbytes=sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in (
                state.qo_indptr,
                state.kv_indptr,
                state.indices,
                state.last_page_len,
            )
        ),
    )
    if not skip_exact_cache:
        _SPARSE_METADATA_CACHE.put(key, state)
    return state, key


def _grouped_row_ranges(metadata: dict[str, Any]) -> list[tuple[int, int]]:
    ranges = []
    for row in metadata.get("repair_ranges") or []:
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            raise SparseRepairAttentionError("invalid grouped repair row")
        ranges.append((start, end))
    return ranges


def _make_flashinfer_row_batch(
    forward_batch: Any,
    *,
    row_index: int,
    token_offset: int,
    start: int,
    end: int,
    metadata: dict[str, Any],
) -> Any:
    length = end - start
    device = forward_batch.input_ids.device
    token_slice = slice(token_offset, token_offset + length)
    req_pool_indices = forward_batch.req_pool_indices[row_index : row_index + 1]
    seq_lens = torch.tensor([end], dtype=torch.int64, device=device)
    return dataclasses.replace(
        forward_batch,
        batch_size=1,
        input_ids=forward_batch.input_ids[token_slice],
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=torch.tensor([end], dtype=torch.int64),
        orig_seq_lens=seq_lens,
        out_cache_loc=forward_batch.out_cache_loc[token_slice],
        seq_lens_sum=end,
        positions=forward_batch.positions[token_slice],
        extend_num_tokens=length,
        extend_seq_lens=torch.tensor([length], dtype=torch.int32, device=device),
        extend_prefix_lens=torch.tensor([start], dtype=torch.int32, device=device),
        extend_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
        extend_prefix_lens_cpu=[start],
        extend_seq_lens_cpu=[length],
        extend_logprob_start_lens_cpu=[length],
        return_logprob=False,
        num_token_non_padded=torch.tensor(length, dtype=torch.int32, device=device),
        num_token_non_padded_cpu=length,
        is_extend_in_batch=True,
        all_extend_in_batch=True,
        pic_sparse_repair_metadata=metadata,
    )


def _try_flashinfer_paged_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    save_kv_cache: bool,
    metadata: dict[str, Any],
) -> torch.Tensor:
    if str(metadata.get("requested_backend")) != "flashinfer":
        raise SparseRepairAttentionError(
            "paged grouped attention must be forced to FlashInfer"
        )
    if metadata.get("actual_backend") == "native":
        # An earlier layer of this request degraded on capacity. Keep every
        # remaining layer on the same operator so one request never mixes
        # backends.
        return None
    ranges = _grouped_row_ranges(metadata)
    if int(forward_batch.batch_size) != len(ranges):
        raise SparseRepairAttentionError(
            "paged repair metadata and ForwardBatch size disagree"
        )
    if sum(end - start for start, end in ranges) != int(q.shape[0]):
        raise SparseRepairAttentionError(
            "paged repair row lengths do not match Q tokens"
        )
    metadata["grouped_attention_execution"] = "flashinfer_paged_prefill_rows"
    metadata["flashinfer_api"] = (
        "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper"
    )
    metadata["q_tokens"] = int(q.shape[0])
    metadata["kv_tokens"] = sum(end for _, end in ranges)
    metadata["max_kv_tokens"] = max(end for _, end in ranges)
    metadata["mask_semantics"] = (
        "paged_token_pages_native_suffix_causal_mask"
    )
    metadata["kv_layout"] = "TokenToKVPool_NHD_page_size_1"
    true_count, mask_elements, density = _repair_ranges_causal_density(ranges)
    metadata["mask_true_count"] = true_count
    metadata["mask_elements"] = mask_elements
    metadata["sparse_density"] = density
    metadata["layer_calls"] = int(metadata.get("layer_calls", 0)) + 1

    error = _compatibility_error(q, k, v, layer, forward_batch)
    if error is not None:
        raise SparseRepairAttentionError(error)
    with (
        _cpu_timing(metadata, "kv_buffer_lookup"),
        sparse_repair_component(forward_batch, "FlashInferKVBufferLookup"),
    ):
        key_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

    requested_workspace_bytes = int(metadata["workspace_bytes"])
    workspace_bytes = _paged_wrapper_workspace_bytes(requested_workspace_bytes)
    metadata["requested_workspace_bytes"] = requested_workspace_bytes
    metadata["workspace_bytes"] = workspace_bytes
    metadata["paged_wrapper_workspace_bytes"] = workspace_bytes
    if not metadata["_planned"]:
        try:
            build_started = time.perf_counter()
            paged_state, metadata_key = _paged_sparse_metadata_state(
                forward_batch, metadata, int(key_buffer.shape[0])
            )
            metadata["metadata_build_ms"] += (
                time.perf_counter() - build_started
            ) * 1000.0
            fixed_split_size = _positive_int_env(
                SPARSE_REPAIR_FIXED_SPLIT_SIZE_ENV, 128
            )
            plan_key = (
                "paged_prefill_rows",
                metadata_key,
                str(q.device),
                workspace_bytes,
                int(layer.tp_q_head_num),
                int(layer.tp_k_head_num),
                int(layer.head_dim),
                str(q.dtype),
                str(key_buffer.dtype),
                float(layer.scaling),
                fixed_split_size,
            )
            reuse_wrapper, wrapper_reuse_config_source = (
                _paged_wrapper_reuse_option(forward_batch)
            )
            wrapper_cache_key = (
                _paged_wrapper_cache_key(
                    q,
                    key_buffer,
                    layer,
                    workspace_bytes=workspace_bytes,
                    fixed_split_size=fixed_split_size,
                )
                if reuse_wrapper
                else plan_key
            )
            state = _WRAPPER_STATE_CACHE.get(wrapper_cache_key)
            wrapper_cache_hit = state is not None
            plan_cache_hit = state is not None and state.plan_key == plan_key
            metadata["wrapper_replan_reuse_enabled"] = reuse_wrapper
            metadata["wrapper_reuse_config_source"] = wrapper_reuse_config_source
            metadata["wrapper_reuse_scope"] = (
                "compatible_shape_replan" if reuse_wrapper else "exact_plan"
            )
            metadata["plan_cache_hit"] = plan_cache_hit
            metadata["plan_cache_miss"] = not plan_cache_hit
            metadata["wrapper_cache_hit"] = wrapper_cache_hit
            metadata["wrapper_cache_miss"] = not wrapper_cache_hit
            state_created = False
            if state is None:
                wrapper_started = time.perf_counter()
                with sparse_repair_component(
                    forward_batch, "FlashInferWrapperCreate"
                ):
                    state = _new_paged_wrapper_state(q.device, workspace_bytes)
                metadata["wrapper_create_ms"] += (
                    time.perf_counter() - wrapper_started
                ) * 1000.0
                state_created = True
            if not plan_cache_hit:
                plan_started = time.perf_counter()
                with sparse_repair_component(
                    forward_batch, "FlashInferSparseAttentionPlan"
                ):
                    state.wrapper.plan(
                        paged_state.qo_indptr,
                        paged_state.kv_indptr,
                        paged_state.indices,
                        paged_state.last_page_len,
                        int(layer.tp_q_head_num),
                        int(layer.tp_k_head_num),
                        int(layer.head_dim),
                        1,
                        causal=True,
                        pos_encoding_mode="NONE",
                        sm_scale=float(layer.scaling),
                        q_data_type=q.dtype,
                        kv_data_type=key_buffer.dtype,
                        o_data_type=q.dtype,
                        non_blocking=False,
                        fixed_split_size=fixed_split_size,
                    )
                metadata["plan_time"] = time.perf_counter() - plan_started
                if state_created:
                    _WRAPPER_STATE_CACHE.put(
                        wrapper_cache_key,
                        state,
                        workspace_bytes=workspace_bytes,
                    )
                state.plan_key = plan_key
            else:
                metadata["plan_time"] = 0.0
            metadata["indptr_length"] = int(paged_state.kv_indptr.numel())
            metadata["indices_length"] = int(paged_state.indices.numel())
            metadata["mask_shape"] = [int(mask_elements)]
            metadata["native_causal_mask"] = True
            metadata["fixed_split_size"] = fixed_split_size
            metadata["page_size"] = 1
            metadata["_planned"] = True
            metadata["_wrapper_state"] = state
            metadata["compatibility_status"] = "pass"
            metadata["flashinfer_backend"] = str(
                getattr(state.wrapper, "_backend", "fa2")
            )
            _PAGED_CAPACITY_MEMO.record_success(
                _paged_capacity_key_for(layer, q, key_buffer, workspace_bytes,
                                        fixed_split_size),
                int(q.shape[0]),
            )
        except Exception as exc:
            capacity = _is_capacity_exhaustion(exc)
            reason = f"FlashInfer paged plan failed: {type(exc).__name__}: {exc}"
            metadata["compatibility_status"] = (
                "plan_capacity_exhausted" if capacity else "plan_failed"
            )
            metadata["fallback_reason"] = reason
            if capacity:
                fixed_split_size = _positive_int_env(
                    SPARSE_REPAIR_FIXED_SPLIT_SIZE_ENV, 128
                )
                learned_cap = _PAGED_CAPACITY_MEMO.record_failure(
                    _paged_capacity_key_for(
                        layer, q, key_buffer, workspace_bytes, fixed_split_size
                    ),
                    int(q.shape[0]),
                )
                metadata["capacity_q_tokens"] = int(q.shape[0])
                metadata["capacity_learned_cap"] = learned_cap
                metadata["capacity_workspace_bytes"] = workspace_bytes
                raise SparseRepairCapacityError(reason) from exc
            raise SparseRepairAttentionError(reason) from exc
    else:
        state = metadata["_wrapper_state"]
        metadata["plan_reuse_calls"] = int(metadata["plan_reuse_calls"]) + 1

    if save_kv_cache:
        with sparse_repair_component(forward_batch, "SparseRepairKVWriteBack"):
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
    try:
        run_started = time.perf_counter()
        with sparse_repair_component(
            forward_batch, "FlashInferSparseAttentionRun"
        ):
            output = state.wrapper.run(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                (key_buffer.unsqueeze(1), value_buffer.unsqueeze(1)),
            )
        metadata["run_time"] += time.perf_counter() - run_started
        metadata["run_calls"] = int(metadata.get("run_calls", 0)) + 1
        metadata["selected_backend"] = "flashinfer_paged_prefill"
        metadata["actual_backend"] = "flashinfer_paged_prefill"
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)
    except Exception as exc:
        capacity = _is_capacity_exhaustion(exc)
        reason = f"FlashInfer paged run failed: {type(exc).__name__}: {exc}"
        metadata["compatibility_status"] = (
            "run_capacity_exhausted" if capacity else "run_failed"
        )
        metadata["fallback_reason"] = reason
        if capacity:
            metadata["capacity_q_tokens"] = int(q.shape[0])
            metadata["capacity_learned_cap"] = _PAGED_CAPACITY_MEMO.record_failure(
                _paged_capacity_key_for(
                    layer,
                    q,
                    key_buffer,
                    workspace_bytes,
                    _positive_int_env(SPARSE_REPAIR_FIXED_SPLIT_SIZE_ENV, 128),
                ),
                int(q.shape[0]),
            )
            raise SparseRepairCapacityError(reason) from exc
        raise SparseRepairAttentionError(reason) from exc


def _try_flashinfer_grouped_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    save_kv_cache: bool,
    metadata: dict[str, Any],
) -> torch.Tensor:
    if os.getenv(SPARSE_REPAIR_GROUPED_ROWS_ENV, "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise SparseRepairAttentionError(
            "FlashInfer grouped-row execution requires "
            f"{SPARSE_REPAIR_GROUPED_ROWS_ENV}=1"
        )
    if str(metadata.get("requested_backend")) != "flashinfer":
        raise SparseRepairAttentionError(
            "FlashInfer grouped-row execution must be forced; auto/hybrid is "
            "not allowed"
        )
    ranges = _grouped_row_ranges(metadata)
    if len(ranges) <= 1 or int(forward_batch.batch_size) != len(ranges):
        raise SparseRepairAttentionError(
            "grouped repair metadata and ForwardBatch size disagree"
        )
    lengths = [end - start for start, end in ranges]
    if sum(lengths) != int(q.shape[0]):
        raise SparseRepairAttentionError(
            "grouped repair row lengths do not match Q tokens"
        )

    grouped_mode = os.getenv(
        SPARSE_REPAIR_GROUPED_ATTENTION_MODE_ENV, "per_row"
    ).strip().lower()
    if grouped_mode not in {"per_row", "batched", "paged_batched"}:
        raise SparseRepairAttentionError(
            f"{SPARSE_REPAIR_GROUPED_ATTENTION_MODE_ENV} must be per_row, "
            f"batched, or paged_batched; got {grouped_mode!r}"
        )
    if grouped_mode == "paged_batched":
        return _try_flashinfer_paged_rows(
            q, k, v, layer, forward_batch, save_kv_cache, metadata
        )
    if grouped_mode == "batched":
        if len(set(lengths)) != 1:
            raise SparseRepairAttentionError(
                "batched FlashInfer grouped attention requires equal repair-row "
                "lengths"
            )
        metadata["grouped_attention_execution"] = (
            "flashinfer_batched_equal_repair_rows"
        )
        metadata["flashinfer_row_block_size"] = lengths[0]
        metadata["q_tokens"] = sum(lengths)
        metadata["kv_tokens"] = sum(end for _, end in ranges)
        metadata["max_kv_tokens"] = max(end for _, end in ranges)
        output = _try_flashinfer_sparse_attention_single(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache,
        )
        if output is None:
            raise SparseRepairAttentionError(
                "batched FlashInfer grouped attention attempted native fallback"
            )
        metadata["actual_backend"] = "flashinfer_block_sparse_grouped_batched"
        return output

    children = metadata.get("_grouped_row_metadata")
    if children is None:
        children = []
        inherited = {
            key: metadata[key]
            for key in (
                "batch_plan_group_id",
                "batch_plan_group_size",
                "batch_execution_unit_size",
                "batch_plan_padding_ratio",
                "batch_plan_execution_mode",
                "batch_plan_row_indices",
                "rowwise_ffn_enabled",
                "rowwise_output_projection_enabled",
                "row_batching_config_source",
                "flashinfer_row_batching_experiment",
            )
            if key in metadata
        }
        for row_index, (start, end) in enumerate(ranges):
            child = make_sparse_repair_metadata(
                forward_batch,
                start=start,
                end=end,
                block_count=int(metadata["block_count"]),
                block_size=int(metadata["block_size"]),
                repair_ranges=[(start, end)],
            )
            child.update(inherited)
            child["grouped_row_index"] = row_index
            child["grouped_row_count"] = len(ranges)
            child["grouped_attention_execution"] = "flashinfer_per_repair_row"
            children.append(child)
        metadata["_grouped_row_metadata"] = children

    outputs = []
    token_offset = 0
    for row_index, ((start, end), child) in enumerate(zip(ranges, children, strict=True)):
        length = end - start
        token_slice = slice(token_offset, token_offset + length)
        child_batch = _make_flashinfer_row_batch(
            forward_batch,
            row_index=row_index,
            token_offset=token_offset,
            start=start,
            end=end,
            metadata=child,
        )
        output = _try_flashinfer_sparse_attention_single(
            q[token_slice],
            k[token_slice],
            v[token_slice],
            layer,
            child_batch,
            save_kv_cache,
        )
        if output is None:
            raise SparseRepairAttentionError(
                f"FlashInfer grouped row {row_index} attempted native fallback"
            )
        outputs.append(output)
        token_offset += length

    metadata["actual_backend"] = "flashinfer_block_sparse_grouped_rows"
    metadata["compatibility_status"] = "pass"
    return torch.cat(outputs, dim=0)


def try_flashinfer_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    save_kv_cache: bool,
) -> torch.Tensor | None:
    metadata = getattr(forward_batch, "pic_sparse_repair_metadata", None)
    try:
        return _dispatch_flashinfer_sparse_attention(
            q, k, v, layer, forward_batch, save_kv_cache, metadata
        )
    except SparseRepairAttentionError as error:
        if metadata is None or not _is_capacity_exhaustion(error):
            raise
        if sparse_repair_capacity_policy() == "strict":
            raise
        # Capacity, not correctness. Native sparse-repair attention computes the
        # same logical mask, so degrading this request keeps the LegoLink result
        # exact while leaving the server available for the next request. The
        # learned ceiling recorded above makes later requests pre-split instead
        # of rediscovering the limit.
        metadata["actual_backend"] = "native"
        metadata["selected_backend"] = "native"
        metadata["capacity_degraded"] = True
        metadata["capacity_degraded_reason"] = str(error)
        metadata["decision_reason"] = "flashinfer_workspace_capacity_exhausted"
        return None


def _dispatch_flashinfer_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    save_kv_cache: bool,
    metadata: dict[str, Any] | None,
) -> torch.Tensor | None:
    grouped_mode = os.getenv(
        SPARSE_REPAIR_GROUPED_ATTENTION_MODE_ENV, "per_row"
    ).strip().lower()
    if (
        metadata
        and grouped_mode == "paged_batched"
        and os.getenv(SPARSE_REPAIR_GROUPED_ROWS_ENV, "0").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        return _try_flashinfer_paged_rows(
            q, k, v, layer, forward_batch, save_kv_cache, metadata
        )
    if (
        metadata
        and int(metadata.get("repair_row_count", 1)) > 1
        and int(getattr(forward_batch, "batch_size", 1)) > 1
    ):
        return _try_flashinfer_grouped_rows(
            q, k, v, layer, forward_batch, save_kv_cache, metadata
        )
    return _try_flashinfer_sparse_attention_single(
        q, k, v, layer, forward_batch, save_kv_cache
    )


def _try_flashinfer_sparse_attention_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    save_kv_cache: bool,
) -> torch.Tensor | None:
    metadata = getattr(forward_batch, "pic_sparse_repair_metadata", None)
    if not metadata:
        return None

    requested = str(metadata.get("requested_backend", "native"))
    hybrid_request_backend = metadata.get("hybrid_request_backend")
    if requested == "hybrid" and hybrid_request_backend == "native":
        metadata["selected_backend"] = "native"
        metadata["actual_backend"] = "native"
        metadata["selector_policy"] = "request_consistent_h100_7b_piecewise_v1"
        metadata["decision_reason"] = metadata.get(
            "hybrid_request_reason", "request_consistency_guard"
        )
        metadata["compatibility_status"] = "selector_native"
        return None
    if metadata.get("actual_backend") == "native":
        return None
    metadata["layer_calls"] = int(metadata.get("layer_calls", 0)) + 1

    start = int(metadata["repair_start"])
    end = int(metadata["repair_end"])
    if (
        metadata.get("grouped_attention_execution")
        == "flashinfer_batched_equal_repair_rows"
    ):
        true_count, mask_elements, density = _repair_ranges_causal_density(
            _grouped_row_ranges(metadata)
        )
        metadata["mask_semantics"] = (
            "bsr_multi_row_physical_slots_explicit_row_local_causal_mask"
        )
    else:
        true_count, mask_elements, density = _logical_causal_density(start, end)
    metadata["mask_true_count"] = true_count
    metadata["mask_elements"] = mask_elements
    metadata["sparse_density"] = density

    force_request_flashinfer = (
        requested == "hybrid"
        and hybrid_request_backend == "flashinfer_block_sparse"
    )
    if requested == "hybrid" and not force_request_flashinfer and (
        int(metadata["repair_tokens"]) > 512
        or density < 0.90
        or int(metadata["kv_tokens"]) < 3200
    ):
        decision = _ATTENTION_SELECTOR.select(
            repair_tokens=int(metadata["repair_tokens"]),
            kv_tokens=int(metadata["kv_tokens"]),
            density=density,
            block_count=int(metadata["block_count"]),
            plan_cache_hit=False,
            metadata_cache_hit=False,
        )
        metadata["selected_backend"] = decision.selected_backend
        metadata["actual_backend"] = decision.selected_backend
        metadata["selector_policy"] = decision.policy
        metadata["decision_reason"] = decision.reason
        metadata["native_estimated_ms"] = decision.native_estimated_ms
        metadata["flashinfer_estimated_ms"] = decision.flashinfer_estimated_ms
        metadata["compatibility_status"] = "selector_native"
        return None

    error = _compatibility_error(q, k, v, layer, forward_batch)
    if error is not None:
        metadata["compatibility_status"] = "runtime_incompatible"
        metadata["fallback_reason"] = error
        metadata["actual_backend"] = "native"
        if requested in {"flashinfer", "hybrid"}:
            raise SparseRepairAttentionError(error)
        return None

    with (
        _cpu_timing(metadata, "kv_buffer_lookup"),
        sparse_repair_component(forward_batch, "FlashInferKVBufferLookup"),
    ):
        key_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
    workspace_bytes = int(metadata["workspace_bytes"])
    if not metadata["_planned"]:
        try:
            sparse_state, metadata_key = _sparse_metadata_state(
                forward_batch, metadata, int(key_buffer.shape[0])
            )
            plan_key = _plan_key(
                metadata_key, q, key_buffer, layer, workspace_bytes
            )
            with (
                _cpu_timing(metadata, "wrapper_state_lookup_or_create"),
                sparse_repair_component(
                    forward_batch, "FlashInferWrapperStateLookupOrCreate"
                ),
            ):
                state = _WRAPPER_STATE_CACHE.get(plan_key)

            plan_cache_hit = state is not None
            metadata["plan_cache_hit"] = plan_cache_hit
            metadata["plan_cache_miss"] = not plan_cache_hit
            metadata["wrapper_cache_hit"] = plan_cache_hit
            metadata["wrapper_cache_miss"] = not plan_cache_hit

            if requested == "hybrid" and not force_request_flashinfer:
                decision = _ATTENTION_SELECTOR.select(
                    repair_tokens=int(metadata["repair_tokens"]),
                    kv_tokens=int(metadata["kv_tokens"]),
                    density=density,
                    block_count=int(metadata["block_count"]),
                    plan_cache_hit=plan_cache_hit,
                    metadata_cache_hit=bool(metadata["metadata_cache_hit"]),
                )
                metadata["selected_backend"] = decision.selected_backend
                metadata["actual_backend"] = decision.selected_backend
                metadata["flashinfer_api"] = (
                    "flashinfer.sparse.BlockSparseAttentionWrapper"
                    if decision.selected_backend == "flashinfer_block_sparse"
                    else None
                )
                metadata["selector_policy"] = decision.policy
                metadata["decision_reason"] = decision.reason
                metadata["native_estimated_ms"] = decision.native_estimated_ms
                metadata["flashinfer_estimated_ms"] = (
                    decision.flashinfer_estimated_ms
                )
                if decision.selected_backend == "native":
                    metadata["compatibility_status"] = "selector_native"
                    return None
            else:
                metadata["selected_backend"] = "flashinfer_block_sparse"
                metadata["actual_backend"] = "flashinfer_block_sparse"
                if force_request_flashinfer:
                    metadata["selector_policy"] = (
                        "request_consistent_h100_7b_piecewise_v1"
                    )
                    metadata["decision_reason"] = metadata.get(
                        "hybrid_request_reason",
                        "request_all_spans_select_flashinfer",
                    )

            if state is None:
                wrapper_started = time.perf_counter()
                with (
                    _cpu_timing(metadata, "wrapper_create"),
                    sparse_repair_component(
                        forward_batch, "FlashInferWrapperCreate"
                    ),
                ):
                    state = _new_wrapper_state(q.device, workspace_bytes)
                metadata["wrapper_create_ms"] += (
                    time.perf_counter() - wrapper_started
                ) * 1000.0
                plan_started = time.perf_counter()
                with (
                    _cpu_timing(metadata, "plan_total"),
                    sparse_repair_component(
                        forward_batch, "FlashInferSparseAttentionPlan"
                    ),
                ):
                    state.wrapper.plan(
                            sparse_state.indptr,
                            sparse_state.indices,
                            int(q.shape[0]),
                            int(key_buffer.shape[0]),
                            int(
                                metadata.get(
                                    "flashinfer_row_block_size", q.shape[0]
                                )
                            ),
                            1,
                            int(layer.tp_q_head_num),
                            int(layer.tp_k_head_num),
                            int(layer.head_dim),
                            mask=sparse_state.mask,
                            causal=False,
                            pos_encoding_mode="NONE",
                            sm_scale=float(layer.scaling),
                            q_data_type=q.dtype,
                            kv_data_type=key_buffer.dtype,
                            o_data_type=q.dtype,
                            non_blocking=False,
                        )
                metadata["plan_time"] = time.perf_counter() - plan_started
                _WRAPPER_STATE_CACHE.put(
                    plan_key,
                    state,
                    workspace_bytes=workspace_bytes,
                )
            else:
                metadata["plan_time"] = 0.0

            metadata["indptr_length"] = int(sparse_state.indptr.numel())
            metadata["indices_length"] = int(sparse_state.indices.numel())
            metadata["mask_shape"] = [
                int(dim) for dim in sparse_state.mask.shape
            ]
            metadata["_planned"] = True
            metadata["_wrapper_state"] = state
            metadata["compatibility_status"] = "pass"
            metadata["flashinfer_backend"] = str(
                getattr(state.wrapper, "_backend", "unknown")
            )
        except Exception as exc:
            capacity = _is_capacity_exhaustion(exc)
            reason = f"FlashInfer plan failed: {type(exc).__name__}: {exc}"
            metadata["compatibility_status"] = (
                "plan_capacity_exhausted" if capacity else "plan_failed"
            )
            metadata["fallback_reason"] = reason
            metadata["actual_backend"] = "native"
            if requested in {"flashinfer", "hybrid"}:
                if capacity:
                    raise SparseRepairCapacityError(reason) from exc
                raise SparseRepairAttentionError(reason) from exc
            return None
    else:
        state = metadata["_wrapper_state"]
        metadata["plan_reuse_calls"] = int(metadata["plan_reuse_calls"]) + 1

    cache_loc = forward_batch.out_cache_loc
    if save_kv_cache:
        with sparse_repair_component(forward_batch, "SparseRepairKVWriteBack"):
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

    try:
        run_started = time.perf_counter()
        with (
            _cpu_timing(metadata, "run_dispatch_total"),
            sparse_repair_component(forward_batch, "FlashInferSparseAttentionRun"),
        ):
            output = state.wrapper.run(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                key_buffer,
                value_buffer,
            )
        metadata["run_time"] += time.perf_counter() - run_started
        metadata["run_calls"] = int(metadata.get("run_calls", 0)) + 1
        with (
            _cpu_timing(metadata, "output_reshape"),
            sparse_repair_component(forward_batch, "FlashInferOutputReshape"),
        ):
            output = output.view(-1, layer.tp_q_head_num * layer.head_dim)
        return output
    except Exception as exc:
        capacity = _is_capacity_exhaustion(exc)
        reason = f"FlashInfer run failed: {type(exc).__name__}: {exc}"
        metadata["compatibility_status"] = (
            "run_capacity_exhausted" if capacity else "run_failed"
        )
        metadata["fallback_reason"] = reason
        metadata["actual_backend"] = "native"
        if requested in {"flashinfer", "hybrid"}:
            if capacity:
                raise SparseRepairCapacityError(reason) from exc
            raise SparseRepairAttentionError(reason) from exc
        return None


def finalize_sparse_repair_metadata(
    batches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded_batches = []
    for metadata in batches:
        children = metadata.get("_grouped_row_metadata")
        expanded_batches.extend(children if children is not None else [metadata])
    batches = expanded_batches
    event_pairs = [
        pair for metadata in batches for pair in metadata.get("_event_pairs", [])
    ]
    if event_pairs:
        event_pairs[-1][2].synchronize()
    output = []
    for metadata in batches:
        components: dict[str, float] = {}
        for name, start, end in metadata.get("_event_pairs", []):
            components[name] = components.get(name, 0.0) + float(
                start.elapsed_time(end)
            )
        public = {
            key: value for key, value in metadata.items() if not key.startswith("_")
        }
        public["component_cuda_ms"] = components
        if str(public.get("actual_backend", "")).startswith("flashinfer_"):
            public["actual_attention_gpu_ms"] = components.get(
                "FlashInferSparseAttentionRun", 0.0
            )
        else:
            public["actual_attention_gpu_ms"] = components.get(
                "NativeSparseAttention", 0.0
            )
        public["cache_snapshot"] = sparse_repair_cache_stats()
        output.append(public)
    _EVENT_POOL.reset()
    return output
