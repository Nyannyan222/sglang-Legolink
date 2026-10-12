"""Experimental EPIC-style position-independent KV linking for SGLang.

It snapshots independently compiled MHA KV blocks, copies them into a linked
request, remaps RoPE keys from block-local to global positions, and causally
recomputes selected contiguous spans with the regular SGLang extend-attention
path. The attention backend writes updated KV values back to the destination
slots. A tokenwise mode remains available as a correctness baseline.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


HOST_TIMING_SCHEMA_VERSION = 1
CUDA_EVENT_SCHEMA_VERSION = 2
PHYSICAL_TOKEN_LAYOUT_VERSION = "noc-single-chat-template-split-v1"
_PROFILE_RANGES_ENABLED = os.getenv(
    "SGLANG_PIC_PROFILE_RANGES_ENABLED", "0"
).lower() in ("1", "true", "yes", "on")
_NVTX_RANGES_ENABLED = os.getenv("SGLANG_PIC_NVTX_ENABLED", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_PROFILE_REGION_NAMES = {
    "SchedulerPrepareLink": "SGLang::request_prepare",
    "SemanticSelection": "SGLang::request_prepare",
    "LinkPlanBuild": "LegoLink::link_prepare",
    "LegoLinkLookup": "LegoLink::lookup",
    "AcquireSourceBlocks": "LegoLink::acquire",
    "DestinationKVAllocation": "LegoLink::destination_allocation",
    "LegoLinkKVFetch": "LegoLink::kv_copy",
    "LegoLinkRoPERemap": "LegoLink::rope_remap",
    "LegoLinkTritonFusion": "LegoLink::triton_fused_kv_link",
    "LegoLinkSparseRecompute": "LegoLink::sparse_recompute",
    "LegoLinkKVWriteBack": "LegoLink::writeback",
    "ReleaseSourceBlocks": "LegoLink::release",
    "NativeDecode": "SGLang::decode_iteration",
    "Sampling": "SGLang::sampling",
    "NativePrefill": "SGLang::prefill",
}
_CUDA_COMPONENT_ALIASES = {
    "model_prefill": "prefill_total_gpu",
    "kv_retrieval": "kv_retrieval_copy",
    "rope_remap": "rope_global_position_remap",
    "triton_fused_kv_link": "triton_fused_kv_link",
    "kv_writeback": "destination_kv_writeback",
    "sparse_recompute": "sparse_recompute_total",
}


_TRACE_DESCRIPTORS: dict[str, int] = {}
_TRACE_DESCRIPTOR_LOCK = threading.Lock()


def _trace_descriptor(path: str) -> int:
    """One open append descriptor per path, per process.

    The previous implementation paid os.makedirs, os.open and os.close on every
    event. At four events per generated token, across a bind mount, that was
    measured on 2026-09-02 as 24.6 s of a 42.6 s route -- 58% -- and it lands on
    the patched deployment alone, because stock SGLang has no trace call to
    make. Instrumentation that large does not observe the system, it replaces
    the thing being observed.

    Holding the descriptor keeps the write itself unchanged: still one O_APPEND
    write per record, still no user-space buffering. That matters more than the
    syscalls saved -- the workspace-capacity investigation read its evidence out
    of a trace whose scheduler had already died on SIGQUIT, and a buffer would
    have eaten exactly the records that explained the crash.
    """

    descriptor = _TRACE_DESCRIPTORS.get(path)
    if descriptor is not None:
        return descriptor
    with _TRACE_DESCRIPTOR_LOCK:
        descriptor = _TRACE_DESCRIPTORS.get(path)
        if descriptor is not None:
            return descriptor
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        _TRACE_DESCRIPTORS[path] = descriptor
        return descriptor


def _append_jsonl(path_template: str, record: dict[str, Any]) -> None:
    path = path_template.format(pid=os.getpid())
    payload = (
        json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    try:
        os.write(_trace_descriptor(path), payload)
    except OSError:
        # A descriptor can go stale -- the file replaced or removed under a
        # long-lived server. Reopen once; give up rather than let tracing take
        # the request down with it.
        with _TRACE_DESCRIPTOR_LOCK:
            stale = _TRACE_DESCRIPTORS.pop(path, None)
            if stale is not None:
                try:
                    os.close(stale)
                except OSError:
                    pass
        try:
            os.write(_trace_descriptor(path), payload)
        except OSError:
            pass


class _HostTimingRecorder:
    """Low-overhead, opt-in host timing for one scheduler/model request."""

    def __init__(self, stage: str, **context: Any):
        self.enabled = _env_flag("SGLANG_PIC_HOST_TIMING_ENABLED", "0")
        self.stage = str(stage)
        self.context = {
            key: value for key, value in context.items() if value is not None
        }
        self.durations_ns: dict[str, int] = {}
        self.started_ns = time.perf_counter_ns() if self.enabled else 0
        self.emitted = False

    @contextlib.contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add_ns(name, time.perf_counter_ns() - started_ns)

    def add_ns(self, name: str, elapsed_ns: int) -> None:
        if self.enabled:
            self.durations_ns[str(name)] = self.durations_ns.get(str(name), 0) + max(
                int(elapsed_ns), 0
            )

    def merge_ns(self, values: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for name, value in values.items():
            self.add_ns(str(name), int(value or 0))

    def update_context(self, **context: Any) -> None:
        if self.enabled:
            self.context.update(
                {key: value for key, value in context.items() if value is not None}
            )

    def emit(self, *, status: str = "ok", **details: Any) -> None:
        if not self.enabled or self.emitted:
            return
        self.emitted = True
        total_ns = max(time.perf_counter_ns() - self.started_ns, 0)
        durations_ns = {key: int(value) for key, value in self.durations_ns.items()}
        durations_ms = {key: value / 1_000_000.0 for key, value in durations_ns.items()}
        record = {
            "schema_version": HOST_TIMING_SCHEMA_VERSION,
            "timestamp_unix_s": time.time(),
            "event": "pic_host_timing",
            "pid": os.getpid(),
            "rank": os.getenv("LOCAL_RANK", os.getenv("RANK", "0")),
            "stage": self.stage,
            "status": str(status),
            "total_ns": total_ns,
            "total_ms": total_ns / 1_000_000.0,
            "durations_ns": durations_ns,
            "durations_ms": durations_ms,
            **self.context,
            **details,
        }
        path = os.getenv("SGLANG_PIC_HOST_TIMING_PATH") or os.getenv(
            "SGLANG_PIC_TRACE_PATH"
        )
        if path:
            _append_jsonl(path, record)


@contextlib.contextmanager
def _profile_range(name: str, **metadata: Any):
    """Emit matching PyTorch and optional NVTX ranges without synchronizing."""

    nvtx_enabled = _NVTX_RANGES_ENABLED and torch.cuda.is_available()
    range_enabled = nvtx_enabled or _PROFILE_RANGES_ENABLED
    if not range_enabled:
        yield
        return
    region_name = _PROFILE_REGION_NAMES.get(name, f"SGLang::{name}")
    active_context = globals().get("_ACTIVE_PROFILE_CONTEXT", {}) or {}
    bounded = {
        key: value
        for key, value in {**active_context, **metadata}.items()
        if key
        in {
            "request_id",
            "task_id",
            "variant",
            "phase",
            "concurrency",
            "k",
            "decode_iteration",
        }
        and value is not None
    }
    nvtx_name = region_name
    if bounded:
        nvtx_name += (
            "[" + ",".join(f"{key}={value}" for key, value in bounded.items()) + "]"
        )
    with _record_function(region_name):
        if nvtx_enabled:
            torch.cuda.nvtx.range_push(nvtx_name)
        try:
            yield
        finally:
            if nvtx_enabled:
                torch.cuda.nvtx.range_pop()


@dataclass
class _CudaEventSpan:
    label: str
    started: Any
    ended: Any
    phase: str = "unknown"
    decode_iteration: Optional[int] = None
    completed: bool = False


class _CudaEventPairPool:
    def __init__(self):
        self._available: list[tuple[Any, Any]] = []

    def acquire(self) -> tuple[Any, Any]:
        if self._available:
            return self._available.pop()
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def release(self, pair: tuple[Any, Any]) -> None:
        self._available.append(pair)


class _CudaEventRecorder:
    def __init__(
        self,
        *,
        context: Optional[dict[str, Any]] = None,
        warmup: bool = False,
    ):
        self.enabled = (
            _env_flag("SGLANG_PIC_CUDA_EVENTS", "0") and torch.cuda.is_available()
        )
        self.spans: list[_CudaEventSpan] = []
        self.pending: list[_CudaEventSpan] = []
        self.finalized = False
        self.summary: dict[str, float] = {}
        self.context = {
            key: value for key, value in (context or {}).items() if value is not None
        }
        self.warmup = bool(warmup)
        self.errors: list[str] = []

    def update_context(self, **context: Any) -> None:
        self.context.update(
            {key: value for key, value in context.items() if value is not None}
        )

    def start(
        self,
        label: str,
        *,
        phase: Optional[str] = None,
        decode_iteration: Optional[int] = None,
    ) -> Optional[_CudaEventSpan]:
        if not self.enabled:
            return None
        pair: Optional[tuple[Any, Any]] = None
        try:
            pair = _CUDA_EVENT_PAIR_POOL.acquire()
            started, ended = pair
            started.record()
            span = _CudaEventSpan(
                label=label,
                started=started,
                ended=ended,
                phase=str(phase or self.context.get("phase") or "unknown"),
                decode_iteration=decode_iteration,
            )
            self.pending.append(span)
            return span
        except Exception as error:
            self.errors.append(f"start {label}: {type(error).__name__}: {error}")
            if pair is not None:
                _CUDA_EVENT_PAIR_POOL.release(pair)
            return None

    def end(self, span: Optional[_CudaEventSpan]) -> None:
        if span is None or span.completed:
            return
        try:
            span.ended.record()
            span.completed = True
            if span in self.pending:
                self.pending.remove(span)
            self.spans.append(span)
        except Exception as error:
            self.errors.append(f"end {span.label}: {type(error).__name__}: {error}")
            if span in self.pending:
                self.pending.remove(span)
            _CUDA_EVENT_PAIR_POOL.release((span.started, span.ended))

    def _emit_component(
        self,
        *,
        component: str,
        elapsed_ms: Optional[float],
        phase: str,
        decode_iteration: Optional[int] = None,
        status: str = "ok",
        derived: bool = False,
        error: Optional[str] = None,
    ) -> None:
        context = {
            key: value
            for key, value in self.context.items()
            if key not in {"phase", "decode_iteration"}
        }
        _emit_pic_trace(
            "cuda_event_component",
            cuda_event_schema_version=CUDA_EVENT_SCHEMA_VERSION,
            measurement_kind="diagnostic_cuda_event",
            profiling_type="cuda_event",
            formal_latency=False,
            diagnostic=True,
            component=component,
            phase=phase,
            decode_iteration=decode_iteration,
            elapsed_ms=elapsed_ms,
            warmup=self.warmup,
            status=status,
            derived=derived,
            error=error,
            **context,
        )

    def finalize(self) -> dict[str, float]:
        if self.finalized:
            return self.summary
        self.finalized = True
        try:
            if self.spans:
                # All pairs are recorded on the request's model stream. One wait
                # on the final event validates every preceding pair. There is no
                # per-component synchronize.
                self.spans[-1].ended.synchronize()
            for span in self.spans:
                try:
                    elapsed_ms = float(span.started.elapsed_time(span.ended))
                    self.summary[span.label] = (
                        self.summary.get(span.label, 0.0) + elapsed_ms
                    )
                    canonical = _CUDA_COMPONENT_ALIASES.get(span.label, span.label)
                    if canonical != span.label:
                        self.summary[canonical] = (
                            self.summary.get(canonical, 0.0) + elapsed_ms
                        )
                    self._emit_component(
                        component=canonical,
                        elapsed_ms=elapsed_ms,
                        phase=span.phase,
                        decode_iteration=span.decode_iteration,
                    )
                except Exception as error:
                    message = f"elapsed {span.label}: {type(error).__name__}: {error}"
                    self.errors.append(message)
                    self._emit_component(
                        component=_CUDA_COMPONENT_ALIASES.get(span.label, span.label),
                        elapsed_ms=None,
                        phase=span.phase,
                        decode_iteration=span.decode_iteration,
                        status="error",
                        error=message,
                    )
                finally:
                    _CUDA_EVENT_PAIR_POOL.release((span.started, span.ended))
        except Exception as error:
            self.errors.append(f"finalize: {type(error).__name__}: {error}")
            logger.warning(
                "CUDA Event diagnostics failed without aborting inference: %s", error
            )
        finally:
            for span in list(self.pending):
                _CUDA_EVENT_PAIR_POOL.release((span.started, span.ended))
                self._emit_component(
                    component=_CUDA_COMPONENT_ALIASES.get(span.label, span.label),
                    elapsed_ms=None,
                    phase=span.phase,
                    decode_iteration=span.decode_iteration,
                    status="incomplete",
                    error="event pair was not completed",
                )
            self.pending.clear()
        total = self.summary.get("legolink_total_gpu")
        if total is not None:
            measured = sum(
                self.summary.get(component, 0.0)
                for component in (
                    "kv_retrieval_copy",
                    "rope_global_position_remap",
                    "triton_fused_kv_link",
                    "sparse_recompute_total",
                    "destination_kv_writeback",
                    "mutable_logits",
                )
            )
            residual = max(total - measured, 0.0)
            self.summary["legolink_residual_gpu"] = residual
            self._emit_component(
                component="legolink_residual_gpu",
                elapsed_ms=residual,
                phase="link",
                derived=True,
            )
        return self.summary


_CUDA_EVENT_PAIR_POOL = _CudaEventPairPool()
_ACTIVE_CUDA_EVENT_RECORDER: Optional[_CudaEventRecorder] = None
_ACTIVE_PROFILE_CONTEXT: dict[str, Any] = {}
_CUDA_EVENT_REQUEST_COUNT = 0
_NATIVE_DECODE_ITERATION_COUNT = 0



def _debug_capture_post_remap_k(pool, destination_slots, full_input_ids, link_plan, context):
    """DEBUG ONLY. Dump post-remap cached K for a few rows, then return.

    Off unless SGLANG_PIC_DEBUG_POST_REMAP_K names an output file. When off this
    returns before touching a tensor, so ordinary requests are unaffected. It
    only reads: every tensor is detached, copied to CPU and never written back.

    Exists to answer one question -- does the K the runtime has after gather and
    RoPE remap equal the K an offline rotation of the cached block predicts --
    at the earliest point where the two can be compared.
    """

    path = os.getenv("SGLANG_PIC_DEBUG_POST_REMAP_K", "").strip()
    if not path:
        return
    try:
        raw = os.getenv("SGLANG_PIC_DEBUG_POST_REMAP_K_POSITIONS", "").strip()
        wanted = [int(value) for value in raw.split(",") if value.strip()]
        layers = [
            int(value)
            for value in os.getenv(
                "SGLANG_PIC_DEBUG_POST_REMAP_K_LAYERS", "0"
            ).split(",")
            if value.strip()
        ]
        block_ranges = link_plan.get("block_ranges") or []
        samples = []
        for position in wanted:
            if position < 0 or position >= len(destination_slots):
                continue
            slot = int(destination_slots[position].item())
            owner = next(
                (
                    entry
                    for entry in block_ranges
                    if int(entry["start_token"]) <= position < int(entry["end_token"])
                ),
                None,
            )
            for layer_id in layers:
                key = pool.get_key_buffer(layer_id)[slot].detach().to("cpu")
                samples.append(
                    {
                        "layer": layer_id,
                        "global_position": position,
                        "local_position": (
                            position - int(owner["start_token"]) if owner else None
                        ),
                        "cache_key": owner.get("cache_key") if owner else None,
                        "semantic_block_id": owner.get("block_id") if owner else None,
                        "token_id": (
                            int(full_input_ids[position])
                            if position < len(full_input_ids)
                            else None
                        ),
                        "destination_slot": slot,
                        "dtype": str(key.dtype),
                        "shape": list(key.shape),
                        "k": key.float().tolist(),
                    }
                )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"checkpoint": "post_remap_k", **context, "samples": samples})
                + "\n"
            )
    except Exception as error:  # never let a debug dump break a request
        logger.warning("post-remap K capture skipped: %s", error)


def _debug_capture_post_repair_kv(
    pool, destination_slots, full_input_ids, link_plan, repaired_positions, context
):
    """DEBUG ONLY. Dump post-repair cached K and V for a few rows, then return.

    Off unless SGLANG_PIC_DEBUG_POST_REPAIR_KV names an output file; when off it
    returns before touching a tensor. Read-only: every tensor is detached,
    copied to CPU, and never written back.

    The post-remap checkpoint answers whether linking put the right angles in
    the right slots. This one answers the next question -- what the cache
    actually holds once sparse repair has run -- which is where a multi-block
    partial-repair failure has to show up if it is a KV problem at all. Each row
    is labelled REPAIRED or STALE from the runtime's own recompute positions
    rather than inferred from the tensor, so a row cannot be mislabelled by the
    very divergence being measured.
    """

    path = os.getenv("SGLANG_PIC_DEBUG_POST_REPAIR_KV", "").strip()
    if not path:
        return
    try:
        raw = os.getenv("SGLANG_PIC_DEBUG_POST_REPAIR_KV_POSITIONS", "").strip()
        wanted = [int(value) for value in raw.split(",") if value.strip()]
        layers = [
            int(value)
            for value in os.getenv(
                "SGLANG_PIC_DEBUG_POST_REPAIR_KV_LAYERS", "0"
            ).split(",")
            if value.strip()
        ]
        repaired = set(int(value) for value in (repaired_positions or ()))
        block_ranges = link_plan.get("block_ranges") or []
        samples = []
        for position in wanted:
            if position < 0 or position >= len(destination_slots):
                continue
            slot = int(destination_slots[position].item())
            owner = next(
                (
                    entry
                    for entry in block_ranges
                    if int(entry["start_token"]) <= position < int(entry["end_token"])
                ),
                None,
            )
            for layer_id in layers:
                key = pool.get_key_buffer(layer_id)[slot].detach().to("cpu")
                value = pool.get_value_buffer(layer_id)[slot].detach().to("cpu")
                samples.append(
                    {
                        "layer": layer_id,
                        "global_position": position,
                        "local_position": (
                            position - int(owner["start_token"]) if owner else None
                        ),
                        "cache_key": owner.get("cache_key") if owner else None,
                        "block_start_token": (
                            int(owner["start_token"]) if owner else None
                        ),
                        "status": "REPAIRED" if position in repaired else "STALE",
                        "token_id": (
                            int(full_input_ids[position])
                            if position < len(full_input_ids)
                            else None
                        ),
                        "destination_slot": slot,
                        "dtype": str(key.dtype),
                        "shape": list(key.shape),
                        "k": key.float().tolist(),
                        "v": value.float().tolist(),
                    }
                )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"checkpoint": "post_repair_kv", **context, "samples": samples}
                )
                + "\n"
            )
    except Exception as error:  # never let a debug dump break a request
        logger.warning("post-repair KV capture skipped: %s", error)


class PICRuntimeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# The pic_link request contract.
#
# `pic_link` reaches the server as an untyped dict, so until this existed a
# misspelt key was accepted in silence and three host builders had drifted into
# three different field sets. The split below is the contract: the host
# describes which semantic blocks to reuse and under what repair policy, and
# this process derives every physical consequence of that choice.
#
# The division is not stylistic. `_prepare_link` already overwrote every
# physical field a host sent, so a host that computed them was writing into a
# value nothing would read -- which is worse than an error, because a stale
# `imp_indices` in a saved plan file looked like it was still steering the run.
# ---------------------------------------------------------------------------

#: Identity and telemetry labels. Never affect what is computed.
PIC_LINK_IDENTITY_FIELDS = frozenset(
    {
        "request_id",
        "task_id",
        "variant",
        "profiling_warmup",
        "concurrency",
    }
)

#: The reuse intent itself: which compiled blocks, in which order.
PIC_LINK_SELECTION_FIELDS = frozenset(
    {
        "cache_keys",
        "block_ids",
    }
)

#: Which compiled blocks this request is allowed to bind to. These qualify the
#: RadixCache lookup key, so a Link only matches a Compile that declared the
#: same values -- see LegoLinkBlockKey.
PIC_LINK_COMPATIBILITY_FIELDS = frozenset(
    {
        "adapter_id",
        "segmentation_policy_version",
        "legolink_policy_version",
    }
)

#: Sparse-repair policy. The host chooses how much of each block to recompute;
#: this process turns that choice into positions.
PIC_LINK_REPAIR_POLICY_FIELDS = frozenset(
    {
        "recompute_mode",
        "recompute_initial_tokens",
        "recompute_first_block",
        "recompute_first_block_tokens",
        "recompute_tokens_per_block",
    }
)

#: The exact token sequence, when the host owns tokenization.
#:
#: These name physical tokens but are not a physical *plan*: they are the host
#: asserting "this is what the chat template produced", and every one of them is
#: checked here against the tokens actually held in the registry before it is
#: used. An override that disagrees raises. Position, offset and slot decisions
#: remain this process's alone.
PIC_LINK_TOKEN_LAYOUT_FIELDS = frozenset(
    {
        "physical_token_layout_version",
        "physical_input_ids",
        "physical_token_count",
        "physical_token_ids_sha256",
        "physical_prefix_token_count",
        "physical_block_segments",
        "expected_complete_token_count",
        "expected_complete_token_ids_sha256",
        # EXPERIMENTAL, see `indexed_mixed_prefill_enabled`.
        "physical_interblock_fresh_token_counts",
    }
)

#: Protected diagnostic switches, read only by the row-batching investigation.
PIC_LINK_DIAGNOSTIC_FIELDS = frozenset(
    {
        "diagnostic_flashinfer_reuse_paged_wrapper_across_plans",
        "diagnostic_flashinfer_skip_exact_paged_metadata_cache",
        "diagnostic_sparse_repair_record_function",
        "diagnostic_sparse_repair_row_batching",
        "diagnostic_sparse_repair_rowwise_ffn",
        "diagnostic_sparse_repair_rowwise_output_projection",
    }
)

#: EXPERIMENTAL, off by default. The host may ask that a successfully
#: reconstructed execution group be promoted into RadixCache under a
#: content-derived group key. Both fields are semantic intent -- which members
#: were grouped, and what the group is called. Every physical decision (slots,
#: positions, ownership, when to refuse) stays in this process, which is why
#: the key is accepted but the slots are not.
PIC_LINK_EXPERIMENTAL_PROMOTION_FIELDS = frozenset(
    {
        "promote_reconstructed_group",
        "execution_group_key",
    }
)

#: Accepted but superseded. recompute_initial_tokens says the same thing.
PIC_LINK_DEPRECATED_ALIASES = frozenset({"legolink_k"})

#: Everything a host may send.
PIC_LINK_GOVERNANCE_FIELDS = frozenset({"execution_mode", "framing_spec", "repair_decision"})

PIC_LINK_REQUEST_FIELDS = (
    PIC_LINK_GOVERNANCE_FIELDS |
    PIC_LINK_IDENTITY_FIELDS
    | PIC_LINK_SELECTION_FIELDS
    | PIC_LINK_COMPATIBILITY_FIELDS
    | PIC_LINK_REPAIR_POLICY_FIELDS
    | PIC_LINK_TOKEN_LAYOUT_FIELDS
    | PIC_LINK_DIAGNOSTIC_FIELDS
    | PIC_LINK_EXPERIMENTAL_PROMOTION_FIELDS
    | PIC_LINK_DEPRECATED_ALIASES
)

#: Derived here, from the registry. A host that sends one of these is describing
#: a layout it cannot know -- block token counts live in this process -- so the
#: value is refused rather than silently replaced.
PIC_LINK_RUNTIME_OWNED_FIELDS = frozenset(
    {
        "attention_ranges",
        "block_ranges",
        "imp_indices",
        "kvlink",
        "mutable_start_token",
        "mutable_token_count",
        "recompute_positions",
        "token_layout_source",
        "total_token_count",
    }
)

#: Written by the scheduler into the same dict after it owns the request.
PIC_LINK_INTERNAL_FIELDS = frozenset({"_scheduler_reserved_cache_keys"})

#: Largest value any integer `pic_link` count may carry. Far above any real
#: token count and far below the point where a Python int stops being a
#: plain machine integer downstream (range bounds, tensor shapes, traces).
PIC_LINK_MAX_INTEGER = 2**31 - 1

#: The sparse-repair recompute modes `PICRuntime.run_link` implements.
PIC_LINK_RECOMPUTE_MODES = frozenset({"batched", "tokenwise", "full_causal"})


def _link_field_error(name: str, problem: str) -> PICRuntimeError:
    return PICRuntimeError(f"pic_link.{name} {problem}.")


def _link_int(
    link_plan: dict[str, Any],
    name: str,
    *,
    nullable: bool = False,
    minimum: int = 0,
    maximum: int = PIC_LINK_MAX_INTEGER,
) -> Optional[int]:
    """A present integer field, strictly: no bool, float, string or list."""

    if name not in link_plan:
        return None
    value = link_plan[name]
    if value is None:
        if nullable:
            return None
        raise _link_field_error(name, "must not be null")
    if isinstance(value, bool) or not isinstance(value, int):
        raise _link_field_error(name, "must be an integer")
    if value < minimum or value > maximum:
        raise _link_field_error(name, f"must be between {minimum} and {maximum}")
    return value


def _link_optional_str(link_plan: dict[str, Any], name: str) -> None:
    value = link_plan.get(name)
    if value is not None and not isinstance(value, str):
        raise _link_field_error(name, "must be a string when present")


def _link_optional_bool(link_plan: dict[str, Any], name: str) -> None:
    value = link_plan.get(name)
    if value is not None and not isinstance(value, bool):
        raise _link_field_error(name, "must be a boolean when present")


def _link_label(link_plan: dict[str, Any], name: str) -> None:
    """Identity and telemetry labels: a string or an integer, never a container."""

    value = link_plan.get(name)
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise _link_field_error(name, "must be a string or an integer when present")


def _link_string_list(
    link_plan: dict[str, Any],
    name: str,
    *,
    required: bool = False,
    nullable: bool = True,
    at_most: Optional[int] = None,
) -> Optional[list[str]]:
    if name not in link_plan:
        if required:
            raise _link_field_error(name, "must be non-empty")
        return None
    value = link_plan[name]
    if value is None:
        if required or not nullable:
            raise _link_field_error(name, "must not be null")
        return None
    if not isinstance(value, list):
        raise _link_field_error(name, "must be a list of strings")
    if required and not value:
        raise _link_field_error(name, "must be non-empty")
    if any(not isinstance(item, str) for item in value):
        raise _link_field_error(name, "must contain only strings")
    if at_most is not None and len(value) > at_most:
        raise _link_field_error(name, f"may hold at most {at_most} entries")
    return value


def validate_pic_link_field_types(link_plan: dict[str, Any]) -> None:
    """Type- and range-check every value a host may send in `pic_link`.

    `pic_link` is copied from the HTTP body into the scheduler unchanged, and
    `_prepare_link` applies `int()`, `len()` and indexing to what it finds. A
    value of the wrong shape therefore used to surface as a TypeError,
    ValueError, OverflowError or KeyError -- none of which the scheduler turns
    into a per-request abort -- so one malformed request ended the shared
    process. Every failure here is a PICRuntimeError, which admission already
    reports as a rejection of that one request.

    Presence rules are the ones `_prepare_link` always had (an absent optional
    field means its default); what changes is that a value which is present
    must be what the field says it is. Governance (`execution_mode`,
    `framing_spec`, `repair_decision`) is validated by `validate_execution_intent`;
    token-layout fields are checked against the registry's tokens where they
    are used; diagnostic switches keep their existing per-request gate.
    """

    cache_keys = _link_string_list(link_plan, "cache_keys", required=True) or []

    for name in ("request_id", "task_id"):
        _link_label(link_plan, name)
    _link_optional_str(link_plan, "variant")
    _link_optional_bool(link_plan, "profiling_warmup")
    _link_int(link_plan, "concurrency", nullable=True)

    _link_string_list(link_plan, "block_ids", at_most=len(cache_keys))

    for name in (
        "adapter_id",
        "segmentation_policy_version",
        "legolink_policy_version",
        "execution_group_key",
    ):
        _link_optional_str(link_plan, name)
    _link_optional_bool(link_plan, "promote_reconstructed_group")

    if "recompute_mode" in link_plan:
        mode = link_plan["recompute_mode"]
        if not isinstance(mode, str):
            raise _link_field_error("recompute_mode", "must be a string")
        if mode.lower() not in PIC_LINK_RECOMPUTE_MODES:
            raise _link_field_error(
                "recompute_mode",
                "must be one of " + ", ".join(sorted(PIC_LINK_RECOMPUTE_MODES)),
            )

    # `recompute_initial_tokens` and its alias are read with `int()` and
    # compared; an explicit null is not "use the default", it is a TypeError.
    _link_int(link_plan, "recompute_initial_tokens")
    _link_int(link_plan, "legolink_k")
    _link_optional_bool(link_plan, "recompute_first_block")
    _link_int(link_plan, "recompute_first_block_tokens", nullable=True)

    if "recompute_tokens_per_block" in link_plan:
        windows = link_plan["recompute_tokens_per_block"]
        if windows is not None:
            if not isinstance(windows, list):
                raise _link_field_error(
                    "recompute_tokens_per_block", "must be a list of integers"
                )
            if len(windows) != len(cache_keys):
                raise _link_field_error(
                    "recompute_tokens_per_block",
                    "must contain one value per cache key",
                )
            for window in windows:
                if (
                    isinstance(window, bool)
                    or not isinstance(window, int)
                    or window < 0
                    or window > PIC_LINK_MAX_INTEGER
                ):
                    raise _link_field_error(
                        "recompute_tokens_per_block",
                        "must contain only non-negative integers within range",
                    )


def group_promotion_enabled() -> bool:
    """EXPERIMENTAL. Whether a reconstructed execution group may be promoted
    into RadixCache.

    Two independent switches, on purpose: the deployment enables the code path
    and the request asks for it. Neither alone promotes anything, so the
    default deployment cannot be talked into persisting a reconstructed state
    by a request field alone.
    """

    return _env_flag("SGLANG_PIC_EXPERIMENTAL_GROUP_PROMOTION", "0")


def indexed_mixed_prefill_enabled() -> bool:
    """EXPERIMENTAL. Whether a Link may carry fresh tokens *between* cached
    blocks, not only before the first and after the last.

    The indexed exact sub-context path finds cached chunks wherever they sit in
    a query, so a request can be `fresh | cached | fresh | cached | fresh`. The
    host sends only how many of its own tokens form each gap; which slots they
    occupy, their positions, and the RoPE delta of every block after them stay
    here. Off unless the deployment sets
    SGLANG_PIC_EXPERIMENTAL_INDEXED_MIXED_PREFILL=1.
    """

    return _env_flag("SGLANG_PIC_EXPERIMENTAL_INDEXED_MIXED_PREFILL", "0")


def _split_interblock_fresh_tokens(
    link_plan: dict[str, Any],
    post_prefix_input_ids: list[int],
    block_count: int,
) -> tuple[list[list[int]], list[int]]:
    """Carve the host's fresh gaps off the front of the post-prefix Link input.

    The Link input is every host-owned token in order: the fresh prefix, then
    the fresh run after block 0, after block 1, ..., then the mutable suffix.
    `physical_interblock_fresh_token_counts` says how long each between-block
    run is. Without the field every run is empty, which is the layout this
    runtime always had.
    """

    raw = link_plan.get("physical_interblock_fresh_token_counts")
    if raw is None:
        return [[] for _ in range(max(block_count - 1, 0))], list(post_prefix_input_ids)
    if not indexed_mixed_prefill_enabled():
        raise PICRuntimeError(
            "pic_link.physical_interblock_fresh_token_counts is experimental; "
            "set SGLANG_PIC_EXPERIMENTAL_INDEXED_MIXED_PREFILL=1 to allow fresh "
            "tokens between cached blocks."
        )
    if (
        not isinstance(raw, list)
        or len(raw) != block_count - 1
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in raw)
    ):
        raise PICRuntimeError(
            "pic_link.physical_interblock_fresh_token_counts must hold one "
            "non-negative token count between each pair of consecutive blocks."
        )
    if link_plan.get("promote_reconstructed_group") and any(raw):
        raise PICRuntimeError(
            "A Link with fresh tokens between cached blocks is not an execution "
            "group and cannot be promoted as one."
        )
    runs: list[list[int]] = []
    cursor = 0
    for count in raw:
        runs.append(list(post_prefix_input_ids[cursor : cursor + count]))
        cursor += count
    if cursor >= len(post_prefix_input_ids):
        raise PICRuntimeError(
            "pic_link.physical_interblock_fresh_token_counts must leave a "
            "non-empty mutable suffix."
        )
    return runs, list(post_prefix_input_ids[cursor:])


def legacy_link_fields_allowed() -> bool:
    """Whether a host-sent physical field downgrades to a warning.

    Plan files written before the contract existed carry block_ranges and
    imp_indices. Set SGLANG_PIC_ALLOW_LEGACY_LINK_FIELDS=1 to replay one
    without regenerating it; the values are still discarded.
    """

    return _env_flag("SGLANG_PIC_ALLOW_LEGACY_LINK_FIELDS", "0")


def validate_pic_link_request(link_plan: dict[str, Any]) -> None:
    """Reject anything the host is not entitled to send.

    Called once, at request arrival, before this process writes its own fields
    into the same dict.
    """

    if not isinstance(link_plan, dict):
        raise PICRuntimeError("pic_link must be an object.")

    # Shapes first: the governance check below reads cache_keys with len().
    validate_pic_link_field_types(link_plan)
    validate_runtime_execution_intent(link_plan)
    physical = sorted(set(link_plan) & PIC_LINK_RUNTIME_OWNED_FIELDS)
    if physical:
        if link_plan.get("execution_mode") == "formal" or not legacy_link_fields_allowed():
            raise PICRuntimeError(
                "pic_link carries fields the runtime derives and the host "
                f"cannot know: {', '.join(physical)}. Send the semantic "
                "selection only; set SGLANG_PIC_ALLOW_LEGACY_LINK_FIELDS=1 to "
                "replay a pre-contract plan file."
            )
        for field_name in physical:
            link_plan.pop(field_name, None)
        _emit_pic_trace(
            "pic_link_legacy_physical_fields_discarded",
            phase="link",
            request_id=link_plan.get("request_id"),
            task_id=link_plan.get("task_id"),
            fields=physical,
        )

    unknown = sorted(
        set(link_plan)
        - PIC_LINK_REQUEST_FIELDS
        - PIC_LINK_RUNTIME_OWNED_FIELDS
        - PIC_LINK_INTERNAL_FIELDS
    )
    if unknown:
        raise PICRuntimeError(
            f"pic_link has unknown fields: {', '.join(unknown)}. "
            "Host correctness evidence and repair-window rationale belong in "
            "the host audit record, not in the runtime request."
        )


def validate_execution_intent(wire, profile=None, *, require_formal=False):
    """Validate policy, then consistency. `profile` is deployment-owned trust.

    canonical_input_hashes are reviewed canonical input evidence, never learned
    from the incoming request. An identity label plus a matching hash is not a
    framing proof. Admission requires that input in the installed profile too.
    """
    import math

    mode = wire.get("execution_mode", "legacy_debug")
    if mode not in ("formal", "legacy_debug"):
        raise ValueError("unknown execution_mode")
    if require_formal and mode != "formal":
        raise ValueError("formal deployment rejects legacy_debug execution")
    formal = require_formal or mode == "formal"
    framing = wire.get("framing_spec")
    repair = wire.get("repair_decision")
    if not formal and framing is None and repair is None:
        return
    required_framing = {"framing_layout_version", "model_id", "tokenizer_fingerprint",
                        "chat_template_sha256", "profile_id"}
    if not isinstance(framing, dict) or set(framing) != required_framing:
        raise ValueError("missing or malformed framing_spec")
    if any(not isinstance(v, str) or not v.strip() for v in framing.values()):
        raise ValueError("framing identity must be nonempty")
    version = framing["framing_layout_version"]
    if version != "noc-single-chat-template-split-v1":
        raise ValueError("unsupported/legacy framing version")
    template = framing["chat_template_sha256"]
    if len(template) != 64 or any(c not in "0123456789abcdef" for c in template):
        raise ValueError("invalid chat template hash")
    if wire.get("physical_token_layout_version") != version:
        raise ValueError("framing and physical layout version disagree")
    fields = {"selected_k", "windows", "policy", "policy_version",
              "framing_layout_version", "quality_profile_id", "evidence_id",
              "reason", "workload_class"}
    if not isinstance(repair, dict) or set(repair) != fields:
        raise ValueError("missing or malformed RepairDecision")
    k, windows = repair["selected_k"], repair["windows"]
    if type(k) is not int or k < 0:
        raise ValueError("selected_k must be a nonnegative integer")
    if (not isinstance(windows, list) or len(windows) != len(wire.get("cache_keys", []))
            or not windows or any(type(v) is not int or v < 0 or v > k for v in windows)):
        raise ValueError("invalid repair window vector")
    if repair["policy_version"] != "repair-decision-v1":
        raise ValueError("unsupported repair policy version")
    if repair["framing_layout_version"] != version:
        raise ValueError("repair/framing incompatibility")
    if repair["policy"] not in ("quality_guarded", "fixed_unvalidated_debug"):
        raise ValueError("unknown repair policy")
    if repair["quality_profile_id"] is not None and not isinstance(
        repair["quality_profile_id"], str
    ):
        # Used as a dict key below; a list or dict would raise TypeError.
        raise ValueError("quality_profile_id must be a string or null")
    if not all(isinstance(repair[x], str) and repair[x].strip()
               for x in ("reason", "workload_class")):
        raise ValueError("missing repair provenance")
    if type(wire.get("recompute_initial_tokens")) is not int or wire["recompute_initial_tokens"] != k:
        raise ValueError("repair K was replaced or defaulted")
    if wire.get("recompute_tokens_per_block") != windows:
        raise ValueError("repair windows were replaced or defaulted")
    if (wire.get("recompute_first_block_tokens") != windows[0]
            or wire.get("recompute_first_block") is not bool(windows[0])):
        raise ValueError("first-block repair disagrees with decision")
    if "legolink_k" in wire and wire["legolink_k"] != k:
        raise ValueError("legacy K alias disagrees with decision")
    if formal or repair["policy"] == "quality_guarded":
        if repair["policy"] != "quality_guarded":
            raise ValueError("formal execution requires measured repair quality")
        if not isinstance(profile, dict) or profile.get("framing_spec") != framing:
            raise ValueError("missing/incompatible deployment framing profile")
        if wire.get("expected_complete_token_ids_sha256") not in profile.get("canonical_input_hashes", []):
            raise ValueError("input has no reviewed canonical framing evidence")
        quality = profile.get("repair_quality_profiles", {}).get(repair["quality_profile_id"])
        if not isinstance(quality, dict):
            raise ValueError("missing repair quality profile")
        for name in ("selected_k", "windows", "workload_class", "evidence_id"):
            if quality.get(name) != repair[name]:
                raise ValueError("incompatible repair quality " + name)
        if (quality.get("framing_spec") != framing or quality.get("status") != "PASS"
                or quality.get("profile_id") != repair["quality_profile_id"]
                or not repair["evidence_id"] or not quality.get("metric")):
            raise ValueError("unvalidated repair quality provenance")
        measured, threshold = quality.get("measured_value"), quality.get("required_threshold")
        if (type(measured) not in (float, int) or type(threshold) not in (float, int)
                or not math.isfinite(measured) or not math.isfinite(threshold)
                or measured < threshold):
            raise ValueError("repair quality threshold not satisfied")


def validate_runtime_execution_intent(link_plan):
    profile = None
    profile_path = os.getenv("SGLANG_PIC_EXECUTION_PROFILE")
    if profile_path:
        with open(profile_path, encoding="utf-8") as handle:
            profile = json.load(handle)
    formal = os.getenv("SGLANG_PIC_FORMAL_EXECUTION", "0") == "1"
    try:
        validate_execution_intent(link_plan, profile, require_formal=formal)
    except ValueError as error:
        raise PICRuntimeError(str(error)) from error

@dataclass(frozen=True)
class ResolvedLinkPlan:
    """The physical plan, derived here and owned here.

    Named so the equivalence tests have something to compare. Two requests that
    express the same reuse intent through different host builders must resolve
    to an identical value of this type.
    """

    cache_keys: tuple[str, ...]
    block_ranges: tuple[tuple[tuple[str, Any], ...], ...]
    mutable_start_token: int
    mutable_token_count: int
    physical_prefix_token_count: int
    total_token_count: int
    recompute_positions: tuple[int, ...]
    recompute_tokens_per_block: tuple[int, ...]
    recompute_first_block: bool
    recompute_initial_tokens: int
    token_layout_source: str
    linked_token_ids: tuple[int, ...]

    @staticmethod
    def freeze_ranges(
        ranges: list[dict[str, Any]]
    ) -> tuple[tuple[tuple[str, Any], ...], ...]:
        """Hashable block ranges, so two resolutions compare by value."""

        return tuple(tuple(sorted(item.items())) for item in ranges)

    @property
    def block_range_dicts(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.block_ranges]

    def apply_to(self, link_plan: dict[str, Any]) -> None:
        """Publish the resolved plan under the keys the rest of the runtime reads.

        imp_indices and kvlink are the paper-side and EPIC-demo-side names for
        the same positions; both are written so neither vocabulary drifts from
        the layout SGLang actually tokenized.
        """

        block_ranges = self.block_range_dicts
        recompute_positions = list(self.recompute_positions)
        link_plan["block_ranges"] = block_ranges
        link_plan["attention_ranges"] = block_ranges
        link_plan["mutable_start_token"] = self.mutable_start_token
        link_plan["mutable_token_count"] = self.mutable_token_count
        link_plan["physical_prefix_token_count"] = self.physical_prefix_token_count
        link_plan["total_token_count"] = self.total_token_count
        link_plan["kvlink"] = recompute_positions
        link_plan["legolink_k"] = self.recompute_initial_tokens
        link_plan["recompute_first_block"] = self.recompute_first_block
        link_plan["recompute_first_block_tokens"] = self.recompute_tokens_per_block[0]
        link_plan["recompute_tokens_per_block"] = list(self.recompute_tokens_per_block)
        link_plan["imp_indices"] = recompute_positions
        link_plan["recompute_positions"] = recompute_positions
        link_plan["token_layout_source"] = self.token_layout_source


def _validate_tp_rank_manifests(manifests: list[dict[str, Any]]) -> None:
    if not manifests:
        raise PICRuntimeError("PIC TP validation received no rank manifests.")
    logical_fields = (
        "cache_keys",
        "block_token_lengths",
        "destination_global_offsets",
        "imp_indices",
    )
    baseline = manifests[0]
    for rank, manifest in enumerate(manifests):
        for field_name in logical_fields:
            if tuple(manifest.get(field_name) or ()) != tuple(
                baseline.get(field_name) or ()
            ):
                raise PICRuntimeError(f"PIC TP rank {rank} disagrees on {field_name}.")
        if not bool(manifest.get("local_lookup_success")):
            raise PICRuntimeError(
                f"PIC TP rank {rank} cannot resolve every local LegoLink handle."
            )


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _sparse_repair_row_batching_options(
    forward_batch: Any,
) -> tuple[bool, bool, bool, str]:
    """Resolve protected per-request row-batching diagnostic options."""

    row_batching = _env_flag("SGLANG_PIC_SPARSE_REPAIR_ROW_BATCHING", "0")
    rowwise_ffn = _env_flag("SGLANG_PIC_SPARSE_REPAIR_ROWWISE_FFN", "0")
    rowwise_output_projection = _env_flag(
        "SGLANG_PIC_SPARSE_REPAIR_ROWWISE_OUTPUT_PROJECTION", "0"
    )
    metadata = getattr(forward_batch, "pic_request_metadata", None)
    link_plan = (
        metadata[0].get("link_plan")
        if isinstance(metadata, list)
        and len(metadata) == 1
        and isinstance(metadata[0], dict)
        else None
    )
    if not isinstance(link_plan, dict):
        return row_batching, rowwise_ffn, rowwise_output_projection, "environment"
    keys = (
        "diagnostic_sparse_repair_row_batching",
        "diagnostic_sparse_repair_rowwise_ffn",
        "diagnostic_sparse_repair_rowwise_output_projection",
    )
    if not any(key in link_plan for key in keys):
        return row_batching, rowwise_ffn, rowwise_output_projection, "environment"
    if not _env_flag("SGLANG_PIC_ALLOW_REQUEST_ROW_BATCHING_OVERRIDE", "0"):
        raise PICRuntimeError(
            "Per-request sparse repair row-batching override is disabled."
        )
    for key in keys:
        if key in link_plan and not isinstance(link_plan[key], bool):
            raise PICRuntimeError(f"pic_link.{key} must be a boolean.")
    row_batching = bool(
        link_plan.get("diagnostic_sparse_repair_row_batching", row_batching)
    )
    rowwise_ffn = bool(
        link_plan.get("diagnostic_sparse_repair_rowwise_ffn", rowwise_ffn)
    )
    rowwise_output_projection = bool(
        link_plan.get(
            "diagnostic_sparse_repair_rowwise_output_projection",
            rowwise_output_projection,
        )
    )
    if (rowwise_ffn or rowwise_output_projection) and not row_batching:
        raise PICRuntimeError(
            "Per-request row-wise compute requires sparse repair row batching."
        )
    return (
        row_batching,
        rowwise_ffn,
        rowwise_output_projection,
        "request_diagnostic_override",
    )


def get_triton_fused_kv_link_mode() -> str:
    mode = os.getenv("SGLANG_PIC_TRITON_FUSED_KV_LINK", "0").strip().lower()
    aliases = {
        "0": "disabled",
        "false": "disabled",
        "off": "disabled",
        "disabled": "disabled",
        "1": "required",
        "true": "required",
        "on": "required",
        "required": "required",
        "auto": "auto",
    }
    normalized = aliases.get(mode)
    if normalized is None:
        raise PICRuntimeError("SGLANG_PIC_TRITON_FUSED_KV_LINK must be 0, 1, or auto.")
    return normalized


def legolink_only_mode_enabled() -> bool:
    """Return whether the patched server accepts only Compile/Link requests."""

    return _env_flag("SGLANG_PIC_LEGOLINK_ONLY", "1")


_WARMUP_CACHE_KEY = "sglang-pic-warmup-block"


def _warmup_shape() -> tuple[int, int]:
    """Block size and LegoLink k the warmup should specialize for.

    The first attempt used a 48-token block and k=8. It halved the cold Compile
    -- 2.564 s to 0.710 -- and did nothing at all for the cold Link, which
    stayed at 8.887 s against a warm 1.8 s. That is the failure this function
    exists to prevent: Triton specializes on constexpr values and FlashInfer
    plans per shape, so a warmup an order of magnitude smaller than the traffic
    compiles kernels the traffic will not use.

    The defaults match the project's standard arm -- k=128 and a block near the
    1,556-token selected contexts the 2026-09-02 runs actually sent. A
    deployment whose traffic looks different should say so rather than warm the
    wrong shape.
    """

    tokens = max(int(os.getenv("SGLANG_PIC_WARMUP_BLOCK_TOKENS", "1536")), 16)
    k = max(int(os.getenv("SGLANG_PIC_WARMUP_LEGOLINK_K", "128")), 1)
    return tokens, min(k, tokens)


def _warmup_text(tokens: int) -> str:
    # Roughly one token per word for this filler; the point is the shape the
    # kernels see, not the content, and nothing reads it back.
    return " ".join(["warmup context block for kernel specialization"] * (tokens // 6))


def legolink_warmup_enabled() -> bool:
    """Whether the strict deployment warms itself with a Compile and a Link.

    SGLang's own warmup is a plain generation request, which LegoLink-only mode
    rejects by design, so the launcher is forced to pass --skip-server-warmup
    and the first real request pays the Triton JIT and the FlashInfer plan
    build. Measured on 2026-09-02 against a warmed Original server: the first
    Compile cost 2.564 s against ~0.28 s once warm, the first Link 7.101 s
    against ~2.0 s, and those 9.665 s were 48% of everything the optimized arm
    spent across six calls. That is a one-time cost being charged to every
    short session, and every Agent Quality session is short.

    The restriction is the patched runtime's own, so the compensation belongs
    here rather than in each launcher that starts the server.
    """

    return _env_flag("SGLANG_PIC_WARMUP", "1")


def legolink_warmup_payloads(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """One Compile and the Link that consumes it, shaped like the real ones.

    Deliberately built from the same request contract the host sends rather
    than by calling kernels directly: a synthetic kernel call can JIT a
    different specialization than the one a real request needs, which would
    warm nothing and hide that it warmed nothing.
    """

    block_tokens, legolink_k = _warmup_shape()
    metadata = {
        "prefill_unit_id": _WARMUP_CACHE_KEY,
        "cache_key": _WARMUP_CACHE_KEY,
        "source_sub_context_id": _WARMUP_CACHE_KEY,
        "context_field": "warmup",
        "functional_role": "warmup",
        "attention_policy": "block_isolated_no_pairwise_attention",
        "compile_attention_policy": "isolated_causal",
        "link_attention_policy": "legolink_sparse_causal",
        "cache_scope": "sub_context_block",
        "kv_reuse_granularity": "sub_context",
        "task_id": "sglang-pic-warmup",
    }
    compile_request = {
        "model": model,
        "messages": [{"role": "user", "content": _warmup_text(block_tokens)}],
        "temperature": 0,
        # Eight, the same count SGLang's own warmup uses, and for the same
        # reason. At max_tokens=1 the only token comes out of the prefill or
        # Link forward and no decode step ever runs, so the decode attention
        # kernels, the CUDA graph replay path and the sampler stay cold. The
        # 2026-09-02 warmup asked for one and left the first real Link at
        # 8.9 s, unchanged.
        "max_tokens": 8,
        "extra_key": _WARMUP_CACHE_KEY,
        "custom_labels": {
            "prefill_unit_id": _WARMUP_CACHE_KEY,
            "cache_key": _WARMUP_CACHE_KEY,
            "context_field": "warmup",
        },
        "sub_context": metadata,
    }
    link_request = {
        "model": model,
        "messages": [{"role": "user", "content": "warmup query"}],
        "temperature": 0,
        "max_tokens": 8,
        "pic_link": {
            "request_id": "sglang-pic-warmup",
            "task_id": "sglang-pic-warmup",
            "block_ids": [_WARMUP_CACHE_KEY],
            "cache_keys": [_WARMUP_CACHE_KEY],
            "recompute_initial_tokens": legolink_k,
            "variant": "sglang_pic_warmup",
        },
    }
    return compile_request, link_request


def execute_legolink_warmup(
    url: str,
    *,
    model: str,
    headers: Optional[dict[str, str]] = None,
    ssl_verify: Any = True,
    timeout_s: float = 600.0,
) -> bool:
    """Pay the JIT before the first real request. Never fail startup for it.

    A warmup that cannot run leaves the server exactly as it was, which is
    today's behaviour, so it is logged and swallowed. Refusing to serve because
    an optimization did not apply would be a worse failure than the one being
    fixed.
    """

    import requests

    compile_request, link_request = legolink_warmup_payloads(model)
    base = url.rstrip("/")
    endpoint = base + "/v1/chat/completions"

    # The skipped-warmup path does not wait for the app to start listening --
    # the native path only waits because its own warmup poll does. Same poll,
    # same 120 s budget.
    for _ in range(120):
        try:
            probe = requests.get(
                base + "/model_info", timeout=5, headers=headers or {}, verify=ssl_verify
            )
            if probe.status_code == 200:
                break
        except Exception:  # noqa: BLE001 - not listening yet
            pass
        time.sleep(1)
    else:
        logger.warning("LegoLink warmup skipped: server did not become reachable")
        return False

    started = time.perf_counter()
    for name, payload in (("compile", compile_request), ("link", link_request)):
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers or {},
                timeout=timeout_s,
                verify=ssl_verify,
            )
        except Exception as error:  # noqa: BLE001 - startup must survive anything
            logger.warning("LegoLink %s warmup could not be sent: %s", name, error)
            return False
        if response.status_code != 200:
            logger.warning(
                "LegoLink %s warmup returned %s: %s",
                name,
                response.status_code,
                response.text[:400],
            )
            return False
    elapsed = time.perf_counter() - started
    logger.info("LegoLink warmup completed in %.2f s", elapsed)
    _emit_pic_trace(
        "legolink_warmup_complete",
        phase="warmup",
        duration_s=elapsed,
        cache_key=_WARMUP_CACHE_KEY,
    )
    return True


def _record_function(name: str):
    return torch.profiler.record_function(name)


def _guard_legolink_release(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._pending_legolink_release = None
        forward_batch = args[0] if args else kwargs.get("forward_batch")
        metadata = getattr(forward_batch, "pic_request_metadata", None) or []
        link_plan = metadata[0].get("link_plan", {}) if metadata else {}
        reserved_cache_keys = [
            str(cache_key)
            for cache_key in (link_plan.get("_scheduler_reserved_cache_keys") or [])
        ]
        tree_cache = getattr(self, "_legolink_cache", None)
        if reserved_cache_keys and tree_cache is not None:
            # Scheduler pins source blocks before request-private destination
            # allocation. Arm cleanup before any GPU-side preparation so an
            # exception early in run_link cannot leak those source pins.
            self._pending_legolink_release = (
                tree_cache,
                [],
                reserved_cache_keys,
            )
        try:
            return method(self, *args, **kwargs)
        finally:
            pending = getattr(self, "_pending_legolink_release", None)
            self._pending_legolink_release = None
            if pending is not None:
                tree_cache, handles, cache_keys = pending
                recorder = getattr(self, "_host_timing_recorder", None)
                release_context = (
                    recorder.span("release_legolink_blocks")
                    if recorder is not None
                    else contextlib.nullcontext()
                )
                with _profile_range("ReleaseSourceBlocks"), release_context:
                    if handles and hasattr(tree_cache, "release_legolink_blocks"):
                        tree_cache.release_legolink_blocks(handles, unpin=True)
                    elif cache_keys and hasattr(tree_cache, "unpin_legolink_blocks"):
                        tree_cache.unpin_legolink_blocks(cache_keys)
                    elif cache_keys and hasattr(tree_cache, "unref_legolink_blocks"):
                        tree_cache.unref_legolink_blocks(cache_keys, unpin=True)
                if reserved_cache_keys and isinstance(link_plan, dict):
                    # The pins just released are the ones the marker names.
                    # Drop it in the same step: a later admission of this
                    # request must pin again, and must not release pins it no
                    # longer holds -- possibly another request's.
                    link_plan.pop("_scheduler_reserved_cache_keys", None)

    return wrapped


def _trace_link_failure(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        global _ACTIVE_CUDA_EVENT_RECORDER, _ACTIVE_PROFILE_CONTEXT
        global _CUDA_EVENT_REQUEST_COUNT
        previous_recorder = _ACTIVE_CUDA_EVENT_RECORDER
        previous_profile_context = _ACTIVE_PROFILE_CONTEXT
        forward_batch = args[0] if args else kwargs.get("forward_batch")
        metadata = getattr(forward_batch, "pic_request_metadata", None) or []
        link_plan = metadata[0].get("link_plan", {}) if metadata else {}
        trace_context = {
            "request_id": link_plan.get("request_id"),
            "task_id": link_plan.get("task_id"),
            "variant": link_plan.get("variant"),
            "phase": "link",
            "concurrency": link_plan.get("concurrency"),
            "k": link_plan.get("legolink_k")
            or link_plan.get("recompute_initial_tokens"),
        }
        _CUDA_EVENT_REQUEST_COUNT += 1
        warmup_requests = max(
            int(os.getenv("SGLANG_PIC_CUDA_EVENT_WARMUP_REQUESTS", "0")), 0
        )
        warmup = bool(link_plan.get("profiling_warmup")) or (
            _CUDA_EVENT_REQUEST_COUNT <= warmup_requests
        )
        recorder = _CudaEventRecorder(context=trace_context, warmup=warmup)
        _ACTIVE_CUDA_EVENT_RECORDER = recorder
        _ACTIVE_PROFILE_CONTEXT = trace_context
        self._cuda_event_recorder = recorder
        host_recorder = _HostTimingRecorder(
            "model_runner",
            request_id=link_plan.get("request_id"),
            task_id=link_plan.get("task_id"),
            variant=link_plan.get("variant"),
            owner_mode=(
                "radixcache_owner"
                if _env_flag("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1")
                else "picruntime_owner"
            ),
        )
        self._host_timing_recorder = host_recorder
        host_wait_started_ns = time.perf_counter_ns() if host_recorder.enabled else 0
        status = "ok"
        error_details: dict[str, Any] = {}
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            status = "error"
            error_details = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _emit_pic_trace(
                "model_link_failure",
                phase="link",
                **getattr(self, "_active_trace_context", {}),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            if host_recorder.enabled:
                host_recorder.add_ns(
                    "picruntime_run_link_host_wait",
                    time.perf_counter_ns() - host_wait_started_ns,
                )
                host_recorder.update_context(
                    **getattr(self, "_active_trace_context", {})
                )
                host_recorder.emit(status=status, **error_details)
            try:
                self._last_cuda_event_summary = recorder.finalize()
            except Exception as error:
                # Diagnostics are never allowed to turn a successful inference
                # into a request failure.
                logger.warning("Ignoring CUDA Event finalization failure: %s", error)
                self._last_cuda_event_summary = {}
            _ACTIVE_CUDA_EVENT_RECORDER = previous_recorder
            _ACTIVE_PROFILE_CONTEXT = previous_profile_context
            self._active_trace_context = {}
            self._host_timing_recorder = None

    return wrapped


class _LazyTraceValues:
    """Defer index materialization until a trace record is actually written.

    ``_emit_pic_trace`` returns immediately when tracing is disabled, but Python
    evaluates its keyword arguments first. Several call sites pass CUDA tensors,
    and ``.cpu()`` on one is a device-to-host copy that synchronises the stream.
    Profiler-free benchmark runs disable tracing, so those synchronisations sat
    inside the measured KV-link path for no observable benefit.

    A callable source defers the work that produces the values as well, which
    matters where a call site would otherwise allocate a concatenated tensor.
    Materialization happens inside ``_emit_pic_trace`` immediately after the
    enabled check, so a written record holds exactly what it held before.
    """

    __slots__ = ("_source",)

    def __init__(self, source: Any) -> None:
        self._source = source

    def materialize(self) -> dict[str, Any]:
        values = self._source() if callable(self._source) else self._source
        limit = max(int(os.getenv("SGLANG_PIC_TRACE_MAX_ITEMS", "32")), 1)
        if isinstance(values, torch.Tensor):
            items = values.detach().reshape(-1).cpu().tolist()
        else:
            items = list(values)
        items = [int(item) for item in items]
        if len(items) <= limit:
            return {"count": len(items), "values": items, "truncated": False}
        head_count = (limit + 1) // 2
        tail_count = limit // 2
        return {
            "count": len(items),
            "values": items[:head_count] + items[-tail_count:],
            "truncated": True,
        }


def _trace_values(values: Any) -> _LazyTraceValues:
    """Return a bounded trace representation of token or KV slot indices.

    The representation is produced only if the record is written; see
    :class:`_LazyTraceValues`.
    """

    return _LazyTraceValues(values)


class _DeferredTraceField:
    """Defer an arbitrary trace field that is expensive to build."""

    __slots__ = ("_factory",)

    def __init__(self, factory: Any) -> None:
        self._factory = factory

    def materialize(self) -> Any:
        return _materialize_trace_details(self._factory())


def _lazy_trace(factory: Any) -> _DeferredTraceField:
    return _DeferredTraceField(factory)


def _materialize_trace_details(value: Any) -> Any:
    """Resolve deferred trace values anywhere inside a record."""

    if isinstance(value, (_LazyTraceValues, _DeferredTraceField)):
        return value.materialize()
    if isinstance(value, dict):
        return {key: _materialize_trace_details(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_materialize_trace_details(item) for item in value]
    return value


def _emit_pic_trace(event: str, **details: Any) -> None:
    """Append one diagnostic event without storing prompts or token contents."""
    raw_path = os.getenv("SGLANG_PIC_TRACE_PATH")
    if not raw_path:
        return
    record = {
        "schema_version": 1,
        "timestamp_unix_s": time.time(),
        "timestamp_monotonic_s": time.perf_counter(),
        "event": event,
        "pid": os.getpid(),
        "rank": os.getenv("LOCAL_RANK", os.getenv("RANK", "0")),
        **_materialize_trace_details(details),
    }
    _append_jsonl(raw_path, record)


def _profile_sync() -> None:
    if (
        os.getenv("SGLANG_PIC_PROFILE_SYNC", "0").lower()
        in (
            "1",
            "true",
            "yes",
        )
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()


def _profile_start() -> float:
    _profile_sync()
    return time.perf_counter()


def _profile_elapsed(started: float) -> float:
    _profile_sync()
    return time.perf_counter() - started


def _cuda_event_start(
    label: str = "unlabeled",
    *,
    phase: Optional[str] = None,
    decode_iteration: Optional[int] = None,
) -> Optional[_CudaEventSpan]:
    if _ACTIVE_CUDA_EVENT_RECORDER is None:
        return None
    return _ACTIVE_CUDA_EVENT_RECORDER.start(
        label,
        phase=phase,
        decode_iteration=decode_iteration,
    )


def _cuda_event_elapsed_ms(started: Optional[_CudaEventSpan]) -> float:
    if started is None:
        return 0.0
    if _ACTIVE_CUDA_EVENT_RECORDER is not None:
        _ACTIVE_CUDA_EVENT_RECORDER.end(started)
    return 0.0


def _finalize_cuda_events(runtime: Any) -> dict[str, float]:
    recorder = getattr(runtime, "_cuda_event_recorder", None)
    return recorder.finalize() if recorder is not None else {}


def _cuda_context_from_forward_batch(
    forward_batch: Any,
    *,
    phase: str,
) -> dict[str, Any]:
    metadata = getattr(forward_batch, "pic_request_metadata", None) or []
    first = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
    link_plan = first.get("link_plan") or {}
    return {
        "request_id": link_plan.get("request_id") or first.get("request_id"),
        "task_id": link_plan.get("task_id") or first.get("task_id"),
        "variant": link_plan.get("variant") or first.get("variant") or "native_sglang",
        "phase": phase,
        "concurrency": getattr(forward_batch, "batch_size", None),
        "k": link_plan.get("legolink_k") or link_plan.get("recompute_initial_tokens"),
        "warmup": bool(link_plan.get("profiling_warmup")),
    }


def _next_decode_iteration() -> int:
    global _NATIVE_DECODE_ITERATION_COUNT
    _NATIVE_DECODE_ITERATION_COUNT += 1
    return _NATIVE_DECODE_ITERATION_COUNT


@contextlib.contextmanager
def _standalone_cuda_phase(
    *,
    phase: str,
    component: str,
    context: Optional[dict[str, Any]] = None,
    decode_iteration: Optional[int] = None,
):
    """Time one native phase when it is not nested inside ``run_link``."""

    global _ACTIVE_CUDA_EVENT_RECORDER, _ACTIVE_PROFILE_CONTEXT
    if phase == "native_decode" and decode_iteration is not None:
        iteration_limit = max(
            int(os.getenv("SGLANG_PIC_CUDA_EVENT_DECODE_ITERATIONS", "8")), 0
        )
        if iteration_limit == 0 or decode_iteration > iteration_limit:
            yield
            return
    if _ACTIVE_CUDA_EVENT_RECORDER is not None:
        # Sparse recompute reuses native forward functions while the link
        # recorder is active. Do not misclassify those kernels as native prefill.
        yield
        return
    previous_profile_context = _ACTIVE_PROFILE_CONTEXT
    phase_context = dict(context or {})
    warmup = bool(phase_context.pop("warmup", False))
    recorder = _CudaEventRecorder(context=phase_context, warmup=warmup)
    _ACTIVE_CUDA_EVENT_RECORDER = recorder
    _ACTIVE_PROFILE_CONTEXT = dict(context or {})
    span = recorder.start(
        component,
        phase=phase,
        decode_iteration=decode_iteration,
    )
    try:
        yield
    finally:
        recorder.end(span)
        try:
            recorder.finalize()
        except Exception as error:
            logger.warning("Ignoring standalone CUDA Event failure: %s", error)
        _ACTIVE_CUDA_EVENT_RECORDER = None
        _ACTIVE_PROFILE_CONTEXT = previous_profile_context


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((out1, out2), dim=-1)
    return torch.stack((out1, out2), dim=-1).flatten(-2)


def _custom_params(sampling_params: Any) -> dict[str, Any]:
    custom_params = getattr(sampling_params, "custom_params", None)
    return custom_params if isinstance(custom_params, dict) else {}


def physical_token_ids_sha256(token_ids: list[int] | tuple[int, ...]) -> str:
    encoded = json.dumps(
        [int(token_id) for token_id in token_ids],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_physical_token_input(
    metadata: Any,
    *,
    label: str,
) -> Optional[list[int]]:
    """Validate an exact token override carried by Compile or Link metadata."""

    if not isinstance(metadata, dict) or "physical_input_ids" not in metadata:
        return None
    version = metadata.get("physical_token_layout_version")
    if version != PHYSICAL_TOKEN_LAYOUT_VERSION:
        raise PICRuntimeError(
            f"{label} physical token layout version is unsupported: {version!r}"
        )
    raw_token_ids = metadata.get("physical_input_ids")
    if not isinstance(raw_token_ids, list) or not raw_token_ids:
        raise PICRuntimeError(
            f"{label}.physical_input_ids must be a non-empty integer list."
        )
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in raw_token_ids
    ):
        raise PICRuntimeError(
            f"{label}.physical_input_ids contains a non-integer token id."
        )
    token_ids = [int(token_id) for token_id in raw_token_ids]
    expected_count = metadata.get("physical_token_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(token_ids)
    ):
        raise PICRuntimeError(
            f"{label} physical token count mismatch: metadata="
            f"{expected_count!r}, actual={len(token_ids)}"
        )
    expected_hash = metadata.get("physical_token_ids_sha256")
    actual_hash = physical_token_ids_sha256(token_ids)
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise PICRuntimeError(
            f"{label} physical token hash mismatch: expected="
            f"{expected_hash!r}, actual={actual_hash}"
        )
    return token_ids


def get_pic_compile_cache_key(sampling_params: Any) -> Optional[str]:
    metadata = get_pic_compile_metadata(sampling_params)
    cache_key = metadata.get("cache_key")
    return str(cache_key) if cache_key else None


def pic_compile_key_in_batch(cache_key: Optional[str], batch_reqs: Any) -> bool:
    """Whether a Compile of ``cache_key`` is already in the prefill batch being built.

    Two Compiles of one cache key must not finish in the same prefill batch.
    The forward captures one runtime snapshot for the key; processing the
    first result rebinds it, registers the block with RadixCache and releases
    the local snapshot, so the second result finds no snapshot and
    ``rebind_reference_slots`` raises -- which takes the scheduler down.
    Byte-identical agents reaching the same turn together produce exactly
    this (Terminal-Bench C=8, three task copies, 2026-09-16). A later
    batch is safe: the second Compile then takes RadixCache's existing
    duplicate-registration path. The scheduler therefore defers a
    same-key Compile to the next batch; nothing about registration,
    ownership or cache-key lifetime changes.
    """

    if cache_key is None:
        return False
    return any(
        get_pic_compile_cache_key(getattr(req, "sampling_params", None)) == cache_key
        for req in batch_reqs
    )


def get_pic_compile_metadata(sampling_params: Any) -> dict[str, Any]:
    metadata = _custom_params(sampling_params).get("sub_context_metadata")
    if not isinstance(metadata, dict):
        return {}
    return metadata


def get_pic_compile_storage_mode(sampling_params: Any) -> str:
    metadata = _custom_params(sampling_params).get("sub_context_metadata")
    raw_mode = metadata.get("pic_storage_mode") if isinstance(metadata, dict) else None
    mode = str(raw_mode or os.getenv("SGLANG_PIC_STORAGE_MODE", "reference")).lower()
    if mode not in ("clone", "reference"):
        raise PICRuntimeError("PIC storage mode must be 'clone' or 'reference'.")
    return mode


def get_pic_link_plan(sampling_params: Any) -> Optional[dict[str, Any]]:
    link_plan = _custom_params(sampling_params).get("pic_link")
    return link_plan if isinstance(link_plan, dict) else None


def per_batch_overlap_enabled() -> bool:
    """Whether PIC suppresses overlap per batch instead of per server.

    Linking cannot run while the scheduler overlaps CPU work with the previous
    batch's forward, so the arrival path rejected any request that arrived on a
    server with overlap scheduling on. That is a server-wide answer to a
    per-batch problem: ``PICRuntime.has_link`` only ever fires on an extend
    batch holding exactly one request, so every decode step in the deployment
    was paying for a constraint that belongs to the link batches alone. In the
    2026-09-02 local terminus2 run that was 1,325 decode steps carrying a
    penalty for 10 links.

    With this enabled the scheduler keeps overlap and drains around PIC batches
    instead, the way upstream already handles speculative decoding with grammar.
    Off by default: it changes how a GPU deployment schedules, and the
    measurement that would justify flipping the default has not been run.
    """

    return _env_flag("SGLANG_PIC_PER_BATCH_OVERLAP", "0")


def link_admission_while_decoding_enabled() -> bool:
    """Whether a Link may be admitted while other requests are decoding.

    The scheduler used to admit a Link only into an otherwise empty engine, so
    a Link waited for every decoding request to finish; at concurrency 8 that
    queue was 97.8% of semantic TTFT. The audit of that guard found no
    invariant needing an empty engine: KV slots, source pins and positions are
    request-private, and forwards are sequential. What run_link does need --
    a batch holding exactly one request -- stays enforced independently of
    this flag. With it on, a Link runs alone in its own extend forward between
    decode steps of other requests; it is not batched with them and does not
    execute concurrently on the GPU.

    Off by default: experimental until the serialized-vs-interleaved
    equivalence gate has passed.
    """

    return _env_flag("SGLANG_PIC_ADMIT_LINK_WHILE_DECODING", "0")


def batch_needs_synchronous_boundary(batch: Any) -> bool:
    """True when this batch carries Compile or Link work.

    Both write into KV the neighbouring batch may still be reading — Link
    overwrites selected positions, Compile registers blocks the RadixCache then
    hands out — so a batch that carries either must not be launched while the
    previous forward is in flight, and its own result must be processed before
    the next batch can reuse what it released. The caller applies this to the
    batch and to its predecessor, which fences both sides.
    """

    for req in getattr(batch, "reqs", None) or ():
        sampling_params = getattr(req, "sampling_params", None)
        if sampling_params is None:
            continue
        if get_pic_link_plan(sampling_params) is not None:
            return True
        if get_pic_compile_cache_key(sampling_params) is not None:
            return True
    return False


def get_pic_request_metadata(req: Any) -> dict[str, Any]:
    """Return PIC metadata only while a request is eligible for model work.

    SGLang represents a rejected request by replacing its prompt with a
    one-token abort sentinel and setting ``to_finish``.  Retaining the original
    PIC link plan in that state makes ModelRunner compare the sentinel against
    the pre-abort linked layout and can terminate the scheduler process.
    """
    if (
        getattr(req, "to_finish", None) is not None
        or getattr(req, "finished_reason", None) is not None
    ):
        return {
            "compile_cache_key": None,
            "compile_storage_mode": None,
            "link_plan": None,
        }
    scheduler_waiting_ns = None
    if _env_flag("SGLANG_PIC_HOST_TIMING_ENABLED", "0"):
        time_stats = getattr(req, "time_stats", None)
        try:
            queue_s = time_stats.get_queueing_time() if time_stats is not None else None
            if queue_s is not None and float(queue_s) >= 0:
                scheduler_waiting_ns = int(float(queue_s) * 1_000_000_000)
        except (AttributeError, TypeError, ValueError):
            scheduler_waiting_ns = None
    return {
        "compile_cache_key": get_pic_compile_cache_key(req.sampling_params),
        "compile_storage_mode": get_pic_compile_storage_mode(req.sampling_params),
        "link_plan": get_pic_link_plan(req.sampling_params),
        "scheduler_waiting_ns": scheduler_waiting_ns,
    }


def validate_pic_link_input_length(
    input_ids: list[int], sampling_params: Any, max_req_input_len: int
) -> Optional[str]:
    """Reject overlength linked layouts instead of position-unsafe truncation."""
    link_plan = get_pic_link_plan(sampling_params)
    linked_input_length = len(input_ids)
    if isinstance(link_plan, dict):
        expected_complete_count = link_plan.get("expected_complete_token_count")
        if (
            isinstance(expected_complete_count, int)
            and not isinstance(expected_complete_count, bool)
            and expected_complete_count > 0
        ):
            linked_input_length = expected_complete_count
    if link_plan is not None and linked_input_length >= max_req_input_len:
        return (
            f"PIC linked input length ({linked_input_length} tokens) exceeds the "
            f"maximum allowed length ({max_req_input_len} tokens). PIC linked "
            "requests cannot be auto-truncated because the link plan is "
            "position-dependent."
        )
    return None


def validate_pic_link_logprob_request(
    sampling_params: Any,
    *,
    return_logprob: bool,
    logprob_start_len: int,
    input_length: int,
) -> Optional[str]:
    """Reject a Link that asks for input-token logprobs, which it cannot return.

    `PICRuntime.run_link` rebuilds its inner extend batches with
    `return_logprob=False`, so the logits output it hands back carries no
    `input_token_logprobs`. The scheduler's result processor nevertheless
    counts `extend_input_len - extend_logprob_start_len` input logprobs for the
    request and asserts the field is present. That assertion is raised from the
    event loop, not from a request handler, and the scheduler process answers
    any such exception by signalling its parent -- one request ended the whole
    server. Serving the request without the logprobs would instead return a
    result that silently lacks what the caller asked for, so it is refused here,
    at admission, as a request-scoped error.

    Input logprobs are wanted exactly when logprobs are on and the effective
    start lies inside the prompt: `-1` means "from the end" and
    `logprob_start_len == input_length` means "none", and neither reaches the
    assertion. A Link request has a request-private radix namespace, so no
    prefix is skipped and the absolute start equals the extend-relative one.
    Requests that are not a Link are untouched.
    """

    if get_pic_link_plan(sampling_params) is None:
        return None
    if not return_logprob:
        return None
    if 0 <= logprob_start_len < input_length:
        return (
            "PIC link requests cannot return input-token logprobs "
            f"(logprob_start_len={logprob_start_len} lies inside the "
            f"{input_length}-token linked input): the linked prefill does not "
            "compute them. Request output logprobs only (logprob_start_len=-1, "
            "and no echo of the prompt)."
        )
    return None


def _contiguous_ranges(positions: list[int]) -> list[tuple[int, int]]:
    """Convert sorted token positions to half-open contiguous ranges."""
    if not positions:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position <= previous:
            raise PICRuntimeError("PIC imp_indices must be strictly increasing.")
        if position != previous + 1:
            ranges.append((start, previous + 1))
            start = position
        previous = position
    ranges.append((start, previous + 1))
    return ranges


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise PICRuntimeError(f"{name} must be an integer; got {raw!r}") from exc
    if value < 0:
        raise PICRuntimeError(f"{name} must be non-negative; got {value}")
    return value


def _split_ranges_by_max_repair_tokens(
    ranges: list[tuple[int, int]],
    *,
    max_tokens: int,
) -> list[tuple[int, int]]:
    """Split long sparse-repair rows without changing logical token order."""

    if max_tokens <= 0:
        return list(ranges)
    split: list[tuple[int, int]] = []
    for start, end in ranges:
        cursor = start
        while cursor < end:
            next_end = min(end, cursor + max_tokens)
            split.append((cursor, next_end))
            cursor = next_end
    return split


def _build_flashinfer_paged_execution_units(
    ranges: list[tuple[int, int]],
    *,
    max_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Group contiguous repair chunks under a FlashInfer workspace-safe cap."""

    if max_tokens <= 0:
        return (tuple(range(len(ranges))),)
    units: list[tuple[int, ...]] = []
    current: list[int] = []
    current_tokens = 0
    for row_index, (start, end) in enumerate(ranges):
        row_tokens = end - start
        if row_tokens <= 0:
            raise PICRuntimeError("Invalid FlashInfer sparse repair range.")
        if current and current_tokens + row_tokens > max_tokens:
            units.append(tuple(current))
            current = []
            current_tokens = 0
        current.append(row_index)
        current_tokens += row_tokens
    if current:
        units.append(tuple(current))
    return tuple(units)


#: Lifecycle of one Compile cache key in the scheduler-side registry.
COMPILE_STATE_ABSENT = "absent"
COMPILE_STATE_COMPILING = "compiling"
COMPILE_STATE_RESIDENT = "resident"


@dataclass
class _CompileRecord:
    """One admitted Compile request: what *it* will finalize, fixed at admission.

    `pin_finished_compile` runs after the request has produced its output and
    must register exactly what that request computed. It used to re-read the
    token tuple from `PICRequestTokenRegistry._blocks[cache_key]`, a map the
    next same-key Compile overwrites at admission, so a second Compile
    admitted while the first was still running changed what the first one
    tried to pin.
    """

    cache_key: str
    token_ids: tuple[int, ...]
    storage_mode: str


def _finish_reason_is_abort(reason: Any) -> bool:
    to_json = getattr(reason, "to_json", None)
    if not callable(to_json):
        return False
    try:
        return to_json().get("type") == "abort"
    except AttributeError:
        return False


class PICRequestTokenRegistry:
    """Scheduler-side token registry used to assemble a linked request.

    A Compile cache key moves through three states, visible as
    `compile_state()`:

    * ``absent`` -- no tokens known for the key;
    * ``compiling`` -- one or more admitted Compile requests have not finished.
      Every one of them carries the same tokens and storage mode: a Compile that
      disagrees is refused at admission instead of overwriting the others, and a
      Compile that agrees is attached as its own record, so duplicated work
      stays harmless and nothing is shared between requests but the key;
    * ``resident`` -- no Compile in flight and the key's tokens are known.

    A Compile that ends without registering (aborted, or its KV never entered
    the radix tree) returns the key to the state it had before the first
    in-flight Compile admitted it.
    """

    def __init__(self, max_blocks: Optional[int] = None):
        self.max_blocks = max_blocks or int(os.getenv("SGLANG_PIC_MAX_BLOCKS", "256"))
        self._blocks: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._storage_modes: dict[str, str] = {}
        self._pinned_blocks: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._tree_cache: Any = None
        self._legolink_compatibility: dict[str, Any] = {}
        # The last plan this registry resolved. Kept so an equivalence test can
        # compare two request paths without reaching into a mutated dict.
        self._last_resolved_link_plan: Optional[ResolvedLinkPlan] = None
        # In-flight Compile requests, by the identity of the owning request.
        self._compile_records: dict[int, _CompileRecord] = {}
        self._inflight_owners: dict[str, list[int]] = {}
        # What `_blocks` / `_storage_modes` held for a key before its first
        # in-flight Compile, restored if every in-flight Compile fails.
        self._precompile_state: dict[
            str, tuple[Optional[tuple[int, ...]], Optional[str]]
        ] = {}

    @property
    def last_resolved_link_plan(self) -> Optional["ResolvedLinkPlan"]:
        return self._last_resolved_link_plan

    def compile_state(self, cache_key: str) -> str:
        if self._inflight_owners.get(cache_key):
            return COMPILE_STATE_COMPILING
        if cache_key in self._blocks:
            return COMPILE_STATE_RESIDENT
        return COMPILE_STATE_ABSENT

    def _admit_compile(
        self,
        cache_key: str,
        token_ids: tuple[int, ...],
        storage_mode: str,
        owner: Any,
    ) -> None:
        """Record one Compile as in flight, or refuse it, before any mutation."""

        owners = self._inflight_owners.get(cache_key)
        if owners:
            first = self._compile_records[owners[0]]
            if first.token_ids != token_ids or first.storage_mode != storage_mode:
                raise PICRuntimeError(
                    f"A Compile for cache_key {cache_key} that differs in tokens "
                    "or storage mode is already in flight. A cache key names one "
                    "block; wait for the first Compile to finish or use a "
                    "different key."
                )
        owner_id = id(owner)
        if owner_id in self._compile_records:
            # The same request admitted twice replaces its own record.
            self._drop_compile_record(owner_id, restore=False)
        owners = self._inflight_owners.setdefault(cache_key, [])
        if not owners:
            self._precompile_state[cache_key] = (
                self._blocks.get(cache_key),
                self._storage_modes.get(cache_key),
            )
        owners.append(owner_id)
        self._compile_records[owner_id] = _CompileRecord(
            cache_key=cache_key, token_ids=token_ids, storage_mode=storage_mode
        )
        try:
            finalizer = weakref.finalize(owner, self._owner_collected, owner_id)
            finalizer.atexit = False
        except TypeError:
            pass  # not weak-referenceable: explicit discard/finish still cleans up

    def _owner_collected(self, owner_id: int) -> None:
        """A request vanished without finishing (for example, aborted in the queue)."""

        self._drop_compile_record(owner_id, restore=True)

    def _drop_compile_record(self, owner_id: int, *, restore: bool) -> Optional[_CompileRecord]:
        record = self._compile_records.pop(owner_id, None)
        if record is None:
            return None
        owners = self._inflight_owners.get(record.cache_key, [])
        if owner_id in owners:
            owners.remove(owner_id)
        if not owners:
            self._inflight_owners.pop(record.cache_key, None)
            previous = self._precompile_state.pop(record.cache_key, None)
            if restore and previous is not None:
                self._restore_precompile_state(record, previous)
        return record

    def _complete_compile_record(self, owner_id: int) -> None:
        """A Compile finished with a block: the key's state is committed.

        Once any Compile of a key has completed, a later failure of an attached
        sibling must not roll the key back to what it held before the first
        one, so the saved state is discarded here even if siblings remain.
        """

        record = self._compile_records.get(owner_id)
        if record is not None:
            self._precompile_state.pop(record.cache_key, None)
        self._drop_compile_record(owner_id, restore=False)

    def _restore_precompile_state(
        self,
        record: _CompileRecord,
        previous: tuple[Optional[tuple[int, ...]], Optional[str]],
    ) -> None:
        # Only undo our own write: if the key now holds other tokens, another
        # path has taken it over and its entry is not ours to remove.
        if self._blocks.get(record.cache_key) != record.token_ids:
            return
        tokens, mode = previous
        if tokens is None:
            self._blocks.pop(record.cache_key, None)
        else:
            self._blocks[record.cache_key] = tokens
        if mode is None:
            self._storage_modes.pop(record.cache_key, None)
        else:
            self._storage_modes[record.cache_key] = mode

    def _find_compile_owner(self, req: Any) -> Optional[int]:
        for candidate in (req, getattr(req, "sampling_params", None)):
            if candidate is not None and id(candidate) in self._compile_records:
                return id(candidate)
        return None

    def discard_compile(self, req: Any) -> None:
        """Forget a Compile that will never reach `pin_finished_compile`."""

        owner_id = self._find_compile_owner(req)
        if owner_id is not None:
            self._drop_compile_record(owner_id, restore=True)

    @staticmethod
    def _release_runtime_snapshot(model_worker: Any, cache_key: str) -> None:
        model_runner = getattr(model_worker, "model_runner", None)
        runtime = getattr(model_runner, "_pic_runtime", None)
        release = getattr(runtime, "release_local_block", None)
        if callable(release):
            release(cache_key)

    def prepare_request(
        self,
        input_ids: list[int],
        sampling_params: Any,
        owner: Any = None,
    ) -> list[int]:
        """Admit one request. `owner` identifies a Compile for its whole life.

        The scheduler passes the request; a caller without one is identified
        by its `sampling_params` object, which `pin_finished_compile` also
        accepts.
        """

        if legolink_only_mode_enabled() and not _env_flag(
            "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1"
        ):
            raise PICRuntimeError(
                "LegoLink-only mode requires RadixCache to own reusable blocks. "
                "Set SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS=1."
            )
        cache_key = get_pic_compile_cache_key(sampling_params)
        host_recorder = _HostTimingRecorder("scheduler")
        with (
            host_recorder.span("request_parsing"),
            host_recorder.span("semantic_block_metadata_parsing"),
        ):
            link_plan = get_pic_link_plan(sampling_params)

        if link_plan is not None:
            if cache_key is not None:
                raise PICRuntimeError(
                    "A request cannot compile a block and link blocks simultaneously."
                )
            host_recorder.update_context(
                request_id=link_plan.get("request_id"),
                task_id=link_plan.get("task_id"),
                variant=link_plan.get("variant"),
            )
            try:
                with (
                    _profile_range("SchedulerPrepareLink"),
                    host_recorder.span("scheduler_link_request_preparation"),
                ):
                    return self._prepare_link(input_ids, link_plan, host_recorder)
            except Exception as error:
                host_recorder.emit(
                    status="error",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise
            finally:
                host_recorder.emit()

        if cache_key is not None:
            metadata = get_pic_compile_metadata(sampling_params)
            physical_input_ids = validate_physical_token_input(
                metadata,
                label="sub_context",
            )
            normalized_input_ids = [int(token_id) for token_id in input_ids]
            if (
                physical_input_ids is not None
                and normalized_input_ids != physical_input_ids
            ):
                raise PICRuntimeError(
                    "Compile request tokens differ from the validated "
                    "physical_input_ids override."
                )
            if physical_input_ids is not None:
                metadata.pop("physical_input_ids", None)
            host_recorder.update_context(
                cache_key=cache_key,
                task_id=metadata.get("task_id"),
                semantic_block_id=metadata.get("semantic_block_id")
                or metadata.get("prefill_unit_id"),
                variant="compile",
            )
            with (
                _profile_range("SemanticSelection"),
                host_recorder.span("semantic_selection"),
            ):
                storage_mode = get_pic_compile_storage_mode(sampling_params)
                # Refuses a conflicting in-flight Compile before anything is
                # overwritten, and fixes what this request will finalize.
                self._admit_compile(
                    cache_key,
                    tuple(normalized_input_ids),
                    storage_mode,
                    owner if owner is not None else sampling_params,
                )
                if storage_mode != "reference":
                    self._release_pin(cache_key)
                self._blocks[cache_key] = tuple(normalized_input_ids)
                self._storage_modes[cache_key] = storage_mode
                self._blocks.move_to_end(cache_key)
                while len(self._blocks) > self.max_blocks:
                    old_cache_key, _ = self._blocks.popitem(last=False)
                    self._storage_modes.pop(old_cache_key, None)
                    self._release_pin(old_cache_key)
            trace_metadata = {
                "cache_key": cache_key,
                "sub_context_id": metadata.get("sub_context_id")
                or metadata.get("source_sub_context_id"),
                "semantic_block_id": metadata.get("semantic_block_id")
                or metadata.get("prefill_unit_id")
                or cache_key,
                "context_field": metadata.get("context_field"),
                "token_count": len(input_ids),
                "attention_policy": metadata.get("attention_policy"),
                "physical_token_layout_version": metadata.get(
                    "physical_token_layout_version"
                ),
                "physical_token_ids_sha256": metadata.get("physical_token_ids_sha256"),
                "segmentation_policy_version": metadata.get(
                    "segmentation_policy_version"
                ),
            }
            _emit_pic_trace(
                "semantic_subcontext_selected",
                phase="compile",
                **trace_metadata,
            )
            _emit_pic_trace(
                "scheduler_compile_registered",
                phase="compile",
                storage_mode=storage_mode,
                **trace_metadata,
            )
            host_recorder.emit()
            return input_ids

        if legolink_only_mode_enabled():
            _emit_pic_trace(
                "legolink_only_request_rejected",
                phase="scheduler",
                reason="missing_compile_or_link_metadata",
                token_count=len(input_ids),
            )
            raise PICRuntimeError(
                "This patched SGLang deployment is LegoLink-only. Generation "
                "requests must either compile one semantic sub-context with "
                "sub_context.cache_key or link previously compiled blocks with "
                "pic_link. Use the separate unpatched SGLang deployment for the "
                "baseline."
            )
        return input_ids

    def pin_finished_compile(
        self,
        req: Any,
        tree_cache: Any,
        model_worker: Any,
    ) -> bool:
        """Register the block one finished Compile request computed.

        Runs from the batch-result loop after the request has already produced
        its output, so an unmet post-condition caused by *the request* must not
        raise: that would end the scheduler, and with it every other request,
        for something that only costs this block its LegoLink name. The tokens
        come from the record this request was admitted with, never from the
        shared per-key map. Returns whether the block was pinned.

        Conditions the request can cause degrade to "not registered" and are
        traced: the Compile was aborted, its KV was kept out of the radix tree
        (a client-set bootstrap_host equal to the upstream sentinel does that),
        or the tree does not hold the whole block. Configuration invariants
        (no ModelRunner runtime to rebind into) still raise.
        """

        cache_key = get_pic_compile_cache_key(req.sampling_params)
        if cache_key is None:
            return False
        owner_id = self._find_compile_owner(req)
        record = self._compile_records.get(owner_id) if owner_id is not None else None
        if record is None or record.cache_key != cache_key:
            _emit_pic_trace(
                "compile_finish_without_admission",
                phase="compile",
                cache_key=cache_key,
            )
            return False
        if record.storage_mode != "reference":
            self._complete_compile_record(owner_id)
            return False

        skip_reason = self._compile_not_registrable_reason(req)
        if skip_reason is not None:
            self._abandon_finished_compile(owner_id, record, model_worker, skip_reason)
            return False
        token_ids = record.token_ids

        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        with _record_function("SGLang::RadixPrefixLookup"):
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=list(token_ids),
                        extra_key=req.extra_key,
                    )
                )
            )
        if len(match_result.device_indices) != len(token_ids):
            self._abandon_finished_compile(
                owner_id,
                record,
                model_worker,
                "radix_tree_missing_block_tokens "
                f"(matched={len(match_result.device_indices)}, "
                f"expected={len(token_ids)})",
            )
            return False

        self._release_pin(cache_key)
        self._tree_cache = tree_cache
        radixcache_owns_blocks = _env_flag("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1")
        lock_result = None
        if not radixcache_owns_blocks:
            lock_result = tree_cache.inc_lock_ref(match_result.last_device_node)
            self._pinned_blocks[cache_key] = (
                match_result.last_device_node,
                lock_result.to_dec_params(),
            )
            self._pinned_blocks.move_to_end(cache_key)

        model_runner = getattr(model_worker, "model_runner", None)
        runtime = getattr(model_runner, "_pic_runtime", None)
        if runtime is None:
            self._release_pin(cache_key)
            raise PICRuntimeError(
                "PIC ModelRunner runtime is unavailable for rebinding."
            )
        snapshot = runtime.rebind_reference_slots(
            cache_key, match_result.device_indices
        )
        runtime.set_legolink_cache(tree_cache)
        if radixcache_owns_blocks and hasattr(tree_cache, "register_legolink_block"):
            metadata = get_pic_compile_metadata(req.sampling_params)
            compatibility = runtime.get_legolink_compatibility(
                adapter_id=getattr(req, "lora_id", None)
            )
            compatibility.update(
                {
                    "segmentation_policy_version": metadata.get(
                        "segmentation_policy_version", "field-first-v1"
                    ),
                    "legolink_policy_version": metadata.get(
                        "legolink_policy_version",
                        compatibility.get("legolink_policy_version"),
                    ),
                }
            )
            self._legolink_compatibility = dict(compatibility)
            registered = self._register_compiled_block(
                tree_cache,
                cache_key=cache_key,
                block_id=metadata.get("prefill_unit_id") or cache_key,
                sub_context_id=metadata.get("sub_context_id")
                or metadata.get("source_sub_context_id")
                or cache_key,
                semantic_block_id=metadata.get("semantic_block_id")
                or metadata.get("prefill_unit_id")
                or cache_key,
                context_field=metadata.get("context_field") or "unknown",
                token_ids=token_ids,
                token_range=(0, len(token_ids)),
                kv_slots=match_result.device_indices,
                snapshot=snapshot,
                radix_node=match_result.last_device_node,
                pinned=False,
                compatibility=compatibility,
            )
            runtime.release_local_block(cache_key)
            if not registered:
                # A resident block already describes different content under
                # this key. The request has already produced its output and its
                # KV is in the radix tree; only the LegoLink name is refused, so
                # this costs the registration and nothing else.
                self._complete_compile_record(owner_id)
                return False
        self._complete_compile_record(owner_id)
        logger.info(
            "SGLang PIC scheduler pinned cache_key=%s tokens=%d",
            cache_key,
            len(token_ids),
        )
        return True

    @staticmethod
    def _compile_not_registrable_reason(req: Any) -> Optional[str]:
        """Why this finished Compile has no block to register, if it has none."""

        if _finish_reason_is_abort(
            getattr(req, "finished_reason", None)
        ) or _finish_reason_is_abort(getattr(req, "to_finish", None)):
            return "compile_aborted"
        if getattr(req, "skip_radix_cache_insert", False) is True:
            return "radix_insert_suppressed"
        return None

    def _abandon_finished_compile(
        self,
        owner_id: int,
        record: _CompileRecord,
        model_worker: Any,
        reason: str,
    ) -> None:
        """Finish a Compile that produced no registrable block, without raising."""

        self._drop_compile_record(owner_id, restore=True)
        if not self._inflight_owners.get(record.cache_key):
            self._release_runtime_snapshot(model_worker, record.cache_key)
        logger.warning(
            "SGLang PIC did not register compiled block cache_key=%s: %s",
            record.cache_key,
            reason,
        )
        _emit_pic_trace(
            "compile_not_registered",
            phase="compile",
            cache_key=record.cache_key,
            reason=reason,
        )

    @staticmethod
    def _register_compiled_block(tree_cache: Any, **kwargs: Any) -> bool:
        """Register a compiled block, and survive a refused duplicate.

        `pin_finished_compile` runs from the batch-result loop, after the
        request has already produced its output, so a raise here does not fail
        a request -- it kills the scheduler. A group key can now be claimed by
        a promotion as well as by a compile, which makes a same-key collision
        reachable in ordinary operation, so it has to degrade to "this block
        does not get a LegoLink name".
        """

        from sglang.srt.mem_cache.radix_cache import (
            LegoLinkDuplicateRegistrationError,
        )

        try:
            tree_cache.register_legolink_block(**kwargs)
        except LegoLinkDuplicateRegistrationError as error:
            logger.warning(
                "SGLang PIC refused a duplicate LegoLink registration for "
                "cache_key=%s: %s",
                kwargs.get("cache_key"),
                error,
            )
            _emit_pic_trace(
                "legolink_duplicate_registration_refused",
                phase="compile",
                cache_key=str(kwargs.get("cache_key")),
                reason=str(error)[:300],
            )
            return False
        return True

    def reserve_link_sources(self, sampling_params: Any) -> list[str]:
        """Validate and pin all source blocks before destination allocation.

        Scheduler admission may reclaim capacity from unpinned semantic blocks.
        Reserving the exact LinkPlan sources first prevents that allocation from
        evicting a block which ModelRunner is about to read.
        """

        link_plan = get_pic_link_plan(sampling_params)
        if (
            link_plan is None
            or not legolink_only_mode_enabled()
            or not _env_flag("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1")
        ):
            return []
        existing = link_plan.get("_scheduler_reserved_cache_keys")
        if isinstance(existing, list) and existing:
            return [str(cache_key) for cache_key in existing]

        tree_cache = self._tree_cache
        if tree_cache is None or not hasattr(tree_cache, "lookup_legolink_blocks"):
            raise PICRuntimeError(
                "LegoLink-only request cannot reserve source blocks because "
                "the RadixCache LegoLink registry is unavailable."
            )

        cache_keys = [
            str(cache_key) for cache_key in (link_plan.get("cache_keys") or [])
        ]
        ranges = (
            link_plan.get("block_ranges") or link_plan.get("attention_ranges") or []
        )
        explicit_block_ids = link_plan.get("block_ids") or []
        block_ids = []
        for index, cache_key in enumerate(cache_keys):
            range_block_id = (
                ranges[index].get("block_id")
                if index < len(ranges) and isinstance(ranges[index], dict)
                else None
            )
            block_ids.append(
                str(
                    range_block_id
                    or (
                        explicit_block_ids[index]
                        if index < len(explicit_block_ids)
                        else cache_key
                    )
                )
            )
        compatibility = dict(self._legolink_compatibility)
        adapter_id = link_plan.get("adapter_id")
        if adapter_id is not None:
            compatibility["adapter_id"] = str(adapter_id)
        segmentation_policy_version = link_plan.get("segmentation_policy_version")
        if segmentation_policy_version is not None:
            compatibility["segmentation_policy_version"] = str(
                segmentation_policy_version
            )
        legolink_policy_version = link_plan.get(
            "legolink_policy_version"
        ) or link_plan.get("physical_token_layout_version")
        if legolink_policy_version is not None:
            compatibility["legolink_policy_version"] = str(legolink_policy_version)
            compatibility["policy_version"] = str(legolink_policy_version)

        lookup_result = tree_cache.lookup_legolink_blocks(
            cache_keys,
            semantic_block_ids=block_ids,
            link_plan=link_plan,
            pin=False,
            compatibility=compatibility or None,
        )
        unavailable = [
            *getattr(lookup_result, "missing_blocks", []),
            *getattr(lookup_result, "stale_blocks", []),
            *getattr(lookup_result, "incompatible_blocks", []),
        ]
        if unavailable:
            _emit_pic_trace(
                "legolink_request_failed_closed",
                phase="scheduler",
                request_id=link_plan.get("request_id"),
                task_id=link_plan.get("task_id"),
                cache_keys=cache_keys,
                missing_blocks=list(getattr(lookup_result, "missing_blocks", [])),
                stale_blocks=list(getattr(lookup_result, "stale_blocks", [])),
                incompatible_blocks=list(
                    getattr(lookup_result, "incompatible_blocks", [])
                ),
                reason="scheduler_source_preflight_failed",
            )
            # The lookup already knows exactly which keys are gone and why. A
            # host that planned reuse from its own metadata cannot know that
            # without being told, so the classification travels out with the
            # abort instead of dying here as prose: the caller can invalidate
            # precisely those keys, recompile them under the same identity and
            # resend. Detection, fail-closed behaviour and eviction policy are
            # unchanged -- this only stops throwing the answer away.
            detail = {
                "noc_residency_recovery": {
                    "reason": "scheduler_source_preflight_failed",
                    "missing_cache_keys": [
                        str(item)
                        for item in getattr(lookup_result, "missing_blocks", [])
                    ],
                    "stale_cache_keys": [
                        str(item)
                        for item in getattr(lookup_result, "stale_blocks", [])
                    ],
                    "incompatible_cache_keys": [
                        str(item)
                        for item in getattr(lookup_result, "incompatible_blocks", [])
                    ],
                }
            }
            raise PICRuntimeError(
                "LegoLink source preflight failed before destination KV "
                "allocation: "
                + ", ".join(str(item) for item in unavailable)
                + " | "
                + json.dumps(detail, separators=(",", ":"))
            )

        handles = list(getattr(lookup_result, "matched_handles", []))
        # The lookup returns one handle per requested *position*. A cache key
        # that appears twice -- the same chunk at two places in one query --
        # is still one source block: reserved once and released once.
        distinct_handles = list(
            {str(handle.cache_key): handle for handle in handles}.values()
        )
        if len(distinct_handles) != len(dict.fromkeys(cache_keys)):
            raise PICRuntimeError(
                "LegoLink source preflight did not return every requested "
                "semantic block."
            )
        tree_cache.pin_legolink_blocks(distinct_handles)
        reserved = [str(handle.cache_key) for handle in distinct_handles]
        allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
        available_size = getattr(allocator, "available_size", None)
        capacity_evictable_size = getattr(tree_cache, "capacity_evictable_size", None)
        if callable(available_size) and callable(capacity_evictable_size):
            required_tokens = int(link_plan.get("total_token_count") or 0)
            allocatable_tokens = int(available_size()) + int(capacity_evictable_size())
            if required_tokens > allocatable_tokens:
                tree_cache.unpin_legolink_blocks(reserved)
                _emit_pic_trace(
                    "legolink_request_failed_closed",
                    phase="scheduler",
                    request_id=link_plan.get("request_id"),
                    task_id=link_plan.get("task_id"),
                    cache_keys=reserved,
                    reason="insufficient_request_private_kv_capacity",
                    required_tokens=required_tokens,
                    allocatable_tokens=allocatable_tokens,
                )
                raise PICRuntimeError(
                    "LegoLink cannot allocate request-private destination KV "
                    "without evicting its reserved source blocks: "
                    f"required={required_tokens}, "
                    f"allocatable={allocatable_tokens}."
                )
        link_plan["_scheduler_reserved_cache_keys"] = reserved
        _emit_pic_trace(
            "scheduler_link_sources_reserved",
            phase="link",
            request_id=link_plan.get("request_id"),
            task_id=link_plan.get("task_id"),
            cache_keys=reserved,
            block_count=len(reserved),
        )
        return reserved

    def release_link_reservation(self, sampling_params: Any) -> None:
        link_plan = get_pic_link_plan(sampling_params)
        if link_plan is None:
            return
        reserved = link_plan.pop("_scheduler_reserved_cache_keys", None)
        if (
            isinstance(reserved, list)
            and reserved
            and self._tree_cache is not None
            and hasattr(self._tree_cache, "unpin_legolink_blocks")
        ):
            self._tree_cache.unpin_legolink_blocks(
                [str(cache_key) for cache_key in reserved]
            )

    def retracted_link_abort_reason(self, req: Any) -> Optional[str]:
        """Decide a retracted request's fate; None means the native re-queue.

        A request that entered through a Link decoded against linked KV. Its
        re-prefill would be prompt + generated tokens, which cannot match the
        linked layout, and a native full prefill would silently change the KV
        it continues from. It fails closed instead. Any reservation it still
        holds is released under the marker's ownership rule, so calling this
        twice releases nothing twice.
        """

        sampling_params = getattr(req, "sampling_params", None)
        link_plan = get_pic_link_plan(sampling_params)
        if link_plan is None:
            return None
        self.release_link_reservation(sampling_params)
        _emit_pic_trace(
            "legolink_request_failed_closed",
            phase="scheduler",
            request_id=link_plan.get("request_id"),
            task_id=link_plan.get("task_id"),
            scheduler_rid=str(getattr(req, "rid", "")),
            cache_keys=[str(key) for key in (link_plan.get("cache_keys") or [])],
            reason="link_request_retracted",
        )
        return (
            "LegoLink request was retracted under KV pressure; its linked KV "
            "layout cannot be replayed, so it fails closed instead of being "
            "re-queued."
        )

    def _release_pin(self, cache_key: str) -> None:
        pinned = self._pinned_blocks.pop(cache_key, None)
        if pinned is not None and self._tree_cache is not None:
            if hasattr(self._tree_cache, "unpin_legolink_blocks"):
                self._tree_cache.unpin_legolink_blocks([cache_key])
            node, dec_params = pinned
            self._tree_cache.dec_lock_ref(node, dec_params)

    def clear(self, model_worker: Any = None) -> None:
        for cache_key in list(self._pinned_blocks):
            self._release_pin(cache_key)
        self._blocks.clear()
        self._storage_modes.clear()
        self._compile_records.clear()
        self._inflight_owners.clear()
        self._precompile_state.clear()
        self._tree_cache = None
        self._legolink_compatibility.clear()
        model_runner = getattr(model_worker, "model_runner", None)
        runtime = getattr(model_runner, "_pic_runtime", None)
        if runtime is not None:
            runtime.clear_registry()

    def _prepare_link(
        self,
        mutable_input_ids: list[int],
        link_plan: dict[str, Any],
        host_recorder: Optional[_HostTimingRecorder] = None,
    ) -> list[int]:
        host_recorder = host_recorder or _HostTimingRecorder("scheduler")
        # Before anything is derived, and before this method writes its own
        # fields into the same dict: after that point a host-sent physical
        # field is indistinguishable from a resolved one.
        validate_pic_link_request(link_plan)
        ordered_keys_started_ns = time.perf_counter_ns() if host_recorder.enabled else 0
        cache_keys = link_plan.get("cache_keys")
        if not isinstance(cache_keys, list) or not cache_keys:
            raise PICRuntimeError("pic_link.cache_keys must be non-empty.")
        cache_keys = [str(cache_key) for cache_key in cache_keys]
        normalized_layout = link_plan.get("physical_token_layout_version") is not None
        validated_link_input = validate_physical_token_input(
            link_plan,
            label="pic_link",
        )
        normalized_mutable_input_ids = [int(token_id) for token_id in mutable_input_ids]
        if normalized_layout:
            if validated_link_input is None:
                raise PICRuntimeError(
                    "normalized pic_link is missing physical_input_ids."
                )
            if normalized_mutable_input_ids != validated_link_input:
                raise PICRuntimeError(
                    "Link request tokens differ from the validated "
                    "physical_input_ids override."
                )
            prefix_count = link_plan.get("physical_prefix_token_count")
            if (
                isinstance(prefix_count, bool)
                or not isinstance(prefix_count, int)
                or prefix_count < 0
                or prefix_count >= len(normalized_mutable_input_ids)
            ):
                raise PICRuntimeError(
                    "pic_link.physical_prefix_token_count must split a "
                    "non-empty physical Link input."
                )
            prefix_input_ids = normalized_mutable_input_ids[:prefix_count]
            mutable_suffix_input_ids = normalized_mutable_input_ids[prefix_count:]
            link_plan.pop("physical_input_ids", None)
            raw_block_segments = link_plan.get("physical_block_segments")
            if (
                not isinstance(raw_block_segments, list)
                or len(raw_block_segments) != len(cache_keys)
                or any(not isinstance(segment, dict) for segment in raw_block_segments)
            ):
                raise PICRuntimeError(
                    "pic_link.physical_block_segments must contain one "
                    "metadata object per cache key."
                )
            physical_block_segments = raw_block_segments
            interblock_input_ids, mutable_suffix_input_ids = (
                _split_interblock_fresh_tokens(
                    link_plan, mutable_suffix_input_ids, len(cache_keys)
                )
            )
        else:
            if link_plan.get("physical_interblock_fresh_token_counts") is not None:
                raise PICRuntimeError(
                    "Fresh tokens between cached blocks require the normalized "
                    "physical token layout."
                )
            prefix_count = 0
            prefix_input_ids = []
            mutable_suffix_input_ids = normalized_mutable_input_ids
            physical_block_segments = []
            interblock_input_ids = [[] for _ in range(max(len(cache_keys) - 1, 0))]
        if host_recorder.enabled:
            host_recorder.add_ns(
                "ordered_cache_key_construction",
                time.perf_counter_ns() - ordered_keys_started_ns,
            )

        with host_recorder.span("token_budget_decision"):
            recompute_initial_tokens = int(
                link_plan["repair_decision"]["selected_k"]
                if link_plan.get("execution_mode") == "formal" else link_plan.get(
                    "recompute_initial_tokens", link_plan.get("legolink_k", 16)
                )
            )
        if recompute_initial_tokens < 0:
            raise PICRuntimeError("recompute_initial_tokens must be non-negative.")
        recompute_first_block = bool(link_plan.get("recompute_first_block", False))
        raw_per_block = link_plan.get("recompute_tokens_per_block")
        if raw_per_block is not None:
            if not isinstance(raw_per_block, list) or len(raw_per_block) != len(
                cache_keys
            ):
                raise PICRuntimeError(
                    "pic_link.recompute_tokens_per_block must contain one value per cache key."
                )
            recompute_tokens_per_block = [int(value) for value in raw_per_block]
            if any(value < 0 for value in recompute_tokens_per_block):
                raise PICRuntimeError(
                    "pic_link.recompute_tokens_per_block values must be non-negative."
                )
        else:
            first_block_tokens = link_plan.get("recompute_first_block_tokens")
            if first_block_tokens is None:
                first_block_tokens = (
                    recompute_initial_tokens if recompute_first_block else 0
                )
            first_block_tokens = int(first_block_tokens)
            if first_block_tokens < 0:
                raise PICRuntimeError(
                    "pic_link.recompute_first_block_tokens must be non-negative."
                )
            recompute_tokens_per_block = [
                first_block_tokens if index == 0 else recompute_initial_tokens
                for index in range(len(cache_keys))
            ]
        recompute_first_block = recompute_tokens_per_block[0] > 0

        full_input_ids: list[int] = list(prefix_input_ids)
        block_ranges: list[dict[str, Any]] = []
        recompute_positions: list[int] = list(range(len(prefix_input_ids)))
        block_ids = link_plan.get("block_ids") or []
        fresh_gap_ranges: list[dict[str, int]] = []

        with (
            _profile_range("LinkPlanBuild"),
            host_recorder.span("link_plan_construction"),
        ):
            for block_index, raw_cache_key in enumerate(cache_keys):
                cache_key = str(raw_cache_key)
                token_ids = None
                if self._tree_cache is not None and hasattr(
                    self._tree_cache, "get_legolink_block_tokens"
                ):
                    token_ids = self._tree_cache.get_legolink_block_tokens(cache_key)
                if token_ids is None:
                    token_ids = self._blocks.get(cache_key)
                if token_ids is None:
                    raise PICRuntimeError(
                        f"PIC block is not compiled in this scheduler: {cache_key}"
                    )
                token_ids = tuple(int(token_id) for token_id in token_ids)
                if normalized_layout:
                    expected_segment = physical_block_segments[block_index]
                    expected_cache_key = expected_segment.get("cache_key")
                    if str(expected_cache_key) != cache_key:
                        raise PICRuntimeError(
                            "physical block segment order/cache key mismatch: "
                            f"expected {cache_key}, got {expected_cache_key!r}"
                        )
                    expected_count = expected_segment.get("physical_token_count")
                    if (
                        isinstance(expected_count, bool)
                        or not isinstance(expected_count, int)
                        or expected_count != len(token_ids)
                    ):
                        raise PICRuntimeError(
                            f"physical block token count mismatch for "
                            f"{cache_key}: expected={expected_count!r}, "
                            f"actual={len(token_ids)}"
                        )
                    expected_hash = expected_segment.get("physical_token_ids_sha256")
                    actual_hash = physical_token_ids_sha256(token_ids)
                    if (
                        not isinstance(expected_hash, str)
                        or expected_hash != actual_hash
                    ):
                        raise PICRuntimeError(
                            f"physical block token hash mismatch for "
                            f"{cache_key}: expected={expected_hash!r}, "
                            f"actual={actual_hash}"
                        )
                if cache_key in self._blocks:
                    self._blocks.move_to_end(cache_key)
                start = len(full_input_ids)
                full_input_ids.extend(token_ids)
                end = len(full_input_ids)
                block_ranges.append(
                    {
                        "block_id": (
                            str(block_ids[block_index])
                            if block_index < len(block_ids)
                            else cache_key
                        ),
                        "cache_key": cache_key,
                        "start_token": start,
                        "end_token": end,
                        "token_count": end - start,
                    }
                )
                block_recompute_tokens = recompute_tokens_per_block[block_index]
                if block_recompute_tokens:
                    recompute_positions.extend(
                        range(start, min(start + block_recompute_tokens, end))
                    )
                # EXPERIMENTAL indexed mixed prefill: host-owned fresh tokens
                # between this block and the next. They are computed from
                # scratch like the prefix and suffix, and every later block is
                # placed -- and RoPE-remapped -- after them.
                if (
                    block_index < len(interblock_input_ids)
                    and interblock_input_ids[block_index]
                ):
                    gap_start = len(full_input_ids)
                    full_input_ids.extend(interblock_input_ids[block_index])
                    recompute_positions.extend(range(gap_start, len(full_input_ids)))
                    fresh_gap_ranges.append(
                        {
                            "after_block_index": block_index,
                            "start_token": gap_start,
                            "end_token": len(full_input_ids),
                        }
                    )

        with host_recorder.span("input_ids_preparation"):
            mutable_start_token = len(full_input_ids)
            full_input_ids.extend(mutable_suffix_input_ids)
            recompute_positions.extend(range(mutable_start_token, len(full_input_ids)))

        if not mutable_suffix_input_ids:
            raise PICRuntimeError("PIC link requires a non-empty mutable query.")
        if normalized_layout:
            expected_complete_count = link_plan.get("expected_complete_token_count")
            if (
                isinstance(expected_complete_count, bool)
                or not isinstance(expected_complete_count, int)
                or expected_complete_count != len(full_input_ids)
            ):
                raise PICRuntimeError(
                    "normalized Link complete token count mismatch: "
                    f"expected={expected_complete_count!r}, "
                    f"actual={len(full_input_ids)}"
                )
            expected_complete_hash = link_plan.get("expected_complete_token_ids_sha256")
            actual_complete_hash = physical_token_ids_sha256(full_input_ids)
            if (
                not isinstance(expected_complete_hash, str)
                or expected_complete_hash != actual_complete_hash
            ):
                raise PICRuntimeError(
                    "normalized Link complete token hash mismatch: "
                    f"expected={expected_complete_hash!r}, "
                    f"actual={actual_complete_hash}"
                )

        with host_recorder.span("global_offset_position_map_construction"):
            resolved = ResolvedLinkPlan(
                cache_keys=tuple(cache_keys),
                block_ranges=ResolvedLinkPlan.freeze_ranges(block_ranges),
                mutable_start_token=mutable_start_token,
                mutable_token_count=len(mutable_suffix_input_ids),
                physical_prefix_token_count=len(prefix_input_ids),
                total_token_count=len(full_input_ids),
                recompute_positions=tuple(recompute_positions),
                recompute_tokens_per_block=tuple(recompute_tokens_per_block),
                recompute_first_block=recompute_first_block,
                recompute_initial_tokens=recompute_initial_tokens,
                token_layout_source=(
                    "single_chat_template_physical_override"
                    if normalized_layout
                    else "sglang_tokenizer_runtime"
                ),
                linked_token_ids=tuple(full_input_ids),
            )
            self._last_resolved_link_plan = resolved
            resolved.apply_to(link_plan)
        _emit_pic_trace(
            "scheduler_link_prepared",
            execution_mode=link_plan.get("execution_mode", "legacy_debug"),
            framing_spec=link_plan.get("framing_spec"),
            repair_decision=link_plan.get("repair_decision"),
            phase="link",
            request_id=link_plan.get("request_id"),
            task_id=link_plan.get("task_id"),
            variant=link_plan.get("variant"),
            cache_keys=[str(item) for item in cache_keys],
            block_ranges=block_ranges,
            physical_token_layout_version=link_plan.get(
                "physical_token_layout_version"
            ),
            physical_prefix_token_count=len(prefix_input_ids),
            physical_token_ids_exact=normalized_layout,
            fresh_gap_ranges=fresh_gap_ranges,
            mutable_start_token=mutable_start_token,
            total_token_count=len(full_input_ids),
            legolink_k=recompute_initial_tokens,
            recompute_first_block=recompute_first_block,
            recompute_tokens_per_block=recompute_tokens_per_block,
            kvlink=_trace_values(recompute_positions),
            imp_indices=_trace_values(recompute_positions),
        )
        return full_input_ids


@dataclass
class PICBlockSnapshot:
    cache_key: str
    token_ids: tuple[int, ...]
    storage_mode: str = "clone"
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    source_slots: Optional[torch.Tensor] = None
    reference_ready: bool = False
    kv_bytes: int = 0

    @property
    def owned_bytes(self) -> int:
        tensors = (*self.keys.values(), *self.values.values())
        tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        slot_bytes = (
            self.source_slots.numel() * self.source_slots.element_size()
            if self.source_slots is not None
            else 0
        )
        return tensor_bytes + slot_bytes


class PICRuntime:
    """ModelRunner-side KV snapshot registry and link executor."""

    def __init__(self, model_runner: Any, max_blocks: Optional[int] = None):
        self.model_runner = model_runner
        self.max_blocks = max_blocks or int(os.getenv("SGLANG_PIC_MAX_BLOCKS", "256"))
        self._blocks: OrderedDict[str, PICBlockSnapshot] = OrderedDict()
        self._legolink_cache: Any = None
        self._rotary_by_layer = self._find_rotary_embeddings(model_runner.model)

    @staticmethod
    def _find_rotary_embeddings(model: Any) -> dict[int, Any]:
        rotary_by_layer: dict[int, Any] = {}
        for module in model.modules():
            attention = getattr(module, "attn", None)
            rotary = getattr(module, "rotary_emb", None)
            layer_id = getattr(attention, "layer_id", None)
            if layer_id is not None and rotary is not None:
                rotary_by_layer[int(layer_id)] = rotary
        return rotary_by_layer

    @staticmethod
    def has_link(forward_batch: Any) -> bool:
        metadata = getattr(forward_batch, "pic_request_metadata", None)
        return bool(
            forward_batch.forward_mode.is_extend()
            and metadata
            and len(metadata) == 1
            and isinstance(metadata[0].get("link_plan"), dict)
        )

    def can_run_link(self, forward_batch: Any) -> bool:
        metadata = getattr(forward_batch, "pic_request_metadata", None)
        is_link = self.has_link(forward_batch)
        if not is_link or not _env_flag("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1"):
            return is_link
        link_plan = metadata[0]["link_plan"]
        cache_keys = [str(item) for item in link_plan.get("cache_keys") or []]
        reserved_cache_keys = [
            str(item)
            for item in (link_plan.get("_scheduler_reserved_cache_keys") or [])
        ]
        if (
            reserved_cache_keys
            and list(dict.fromkeys(cache_keys)) == reserved_cache_keys
        ):
            return True
        ranges = (
            link_plan.get("block_ranges") or link_plan.get("attention_ranges") or []
        )
        tree_cache = getattr(self, "_legolink_cache", None)
        if tree_cache is None or not hasattr(tree_cache, "get_legolink_block"):
            recorder = _HostTimingRecorder(
                "fallback",
                request_id=link_plan.get("request_id"),
                task_id=link_plan.get("task_id"),
            )
            with recorder.span("fallback_preparation"):
                pass
            if legolink_only_mode_enabled():
                _emit_pic_trace(
                    "legolink_request_failed_closed",
                    phase="link",
                    request_id=link_plan.get("request_id"),
                    task_id=link_plan.get("task_id"),
                    cache_keys=cache_keys,
                    reason="legolink_cache_unavailable",
                )
                recorder.emit(status="error", reason="legolink_cache_unavailable")
                raise PICRuntimeError(
                    "LegoLink-only request cannot continue because the "
                    "RadixCache LegoLink registry is unavailable."
                )
            recorder.emit(status="fallback", reason="legolink_cache_unavailable")
            return False
        unavailable: list[str] = []
        for index, cache_key in enumerate(cache_keys):
            handle = tree_cache.get_legolink_block(cache_key)
            expected_id = (
                str(ranges[index].get("block_id") or cache_key)
                if index < len(ranges)
                else cache_key
            )
            if handle is None or handle.semantic_block_id not in (
                expected_id,
                cache_key,
            ):
                unavailable.append(cache_key)
        if unavailable:
            recorder = _HostTimingRecorder(
                "fallback",
                request_id=link_plan.get("request_id"),
                task_id=link_plan.get("task_id"),
                variant=link_plan.get("variant"),
            )
            fallback_started_ns = time.perf_counter_ns() if recorder.enabled else 0
            fallback_tokens = int(link_plan.get("total_token_count") or 0)
            details = {
                "phase": "link",
                "request_id": link_plan.get("request_id"),
                "task_id": link_plan.get("task_id"),
                "cache_keys": cache_keys,
                "missing_blocks": unavailable,
                "fallback_tokens": fallback_tokens,
                "reason": "missing_stale_or_incompatible_block",
            }
            if recorder.enabled:
                recorder.add_ns(
                    "fallback_preparation",
                    time.perf_counter_ns() - fallback_started_ns,
                )
            if legolink_only_mode_enabled():
                _emit_pic_trace("legolink_request_failed_closed", **details)
                recorder.emit(
                    status="error",
                    reason="missing_stale_or_incompatible_block",
                    missing_blocks=unavailable,
                )
                raise PICRuntimeError(
                    "LegoLink-only request cannot continue because required "
                    "RadixCache blocks are missing, stale, or incompatible: "
                    + ", ".join(unavailable)
                )
            if hasattr(tree_cache, "record_legolink_fallback"):
                tree_cache.record_legolink_fallback(fallback_tokens)
            _emit_pic_trace("radix_legolink_fallback_prefill", **details)
            _emit_pic_trace("legolink_fallback_to_full_prefill", **details)
            recorder.emit(
                status="fallback",
                reason="missing_stale_or_incompatible_block",
                missing_blocks=unavailable,
            )
            return False
        return True

    def capture_compile_blocks(self, forward_batch: Any) -> None:
        if not forward_batch.forward_mode.is_extend():
            return
        metadata = getattr(forward_batch, "pic_request_metadata", None)
        full_input_ids = getattr(forward_batch, "full_input_ids", None)
        if not metadata or not full_input_ids:
            return

        compile_indices = [
            index
            for index, request_metadata in enumerate(metadata)
            if request_metadata.get("compile_cache_key")
            and request_metadata.get("link_plan") is None
        ]
        if not compile_indices:
            return

        pool = forward_batch.token_to_kv_pool
        self._validate_pool(pool)
        for request_index in compile_indices:
            request_metadata = metadata[request_index]
            cache_key = request_metadata.get("compile_cache_key")
            storage_mode = request_metadata.get("compile_storage_mode") or "reference"
            token_ids = tuple(
                int(token_id) for token_id in full_input_ids[request_index]
            )
            seq_len = len(token_ids)
            req_pool_idx = int(forward_batch.req_pool_indices[request_index].item())
            slots = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :seq_len
            ].long()
            keys: dict[int, torch.Tensor] = {}
            values: dict[int, torch.Tensor] = {}
            kv_bytes = 0
            if storage_mode == "clone":
                for layer_id in range(
                    self.model_runner.start_layer, self.model_runner.end_layer
                ):
                    keys[layer_id] = pool.get_key_buffer(layer_id)[slots].clone()
                    values[layer_id] = pool.get_value_buffer(layer_id)[slots].clone()
                    kv_bytes += (
                        keys[layer_id].numel() * keys[layer_id].element_size()
                        + values[layer_id].numel() * values[layer_id].element_size()
                    )
                source_slots = None
                reference_ready = False
            elif storage_mode == "reference":
                source_slots = slots.clone()
                reference_ready = False
                for layer_id in range(
                    self.model_runner.start_layer, self.model_runner.end_layer
                ):
                    key_buffer = pool.get_key_buffer(layer_id)
                    value_buffer = pool.get_value_buffer(layer_id)
                    kv_bytes += seq_len * (
                        key_buffer[0].numel() * key_buffer.element_size()
                        + value_buffer[0].numel() * value_buffer.element_size()
                    )
            else:
                raise PICRuntimeError(f"Unsupported PIC storage mode: {storage_mode}")

            snapshot = PICBlockSnapshot(
                cache_key=str(cache_key),
                token_ids=token_ids,
                storage_mode=storage_mode,
                keys=keys,
                values=values,
                source_slots=source_slots,
                reference_ready=reference_ready,
                kv_bytes=kv_bytes,
            )
            self._blocks[str(cache_key)] = snapshot
            logger.info(
                "SGLang PIC compile captured cache_key=%s mode=%s tokens=%d "
                "registry_bytes=%d kv_bytes=%d avoided_bytes=%d",
                cache_key,
                storage_mode,
                seq_len,
                snapshot.owned_bytes,
                kv_bytes,
                kv_bytes - snapshot.owned_bytes,
            )
            _emit_pic_trace(
                "model_compile_captured",
                phase="compile",
                cache_key=str(cache_key),
                storage_mode=storage_mode,
                token_count=seq_len,
                source_slots=_trace_values(slots),
                layer_start=self.model_runner.start_layer,
                layer_end=self.model_runner.end_layer,
                registry_bytes=snapshot.owned_bytes,
                represented_kv_bytes=kv_bytes,
            )
            self._blocks.move_to_end(str(cache_key))
            while len(self._blocks) > self.max_blocks:
                self._blocks.popitem(last=False)

    def rebind_reference_slots(
        self, cache_key: str, source_slots: torch.Tensor
    ) -> PICBlockSnapshot:
        snapshot = self._blocks.get(str(cache_key))
        if snapshot is None:
            raise PICRuntimeError(f"PIC cannot rebind missing block {cache_key}.")
        if snapshot.storage_mode != "reference":
            return snapshot
        if len(source_slots) != len(snapshot.token_ids):
            raise PICRuntimeError(f"PIC reference slot count mismatch for {cache_key}.")
        snapshot.source_slots = source_slots.long().clone()
        snapshot.reference_ready = True
        self._blocks.move_to_end(str(cache_key))
        registry_bytes, kv_bytes = self.registry_memory_bytes()
        logger.info(
            "SGLang PIC reference rebound cache_key=%s tokens=%d "
            "registry_bytes=%d represented_kv_bytes=%d avoided_bytes=%d",
            cache_key,
            len(snapshot.token_ids),
            registry_bytes,
            kv_bytes,
            kv_bytes - registry_bytes,
        )
        _emit_pic_trace(
            "model_reference_rebound",
            phase="compile",
            cache_key=str(cache_key),
            token_count=len(snapshot.token_ids),
            source_slots=_trace_values(snapshot.source_slots),
        )
        return snapshot

    def set_legolink_cache(self, tree_cache: Any) -> None:
        self._legolink_cache = tree_cache

    def get_legolink_compatibility(
        self, *, adapter_id: Optional[str] = None
    ) -> dict[str, Any]:
        model_config = getattr(self.model_runner, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        model_fingerprint = (
            getattr(model_config, "model_path", None)
            or getattr(model_config, "model", None)
            or getattr(hf_config, "_name_or_path", None)
            or type(getattr(self.model_runner, "model", None)).__name__
        )
        revision = (
            getattr(model_config, "revision", None)
            or getattr(hf_config, "_commit_hash", None)
            or "runtime-local"
        )
        tokenizer_fingerprint = os.getenv(
            "SGLANG_PIC_TOKENIZER_FINGERPRINT",
            f"{model_fingerprint}:{getattr(hf_config, 'vocab_size', 'unknown')}",
        )
        rope_values = {
            "rope_theta": getattr(hf_config, "rope_theta", None),
            "rope_scaling": getattr(hf_config, "rope_scaling", None),
            "max_position_embeddings": getattr(
                hf_config, "max_position_embeddings", None
            ),
        }
        rope_fingerprint = json.dumps(
            rope_values, sort_keys=True, default=str, separators=(",", ":")
        )
        kv_dtype = (
            getattr(self.model_runner, "kv_cache_dtype", None)
            or getattr(model_config, "dtype", None)
            or getattr(self.model_runner, "dtype", None)
            or "runtime-local"
        )
        attention_backend = getattr(self.model_runner, "attn_backend", None)
        attention_backend_fingerprint = (
            f"{type(attention_backend).__module__}."
            f"{type(attention_backend).__qualname__}"
            if attention_backend is not None
            else "runtime-local"
        )
        policy_version = os.getenv("SGLANG_PIC_POLICY_VERSION", "legolink-radix-v1")
        return {
            "model_fingerprint": f"{model_fingerprint}@{revision}",
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "kv_dtype": str(kv_dtype),
            "rope_fingerprint": rope_fingerprint,
            "attention_backend_fingerprint": attention_backend_fingerprint,
            "adapter_id": str(adapter_id or "none"),
            "tp_world_size": int(getattr(self.model_runner, "tp_size", 1)),
            "segmentation_policy_version": os.getenv(
                "SGLANG_PIC_SEGMENTATION_POLICY_VERSION", "field-first-v1"
            ),
            "legolink_policy_version": policy_version,
            "policy_version": policy_version,
        }

    def get_legolink_block(self, cache_key: str) -> Optional[PICBlockSnapshot]:
        tree_cache = getattr(self, "_legolink_cache", None)
        if tree_cache is not None and hasattr(
            tree_cache, "get_legolink_block_snapshot"
        ):
            snapshot = tree_cache.get_legolink_block_snapshot(str(cache_key))
            if snapshot is not None:
                return snapshot
        if _env_flag("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1"):
            return None
        return self._blocks.get(str(cache_key))

    def release_local_block(self, cache_key: str) -> Optional[PICBlockSnapshot]:
        return self._blocks.pop(str(cache_key), None)

    def registry_memory_bytes(self) -> tuple[int, int]:
        tree_cache = getattr(self, "_legolink_cache", None)
        if tree_cache is not None and hasattr(tree_cache, "legolink_memory_bytes"):
            radix_bytes = tree_cache.legolink_memory_bytes()
            if radix_bytes != (0, 0):
                return radix_bytes
        return (
            sum(snapshot.owned_bytes for snapshot in self._blocks.values()),
            sum(snapshot.kv_bytes for snapshot in self._blocks.values()),
        )

    def clear_registry(self) -> None:
        self._blocks.clear()
        tree_cache = getattr(self, "_legolink_cache", None)
        if tree_cache is not None and hasattr(tree_cache, "clear_legolink_blocks"):
            tree_cache.clear_legolink_blocks()

    def _validate_tp_link_plan(
        self, link_plan: dict[str, Any], imp_indices: list[int]
    ) -> None:
        if not _env_flag("SGLANG_PIC_VALIDATE_TP", "0"):
            return
        tp_size = int(getattr(self.model_runner, "tp_size", 1))
        if tp_size <= 1:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            raise PICRuntimeError("PIC TP validation requires torch.distributed.")
        ranges = (
            link_plan.get("block_ranges") or link_plan.get("attention_ranges") or []
        )
        cache_keys = [str(item) for item in link_plan.get("cache_keys") or []]
        manifest = {
            "cache_keys": cache_keys,
            "block_token_lengths": [
                int(item["end_token"]) - int(item["start_token"]) for item in ranges
            ],
            "destination_global_offsets": [int(item["start_token"]) for item in ranges],
            "imp_indices": list(imp_indices),
            "local_lookup_success": all(
                self.get_legolink_block(cache_key) is not None
                for cache_key in cache_keys
            ),
        }
        manifests: list[Optional[dict[str, Any]]] = [None] * tp_size
        group = getattr(getattr(self.model_runner, "tp_group", None), "cpu_group", None)
        dist.all_gather_object(manifests, manifest, group=group)
        _validate_tp_rank_manifests([item for item in manifests if item is not None])

    @_trace_link_failure
    @_guard_legolink_release
    def run_link(self, forward_batch: Any) -> Any:
        started = _profile_start()
        host_recorder = getattr(
            self, "_host_timing_recorder", None
        ) or _HostTimingRecorder("model_runner")
        self._last_legolink_execution_handles = []
        self._last_legolink_lookup_elapsed_s = 0.0
        with host_recorder.span("semantic_block_metadata_parsing"):
            self._validate_link_batch(forward_batch)
            metadata = forward_batch.pic_request_metadata[0]
            link_plan = metadata["link_plan"]
            validate_runtime_execution_intent(link_plan)
            if link_plan.get("execution_mode") == "formal":
                actual_model = getattr(getattr(self.model_runner, "model_config", None), "model_path", None)
                if actual_model != link_plan["framing_spec"]["model_id"]:
                    raise PICRuntimeError("formal framing profile does not match loaded model")
            full_input_ids = forward_batch.full_input_ids[0]
        if metadata.get("scheduler_waiting_ns") is not None:
            host_recorder.add_ns(
                "scheduler_waiting_time", int(metadata["scheduler_waiting_ns"])
            )
        total_token_count = int(link_plan["total_token_count"])
        scheduler_rids = [str(rid) for rid in (forward_batch.rids or [])]
        if len(full_input_ids) != total_token_count:
            raise PICRuntimeError(
                "Scheduler/model token layout mismatch for PIC link request."
            )

        with (
            _profile_range("DestinationKVAllocation"),
            host_recorder.span("request_private_destination_kv_allocation"),
        ):
            # Allocation is performed by the native scheduler before ModelRunner.
            # This range measures the request-private allocation handoff and exact
            # destination-slot materialization used by LegoLink.
            req_pool_idx = int(forward_batch.req_pool_indices[0].item())
            destination_slots = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :total_token_count
            ].long()
        _emit_pic_trace(
            "destination_kv_allocated",
            phase="link",
            request_id=link_plan.get("request_id"),
            task_id=link_plan.get("task_id"),
            total_token_count=total_token_count,
            request_private=True,
            destination_slots=_trace_values(destination_slots),
        )
        imp_indices = [int(position) for position in link_plan["imp_indices"]]
        if not imp_indices or imp_indices[-1] != total_token_count - 1:
            raise PICRuntimeError(
                "PIC imp_indices must end at the final mutable-query token."
            )
        self._validate_tp_link_plan(link_plan, imp_indices)
        recompute_mode = str(
            link_plan.get(
                "recompute_mode",
                os.getenv("SGLANG_PIC_RECOMPUTE_MODE", "batched"),
            )
        ).lower()
        trace_context = {
            "request_id": link_plan.get("request_id"),
            "task_id": link_plan.get("task_id"),
            "variant": link_plan.get("variant"),
            "link_trace_id": f"{os.getpid()}-{time.time_ns()}",
            "rids": scheduler_rids,
        }
        self._active_trace_context = trace_context
        block_ranges = (
            link_plan.get("block_ranges") or link_plan.get("attention_ranges") or []
        )
        self._active_sparse_repair_block_count = len(block_ranges)
        self._active_sparse_repair_block_size = max(
            (
                int(item.get("end_token", 0)) - int(item.get("start_token", 0))
                for item in block_ranges
            ),
            default=0,
        )
        self._active_sparse_repair_batches = []
        self._request_capacity_degraded = False
        _emit_pic_trace(
            "model_link_start",
            execution_mode=link_plan.get("execution_mode", "legacy_debug"),
            framing_spec=link_plan.get("framing_spec"),
            repair_decision=link_plan.get("repair_decision"),
            phase="link",
            **trace_context,
            recompute_mode=recompute_mode,
            cache_keys=[str(item) for item in link_plan["cache_keys"]],
            block_ranges=link_plan.get("block_ranges")
            or link_plan.get("attention_ranges"),
            destination_slots=_trace_values(destination_slots),
            kvlink=_trace_values(imp_indices),
            imp_indices=_trace_values(imp_indices),
            # Deferred: _contiguous_ranges walks every repair position, and the
            # result is only ever read from a written trace record.
            recompute_ranges=_lazy_trace(
                lambda: [
                    {"start_token": start, "end_token": end}
                    for start, end in _contiguous_ranges(imp_indices)
                ]
            ),
            total_token_count=total_token_count,
        )
        prepare_s = _profile_elapsed(started)
        if recompute_mode == "full_causal":
            recompute_started = _profile_start()
            recompute_cuda_started = _cuda_event_start("model_prefill")
            output, forward_count = self._run_full_causal(
                forward_batch,
                full_input_ids,
                destination_slots,
            )
            recompute_cuda_ms = _cuda_event_elapsed_ms(recompute_cuda_started)
            cuda_summary = _finalize_cuda_events(self)
            recompute_cuda_ms = cuda_summary.get("model_prefill", 0.0)
            recompute_s = _profile_elapsed(recompute_started)
            elapsed_s = _profile_elapsed(started)
            logger.info(
                "SGLang PIC link complete mode=%s linked_tokens=%d "
                "recompute_tokens=%d model_forwards=%d elapsed_s=%.6f",
                recompute_mode,
                total_token_count,
                total_token_count,
                forward_count,
                elapsed_s,
            )
            _emit_pic_trace(
                "model_link_complete",
                phase="link",
                **trace_context,
                recompute_mode=recompute_mode,
                linked_tokens=total_token_count,
                reused_tokens=0,
                recompute_tokens=total_token_count,
                evicted_tokens=0,
                model_forwards=forward_count,
                elapsed_s=elapsed_s,
                timing_breakdown_s={
                    "prepare": prepare_s,
                    "kv_retrieval": 0.0,
                    "rope_remap": 0.0,
                    "kv_writeback": 0.0,
                    "kv_link_total": 0.0,
                    "sparse_recompute": recompute_s,
                    "total": elapsed_s,
                },
                timing_breakdown_cuda_ms={
                    "kv_retrieval": 0.0,
                    "rope_remap": 0.0,
                    "kv_writeback": 0.0,
                    "kv_link_total": 0.0,
                    "sparse_recompute": recompute_cuda_ms,
                },
            )
            with (
                _profile_range("NativeDecode"),
                host_recorder.span("native_decode_handoff"),
            ):
                return output

        kv_link_started = _profile_start()
        self._last_legolink_lookup_evicted_tokens = 0
        with host_recorder.span("legolink_link_host_total"):
            kv_timing = self._copy_linked_kv(
                forward_batch.token_to_kv_pool,
                destination_slots,
                link_plan,
            )
        kv_link_s = _profile_elapsed(kv_link_started)
        _debug_capture_post_remap_k(
            forward_batch.token_to_kv_pool,
            destination_slots,
            full_input_ids,
            link_plan,
            {
                "request_id": link_plan.get("request_id"),
                "task_id": link_plan.get("task_id"),
                "total_token_count": total_token_count,
            },
        )
        logger.info(
            "SGLang PIC link start blocks=%d linked_tokens=%d reused_tokens=%d "
            "recompute_tokens=%d",
            len(link_plan["cache_keys"]),
            total_token_count,
            total_token_count - len(imp_indices),
            len(imp_indices),
        )

        recompute_started = _profile_start()
        recompute_cuda_started = _cuda_event_start("sparse_recompute")
        with _profile_range("LegoLinkSparseRecompute"):
            if recompute_mode == "batched":
                output, forward_count = self._run_batched_recompute(
                    forward_batch,
                    full_input_ids,
                    destination_slots,
                    imp_indices,
                )
            elif recompute_mode == "tokenwise":
                output, forward_count = self._run_tokenwise_recompute(
                    forward_batch,
                    full_input_ids,
                    destination_slots,
                    imp_indices,
                )
            else:
                raise PICRuntimeError(
                    "PIC recompute_mode must be 'batched', 'tokenwise', "
                    "or 'full_causal'."
                )
        recompute_cuda_ms = _cuda_event_elapsed_ms(recompute_cuda_started)
        _debug_capture_post_repair_kv(
            forward_batch.token_to_kv_pool,
            destination_slots,
            full_input_ids,
            link_plan,
            imp_indices,
            {
                "request_id": link_plan.get("request_id"),
                "task_id": link_plan.get("task_id"),
                "total_token_count": total_token_count,
                "repaired_token_count": len(imp_indices),
            },
        )
        # EXPERIMENTAL, off unless the deployment and the request both ask.
        # The grouped state exists exactly here -- reconstructed, repaired, and
        # about to be discarded with the request's slots. Promotion runs after
        # repair and before decode so that a failure to persist can only cost a
        # trace row, never an answer.
        promotion = None
        if group_promotion_enabled() and link_plan.get(
            "promote_reconstructed_group"
        ):
            with host_recorder.span("execution_group_promotion"):
                promotion = self._promote_reconstructed_group(
                    pool=forward_batch.token_to_kv_pool,
                    destination_slots=destination_slots,
                    full_input_ids=full_input_ids,
                    link_plan=link_plan,
                    recompute_tokens_per_block=list(
                        link_plan.get("recompute_tokens_per_block") or []
                    ),
                    trace_context=trace_context,
                )

        from sglang.srt.flashinfer_sparse_repair import (
            finalize_sparse_repair_metadata,
        )

        sparse_attention_summaries = finalize_sparse_repair_metadata(
            self._active_sparse_repair_batches
        )
        for summary in sparse_attention_summaries:
            _emit_pic_trace(
                "sparse_repair_attention",
                phase="link",
                **trace_context,
                **summary,
            )
        _cuda_event_elapsed_ms(kv_timing.pop("_legolink_total_span", None))
        cuda_summary = _finalize_cuda_events(self)
        kv_link_cuda_ms = cuda_summary.get("kv_link_total", 0.0)
        recompute_cuda_ms = cuda_summary.get("sparse_recompute", 0.0)
        recompute_s = _profile_elapsed(recompute_started)
        elapsed_s = _profile_elapsed(started)
        execution_handles = list(getattr(self, "_last_legolink_execution_handles", []))
        tree_cache = getattr(self, "_legolink_cache", None)
        if execution_handles and hasattr(tree_cache, "record_legolink_execution"):
            from sglang.srt.mem_cache.radix_cache import LegoLinkExecutionResult

            block_ranges = (
                link_plan.get("block_ranges") or link_plan.get("attention_ranges") or []
            )
            copied_tokens = sum(
                max(
                    int(block_range.get("end_token", 0))
                    - int(block_range.get("start_token", 0)),
                    0,
                )
                for block_range in block_ranges
            )
            remapped_tokens = sum(
                max(
                    int(block_range.get("end_token", 0))
                    - int(block_range.get("start_token", 0)),
                    0,
                )
                for block_range in block_ranges[1:]
            )
            recompute_positions = set(imp_indices)
            per_block_accounting: dict[str, dict[str, int]] = {}
            for cache_key, block_range in zip(
                link_plan["cache_keys"], block_ranges, strict=False
            ):
                start_token = int(block_range.get("start_token", 0))
                end_token = int(block_range.get("end_token", start_token))
                block_tokens = max(end_token - start_token, 0)
                recomputed = sum(
                    int(position in recompute_positions)
                    for position in range(start_token, end_token)
                )
                per_block_accounting[str(cache_key)] = {
                    "actual_reused_tokens": block_tokens - recomputed,
                    "actual_recomputed_tokens": recomputed,
                }
            with (
                _profile_range("ExecutionAccounting"),
                host_recorder.span("record_legolink_execution"),
            ):
                tree_cache.record_legolink_execution(
                    execution_handles,
                    LegoLinkExecutionResult(
                        actual_reused_tokens=total_token_count - len(imp_indices),
                        actual_recomputed_tokens=len(imp_indices),
                        copied_tokens=copied_tokens,
                        remapped_tokens=remapped_tokens,
                        writeback_tokens=copied_tokens,
                        linked_blocks=[str(item) for item in link_plan["cache_keys"]],
                        per_block_accounting=per_block_accounting,
                    ),
                )
        logger.info(
            "SGLang PIC link complete mode=%s linked_tokens=%d "
            "recompute_tokens=%d model_forwards=%d elapsed_s=%.6f",
            recompute_mode,
            total_token_count,
            len(imp_indices),
            forward_count,
            elapsed_s,
        )
        _emit_pic_trace(
            "model_link_complete",
            phase="link",
            **trace_context,
            recompute_mode=recompute_mode,
            linked_tokens=total_token_count,
            reused_tokens=total_token_count - len(imp_indices),
            recompute_tokens=len(imp_indices),
            evicted_tokens=getattr(self, "_last_legolink_lookup_evicted_tokens", 0),
            model_forwards=forward_count,
            elapsed_s=elapsed_s,
            timing_breakdown_s={
                "prepare": prepare_s,
                "legolink_lookup": getattr(
                    self, "_last_legolink_lookup_elapsed_s", 0.0
                ),
                "kv_retrieval": kv_timing["kv_retrieval_s"],
                "rope_remap": kv_timing["rope_remap_s"],
                "kv_writeback": kv_timing["kv_writeback_s"],
                "triton_fused_kv_link": kv_timing.get("triton_fused_kv_link_s", 0.0),
                "kv_link_total": kv_link_s,
                "sparse_recompute": recompute_s,
                "total": elapsed_s,
            },
            timing_breakdown_cuda_ms={
                "kv_retrieval": cuda_summary.get("kv_retrieval", 0.0),
                "rope_remap": cuda_summary.get("rope_remap", 0.0),
                "kv_writeback": cuda_summary.get("kv_writeback", 0.0),
                "triton_fused_kv_link": cuda_summary.get("triton_fused_kv_link", 0.0),
                "kv_link_total": kv_link_cuda_ms,
                "sparse_recompute": recompute_cuda_ms,
                "legolink_total_gpu": cuda_summary.get("legolink_total_gpu", 0.0),
                "legolink_residual_gpu": cuda_summary.get("legolink_residual_gpu", 0.0),
                "model_prefill": cuda_summary.get("model_prefill", 0.0),
                "attention": cuda_summary.get("attention", 0.0),
                "ffn": cuda_summary.get("ffn", 0.0),
                "lm_head": cuda_summary.get("lm_head", 0.0),
                "sampling": cuda_summary.get("sampling", 0.0),
            },
        )
        with (
            _profile_range("NativeDecode"),
            host_recorder.span("native_decode_handoff"),
        ):
            return output

    def _run_full_causal(
        self,
        forward_batch: Any,
        full_input_ids: list[int],
        destination_slots: torch.Tensor,
    ) -> tuple[Any, int]:
        full_batch = self._make_extend_batch(
            forward_batch,
            token_ids=full_input_ids,
            start=0,
            end=len(full_input_ids),
            destination_slots=destination_slots,
        )
        started = _profile_start()
        output, _ = self.model_runner.forward_extend(full_batch)
        elapsed_s = _profile_elapsed(started)
        _emit_pic_trace(
            "full_causal_writeback",
            phase="link",
            **getattr(self, "_active_trace_context", {}),
            execution="forward_extend",
            start_token=0,
            end_token=len(full_input_ids),
            token_count=len(full_input_ids),
            destination_slots=_trace_values(destination_slots),
            kv_writeback="sglang_attention_backend",
            elapsed_s=elapsed_s,
        )
        return output, 1

    def _run_batched_recompute(
        self,
        forward_batch: Any,
        full_input_ids: list[int],
        destination_slots: torch.Tensor,
        imp_indices: list[int],
    ) -> tuple[Any, int]:
        if not hasattr(self, "_active_sparse_repair_batches"):
            self._active_sparse_repair_batches = []
        ranges = _contiguous_ranges(imp_indices)
        from sglang.srt.flashinfer_sparse_repair import (
            _component_ranges_option,
            effective_paged_batch_q_token_cap,
            paged_attention_path_reachable,
            paged_batch_capacity_snapshot,
            requested_sparse_repair_backend,
            select_hybrid_request_backend,
        )
        from sglang.srt.sparse_repair_compute import (
            build_sparse_repair_execution_units,
            build_unified_sparse_repair_metadata_view,
            plan_sparse_repair_batches,
        )

        requested_backend = requested_sparse_repair_backend()
        record_component_ranges, component_ranges_source = (
            _component_ranges_option(forward_batch)
        )
        (
            row_batching_enabled,
            rowwise_ffn_enabled,
            rowwise_output_projection_enabled,
            row_batching_config_source,
        ) = _sparse_repair_row_batching_options(forward_batch)
        flashinfer_row_batching = (
            row_batching_enabled
            and requested_backend == "flashinfer"
            and _env_flag(
                "SGLANG_PIC_ALLOW_FLASHINFER_ROW_BATCHING_EXPERIMENT", "0"
            )
        )
        flashinfer_full_row_batching = (
            flashinfer_row_batching
            and os.getenv(
                "SGLANG_PIC_FLASHINFER_GROUPED_ATTENTION_MODE", "per_row"
            ).strip().lower()
            == "paged_batched"
            and _env_flag(
                "SGLANG_PIC_FLASHINFER_INCLUDE_FINAL_ROW_IN_BATCH", "0"
            )
        )
        dense_grouped_ffn_experiment = (
            flashinfer_full_row_batching
            and not rowwise_ffn_enabled
            and _env_flag(
                "SGLANG_PIC_ALLOW_DENSE_GROUPED_FFN_EXPERIMENT", "0"
            )
        )
        if flashinfer_full_row_batching and not (
            (rowwise_ffn_enabled or dense_grouped_ffn_experiment)
            and rowwise_output_projection_enabled
        ):
            raise PICRuntimeError(
                "Full FlashInfer repair-row batching requires row-wise FFN "
                "or the explicit dense-grouped-FFN experiment, plus row-wise "
                "output projection."
            )
        if row_batching_enabled and requested_backend != "native" and not flashinfer_row_batching:
            raise PICRuntimeError(
                "Experimental sparse repair row batching requires native "
                "attention, or forced FlashInfer plus "
                "SGLANG_PIC_ALLOW_FLASHINFER_ROW_BATCHING_EXPERIMENT=1. "
                "Auto/hybrid fallback is not allowed."
            )
        original_sparse_repair_ranges = list(ranges)
        flashinfer_max_paged_batch_q_tokens = 0
        # The attention dispatcher selects paged prefill from environment alone.
        # Split whenever that path is reachable, not only when full row batching
        # is on, or a request that overrides row batching off still reaches
        # paged prefill with unsplit rows and overflows the workspace.
        paged_path_reachable = paged_attention_path_reachable(requested_backend)
        if flashinfer_full_row_batching or paged_path_reachable:
            flashinfer_max_paged_batch_q_tokens = (
                effective_paged_batch_q_token_cap()
            )
            ranges = _split_ranges_by_max_repair_tokens(
                ranges,
                max_tokens=flashinfer_max_paged_batch_q_tokens,
            )
        batch_plan = plan_sparse_repair_batches(ranges)
        if flashinfer_full_row_batching:
            execution_units = _build_flashinfer_paged_execution_units(
                ranges,
                max_tokens=flashinfer_max_paged_batch_q_tokens,
            )
        else:
            execution_units = (
                build_sparse_repair_execution_units(batch_plan)
                if row_batching_enabled
                else tuple((row_index,) for row_index in range(len(ranges)))
            )
        execution_mode = (
            "experimental_flashinfer_per_row_grouped_forward_extend"
            if flashinfer_row_batching
            else (
                "experimental_native_grouped_forward_extend"
                if row_batching_enabled
                else batch_plan.execution_mode
            )
        )
        batch_plan_trace = batch_plan.trace_metadata()
        batch_plan_trace["execution_mode"] = execution_mode
        batch_plan_trace["execution_units"] = [list(unit) for unit in execution_units]
        batch_plan_trace["rowwise_ffn_enabled"] = rowwise_ffn_enabled
        batch_plan_trace["dense_grouped_ffn_experiment"] = (
            dense_grouped_ffn_experiment
        )
        batch_plan_trace["rowwise_output_projection_enabled"] = (
            rowwise_output_projection_enabled
        )
        batch_plan_trace["row_batching_config_source"] = row_batching_config_source
        batch_plan_trace["flashinfer_row_batching_experiment"] = (
            flashinfer_row_batching
        )
        batch_plan_trace["flashinfer_final_row_batched"] = (
            flashinfer_full_row_batching
        )
        batch_plan_trace["original_repair_ranges"] = [
            {"start_token": row_start, "end_token": row_end}
            for row_start, row_end in original_sparse_repair_ranges
        ]
        batch_plan_trace["flashinfer_max_paged_batch_q_tokens"] = (
            flashinfer_max_paged_batch_q_tokens
        )
        batch_plan_trace["flashinfer_workspace_split_applied"] = (
            ranges != original_sparse_repair_ranges
        )
        batch_plan_trace["flashinfer_paged_path_reachable"] = paged_path_reachable
        batch_plan_trace["flashinfer_paged_capacity"] = (
            paged_batch_capacity_snapshot()
        )
        _emit_pic_trace(
            "sparse_repair_batch_plan",
            phase="link",
            **getattr(self, "_active_trace_context", {}),
            **batch_plan_trace,
        )
        state_capture = None
        state_capture_initialized = False
        capture_metadata_view = requested_backend != "native" or _env_flag(
            "SGLANG_PIC_SPARSE_REPAIR_METADATA_VIEW_ENABLED", "0"
        )
        hybrid_request_backend = None
        hybrid_request_reason = None
        if requested_backend == "hybrid":
            hybrid_request_backend, hybrid_request_reason = (
                select_hybrid_request_backend(
                    ranges,
                    block_count=int(
                        getattr(self, "_active_sparse_repair_block_count", 0)
                    ),
                )
            )
        output = None
        for unit in execution_units:
            unit_ranges = [ranges[row_index] for row_index in unit]
            start = unit_ranges[0][0]
            end = unit_ranges[-1][1]
            grouped = len(unit) > 1
            if grouped:
                extend_batch = self._make_grouped_extend_batch(
                    forward_batch,
                    full_input_ids=full_input_ids,
                    ranges=unit_ranges,
                    destination_slots=destination_slots,
                )
            else:
                extend_batch = self._make_extend_batch(
                    forward_batch,
                    token_ids=full_input_ids[start:end],
                    start=start,
                    end=end,
                    destination_slots=destination_slots[start:end],
                )
            from sglang.srt.flashinfer_sparse_repair import (
                make_sparse_repair_metadata,
            )

            unified_view = None
            if capture_metadata_view and not grouped:
                unified_view = build_unified_sparse_repair_metadata_view(
                    extend_batch,
                    start=start,
                    end=end,
                    block_count=int(
                        getattr(self, "_active_sparse_repair_block_count", 0)
                    ),
                    block_size=int(
                        getattr(self, "_active_sparse_repair_block_size", 0)
                    ),
                    request_id=str(
                        getattr(self, "_active_trace_context", {}).get(
                            "request_id", "unknown"
                        )
                    ),
                )
            extend_batch.pic_sparse_repair_metadata = make_sparse_repair_metadata(
                extend_batch,
                start=start,
                end=end,
                block_count=int(getattr(self, "_active_sparse_repair_block_count", 0)),
                block_size=int(getattr(self, "_active_sparse_repair_block_size", 0)),
                unified_view=unified_view,
                repair_ranges=unit_ranges,
                record_component_ranges=record_component_ranges,
                component_ranges_source=component_ranges_source,
            )
            group = batch_plan.groups[batch_plan.row_to_group[unit[0]]]
            extend_batch.pic_sparse_repair_metadata.update(
                {
                    "batch_plan_group_id": group.group_id,
                    "batch_plan_group_size": len(group.row_indices),
                    "batch_execution_unit_size": len(unit),
                    "batch_plan_padding_ratio": group.padding_ratio,
                    "batch_plan_execution_mode": execution_mode,
                    "batch_plan_row_indices": list(unit),
                    "rowwise_ffn_enabled": rowwise_ffn_enabled,
                    "rowwise_output_projection_enabled": (
                        rowwise_output_projection_enabled
                    ),
                    "row_batching_config_source": row_batching_config_source,
                    "flashinfer_row_batching_experiment": (
                        flashinfer_row_batching
                    ),
                    "flashinfer_final_row_batched": (
                        flashinfer_full_row_batching
                    ),
                }
            )
            if hybrid_request_backend is not None:
                extend_batch.pic_sparse_repair_metadata.update(
                    {
                        "hybrid_request_backend": hybrid_request_backend,
                        "hybrid_request_reason": hybrid_request_reason,
                    }
                )
            if getattr(self, "_request_capacity_degraded", False):
                # An earlier execution unit of this request degraded on
                # capacity. Finish the whole request on native attention rather
                # than mixing operators between units, so the request is
                # cleanly one configuration or the other.
                extend_batch.pic_sparse_repair_metadata["actual_backend"] = "native"
                extend_batch.pic_sparse_repair_metadata["selected_backend"] = "native"
                extend_batch.pic_sparse_repair_metadata["decision_reason"] = (
                    "request_already_capacity_degraded"
                )
            self._active_sparse_repair_batches.append(
                extend_batch.pic_sparse_repair_metadata
            )
            if not state_capture_initialized:
                from sglang.srt.sparse_repair_compute import (
                    SparseRepairStateCapture,
                )

                state_capture = SparseRepairStateCapture.create_if_enabled(
                    self.model_runner,
                    os.getenv("SGLANG_PIC_SPARSE_REPAIR_STATE_DUMP_DIR"),
                )
                state_capture_initialized = True
            started = _profile_start()
            try:
                output, _ = self.model_runner.forward_extend(extend_batch)
            except Exception:
                if state_capture is not None:
                    state_capture.close()
                raise
            if extend_batch.pic_sparse_repair_metadata.get("capacity_degraded"):
                self._request_capacity_degraded = True
            if flashinfer_full_row_batching and unit[-1] == len(ranges) - 1:
                output = self._select_final_grouped_request_output(
                    output, grouped_batch_size=len(unit)
                )
            elapsed_s = _profile_elapsed(started)
            _emit_pic_trace(
                "sparse_recompute_writeback",
                phase="link",
                **getattr(self, "_active_trace_context", {}),
                execution=("grouped_forward_extend" if grouped else "forward_extend"),
                start_token=start,
                end_token=end,
                token_count=sum(
                    row_end - row_start for row_start, row_end in unit_ranges
                ),
                repair_ranges=[
                    {"start_token": row_start, "end_token": row_end}
                    for row_start, row_end in unit_ranges
                ],
                global_positions=_trace_values(
                    lambda: [
                        position
                        for row_start, row_end in unit_ranges
                        for position in range(row_start, row_end)
                    ]
                ),
                # Deferred: this concatenation allocated a tensor on every
                # repair unit even with tracing disabled.
                destination_slots=_trace_values(
                    lambda: torch.cat(
                        [
                            destination_slots[row_start:row_end]
                            for row_start, row_end in unit_ranges
                        ]
                    )
                ),
                kv_writeback="sglang_attention_backend",
                elapsed_s=elapsed_s,
            )
        if state_capture is not None:
            state_capture.close()
            state_path = state_capture.save(
                forward_batch=forward_batch,
                destination_slots=destination_slots,
                repair_positions=imp_indices,
                execution_units=execution_units,
                output=output,
            )
            _emit_pic_trace(
                "sparse_repair_state_dump",
                phase="link",
                **getattr(self, "_active_trace_context", {}),
                path=str(state_path),
                repair_tokens=len(imp_indices),
                layer_count=len(state_capture.layer_outputs),
                diagnostic_only=True,
            )
        return output, len(execution_units)

    @staticmethod
    def _select_final_grouped_request_output(
        output: Any, *, grouped_batch_size: int
    ) -> Any:
        """Return only the original request's final-row next-token logits."""

        logits_output = getattr(output, "logits_output", output)
        logits = getattr(logits_output, "next_token_logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise PICRuntimeError(
                "Full repair-row batching requires tensor next-token logits."
            )
        if int(logits.shape[0]) != grouped_batch_size:
            raise PICRuntimeError(
                "Grouped repair logits do not match the repair-row batch size: "
                f"rows={grouped_batch_size}, logits={int(logits.shape[0])}."
            )
        logits_output.next_token_logits = logits[-1:]
        return output

    def _run_tokenwise_recompute(
        self,
        forward_batch: Any,
        full_input_ids: list[int],
        destination_slots: torch.Tensor,
        imp_indices: list[int],
    ) -> tuple[Any, int]:
        if not hasattr(self, "_active_sparse_repair_batches"):
            self._active_sparse_repair_batches = []
        from sglang.srt.flashinfer_sparse_repair import (
            _component_ranges_option,
            requested_sparse_repair_backend,
        )
        from sglang.srt.sparse_repair_compute import (
            build_unified_sparse_repair_metadata_view,
        )

        capture_metadata_view = (
            requested_sparse_repair_backend() != "native"
            or _env_flag("SGLANG_PIC_SPARSE_REPAIR_METADATA_VIEW_ENABLED", "0")
        )
        record_component_ranges, component_ranges_source = (
            _component_ranges_option(forward_batch)
        )
        output = None
        started = _profile_start()
        for position in imp_indices:
            one_token_batch = self._make_one_token_batch(
                forward_batch,
                token_id=int(full_input_ids[position]),
                position=position,
                destination_slot=destination_slots[position],
            )
            from sglang.srt.flashinfer_sparse_repair import (
                make_sparse_repair_metadata,
            )

            unified_view = None
            if capture_metadata_view:
                unified_view = build_unified_sparse_repair_metadata_view(
                    one_token_batch,
                    start=position,
                    end=position + 1,
                    block_count=int(
                        getattr(self, "_active_sparse_repair_block_count", 0)
                    ),
                    block_size=int(
                        getattr(self, "_active_sparse_repair_block_size", 0)
                    ),
                    request_id=str(
                        getattr(self, "_active_trace_context", {}).get(
                            "request_id", "unknown"
                        )
                    ),
                )
            one_token_batch.pic_sparse_repair_metadata = make_sparse_repair_metadata(
                one_token_batch,
                start=position,
                end=position + 1,
                block_count=int(getattr(self, "_active_sparse_repair_block_count", 0)),
                block_size=int(getattr(self, "_active_sparse_repair_block_size", 0)),
                unified_view=unified_view,
                record_component_ranges=record_component_ranges,
                component_ranges_source=component_ranges_source,
            )
            self._active_sparse_repair_batches.append(
                one_token_batch.pic_sparse_repair_metadata
            )
            output = self.model_runner.forward_decode(one_token_batch)
        elapsed_s = _profile_elapsed(started)
        _emit_pic_trace(
            "sparse_recompute_writeback",
            phase="link",
            **getattr(self, "_active_trace_context", {}),
            execution="forward_decode_tokenwise",
            token_count=len(imp_indices),
            global_positions=_trace_values(imp_indices),
            destination_slots=_trace_values(destination_slots[imp_indices]),
            kv_writeback="sglang_attention_backend",
            elapsed_s=elapsed_s,
        )
        return output, len(imp_indices)

    def _copy_linked_kv(
        self,
        pool: Any,
        destination_slots: torch.Tensor,
        link_plan: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_pool(pool)
        ranges = link_plan.get("block_ranges") or link_plan.get("attention_ranges")
        cache_keys = link_plan.get("cache_keys")
        if not isinstance(ranges, list) or not isinstance(cache_keys, list):
            raise PICRuntimeError("PIC link plan is missing block ranges/cache keys.")
        if len(ranges) != len(cache_keys):
            raise PICRuntimeError("PIC block range and cache key counts differ.")

        timings = {
            "kv_retrieval_s": 0.0,
            "rope_remap_s": 0.0,
            "kv_writeback_s": 0.0,
            "triton_fused_kv_link_s": 0.0,
            "kv_retrieval_cuda_ms": 0.0,
            "rope_remap_cuda_ms": 0.0,
            "kv_writeback_cuda_ms": 0.0,
            "triton_fused_kv_link_cuda_ms": 0.0,
        }
        lookup_blocks: dict[str, Any] = {}
        lookup_cache_keys: list[str] = []
        lookup_handles: list[Any] = []
        tree_cache = getattr(self, "_legolink_cache", None)
        host_recorder = getattr(
            self, "_host_timing_recorder", None
        ) or _HostTimingRecorder("model_runner")
        radixcache_owns_blocks = bool(
            tree_cache is not None
            and hasattr(tree_cache, "lookup_legolink_blocks")
            and _env_flag("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1")
        )
        if legolink_only_mode_enabled() and not radixcache_owns_blocks:
            raise PICRuntimeError(
                "LegoLink-only execution requires RadixCache block lookup; "
                "the legacy PICRuntime block-owner path is disabled."
            )
        if radixcache_owns_blocks:
            lookup_cache_keys = [str(cache_key) for cache_key in cache_keys]
            block_ids = [
                str(block_range.get("block_id", raw_cache_key))
                for block_range, raw_cache_key in zip(ranges, cache_keys)
            ]
            lookup_compatibility = self.get_legolink_compatibility(
                adapter_id=link_plan.get("adapter_id")
            )
            segmentation_policy_version = link_plan.get("segmentation_policy_version")
            if segmentation_policy_version is not None:
                lookup_compatibility["segmentation_policy_version"] = str(
                    segmentation_policy_version
                )
            legolink_policy_version = link_plan.get(
                "legolink_policy_version"
            ) or link_plan.get("physical_token_layout_version")
            if legolink_policy_version is not None:
                lookup_compatibility["legolink_policy_version"] = str(
                    legolink_policy_version
                )
                lookup_compatibility["policy_version"] = str(legolink_policy_version)
            with (
                _profile_range("LegoLinkLookup"),
                host_recorder.span("radixcache_lookup_legolink_blocks"),
            ):
                lookup_result = tree_cache.lookup_legolink_blocks(
                    lookup_cache_keys,
                    semantic_block_ids=block_ids,
                    link_plan=link_plan,
                    pin=False,
                    compatibility=lookup_compatibility,
                )
            host_recorder.merge_ns(
                getattr(lookup_result, "timing_breakdown_ns", {}) or {}
            )
            self._last_legolink_lookup_elapsed_s = float(
                getattr(lookup_result, "lookup_elapsed_s", 0.0) or 0.0
            )
            lookup_handles = list(getattr(lookup_result, "matched_handles", []))
            with host_recorder.span("missing_stale_incompatible_classification"):
                unavailable = [
                    *lookup_result.missing_blocks,
                    *getattr(lookup_result, "stale_blocks", []),
                    *getattr(lookup_result, "incompatible_blocks", []),
                ]
            if unavailable:
                missing = ", ".join(unavailable)
                raise PICRuntimeError(
                    f"PIC RadixCache lookup cannot safely link block(s): {missing}"
                )
            if hasattr(tree_cache, "acquire_legolink_blocks"):
                with (
                    _profile_range("AcquireSourceBlocks"),
                    host_recorder.span("acquire_legolink_blocks"),
                ):
                    scheduler_reserved = bool(
                        link_plan.get("_scheduler_reserved_cache_keys")
                    )
                    lookup_handles = tree_cache.acquire_legolink_blocks(
                        lookup_handles, pin=not scheduler_reserved
                    )
            self._pending_legolink_release = (
                tree_cache,
                lookup_handles,
                lookup_cache_keys,
            )
            self._last_legolink_execution_handles = list(lookup_handles)
            with host_recorder.span("kv_slot_range_materialization"):
                for cache_key, metadata in lookup_result.matched_blocks.items():
                    snapshot = getattr(metadata, "snapshot", None)
                    if snapshot is None:
                        raise PICRuntimeError(
                            f"PIC RadixCache block has no runtime snapshot: {cache_key}"
                        )
                    kv_slots = lookup_result.kv_slot_ranges.get(cache_key)
                    if kv_slots is not None and snapshot.storage_mode == "reference":
                        snapshot.source_slots = kv_slots.long().clone()
                        snapshot.reference_ready = True
                    lookup_blocks[cache_key] = snapshot

        triton_fusion_mode = get_triton_fused_kv_link_mode()
        if (
            _env_flag("SGLANG_PIC_BATCH_BLOCK_KV_LINK", "1") and len(ranges) > 1
        ) or triton_fusion_mode != "disabled":
            return self._copy_linked_kv_batched_blocks(
                pool=pool,
                destination_slots=destination_slots,
                ranges=ranges,
                cache_keys=cache_keys,
                repair_positions=[
                    int(position) for position in (link_plan.get("imp_indices") or [])
                ],
                lookup_blocks=lookup_blocks,
                radixcache_owns_blocks=radixcache_owns_blocks,
                host_recorder=host_recorder,
            )

        # Start the aggregate GPU link range only after all host-only lookup,
        # validation, acquire, and slot materialization work has completed.
        # This prevents CPU queue/lookup idle time from inflating CUDA Events.
        legolink_total_span = _cuda_event_start("legolink_total_gpu", phase="link")
        kv_link_cuda_started = _cuda_event_start("kv_link_total")
        for block_index, (block_range, raw_cache_key) in enumerate(
            zip(ranges, cache_keys)
        ):
            cache_key = str(raw_cache_key)
            snapshot = lookup_blocks.get(cache_key)
            if snapshot is None and not radixcache_owns_blocks:
                snapshot = self._blocks.get(cache_key)
            if snapshot is None:
                raise PICRuntimeError(
                    f"PIC KV block is not captured in this model worker: {cache_key}"
                )
            if cache_key in self._blocks:
                self._blocks.move_to_end(cache_key)
            start = int(block_range["start_token"])
            end = int(block_range["end_token"])
            if end - start != len(snapshot.token_ids):
                raise PICRuntimeError(
                    f"PIC token count mismatch for block {cache_key}: "
                    f"plan={end - start}, snapshot={len(snapshot.token_ids)}"
                )
            with host_recorder.span("global_offset_position_map_construction"):
                block_slots = destination_slots[start:end]
                local_positions = torch.arange(
                    end - start, dtype=torch.long, device=block_slots.device
                )
                global_positions = local_positions + start
                # Remap iff the cached rows carry RoPE angles for positions
                # other than the ones this request needs. `global_positions`
                # is `local_positions + start` one line above, so `start` is
                # exactly the local->global delta and this scalar test is
                # equivalent to `(local_positions != global_positions).any()`
                # without the device round trip. Block ordinal does not answer
                # the question: under normalized framing block 0 sits behind a
                # chat prefix at a nonzero offset and must be remapped.
                needs_rope_remap = start != 0
            block_timing = {
                "kv_retrieval_s": 0.0,
                "rope_remap_s": 0.0,
                "kv_writeback_s": 0.0,
                "kv_retrieval_cuda_ms": 0.0,
                "rope_remap_cuda_ms": 0.0,
                "kv_writeback_cuda_ms": 0.0,
            }
            for layer_id in range(
                self.model_runner.start_layer, self.model_runner.end_layer
            ):
                retrieval_started = _profile_start()
                retrieval_cuda_started = _cuda_event_start("kv_retrieval")
                with _profile_range("LegoLinkKVFetch"):
                    with _record_function("SGLang::LegoLinkKVCopy"):
                        if snapshot.storage_mode == "reference":
                            if (
                                not snapshot.reference_ready
                                or snapshot.source_slots is None
                            ):
                                raise PICRuntimeError(
                                    f"PIC reference block is not pinned: {cache_key}"
                                )
                            source_slots = snapshot.source_slots
                            key = pool.get_key_buffer(layer_id).index_select(
                                0, source_slots
                            )
                            value = pool.get_value_buffer(layer_id).index_select(
                                0, source_slots
                            )
                        else:
                            key = snapshot.keys[layer_id]
                            value = snapshot.values[layer_id]
                retrieval_cuda_ms = _cuda_event_elapsed_ms(retrieval_cuda_started)
                retrieval_s = _profile_elapsed(retrieval_started)
                timings["kv_retrieval_s"] += retrieval_s
                timings["kv_retrieval_cuda_ms"] += retrieval_cuda_ms
                block_timing["kv_retrieval_s"] += retrieval_s
                block_timing["kv_retrieval_cuda_ms"] += retrieval_cuda_ms
                if needs_rope_remap:
                    rope_started = _profile_start()
                    rope_cuda_started = _cuda_event_start("rope_remap")
                    with _profile_range("LegoLinkRoPERemap"):
                        key = self._remap_key_positions(
                            layer_id,
                            key,
                            local_positions,
                            global_positions,
                            max_global_position=end - 1,
                        )
                    rope_cuda_ms = _cuda_event_elapsed_ms(rope_cuda_started)
                    rope_s = _profile_elapsed(rope_started)
                    timings["rope_remap_s"] += rope_s
                    timings["rope_remap_cuda_ms"] += rope_cuda_ms
                    block_timing["rope_remap_s"] += rope_s
                    block_timing["rope_remap_cuda_ms"] += rope_cuda_ms
                    _emit_pic_trace(
                        "rope_global_position_remap",
                        phase="link",
                        **getattr(self, "_active_trace_context", {}),
                        block_index=block_index,
                        block_id=block_range.get("block_id"),
                        cache_key=cache_key,
                        layer_id=layer_id,
                        local_start_token=0,
                        global_offset=start,
                        token_count=end - start,
                        elapsed_s=rope_s,
                        elapsed_cuda_ms=rope_cuda_ms,
                    )
                writeback_started = _profile_start()
                writeback_cuda_started = _cuda_event_start("kv_writeback")
                with _profile_range("LegoLinkKVWriteBack"):
                    pool.get_key_buffer(layer_id).index_copy_(0, block_slots, key)
                    pool.get_value_buffer(layer_id).index_copy_(0, block_slots, value)
                writeback_cuda_ms = _cuda_event_elapsed_ms(writeback_cuda_started)
                writeback_s = _profile_elapsed(writeback_started)
                timings["kv_writeback_s"] += writeback_s
                timings["kv_writeback_cuda_ms"] += writeback_cuda_ms
                block_timing["kv_writeback_s"] += writeback_s
                block_timing["kv_writeback_cuda_ms"] += writeback_cuda_ms
            _emit_pic_trace(
                "kv_block_linked",
                phase="link",
                **getattr(self, "_active_trace_context", {}),
                block_index=block_index,
                block_id=block_range.get("block_id"),
                cache_key=cache_key,
                storage_mode=snapshot.storage_mode,
                local_start_token=0,
                local_end_token=end - start,
                global_offset=start,
                global_end_token=end,
                source_slots=(
                    _trace_values(snapshot.source_slots)
                    if snapshot.source_slots is not None
                    else {"count": end - start, "values": [], "truncated": True}
                ),
                destination_slots=_trace_values(block_slots),
                rope_remapped=needs_rope_remap,
                layers_written=self.model_runner.end_layer
                - self.model_runner.start_layer,
                kv_writeback="index_copy",
                timing_breakdown_s=block_timing,
            )
        _cuda_event_elapsed_ms(kv_link_cuda_started)
        timings["_legolink_total_span"] = legolink_total_span
        return timings

    def _copy_linked_kv_batched_blocks(
        self,
        *,
        pool: Any,
        destination_slots: torch.Tensor,
        ranges: list[dict[str, Any]],
        cache_keys: list[Any],
        repair_positions: list[int],
        lookup_blocks: dict[str, Any],
        radixcache_owns_blocks: bool,
        host_recorder: _HostTimingRecorder,
    ) -> dict[str, Any]:
        """Link all semantic blocks with one retrieval/write-back per layer."""

        timings = {
            "kv_retrieval_s": 0.0,
            "rope_remap_s": 0.0,
            "kv_writeback_s": 0.0,
            "triton_fused_kv_link_s": 0.0,
            "kv_retrieval_cuda_ms": 0.0,
            "rope_remap_cuda_ms": 0.0,
            "kv_writeback_cuda_ms": 0.0,
            "triton_fused_kv_link_cuda_ms": 0.0,
        }
        block_specs: list[dict[str, Any]] = []
        unique_repair_positions = sorted(set(repair_positions))
        with host_recorder.span("global_offset_position_map_construction"):
            for block_index, (block_range, raw_cache_key) in enumerate(
                zip(ranges, cache_keys)
            ):
                cache_key = str(raw_cache_key)
                snapshot = lookup_blocks.get(cache_key)
                if snapshot is None and not radixcache_owns_blocks:
                    snapshot = self._blocks.get(cache_key)
                if snapshot is None:
                    raise PICRuntimeError(
                        "PIC KV block is not captured in this model worker: "
                        f"{cache_key}"
                    )
                if cache_key in self._blocks:
                    self._blocks.move_to_end(cache_key)
                start = int(block_range["start_token"])
                end = int(block_range["end_token"])
                token_count = end - start
                if token_count != len(snapshot.token_ids):
                    raise PICRuntimeError(
                        f"PIC token count mismatch for block {cache_key}: "
                        f"plan={token_count}, snapshot={len(snapshot.token_ids)}"
                    )
                if snapshot.storage_mode == "reference" and (
                    not snapshot.reference_ready or snapshot.source_slots is None
                ):
                    raise PICRuntimeError(
                        f"PIC reference block is not pinned: {cache_key}"
                    )
                block_slots = destination_slots[start:end]
                local_positions = torch.arange(
                    token_count, dtype=torch.long, device=block_slots.device
                )
                block_specs.append(
                    {
                        "block_index": block_index,
                        "block_range": block_range,
                        "cache_key": cache_key,
                        "snapshot": snapshot,
                        "start": start,
                        "end": end,
                        "token_count": token_count,
                        "destination_slots": block_slots,
                        "local_positions": local_positions,
                        "global_positions": local_positions + start,
                        # See `_copy_linked_kv`: `start` is the local->global
                        # position delta, not merely the block's offset in the
                        # plan, so it alone decides whether this block's cached
                        # RoPE angles are already the ones the request needs.
                        "needs_rope_remap": start != 0,
                        "repair_token_count": sum(
                            start <= position < end
                            for position in unique_repair_positions
                        ),
                    }
                )

            all_destination_slots = torch.cat(
                [spec["destination_slots"] for spec in block_specs]
            )
            all_local_positions = torch.cat(
                [spec["local_positions"] for spec in block_specs]
            )
            all_global_positions = torch.cat(
                [spec["global_positions"] for spec in block_specs]
            )
            all_reference = all(
                spec["snapshot"].storage_mode == "reference" for spec in block_specs
            )
            all_source_slots = (
                torch.cat([spec["snapshot"].source_slots for spec in block_specs])
                if all_reference
                else None
            )
            boundary_repair_aware = _env_flag(
                "SGLANG_PIC_TRITON_BOUNDARY_REPAIR_AWARE", "0"
            )
            repair_offsets: list[int] = []
            concatenated_offset = 0
            if boundary_repair_aware:
                for spec in block_specs:
                    repair_offsets.extend(
                        concatenated_offset + position - spec["start"]
                        for position in unique_repair_positions
                        if spec["start"] <= position < spec["end"]
                    )
                    concatenated_offset += int(spec["token_count"])
            all_repair_mask = None
            if repair_offsets:
                repair_mask_cpu = torch.zeros(
                    int(all_destination_slots.numel()), dtype=torch.uint8
                )
                repair_mask_cpu[torch.tensor(repair_offsets, dtype=torch.long)] = 1
                all_repair_mask = repair_mask_cpu.to(
                    device=all_destination_slots.device,
                    non_blocking=True,
                )

        # The leading run of tokens whose local position already equals its
        # global position; both the torch tail slice and the Triton kernel skip
        # the remap for exactly these. Only block 0 can begin at global 0, and
        # when it does not, there is no identity prefix at all.
        prefix_token_count = (
            0
            if block_specs[0]["needs_rope_remap"]
            else int(block_specs[0]["token_count"])
        )
        tail_token_count = int(all_local_positions.numel()) - prefix_token_count
        repair_token_count = len(repair_offsets)
        triton_boundary_repair_active = all_repair_mask is not None
        max_global_position = max(int(spec["end"]) for spec in block_specs) - 1
        layer_count = self.model_runner.end_layer - self.model_runner.start_layer
        triton_fusion_mode = get_triton_fused_kv_link_mode()
        compare_triton_torch_kv = _env_flag(
            "SGLANG_PIC_COMPARE_TRITON_TORCH_KV_LINK", "0"
        )
        triton_fusion_enabled = False
        triton_fusion_reason = "disabled"
        triton_eligibility_checked = False
        triton_fused_kv_link = None
        triton_traffic: dict[str, Any] = {}
        triton_launch_config: dict[str, Any] = {}
        first_layer = self.model_runner.start_layer
        first_key_buffer = pool.get_key_buffer(first_layer)
        first_value_buffer = pool.get_value_buffer(first_layer)
        if triton_fusion_mode != "disabled":
            triton_eligibility_checked = True
            if not all_reference or all_source_slots is None:
                triton_fusion_reason = "reference_storage_required"
            else:
                try:
                    from sglang.srt.legolink_triton import (
                        check_fused_kv_link_eligibility,
                        estimate_kv_traffic_bytes,
                        fused_kv_link_inplace,
                        resolve_fused_kv_launch_config,
                    )

                    first_rotary = self._rotary_by_layer.get(first_layer)
                    if first_rotary is None or not hasattr(
                        first_rotary, "cos_sin_cache"
                    ):
                        triton_fusion_reason = "supported_rotary_required"
                    else:
                        first_rotary._ensure_cos_sin_cache_length(
                            int(max_global_position)
                        )
                        eligibility = check_fused_kv_link_eligibility(
                            key_buffer=first_key_buffer,
                            value_buffer=first_value_buffer,
                            source_slots=all_source_slots,
                            destination_slots=all_destination_slots,
                            local_positions=all_local_positions,
                            global_positions=all_global_positions,
                            cos_sin_cache=first_rotary.cos_sin_cache,
                            rotary_dim=int(first_rotary.rotary_dim),
                            repair_mask=all_repair_mask,
                        )
                        triton_fusion_enabled = eligibility.supported
                        triton_fusion_reason = eligibility.reason
                        triton_fused_kv_link = fused_kv_link_inplace
                        triton_work_items = int(first_key_buffer.shape[1]) * (
                            int(first_rotary.rotary_dim) // 2
                            + int(first_key_buffer.shape[2])
                            - int(first_rotary.rotary_dim)
                            + int(first_value_buffer.shape[2])
                        )
                        launch_config = resolve_fused_kv_launch_config(
                            work_items=triton_work_items,
                            device=first_key_buffer.device,
                        )
                        triton_launch_config = {
                            "triton_block_size": launch_config.block_size,
                            "triton_num_warps": launch_config.num_warps,
                            "triton_num_stages": launch_config.num_stages,
                            "triton_launch_config_source": launch_config.source,
                            "triton_work_items_per_token": triton_work_items,
                            "triton_device_capability": ".".join(
                                str(value)
                                for value in torch.cuda.get_device_capability(
                                    first_key_buffer.device
                                )
                            ),
                        }
                        triton_traffic = estimate_kv_traffic_bytes(
                            token_count=int(all_source_slots.numel()),
                            key_row_bytes=first_key_buffer[0].numel()
                            * first_key_buffer.element_size(),
                            value_row_bytes=first_value_buffer[0].numel()
                            * first_value_buffer.element_size(),
                            remapped_token_count=tail_token_count,
                            repair_token_count=repair_token_count,
                        )
                except (ImportError, RuntimeError) as error:
                    triton_fusion_reason = (
                        f"initialization_failed:{type(error).__name__}"
                    )
        requested_implementation = {
            "disabled": "torch_block_batched_legolink",
            "required": "triton_fused_legolink",
            "auto": "triton_fused_legolink_auto",
        }[triton_fusion_mode]
        actual_implementation = (
            "triton_fused_legolink"
            if triton_fusion_enabled
            else "torch_block_batched_legolink"
        )
        fallback_reason = (
            triton_fusion_reason
            if triton_fusion_mode == "auto" and not triton_fusion_enabled
            else None
        )
        _emit_pic_trace(
            "legolink_kv_dispatch",
            phase="link",
            **getattr(self, "_active_trace_context", {}),
            requested_implementation=requested_implementation,
            actual_implementation=actual_implementation,
            triton_mode=triton_fusion_mode,
            triton_supported=(
                triton_fusion_enabled if triton_eligibility_checked else None
            ),
            dispatch_reason=triton_fusion_reason,
            fallback_reason=fallback_reason,
            token_count=int(all_destination_slots.numel()),
            block_count=len(block_specs),
            layer_count=layer_count,
            kv_heads=int(first_key_buffer.shape[1]),
            key_head_dimension=int(first_key_buffer.shape[2]),
            value_head_dimension=int(first_value_buffer.shape[2]),
            kv_dtype=str(first_key_buffer.dtype),
            kernel_configuration=(
                {
                    "block_size": 256,
                    "num_warps": 4,
                    "num_stages": 2,
                }
                if triton_fusion_enabled
                else None
            ),
            boundary_repair_mode=(
                "skip_stale_kv_then_sparse_forward_extend"
                if triton_boundary_repair_active
                else "separate_sparse_forward_extend"
            ),
        )
        if triton_fusion_mode == "required" and not triton_fusion_enabled:
            raise PICRuntimeError(
                "Required Triton LegoLink fusion is unavailable: "
                f"{triton_fusion_reason}"
            )
        if triton_fusion_mode == "auto" and not triton_fusion_enabled:
            _emit_pic_trace(
                "triton_fusion_skipped",
                phase="link",
                **getattr(self, "_active_trace_context", {}),
                reason=triton_fusion_reason,
            )

        legolink_total_span = _cuda_event_start("legolink_total_gpu", phase="link")
        kv_link_cuda_started = _cuda_event_start("kv_link_total")
        for layer_id in range(
            self.model_runner.start_layer, self.model_runner.end_layer
        ):
            if triton_fusion_enabled:
                rotary = self._rotary_by_layer.get(layer_id)
                if rotary is None or not hasattr(rotary, "cos_sin_cache"):
                    raise PICRuntimeError(
                        f"Layer {layer_id} has no supported rotary embedding for PIC."
                    )
                rotary._ensure_cos_sin_cache_length(int(max_global_position))
                fused_started = _profile_start()
                fused_cuda_started = _cuda_event_start("triton_fused_kv_link")
                with _profile_range("LegoLinkTritonFusion"):
                    with _record_function("SGLang::LegoLinkTritonFusion"):
                        layer_key_buffer = (
                            first_key_buffer
                            if layer_id == first_layer
                            else pool.get_key_buffer(layer_id)
                        )
                        layer_value_buffer = (
                            first_value_buffer
                            if layer_id == first_layer
                            else pool.get_value_buffer(layer_id)
                        )
                        triton_fused_kv_link(
                            key_buffer=layer_key_buffer,
                            value_buffer=layer_value_buffer,
                            source_slots=all_source_slots,
                            destination_slots=all_destination_slots,
                            local_positions=all_local_positions,
                            global_positions=all_global_positions,
                            cos_sin_cache=rotary.cos_sin_cache,
                            prefix_token_count=prefix_token_count,
                            rotary_dim=int(rotary.rotary_dim),
                            is_neox_style=bool(rotary.is_neox_style),
                            repair_mask=all_repair_mask,
                            block_size=triton_launch_config["triton_block_size"],
                            num_warps=triton_launch_config["triton_num_warps"],
                            num_stages=triton_launch_config["triton_num_stages"],
                        )
                        if compare_triton_torch_kv:
                            with _profile_range("LegoLinkTritonTorchKVCompare"):
                                reference_key = layer_key_buffer.index_select(
                                    0, all_source_slots
                                )
                                reference_value = layer_value_buffer.index_select(
                                    0, all_source_slots
                                )
                                if tail_token_count:
                                    reference_tail = self._remap_key_positions(
                                        layer_id,
                                        reference_key[prefix_token_count:],
                                        all_local_positions[prefix_token_count:],
                                        all_global_positions[prefix_token_count:],
                                        max_global_position=max_global_position,
                                    )
                                    reference_key = torch.cat(
                                        (
                                            reference_key[:prefix_token_count],
                                            reference_tail,
                                        )
                                    )
                                compare_mask = (
                                    all_repair_mask == 0
                                    if all_repair_mask is not None
                                    else torch.ones_like(
                                        all_destination_slots, dtype=torch.bool
                                    )
                                )
                                compare_token_indices = torch.arange(
                                    int(all_destination_slots.numel()),
                                    device=all_destination_slots.device,
                                    dtype=torch.long,
                                )[compare_mask]
                                compare_destinations = all_destination_slots[
                                    compare_mask
                                ]
                                actual_key = layer_key_buffer.index_select(
                                    0, compare_destinations
                                )
                                actual_value = layer_value_buffer.index_select(
                                    0, compare_destinations
                                )
                                reference_key = reference_key[compare_mask]
                                reference_value = reference_value[compare_mask]
                                key_diff = (
                                    actual_key.float() - reference_key.float()
                                ).abs()
                                value_diff = (
                                    actual_value.float() - reference_value.float()
                                ).abs()
                                max_token_index = None
                                max_head_index = None
                                max_dim_index = None
                                max_source_slot = None
                                max_destination_slot = None
                                max_local_position = None
                                max_global_position_observed = None
                                max_actual_key = None
                                max_reference_key = None
                                max_pair_index = None
                                max_pair_first_dim = None
                                max_pair_second_dim = None
                                max_pair_side = None
                                max_source_pair_first = None
                                max_source_pair_second = None
                                max_actual_pair_first = None
                                max_actual_pair_second = None
                                max_reference_pair_first = None
                                max_reference_pair_second = None
                                max_local_cos = None
                                max_local_sin = None
                                max_global_cos = None
                                max_global_sin = None
                                max_delta_cos = None
                                max_delta_sin = None
                                if key_diff.numel():
                                    flat_index = int(key_diff.argmax().item())
                                    per_token = int(
                                        key_diff.shape[1] * key_diff.shape[2]
                                    )
                                    token_offset = flat_index // per_token
                                    remainder = flat_index - token_offset * per_token
                                    max_head_index = remainder // int(key_diff.shape[2])
                                    max_dim_index = remainder - max_head_index * int(
                                        key_diff.shape[2]
                                    )
                                    max_token_index = int(
                                        compare_token_indices[token_offset].item()
                                    )
                                    max_source_slot = int(
                                        all_source_slots[max_token_index].item()
                                    )
                                    max_destination_slot = int(
                                        all_destination_slots[max_token_index].item()
                                    )
                                    max_local_position = int(
                                        all_local_positions[max_token_index].item()
                                    )
                                    max_global_position_observed = int(
                                        all_global_positions[max_token_index].item()
                                    )
                                    max_actual_key = float(
                                        actual_key[
                                            token_offset,
                                            max_head_index,
                                            max_dim_index,
                                        ]
                                        .float()
                                        .item()
                                    )
                                    max_reference_key = float(
                                        reference_key[
                                            token_offset,
                                            max_head_index,
                                            max_dim_index,
                                        ]
                                        .float()
                                        .item()
                                    )
                                    rotary_dim_for_trace = int(rotary.rotary_dim)
                                    if max_dim_index < rotary_dim_for_trace:
                                        rotary_half_for_trace = (
                                            rotary_dim_for_trace // 2
                                        )
                                        if bool(rotary.is_neox_style):
                                            if max_dim_index < rotary_half_for_trace:
                                                max_pair_index = max_dim_index
                                                max_pair_side = "first"
                                            else:
                                                max_pair_index = (
                                                    max_dim_index
                                                    - rotary_half_for_trace
                                                )
                                                max_pair_side = "second"
                                            max_pair_first_dim = max_pair_index
                                            max_pair_second_dim = (
                                                max_pair_index + rotary_half_for_trace
                                            )
                                        else:
                                            max_pair_index = max_dim_index // 2
                                            max_pair_side = (
                                                "first"
                                                if max_dim_index % 2 == 0
                                                else "second"
                                            )
                                            max_pair_first_dim = max_pair_index * 2
                                            max_pair_second_dim = max_pair_first_dim + 1
                                        max_source_pair_first = float(
                                            layer_key_buffer[
                                                max_source_slot,
                                                max_head_index,
                                                max_pair_first_dim,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_source_pair_second = float(
                                            layer_key_buffer[
                                                max_source_slot,
                                                max_head_index,
                                                max_pair_second_dim,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_actual_pair_first = float(
                                            actual_key[
                                                token_offset,
                                                max_head_index,
                                                max_pair_first_dim,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_actual_pair_second = float(
                                            actual_key[
                                                token_offset,
                                                max_head_index,
                                                max_pair_second_dim,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_reference_pair_first = float(
                                            reference_key[
                                                token_offset,
                                                max_head_index,
                                                max_pair_first_dim,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_reference_pair_second = float(
                                            reference_key[
                                                token_offset,
                                                max_head_index,
                                                max_pair_second_dim,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        trace_cache = rotary.cos_sin_cache
                                        max_local_cos = float(
                                            trace_cache[
                                                max_local_position,
                                                max_pair_index,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_local_sin = float(
                                            trace_cache[
                                                max_local_position,
                                                max_pair_index + rotary_half_for_trace,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_global_cos = float(
                                            trace_cache[
                                                max_global_position_observed,
                                                max_pair_index,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_global_sin = float(
                                            trace_cache[
                                                max_global_position_observed,
                                                max_pair_index + rotary_half_for_trace,
                                            ]
                                            .float()
                                            .item()
                                        )
                                        max_delta_cos = (
                                            max_global_cos * max_local_cos
                                            + max_global_sin * max_local_sin
                                        )
                                        max_delta_sin = (
                                            max_global_sin * max_local_cos
                                            - max_global_cos * max_local_sin
                                        )
                                atol = (
                                    2e-2
                                    if layer_key_buffer.dtype == torch.bfloat16
                                    else 5e-3
                                )
                                _emit_pic_trace(
                                    "triton_torch_kv_link_equivalence",
                                    phase="link",
                                    **getattr(self, "_active_trace_context", {}),
                                    layer_id=layer_id,
                                    compared_tokens=int(compare_destinations.numel()),
                                    skipped_repair_tokens=repair_token_count,
                                    key_max_abs_error=float(
                                        key_diff.max().item()
                                        if key_diff.numel()
                                        else 0.0
                                    ),
                                    key_mean_abs_error=float(
                                        key_diff.mean().item()
                                        if key_diff.numel()
                                        else 0.0
                                    ),
                                    value_max_abs_error=float(
                                        value_diff.max().item()
                                        if value_diff.numel()
                                        else 0.0
                                    ),
                                    value_mean_abs_error=float(
                                        value_diff.mean().item()
                                        if value_diff.numel()
                                        else 0.0
                                    ),
                                    key_allclose=bool(
                                        torch.allclose(
                                            actual_key,
                                            reference_key,
                                            atol=atol,
                                            rtol=atol,
                                        )
                                    ),
                                    value_allclose=bool(
                                        torch.equal(actual_value, reference_value)
                                    ),
                                    atol=atol,
                                    max_token_index=max_token_index,
                                    max_head_index=max_head_index,
                                    max_dim_index=max_dim_index,
                                    max_source_slot=max_source_slot,
                                    max_destination_slot=max_destination_slot,
                                    max_local_position=max_local_position,
                                    max_global_position=(max_global_position_observed),
                                    max_is_remapped=(
                                        bool(
                                            max_token_index is not None
                                            and max_token_index >= prefix_token_count
                                        )
                                    ),
                                    max_actual_key=max_actual_key,
                                    max_reference_key=max_reference_key,
                                    max_pair_index=max_pair_index,
                                    max_pair_first_dim=max_pair_first_dim,
                                    max_pair_second_dim=max_pair_second_dim,
                                    max_pair_side=max_pair_side,
                                    max_source_pair_first=max_source_pair_first,
                                    max_source_pair_second=max_source_pair_second,
                                    max_actual_pair_first=max_actual_pair_first,
                                    max_actual_pair_second=max_actual_pair_second,
                                    max_reference_pair_first=max_reference_pair_first,
                                    max_reference_pair_second=max_reference_pair_second,
                                    max_local_cos=max_local_cos,
                                    max_local_sin=max_local_sin,
                                    max_global_cos=max_global_cos,
                                    max_global_sin=max_global_sin,
                                    max_delta_cos=max_delta_cos,
                                    max_delta_sin=max_delta_sin,
                                )
                fused_cuda_ms = _cuda_event_elapsed_ms(fused_cuda_started)
                fused_s = _profile_elapsed(fused_started)
                timings["triton_fused_kv_link_s"] += fused_s
                timings["triton_fused_kv_link_cuda_ms"] += fused_cuda_ms
                continue

            retrieval_started = _profile_start()
            retrieval_cuda_started = _cuda_event_start("kv_retrieval")
            with _profile_range("LegoLinkKVFetch"):
                with _record_function("SGLang::LegoLinkKVCopy"):
                    key_buffer = (
                        first_key_buffer
                        if layer_id == first_layer
                        else pool.get_key_buffer(layer_id)
                    )
                    value_buffer = (
                        first_value_buffer
                        if layer_id == first_layer
                        else pool.get_value_buffer(layer_id)
                    )
                    if all_reference:
                        key = key_buffer.index_select(0, all_source_slots)
                        value = value_buffer.index_select(0, all_source_slots)
                    else:
                        key_chunks = []
                        value_chunks = []
                        for spec in block_specs:
                            snapshot = spec["snapshot"]
                            if snapshot.storage_mode == "reference":
                                source_slots = snapshot.source_slots
                                key_chunks.append(
                                    key_buffer.index_select(0, source_slots)
                                )
                                value_chunks.append(
                                    value_buffer.index_select(0, source_slots)
                                )
                            else:
                                key_chunks.append(snapshot.keys[layer_id])
                                value_chunks.append(snapshot.values[layer_id])
                        key = torch.cat(key_chunks)
                        value = torch.cat(value_chunks)
            retrieval_cuda_ms = _cuda_event_elapsed_ms(retrieval_cuda_started)
            retrieval_s = _profile_elapsed(retrieval_started)
            timings["kv_retrieval_s"] += retrieval_s
            timings["kv_retrieval_cuda_ms"] += retrieval_cuda_ms

            if tail_token_count:
                rope_started = _profile_start()
                rope_cuda_started = _cuda_event_start("rope_remap")
                with _profile_range("LegoLinkRoPERemap"):
                    remapped_tail = self._remap_key_positions(
                        layer_id,
                        key[prefix_token_count:],
                        all_local_positions[prefix_token_count:],
                        all_global_positions[prefix_token_count:],
                        max_global_position=max_global_position,
                    )
                    key = torch.cat((key[:prefix_token_count], remapped_tail))
                rope_cuda_ms = _cuda_event_elapsed_ms(rope_cuda_started)
                rope_s = _profile_elapsed(rope_started)
                timings["rope_remap_s"] += rope_s
                timings["rope_remap_cuda_ms"] += rope_cuda_ms

            writeback_started = _profile_start()
            writeback_cuda_started = _cuda_event_start("kv_writeback")
            with _profile_range("LegoLinkKVWriteBack"):
                key_buffer.index_copy_(0, all_destination_slots, key)
                value_buffer.index_copy_(0, all_destination_slots, value)
            writeback_cuda_ms = _cuda_event_elapsed_ms(writeback_cuda_started)
            writeback_s = _profile_elapsed(writeback_started)
            timings["kv_writeback_s"] += writeback_s
            timings["kv_writeback_cuda_ms"] += writeback_cuda_ms

        _cuda_event_elapsed_ms(kv_link_cuda_started)
        if triton_fusion_enabled:
            _emit_pic_trace(
                "triton_fused_kv_link",
                phase="link",
                **getattr(self, "_active_trace_context", {}),
                execution=(
                    "fused_gather_rope_scatter_boundary_repair_aware"
                    if triton_boundary_repair_active
                    else "fused_gather_rope_scatter"
                ),
                token_count=int(all_source_slots.numel()),
                boundary_repair_token_count=repair_token_count,
                boundary_repair_aware=triton_boundary_repair_active,
                boundary_repair_execution="sglang_forward_extend",
                prefix_token_count=prefix_token_count,
                remapped_token_count=tail_token_count,
                layers_written=layer_count,
                kernel_launches=layer_count,
                elapsed_s=timings["triton_fused_kv_link_s"],
                elapsed_cuda_ms=timings["triton_fused_kv_link_cuda_ms"],
                **triton_launch_config,
                **triton_traffic,
            )
        total_tokens = sum(int(spec["token_count"]) for spec in block_specs)
        tail_tokens = max(total_tokens - prefix_token_count, 0)
        for spec in block_specs:
            token_share = (
                int(spec["token_count"]) / total_tokens if total_tokens else 0.0
            )
            rope_share = (
                int(spec["token_count"]) / tail_tokens
                if spec["needs_rope_remap"] and tail_tokens
                else 0.0
            )
            block_timing = {
                "kv_retrieval_s": timings["kv_retrieval_s"] * token_share,
                "rope_remap_s": timings["rope_remap_s"] * rope_share,
                "kv_writeback_s": timings["kv_writeback_s"] * token_share,
                "triton_fused_kv_link_s": timings["triton_fused_kv_link_s"]
                * token_share,
                "kv_retrieval_cuda_ms": timings["kv_retrieval_cuda_ms"] * token_share,
                "rope_remap_cuda_ms": timings["rope_remap_cuda_ms"] * rope_share,
                "kv_writeback_cuda_ms": timings["kv_writeback_cuda_ms"] * token_share,
                "triton_fused_kv_link_cuda_ms": timings["triton_fused_kv_link_cuda_ms"]
                * token_share,
            }
            snapshot = spec["snapshot"]
            if spec["needs_rope_remap"]:
                _emit_pic_trace(
                    "rope_global_position_remap",
                    phase="link",
                    **getattr(self, "_active_trace_context", {}),
                    block_index=spec["block_index"],
                    block_id=spec["block_range"].get("block_id"),
                    cache_key=spec["cache_key"],
                    local_start_token=0,
                    global_offset=spec["start"],
                    token_count=spec["token_count"],
                    layers_remapped=layer_count,
                    execution="layerwise_batched_blocks",
                    timing_attribution="proportional_tokens",
                    elapsed_s=block_timing["rope_remap_s"],
                    elapsed_cuda_ms=block_timing["rope_remap_cuda_ms"],
                )
            _emit_pic_trace(
                "kv_block_linked",
                phase="link",
                **getattr(self, "_active_trace_context", {}),
                block_index=spec["block_index"],
                block_id=spec["block_range"].get("block_id"),
                cache_key=spec["cache_key"],
                storage_mode=snapshot.storage_mode,
                local_start_token=0,
                local_end_token=spec["token_count"],
                global_offset=spec["start"],
                global_end_token=spec["end"],
                source_slots=(
                    _trace_values(snapshot.source_slots)
                    if snapshot.source_slots is not None
                    else {
                        "count": spec["token_count"],
                        "values": [],
                        "truncated": True,
                    }
                ),
                destination_slots=_trace_values(spec["destination_slots"]),
                rope_remapped=spec["needs_rope_remap"],
                layers_written=layer_count,
                kv_writeback=(
                    ("triton_fused_gather_rope_scatter_boundary_repair_aware")
                    if triton_fusion_enabled and triton_boundary_repair_active
                    else (
                        "triton_fused_gather_rope_scatter"
                        if triton_fusion_enabled
                        else "index_copy"
                    )
                ),
                execution=(
                    "layerwise_triton_fused_boundary_repair_aware"
                    if triton_fusion_enabled and triton_boundary_repair_active
                    else (
                        "layerwise_triton_fused"
                        if triton_fusion_enabled
                        else "layerwise_batched_blocks"
                    )
                ),
                boundary_repair_token_count=(
                    int(spec["repair_token_count"])
                    if triton_fusion_enabled and triton_boundary_repair_active
                    else 0
                ),
                timing_attribution="proportional_tokens",
                timing_breakdown_s=block_timing,
            )
        timings["_legolink_total_span"] = legolink_total_span
        return timings

    def _promote_reconstructed_group(
        self,
        *,
        pool: Any,
        destination_slots: torch.Tensor,
        full_input_ids: list[int],
        link_plan: dict[str, Any],
        recompute_tokens_per_block: list[int],
        trace_context: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """EXPERIMENTAL. Make a reconstructed execution group durable.

        A Link already builds the grouped state -- member KV copied to its
        global offset, RoPE remapped, selected rows recomputed -- and then
        throws it away, because ``register_legolink_block`` is only ever
        reached from the compile path. This is the missing half: after a
        *fully* reconstructed group, copy the grouped rows into cache-owned
        slots and register them, so the next request linking the same members
        finds a block instead of rebuilding one.

        Two things make this a copy rather than an ownership transfer of the
        request's own slots, and neither is about speed:

        * **Aliasing.** The request's slots are inserted into the radix tree by
          ``cache_finished_req`` under the *whole* request sequence. Handing
          the same slots to a group block would give two cache entries one
          buffer, and the rotation below would then corrupt the other one.
        * **Position frame.** Those rows carry RoPE angles for their global
          positions. A compiled block carries block-local angles, because
          ``_copy_linked_kv`` rotates local->global at link time. Registering
          globally-rotated rows would double-rotate them on the next link, so
          the copy is rotated back into the local frame first and a promoted
          block is interchangeable with a compiled one.

        Fails closed everywhere: a skipped promotion is a trace row and a
        served request, never a partial cache entry.
        """

        def skip(reason: str, **details: Any) -> None:
            _emit_pic_trace(
                "group_promotion_skipped",
                phase="link",
                **trace_context,
                reason=reason,
                **details,
            )

        group_key = link_plan.get("execution_group_key")
        if not isinstance(group_key, str) or not group_key.strip():
            skip("no_execution_group_key")
            return None
        group_key = group_key.strip()

        cache_keys = [str(key) for key in (link_plan.get("cache_keys") or [])]
        if len(cache_keys) < 2:
            # A singleton group *is* its member's block. A second name for the
            # same KV would duplicate an entry that already exists and works.
            skip("singleton_group", member_count=len(cache_keys))
            return None

        ranges = link_plan.get("block_ranges") or link_plan.get("attention_ranges") or []
        if len(ranges) != len(cache_keys):
            skip("block_range_mismatch")
            return None

        # Only a fully reconstructed group may become durable. Every member
        # after the first must have been recomputed in its entirety; a
        # partially refreshed group answers correctly often enough to be
        # tempting, and is exactly what must not be persisted.
        for index, (block_range, window) in enumerate(
            zip(ranges[1:], recompute_tokens_per_block[1:]), start=1
        ):
            token_count = int(block_range["end_token"]) - int(block_range["start_token"])
            if int(window) < token_count:
                skip(
                    "reconstruction_below_promotion_policy",
                    member_index=index,
                    recomputed=int(window),
                    token_count=token_count,
                )
                return None

        tree_cache = getattr(self, "_legolink_cache", None)
        if tree_cache is None or not hasattr(tree_cache, "register_legolink_block"):
            skip("legolink_registry_unavailable")
            return None
        allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
        if allocator is None or not hasattr(allocator, "alloc"):
            skip("allocator_unavailable")
            return None
        if int(getattr(tree_cache, "page_size", 1) or 1) != 1:
            # A paged tree aligns keys down, so the block's tail would be
            # registered without its KV. Refuse rather than register short.
            skip("page_size_unsupported", page_size=int(tree_cache.page_size))
            return None

        group_start = int(ranges[0]["start_token"])
        group_end = int(ranges[-1]["end_token"])
        token_count = group_end - group_start
        group_token_ids = [int(token) for token in full_input_ids[group_start:group_end]]
        if token_count <= 0 or len(group_token_ids) != token_count:
            skip("group_token_range_invalid")
            return None
        if getattr(
            tree_cache, "has_legolink_block", None
        ) and tree_cache.has_legolink_block(group_key):
            # Somebody already promoted this group. First registration wins and
            # this reconstruction is discarded rather than overwriting it.
            skip("already_registered", cache_key=group_key, token_count=token_count)
            return None

        _emit_pic_trace(
            "group_promotion_started",
            phase="link",
            **trace_context,
            cache_key=group_key,
            member_cache_keys=cache_keys,
            token_count=token_count,
            group_start_token=group_start,
        )

        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import InsertParams, RadixKey

        started = _profile_start()
        source_slots = destination_slots[group_start:group_end].long()

        def available(pool_allocator: Any) -> Optional[int]:
            size = getattr(pool_allocator, "available_size", None)
            try:
                return int(size()) if callable(size) else None
            except Exception:  # noqa: BLE001
                return None

        # Make room the way every ordinary allocation does, instead of refusing
        # whenever the pool happens to be full. This is the scheduler's own
        # helper: it evicts unlocked radix leaves and LegoLink blocks with no
        # reader and no pin, so neither the members this Link is reading
        # (reserved) nor this request's rows (not in the tree yet) can be what
        # it frees.
        available_before = available(allocator)
        evicted_tokens = 0
        if available_before is not None and available_before < token_count:
            try:
                from sglang.srt.mem_cache.common import evict_from_tree_cache

                evict_from_tree_cache(tree_cache, token_count)
            except Exception as error:  # noqa: BLE001 - never fail a served request
                skip(
                    "eviction_failed",
                    token_count=token_count,
                    error=str(error)[:300],
                )
                return None
            available_after = available(allocator)
            evicted_tokens = max(0, int(available_after or 0) - available_before)
            _emit_pic_trace(
                "group_promotion_eviction",
                phase="link",
                **trace_context,
                cache_key=group_key,
                token_count=token_count,
                available_before=available_before,
                available_after=available_after,
                evicted_tokens=evicted_tokens,
            )
        new_slots = allocator.alloc(token_count)
        if new_slots is None or len(new_slots) != token_count:
            if new_slots is not None:
                allocator.free(new_slots)
            skip(
                "allocation_failed",
                token_count=token_count,
                available_before=available_before,
                evicted_tokens=evicted_tokens,
            )
            return None
        new_slots = new_slots.long()

        # Once `insert` runs, the tree owns these rows. Freeing them on a later
        # error would be a double free, so the error path has to know which
        # side of that line it is on.
        inserted = False

        try:
            global_positions = torch.arange(
                group_start, group_end, device=source_slots.device, dtype=torch.int64
            )
            local_positions = torch.arange(
                token_count, device=source_slots.device, dtype=torch.int64
            )
            for layer_id in range(
                self.model_runner.start_layer, self.model_runner.end_layer
            ):
                key = pool.get_key_buffer(layer_id).index_select(0, source_slots)
                value = pool.get_value_buffer(layer_id).index_select(0, source_slots)
                # global -> local: the exact inverse of what the next link does.
                key = self._remap_key_positions(
                    layer_id,
                    key,
                    global_positions,
                    local_positions,
                    max_global_position=max(group_end, token_count),
                )
                pool.get_key_buffer(layer_id).index_copy_(0, new_slots, key)
                pool.get_value_buffer(layer_id).index_copy_(0, new_slots, value)

            radix_key = RadixKey(token_ids=list(group_token_ids), extra_key=None)
            result = tree_cache.insert(
                InsertParams(key=radix_key, value=new_slots.to(torch.int64).clone())
            )
            inserted = True
            prefix_len = int(getattr(result, "prefix_len", 0) or 0)
            if prefix_len:
                # Rows already in the tree belong to the tree, not to this copy.
                allocator.free(new_slots[:prefix_len])
            match_result = tree_cache.match_prefix(MatchPrefixParams(key=radix_key))
            if len(match_result.device_indices) != token_count:
                skip(
                    "radix_insert_incomplete",
                    matched=int(len(match_result.device_indices)),
                    token_count=token_count,
                )
                return None
            # The promoted block must declare the compatibility the request
            # that reuses it will declare, or `lookup_legolink_blocks` rejects
            # it as a fingerprint mismatch and the reuse never happens. This is
            # the same construction `pin_finished_compile` uses for a compiled
            # block -- the deployment's fingerprints, overlaid with the policy
            # versions this request named.
            compatibility = self.get_legolink_compatibility(
                adapter_id=link_plan.get("adapter_id")
            )
            for field in ("segmentation_policy_version", "legolink_policy_version"):
                declared = link_plan.get(field)
                if declared is not None:
                    compatibility[field] = str(declared)
            # A registered block is read through its snapshot, so a promoted
            # one needs the same reference-mode handle `rebind_reference_slots`
            # gives a compiled block: the cache-owned slots, marked ready. The
            # KV itself stays in the pool; the snapshot only says where.
            snapshot = PICBlockSnapshot(
                cache_key=group_key,
                token_ids=tuple(group_token_ids),
                storage_mode="reference",
                source_slots=match_result.device_indices.long().clone(),
                reference_ready=True,
            )
            metadata = tree_cache.register_legolink_block(
                cache_key=group_key,
                block_id=group_key,
                sub_context_id=group_key,
                semantic_block_id=group_key,
                context_field=str(link_plan.get("context_field") or "execution_group"),
                token_ids=tuple(group_token_ids),
                token_range=(0, token_count),
                kv_slots=match_result.device_indices,
                snapshot=snapshot,
                radix_node=match_result.last_device_node,
                recomputed_tokens=token_count,
                pinned=False,
                compatibility=compatibility,
            )
        except Exception as error:  # noqa: BLE001 - never fail a served request
            if not inserted:
                try:
                    allocator.free(new_slots)
                except Exception:  # noqa: BLE001
                    pass
            skip("promotion_failed", error=str(error)[:300], inserted=inserted)
            return None

        elapsed_s = _profile_elapsed(started)
        record = {
            "cache_key": group_key,
            "member_cache_keys": cache_keys,
            "token_count": token_count,
            "group_start_token": group_start,
            "promotion_elapsed_s": round(elapsed_s, 6),
            # False when a resident block won the key -- a race, since the
            # `has_legolink_block` check above already skips the ordinary case.
            "became_authoritative": bool(
                getattr(metadata, "cache_key", None) == group_key
                and getattr(metadata, "context_field", None)
                == str(link_plan.get("context_field") or "execution_group")
            ),
            # Metadata only. Serving must not branch on where a block came
            # from: a promoted block and a compiled block are the same thing.
            "origin": "reconstructed",
        }
        _emit_pic_trace(
            "group_registered",
            phase="link",
            **trace_context,
            **record,
            block_id=getattr(metadata, "block_id", group_key),
        )
        logger.info(
            "SGLang PIC promoted reconstructed group cache_key=%s tokens=%d "
            "members=%d elapsed_s=%.4f",
            group_key,
            token_count,
            len(cache_keys),
            elapsed_s,
        )
        return record

    def _remap_key_positions(
        self,
        layer_id: int,
        key: torch.Tensor,
        local_positions: torch.Tensor,
        global_positions: torch.Tensor,
        max_global_position: Optional[int] = None,
    ) -> torch.Tensor:
        rotary = self._rotary_by_layer.get(layer_id)
        if rotary is None or not hasattr(rotary, "cos_sin_cache"):
            raise PICRuntimeError(
                f"Layer {layer_id} has no supported rotary embedding for PIC."
            )
        if max_global_position is None:
            max_global_position = int(global_positions[-1].item())
        rotary._ensure_cos_sin_cache_length(int(max_global_position))
        cache = rotary.cos_sin_cache
        local_cos, local_sin = cache.index_select(0, local_positions).chunk(2, dim=-1)
        global_cos, global_sin = cache.index_select(0, global_positions).chunk(
            2, dim=-1
        )
        rotary_dim = int(rotary.rotary_dim)
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]
        # R(global) @ inv(R(local)) expressed as one rotation. This is
        # equivalent to inverse-then-forward RoPE but avoids a second BF16
        # rounding step.
        delta_cos = global_cos * local_cos + global_sin * local_sin
        delta_sin = global_sin * local_cos - global_cos * local_sin
        rerotated = _apply_rotary_emb(
            key_rot, delta_cos, delta_sin, rotary.is_neox_style
        )
        return torch.cat((rerotated, key_pass), dim=-1)

    @staticmethod
    def _make_one_token_batch(
        forward_batch: Any,
        *,
        token_id: int,
        position: int,
        destination_slot: torch.Tensor,
    ) -> Any:
        device = forward_batch.input_ids.device
        position_dtype = (
            forward_batch.positions.dtype
            if forward_batch.positions is not None
            else torch.int64
        )
        return dataclasses.replace(
            forward_batch,
            forward_mode=type(forward_batch.forward_mode).DECODE,
            batch_size=1,
            input_ids=torch.tensor([token_id], dtype=torch.long, device=device),
            seq_lens=torch.tensor([position + 1], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([position + 1], dtype=torch.int64),
            out_cache_loc=destination_slot.reshape(1),
            seq_lens_sum=position + 1,
            positions=torch.tensor([position], dtype=position_dtype, device=device),
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_start_loc=None,
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            extend_logprob_start_lens_cpu=None,
            input_embeds=None,
            replace_embeds=None,
            replace_positions=None,
            out_cache_loc_swa=None,
            spec_info=None,
            global_num_tokens_cpu=None,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
            num_token_non_padded=torch.tensor(1, dtype=torch.int32, device=device),
            num_token_non_padded_cpu=1,
            pic_request_metadata=None,
            full_input_ids=None,
        )

    @staticmethod
    def _make_grouped_extend_batch(
        forward_batch: Any,
        *,
        full_input_ids: list[int],
        ranges: list[tuple[int, int]],
        destination_slots: torch.Tensor,
    ) -> Any:
        if not ranges:
            raise PICRuntimeError("Grouped PIC recompute requires at least one range.")
        previous_end = -1
        for start, end in ranges:
            if start < 0 or end <= start or start < previous_end:
                raise PICRuntimeError(
                    "Grouped PIC recompute ranges must be ordered and disjoint."
                )
            if end > len(full_input_ids) or end > int(destination_slots.numel()):
                raise PICRuntimeError(
                    "Grouped PIC recompute range exceeds input layout."
                )
            previous_end = end

        device = forward_batch.input_ids.device
        position_dtype = (
            forward_batch.positions.dtype
            if forward_batch.positions is not None
            else torch.int64
        )
        forward_mode = type(forward_batch.forward_mode).EXTEND
        starts = [start for start, _ in ranges]
        ends = [end for _, end in ranges]
        lengths = [end - start for start, end in ranges]
        input_ids = torch.tensor(
            [
                int(full_input_ids[position])
                for start, end in ranges
                for position in range(start, end)
            ],
            dtype=torch.long,
            device=device,
        )
        positions = torch.cat(
            [
                torch.arange(start, end, dtype=position_dtype, device=device)
                for start, end in ranges
            ]
        )
        grouped_slots = torch.cat(
            [destination_slots[start:end] for start, end in ranges]
        )
        seq_lens = torch.tensor(ends, dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        extend_start_loc = torch.zeros(len(ranges), dtype=torch.int32, device=device)
        if len(ranges) > 1:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
        req_pool_indices = forward_batch.req_pool_indices
        if req_pool_indices is None or int(req_pool_indices.numel()) != 1:
            raise PICRuntimeError(
                "Grouped PIC recompute requires one source request-pool index."
            )
        req_pool_indices = req_pool_indices.repeat(len(ranges))
        total_extend_tokens = sum(lengths)
        return dataclasses.replace(
            forward_batch,
            forward_mode=forward_mode,
            global_forward_mode=forward_mode,
            batch_size=len(ranges),
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor(ends, dtype=torch.int64),
            orig_seq_lens=seq_lens,
            out_cache_loc=grouped_slots,
            seq_lens_sum=sum(ends),
            positions=positions,
            extend_num_tokens=total_extend_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=torch.tensor(starts, dtype=torch.int32, device=device),
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=starts,
            extend_seq_lens_cpu=lengths,
            extend_logprob_start_lens_cpu=lengths,
            return_logprob=False,
            input_embeds=None,
            replace_embeds=None,
            replace_positions=None,
            out_cache_loc_swa=None,
            spec_info=None,
            global_num_tokens_cpu=None,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
            num_token_non_padded=torch.tensor(
                total_extend_tokens, dtype=torch.int32, device=device
            ),
            num_token_non_padded_cpu=total_extend_tokens,
            is_extend_in_batch=True,
            all_extend_in_batch=True,
            pic_request_metadata=None,
            full_input_ids=None,
        )

    @staticmethod
    def _make_extend_batch(
        forward_batch: Any,
        *,
        token_ids: list[int],
        start: int,
        end: int,
        destination_slots: torch.Tensor,
    ) -> Any:
        if start < 0 or end <= start or end - start != len(token_ids):
            raise PICRuntimeError("Invalid PIC contiguous recompute range.")
        device = forward_batch.input_ids.device
        position_dtype = (
            forward_batch.positions.dtype
            if forward_batch.positions is not None
            else torch.int64
        )
        forward_mode = type(forward_batch.forward_mode).EXTEND
        extend_len = end - start
        seq_lens = torch.tensor([end], dtype=torch.int64, device=device)
        return dataclasses.replace(
            forward_batch,
            forward_mode=forward_mode,
            global_forward_mode=forward_mode,
            batch_size=1,
            input_ids=torch.tensor(token_ids, dtype=torch.long, device=device),
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor([end], dtype=torch.int64),
            orig_seq_lens=seq_lens,
            out_cache_loc=destination_slots,
            seq_lens_sum=end,
            positions=torch.arange(start, end, dtype=position_dtype, device=device),
            extend_num_tokens=extend_len,
            extend_seq_lens=torch.tensor(
                [extend_len], dtype=torch.int32, device=device
            ),
            extend_prefix_lens=torch.tensor([start], dtype=torch.int32, device=device),
            extend_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
            extend_prefix_lens_cpu=[start],
            extend_seq_lens_cpu=[extend_len],
            extend_logprob_start_lens_cpu=[extend_len],
            return_logprob=False,
            input_embeds=None,
            replace_embeds=None,
            replace_positions=None,
            out_cache_loc_swa=None,
            spec_info=None,
            global_num_tokens_cpu=None,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
            num_token_non_padded=torch.tensor(
                extend_len, dtype=torch.int32, device=device
            ),
            num_token_non_padded_cpu=extend_len,
            is_extend_in_batch=True,
            all_extend_in_batch=True,
            pic_request_metadata=None,
            full_input_ids=None,
        )

    def _validate_link_batch(self, forward_batch: Any) -> None:
        if forward_batch.batch_size != 1:
            raise PICRuntimeError("PIC link currently supports batch_size=1 only.")
        if not forward_batch.forward_mode.is_extend():
            raise PICRuntimeError("PIC link must enter ModelRunner as an extend batch.")
        if forward_batch.out_cache_loc_swa is not None:
            raise PICRuntimeError("PIC link does not yet support sliding-window KV.")
        if getattr(self.model_runner, "model_is_mrope", False):
            raise PICRuntimeError("PIC link does not yet support multimodal RoPE.")
        if forward_batch.spec_info is not None:
            raise PICRuntimeError("PIC link does not yet support speculative decoding.")
        if int(getattr(self.model_runner, "pp_size", 1)) > 1:
            raise PICRuntimeError("PIC link does not yet support pipeline parallelism.")

    @staticmethod
    def _validate_pool(pool: Any) -> None:
        if not all(
            hasattr(pool, method) for method in ("get_key_buffer", "get_value_buffer")
        ):
            raise PICRuntimeError("PIC link requires an MHA-style KV pool.")
        if getattr(pool, "store_dtype", None) != getattr(pool, "dtype", None):
            raise PICRuntimeError("PIC link does not yet support quantized KV cache.")
