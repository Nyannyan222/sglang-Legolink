"""Opt-in profiling helpers shared by NOC control-plane modules.

The helpers in this module are deliberately independent from request latency
measurement.  They add diagnostic host spans, PyTorch ``record_function``
regions, and optional NVTX ranges only when their corresponding environment
flags are enabled.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = 1
_HOST_TIMING_LOCK = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _bounded_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    allowed = (
        "request_id",
        "task_id",
        "variant",
        "phase",
        "concurrency",
        "k",
        "decode_iteration",
    )
    return {
        key: metadata[key]
        for key in allowed
        if metadata.get(key) is not None
    }


def nvtx_range_name(name: str, metadata: dict[str, Any] | None = None) -> str:
    """Return a stable NVTX name containing bounded request metadata."""

    values = _bounded_metadata(metadata)
    if not values:
        return name
    suffix = ",".join(f"{key}={values[key]}" for key in values)
    return f"{name}[{suffix}]"


@contextlib.contextmanager
def profile_range(
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Emit an exact PyTorch region and an optional metadata-rich NVTX range."""

    profile_enabled = _env_flag("NOC_PROFILE_RANGES_ENABLED")
    nvtx_enabled = _env_flag("NOC_NVTX_ENABLED")
    if not profile_enabled and not nvtx_enabled:
        yield
        return

    torch = None
    try:
        import torch as imported_torch

        torch = imported_torch
    except (ImportError, OSError):
        pass

    record_context = (
        torch.profiler.record_function(name)
        if profile_enabled and torch is not None
        else contextlib.nullcontext()
    )
    can_nvtx = bool(
        nvtx_enabled
        and torch is not None
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "nvtx")
    )
    with record_context:
        if can_nvtx:
            torch.cuda.nvtx.range_push(nvtx_range_name(name, metadata))
        try:
            yield
        finally:
            if can_nvtx:
                torch.cuda.nvtx.range_pop()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _HOST_TIMING_LOCK, path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class HostTimingRecorder:
    """Nested ``perf_counter_ns`` spans for NOC CPU control-plane diagnostics."""

    def __init__(
        self,
        stage: str,
        *,
        metadata: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.enabled = _env_flag("NOC_HOST_TIMING_ENABLED")
        self.stage = stage
        self.metadata = _bounded_metadata(metadata)
        configured_path = output_path or os.getenv("NOC_HOST_TIMING_PATH")
        self.output_path = Path(configured_path) if configured_path else None
        self.started_ns = time.perf_counter_ns() if self.enabled else 0
        # Wall-clock anchor for started_ns: a span's absolute time is
        # started_unix_ns + (span_ns - started_ns). Lets host rows be placed on
        # the same axis as other processes' wall-clock events.
        self.started_unix_ns = time.time_ns() if self.enabled else 0
        self.spans: list[dict[str, Any]] = []
        self._stack: list[tuple[str, int, int]] = []
        self._emitted = False

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started_ns = time.perf_counter_ns()
        depth = len(self._stack)
        self._stack.append((name, started_ns, depth))
        try:
            yield
        finally:
            ended_ns = time.perf_counter_ns()
            active_name, active_start, active_depth = self._stack.pop()
            self.spans.append(
                {
                    "name": active_name,
                    "depth": active_depth,
                    "started_ns": active_start,
                    "ended_ns": ended_ns,
                    "elapsed_ns": max(ended_ns - active_start, 0),
                }
            )

    def emit(self, *, status: str = "ok", error: str | None = None) -> dict[str, Any]:
        if not self.enabled or self._emitted:
            return {}
        self._emitted = True
        ended_ns = time.perf_counter_ns()
        row = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "metric_level": "request",
            "measurement_kind": "diagnostic_host_timing",
            "profiling_type": "host_timing",
            "formal_latency": False,
            "diagnostic": True,
            "event": "noc_host_timing",
            "stage": self.stage,
            "status": status,
            "started_ns": self.started_ns,
            "started_unix_ns": self.started_unix_ns,
            "ended_ns": ended_ns,
            "wall_elapsed_ns": max(ended_ns - self.started_ns, 0),
            "spans": list(self.spans),
            **self.metadata,
        }
        if error:
            row["error"] = error
        if self.output_path is not None:
            _append_jsonl(self.output_path, row)
        return row


@dataclass(frozen=True)
class ProfilerWindowConfig:
    """Validated controlled-window configuration for PyTorch Profiler runs."""

    wait: int = 1
    warmup: int = 2
    active: int = 5
    repeat: int = 1
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False

    def __post_init__(self) -> None:
        if min(self.wait, self.warmup, self.active, self.repeat) < 0:
            raise ValueError("profiler schedule values cannot be negative")
        if self.active < 1 or self.repeat < 1:
            raise ValueError("profiler active and repeat must be at least one")

    @property
    def total_steps(self) -> int:
        return (self.wait + self.warmup + self.active) * self.repeat

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NsightRunConfig:
    """Serializable Nsight Systems command configuration."""

    output: str
    command: tuple[str, ...]
    trace: str = "cuda,nvtx,osrt"
    capture_range: str = "cudaProfilerApi"
    capture_range_end: str = "stop"
    sample: str = "none"
    force_overwrite: bool = True

    def nsys_command(self) -> list[str]:
        return [
            "nsys",
            "profile",
            f"--trace={self.trace}",
            f"--capture-range={self.capture_range}",
            f"--capture-range-end={self.capture_range_end}",
            f"--sample={self.sample}",
            f"--force-overwrite={'true' if self.force_overwrite else 'false'}",
            f"--output={self.output}",
            "--",
            *self.command,
        ]

    def manifest(self, *, role: str) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "metric_level": "nsight_run",
            "measurement_kind": "diagnostic_nsight_systems",
            "profiling_type": "nsight_systems",
            "formal_latency": False,
            "diagnostic": True,
            "role": role,
            "output": self.output,
            "trace": self.trace,
            "capture_range": self.capture_range,
            "capture_range_end": self.capture_range_end,
            "sample": self.sample,
            "command": self.nsys_command(),
        }

