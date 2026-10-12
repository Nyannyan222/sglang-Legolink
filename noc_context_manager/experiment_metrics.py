"""Aggregation and artifact helpers for formal NOC/LegoLink experiments."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORMAL_KINDS = {
    "formal_latency",
    "paired_replay",
    "official_agent",
    "wildclaw_long_running",
}
ORIGINAL_VARIANTS = {
    "full_context_original_sglang",
    "selected_context_original_sglang",
    "original_sglang",
}
LEGOLINK_ONLY_FIELDS = (
    "compiled_blocks",
    "compiled_tokens",
    "cache_hits",
    "cache_misses",
    "linked_tokens",
    "reused_tokens",
    "recomputed_tokens",
    "cache_reuse_ratio",
    "recompute_ratio",
    "average_reuse_count_per_block",
    "superseded_blocks",
    "invalidated_blocks",
    "evicted_blocks",
    "evicted_tokens",
    "fallback_count",
    "recovery_success_count",
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summary_stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    samples = [float(value) for value in values if value is not None]
    return {
        "sample_count": len(samples),
        "mean": statistics.mean(samples) if samples else None,
        "std": statistics.stdev(samples)
        if len(samples) > 1
        else 0.0
        if samples
        else None,
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
    }


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("warmup", False)
    normalized.setdefault(
        "formal_latency", normalized.get("measurement_kind") in FORMAL_KINDS
    )
    normalized.setdefault("diagnostic", not normalized["formal_latency"])
    if normalized.get("variant") in ORIGINAL_VARIANTS:
        for field in LEGOLINK_ONLY_FIELDS:
            normalized[field] = None
    return normalized


def formal_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        normalize_row(row)
        for row in rows
        if not bool(row.get("warmup"))
        and row.get("measurement_kind") in FORMAL_KINDS
        and not bool(row.get("diagnostic"))
    ]


def _sum(rows: list[dict[str, Any]], field: str) -> int:
    return sum(int(row.get(field) or 0) for row in rows)


def aggregate_agent_steps(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = normalize_row(raw)
        grouped[
            (
                row.get("variant"),
                row.get("task_id"),
                row.get("trial_id"),
                row.get("concurrency") or 1,
            )
        ].append(row)
    output = []
    for (variant, task_id, trial_id, concurrency), items in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        latency = summary_stats(row.get("step_latency_ms") for row in items)
        output.append(
            normalize_row(
                {
                    "metric_level": "task",
                    "measurement_kind": "official_agent",
                    "variant": variant,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "concurrency": concurrency,
                    "agent_steps": len(items),
                    "planner_calls": _sum(items, "planner_calls"),
                    "replanner_calls": _sum(items, "replanner_calls"),
                    "llm_calls": _sum(items, "llm_calls"),
                    "tool_calls": _sum(items, "tool_calls"),
                    "successful_tool_calls": _sum(items, "successful_tool_calls"),
                    "failed_tool_calls": _sum(items, "failed_tool_calls"),
                    "repeated_action_count": _sum(items, "repeated_action_count"),
                    "invalid_tool_call_count": _sum(items, "invalid_tool_call_count"),
                    "step_latency_ms": latency,
                }
            )
        )
    return output


def aggregate_experiment(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = normalize_row(raw)
        if row.get("warmup"):
            continue
        grouped[
            (
                row.get("variant"),
                row.get("measurement_kind"),
                row.get("concurrency") or 1,
                row.get("phase"),
                row.get("k"),
            )
        ].append(row)
    result = []
    for (variant, kind, concurrency, phase, k), items in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        request = summary_stats(row.get("request_latency_ms") for row in items)
        e2e = summary_stats(row.get("e2e_latency_ms") for row in items)
        wall_ms = sum(float(row.get("e2e_latency_ms") or 0.0) for row in items)
        completed = sum(
            bool(row.get("task_success") or row.get("verifier_pass")) for row in items
        )
        reused = None if variant in ORIGINAL_VARIANTS else _sum(items, "reused_tokens")
        recomputed = (
            None if variant in ORIGINAL_VARIANTS else _sum(items, "recomputed_tokens")
        )
        linked = (
            reused + recomputed
            if reused is not None and recomputed is not None
            else None
        )
        result.append(
            normalize_row(
                {
                    "metric_level": "experiment",
                    "measurement_kind": kind,
                    "variant": variant,
                    "concurrency": concurrency,
                    "phase": phase,
                    "k": k,
                    "sample_count": len(items),
                    "request_latency_ms_stats": request,
                    "e2e_latency_ms_stats": e2e,
                    "completed_tasks": completed,
                    "failed_tasks": len(items) - completed,
                    "verifier_pass_rate": completed / len(items) if items else None,
                    "steps_per_minute": (
                        60_000.0 * _sum(items, "agent_steps") / wall_ms
                        if wall_ms
                        else None
                    ),
                    "tasks_per_hour": (
                        3_600_000.0 * len(items) / wall_ms if wall_ms else None
                    ),
                    "successful_tasks_per_gpu_hour": (
                        3_600_000.0 * completed / wall_ms if wall_ms else None
                    ),
                    "reused_tokens": reused,
                    "recomputed_tokens": recomputed,
                    "cache_reuse_ratio": (reused / linked if linked else None),
                    "recompute_ratio": (recomputed / linked if linked else None),
                }
            )
        )
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(normalize_row(row), ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(
        {
            field
            for row in rows
            for field in row
            if not isinstance(row[field], (dict, list))
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
