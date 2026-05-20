#!/usr/bin/env python3
"""Summarize what the OpenClaw SGLang work proves so far.

The report intentionally separates:

1. observed local SGLang behavior,
2. observed native-prefix limitation under order permutation, and
3. simulated behavior of the intended semantic block-key KV cache.

This avoids overstating the current result as a completed runtime-level
non-prefix KV reuse patch.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BLOCK_REPLAY = Path(
    "benchmark_results/sglang_block_prefill_runs/sglang_block_prefill_20260520_153110"
)
DEFAULT_ORDER_RUNTIME = Path(
    "benchmark_results/wildclaw_order_permutation_runtime_runs/"
    "wildclaw_sglang_runtime_20260515_171823/wildclaw_sglang_runtime_results.csv"
)
DEFAULT_POLICY_SIM = Path("benchmark_results/sglang_block_kv_reuse_simulation_20260521")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def summarize_openclaw_events(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        block_id = event.get("openclaw_prefill_unit_id") or ""
        if block_id:
            by_block[block_id].append(event)

    rows = []
    for block_id, block_events in sorted(by_block.items()):
        first = block_events[0]
        last = block_events[-1]
        token_count = as_float(first.get("input_token_len"))
        first_cached = as_float(first.get("cached_tokens"))
        last_cached = as_float(last.get("cached_tokens"))
        rows.append(
            {
                "block_id": block_id,
                "context_field": first.get("openclaw_context_field", ""),
                "events": len(block_events),
                "token_count": int(token_count),
                "first_cached_tokens": int(first_cached),
                "last_cached_tokens": int(last_cached),
                "first_cached_ratio": first_cached / token_count if token_count else 0.0,
                "last_cached_ratio": last_cached / token_count if token_count else 0.0,
            }
        )
    return rows


def summarize_order_runtime(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            grouped[row.get("condition", "")].append(row)

    summary = []
    for condition, condition_rows in sorted(grouped.items()):
        prompt = sum(as_float(row.get("prompt_tokens")) for row in condition_rows)
        cached = sum(as_float(row.get("cached_tokens")) for row in condition_rows)
        prefill = sum(as_float(row.get("estimated_prefill_tokens")) for row in condition_rows)
        summary.append(
            {
                "condition": condition,
                "rows": len(condition_rows),
                "prompt_tokens": int(prompt),
                "cached_tokens": int(cached),
                "estimated_prefill_tokens": int(prefill),
                "cached_ratio": cached / prompt if prompt else 0.0,
            }
        )
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    block_summary = payload["block_replay"]["blocks"]
    metadata_summary = payload["block_replay"]["metadata_summary"]
    order_runtime = payload["order_permutation_runtime"]
    policy_sim = payload["policy_simulation"]

    lines = [
        "# OpenClaw Method Effect Validation",
        "",
        "## Current Verdict",
        "",
        "The current implementation proves runtime feasibility and observability,",
        "but it does not yet prove that OpenClaw is faster than original SGLang",
        "at the actual KV-cache implementation level.",
        "",
        "What is proven:",
        "",
        "- OpenClaw semantic block metadata reaches SGLang request parsing.",
        "- The scheduler cache-lookup log can identify OpenClaw block id, cache key, and context field.",
        "- Replaying the same OpenClaw block produces full prefix-cache hits under SGLang's existing cache.",
        "- Native prefix-cache reuse drops when semantic blocks are reordered.",
        "",
        "What is not proven yet:",
        "",
        "- True runtime-level semantic non-prefix KV reuse.",
        "- Order-independent KV reuse inside SGLang's scheduler/KV cache.",
        "- End-to-end latency improvement over original SGLang after a real non-prefix patch.",
        "",
        "## Local OpenClaw Metadata Replay",
        "",
        f"- Cache lookup events: {metadata_summary['cache_lookup_events']}",
        f"- OpenClaw cache lookup events: {metadata_summary['openclaw_cache_lookup_events']}",
        f"- Metadata complete: {metadata_summary['metadata_complete']}",
        f"- Extra-key mismatches: {metadata_summary['events_extra_key_mismatch_cache_key']}",
        "",
        "| context field | events | tokens | first cached | last cached | first ratio | last ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in block_summary:
        lines.append(
            "| {context_field} | {events} | {token_count} | {first_cached_tokens} | "
            "{last_cached_tokens} | {first_ratio} | {last_ratio} |".format(
                **row,
                first_ratio=percent(row["first_cached_ratio"]),
                last_ratio=percent(row["last_cached_ratio"]),
            )
        )

    lines.extend(
        [
            "",
            "## Native SGLang Prefix Limitation",
            "",
            "This is an actual SGLang runtime replay using reordered semantic contexts.",
            "",
            "| condition | rows | prompt tokens | cached tokens | prefill tokens | cached ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in order_runtime:
        display_row = {
            **row,
            "cached_ratio_display": percent(row["cached_ratio"]),
        }
        lines.append(
            "| {condition} | {rows} | {prompt_tokens} | {cached_tokens} | "
            "{estimated_prefill_tokens} | {cached_ratio_display} |".format(**display_row)
        )

    lines.extend(
        [
            "",
            "## Intended Block-Key KV Policy Simulation",
            "",
            "This is a policy simulation of the target implementation, not a completed SGLang patch.",
            "",
            "| sequence | blocks | tokens | prefix reuse | block-key reuse | block-key prefill |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in policy_sim:
        lines.append(
            "| {sequence_id} | {blocks} | {tokens} | {prefix_reuse} | "
            "{block_reuse} | {block_kv_prefill_tokens} |".format(
                **row,
                prefix_reuse=percent(as_float(row["prefix_baseline_reuse_ratio"])),
                block_reuse=percent(as_float(row["block_kv_reuse_ratio"])),
            )
        )

    lines.extend(
        [
            "",
            "## Next Patch Required",
            "",
            "The next implementation step is to patch SGLang internals so that",
            "`openclaw_cache_key` indexes runtime-owned KV blocks independently",
            "from prompt order. The expected acceptance test is:",
            "",
            "- same semantic blocks in same order: original SGLang and OpenClaw both hit;",
            "- same semantic blocks in different order: original SGLang prefix reuse drops;",
            "- patched OpenClaw block KV reuse still hits by semantic `cache_key`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-replay-dir", type=Path, default=DEFAULT_BLOCK_REPLAY)
    parser.add_argument("--order-runtime-csv", type=Path, default=DEFAULT_ORDER_RUNTIME)
    parser.add_argument("--policy-sim-dir", type=Path, default=DEFAULT_POLICY_SIM)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/openclaw_method_effect_validation"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_summary = read_json(
        args.block_replay_dir
        / "openclaw_log_verification"
        / "openclaw_cache_lookup_summary.json"
    )
    block_events = read_csv(
        args.block_replay_dir
        / "openclaw_log_verification"
        / "openclaw_cache_lookup_events.csv"
    )
    order_runtime_rows = read_csv(args.order_runtime_csv)
    policy_sim_rows = read_csv(args.policy_sim_dir / "block_kv_reuse_summary.csv")

    payload = {
        "block_replay": {
            "metadata_summary": metadata_summary,
            "blocks": summarize_openclaw_events(block_events),
        },
        "order_permutation_runtime": summarize_order_runtime(order_runtime_rows),
        "policy_simulation": policy_sim_rows,
    }
    write_json(args.output_dir / "openclaw_method_effect_validation.json", payload)
    write_report(args.output_dir / "openclaw_method_effect_validation_report.md", payload)

    print("OpenClaw method effect validation")
    print("=================================")
    print(f"Metadata complete: {metadata_summary.get('metadata_complete')}")
    print(f"OpenClaw events: {metadata_summary.get('openclaw_cache_lookup_events')}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
