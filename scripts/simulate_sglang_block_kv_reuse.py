#!/usr/bin/env python3
"""Simulate order-invariant block KV reuse from SGLang block payloads.

This script demonstrates the intended SGLang runtime behavior before editing
the actual scheduler/KV-cache internals:

- prefix baseline only reuses blocks while the sequence prefix is unchanged,
- block-key KV reuse hits any previously stored semantic block by `cache_key`,
  even when the same blocks appear in a different order.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noc_context_manager.block_kv_cache import (
    OrderInvariantBlockKVCache,
    prefix_reuse_tokens,
    summarize_lookups,
)


DEFAULT_PAYLOADS = Path("benchmark_results/sglang_block_runtime_plan/sglang_prefill_request_payloads.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def payload_to_block(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("openclaw_sub_context") or {}
    return {
        "block_id": str(meta.get("prefill_unit_id") or ""),
        "cache_key": str(meta.get("cache_key") or payload.get("extra_key") or ""),
        "context_field": str(meta.get("context_field") or ""),
        "functional_role": str(meta.get("functional_role") or ""),
        "token_count": int(meta.get("token_count") or 0),
    }


def build_sequences(blocks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "original": blocks,
        "repeat_same_order": list(blocks),
        "permuted_reverse_order": list(reversed(blocks)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_simulation(blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sequences = build_sequences(blocks)
    cache = OrderInvariantBlockKVCache()
    lookup_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    previous_sequence: list[dict[str, Any]] | None = None

    for sequence_id, sequence_blocks in sequences.items():
        lookups = []
        for position, block in enumerate(sequence_blocks):
            lookups.append(
                cache.lookup_or_store(
                    sequence_id=sequence_id,
                    position=position,
                    block_id=block["block_id"],
                    cache_key=block["cache_key"],
                    token_count=block["token_count"],
                )
            )
        lookup_rows.extend(lookup.to_dict() for lookup in lookups)

        block_summary = summarize_lookups(lookups)
        prefix_tokens = (
            prefix_reuse_tokens(previous_sequence, sequence_blocks)
            if previous_sequence is not None
            else 0
        )
        token_total = block_summary["tokens"]
        summary_rows.append(
            {
                "sequence_id": sequence_id,
                "blocks": block_summary["blocks"],
                "tokens": token_total,
                "prefix_baseline_reused_tokens": prefix_tokens,
                "prefix_baseline_reuse_ratio": prefix_tokens / token_total if token_total else 0.0,
                "block_kv_reused_tokens": block_summary["reused_tokens"],
                "block_kv_prefill_tokens": block_summary["prefill_tokens"],
                "block_kv_reuse_ratio": block_summary["reuse_ratio"],
                "block_kv_cache_hits": block_summary["cache_hits"],
            }
        )
        previous_sequence = sequence_blocks

    return lookup_rows, summary_rows, {"sequence_count": len(sequences), "block_count": len(blocks)}


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SGLang Block KV Reuse Simulation",
        "",
        "## Purpose",
        "",
        "This compares native prefix-style reuse with the intended OpenClaw",
        "block-key KV reuse policy. The key test is the permuted sequence:",
        "the same semantic blocks appear in a different order.",
        "",
        "## Result",
        "",
        "| sequence | blocks | tokens | prefix reuse | block KV reuse | block KV prefill |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {sequence_id} | {blocks} | {tokens} | {prefix_baseline_reuse_ratio:.1%} | "
            "{block_kv_reuse_ratio:.1%} | {block_kv_prefill_tokens} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `repeat_same_order` should be easy for ordinary prefix cache.",
            "- `permuted_reverse_order` is the important case: prefix reuse drops",
            "  because order changed, while block-key KV reuse should still hit by",
            "  `cache_key`.",
            "- This is a policy simulation, not an SGLang internal patch yet.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/sglang_block_kv_reuse_simulation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = read_jsonl(args.payloads)
    blocks = [payload_to_block(payload) for payload in payloads]
    lookup_rows, summary_rows, metadata = run_simulation(blocks)

    write_csv(args.output_dir / "block_kv_lookup_rows.csv", lookup_rows)
    write_csv(args.output_dir / "block_kv_reuse_summary.csv", summary_rows)
    (args.output_dir / "block_kv_reuse_simulation.json").write_text(
        json.dumps(
            {
                "payloads": args.payloads.as_posix(),
                "metadata": metadata,
                "summary": summary_rows,
                "lookups": lookup_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.output_dir / "block_kv_reuse_simulation_report.md", summary_rows)

    print("SGLang block KV reuse simulation")
    print("================================")
    for row in summary_rows:
        print(
            "{sequence_id}: prefix={prefix_baseline_reuse_ratio:.1%}, "
            "block_kv={block_kv_reuse_ratio:.1%}, prefill={block_kv_prefill_tokens}".format(**row)
        )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
