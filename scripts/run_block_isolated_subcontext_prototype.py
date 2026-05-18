#!/usr/bin/env python3
"""Run a block-isolated sub-context prefill/cache prototype.

This script proves two runtime-facing properties over field/function
sub-contexts:

1. Each sub-context can be prefilled and cached independently.
2. The block-level attention contract has zero allowed pairwise attention
   edges between different sub-contexts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noc_context_manager.runtime_isolation import (
    IndependentPrefillCache,
    block_isolation_report,
    build_blocks,
)


DEFAULT_INPUT = Path("benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def select_rows(rows: list[dict[str, Any]], task_id: str, max_blocks: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if not task_id or row.get("task_id") == task_id]
    candidates.sort(
        key=lambda row: (
            int((row.get("selection_metadata") or {}).get("field_order", 99)),
            row.get("context_field", ""),
            row.get("id", ""),
        )
    )
    return candidates[:max_blocks]


def flatten_results(results: list[dict[str, Any]], replay_index: int) -> list[dict[str, Any]]:
    return [dict(result, replay_index=replay_index) for result in results]


def summarize_prefill(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_total = sum(int(row["token_count"]) for row in rows)
    prefill_total = sum(int(row["prefill_tokens"]) for row in rows)
    reused_total = sum(int(row["reused_tokens"]) for row in rows)
    return {
        "rows": len(rows),
        "token_total": token_total,
        "prefill_total": prefill_total,
        "reused_total": reused_total,
        "cache_hit_count": sum(1 for row in rows if row["cache_hit"]),
        "cache_hit_ratio": reused_total / token_total if token_total else 0.0,
    }


def write_report(
    path: Path,
    *,
    selected_rows: list[dict[str, Any]],
    first_summary: dict[str, Any],
    second_summary: dict[str, Any],
    isolation: dict[str, Any],
    output_dir: Path,
) -> None:
    field_counts: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        field_counts[str(row.get("context_field", ""))] += 1

    lines = [
        "# Block-Isolated Sub-context Runtime Prototype",
        "",
        "## Purpose",
        "",
        "This prototype models the runtime contract needed before patching SGLang internals:",
        "",
        "- each field/function sub-context is an independent prefill unit,",
        "- each unit has its own cache key,",
        "- different sub-context blocks have zero allowed pairwise attention edges.",
        "",
        "## Selected Blocks",
        "",
        f"- Blocks: {len(selected_rows)}",
        f"- Context fields: {', '.join(f'`{key}`={value}' for key, value in sorted(field_counts.items()))}",
        "",
        "## Independent Prefill / Cache",
        "",
        "| replay | blocks | tokens | prefill tokens | reused tokens | cache hit ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| 1 | {first_summary['rows']} | {first_summary['token_total']} | "
            f"{first_summary['prefill_total']} | {first_summary['reused_total']} | "
            f"{first_summary['cache_hit_ratio']:.1%} |"
        ),
        (
            f"| 2 | {second_summary['rows']} | {second_summary['token_total']} | "
            f"{second_summary['prefill_total']} | {second_summary['reused_total']} | "
            f"{second_summary['cache_hit_ratio']:.1%} |"
        ),
        "",
        "## Block-Isolated Attention",
        "",
        f"- Total tokens: {isolation['token_count']}",
        f"- Intra-block allowed edges: {isolation['intra_block_allowed_edges']}",
        f"- Cross-block allowed edges: {isolation['cross_block_allowed_edges']}",
        f"- Cross-block disallowed edges: {isolation['cross_block_disallowed_edges']}",
        f"- No pairwise attention check: {'PASS' if isolation['pairwise_attention_ok'] else 'FAIL'}",
        "",
        "## Output Files",
        "",
        f"- `{(output_dir / 'prefill_results.csv').as_posix()}`",
        f"- `{(output_dir / 'block_pair_attention_matrix.csv').as_posix()}`",
        f"- `{(output_dir / 'block_isolation_runtime_prototype.json').as_posix()}`",
        "",
        "## Caveat",
        "",
        "This is a runtime-facing prototype and validation harness. It proves the block contract and attention mask semantics, but it is not yet an SGLang kernel/scheduler patch.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/block_isolated_subcontext_runtime"))
    parser.add_argument("--task-id", default="04_Search_Retrieval_task_2_conflicting_handling")
    parser.add_argument("--max-blocks", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input_jsonl)
    selected_rows = select_rows(rows, args.task_id, args.max_blocks)
    blocks = build_blocks(selected_rows)

    cache = IndependentPrefillCache()
    first = [cache.prefill(block).to_dict() for block in blocks]
    second = [cache.prefill(block).to_dict() for block in blocks]
    isolation = block_isolation_report(blocks).to_dict()

    prefill_rows = [*flatten_results(first, 1), *flatten_results(second, 2)]
    write_csv(args.output_dir / "prefill_results.csv", prefill_rows)
    write_csv(args.output_dir / "block_pair_attention_matrix.csv", isolation["block_pair_matrix"])
    (args.output_dir / "block_isolation_runtime_prototype.json").write_text(
        json.dumps(
            {
                "input_jsonl": args.input_jsonl.as_posix(),
                "task_id": args.task_id,
                "selected_sub_context_ids": [row.get("id", "") for row in selected_rows],
                "blocks": [block.to_dict() for block in blocks],
                "prefill_replay_1": first,
                "prefill_replay_2": second,
                "isolation": isolation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "block_isolation_runtime_report.md",
        selected_rows=selected_rows,
        first_summary=summarize_prefill(first),
        second_summary=summarize_prefill(second),
        isolation=isolation,
        output_dir=args.output_dir,
    )

    print("Block-isolated sub-context runtime prototype")
    print("===========================================")
    print(f"Selected blocks: {len(blocks)}")
    print(f"Tokens: {isolation['token_count']}")
    print(f"Cross-block allowed edges: {isolation['cross_block_allowed_edges']}")
    print(f"Replay-2 cache hit ratio: {summarize_prefill(second)['cache_hit_ratio']:.1%}")
    print(f"Report: {args.output_dir / 'block_isolation_runtime_report.md'}")
    return 0 if isolation["cross_block_allowed_edges"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
