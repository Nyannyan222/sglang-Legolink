#!/usr/bin/env python3
"""Prepare an SGLang-facing runtime plan for block-isolated sub-contexts.

This does not modify SGLang. It produces the concrete metadata and request
payloads that an SGLang scheduler / KV-cache patch should consume:

- one prefill request per sub-context block,
- `extra_key == cache_key` for block-level cache lookup,
- block attention ranges,
- a scheduler/KV-cache contract,
- validation that cross-block attention is disallowed.
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

from noc_context_manager.runtime_isolation import build_blocks
from noc_context_manager.sglang_bridge import (
    build_sglang_block_runtime_plan,
    validate_sglang_block_runtime_plan,
)


DEFAULT_INPUT = Path("benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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
    selected = [row for row in rows if not task_id or row.get("task_id") == task_id]
    selected.sort(
        key=lambda row: (
            int((row.get("selection_metadata") or {}).get("field_order", 99)),
            row.get("context_field", ""),
            row.get("id", ""),
        )
    )
    return selected[:max_blocks] if max_blocks else selected


def write_contract_markdown(path: Path, plan: dict[str, Any], validation_errors: list[str]) -> None:
    isolation = plan["isolation_report"]
    contract = plan["scheduler_contract"]
    lines = [
        "# SGLang Block-Isolated Sub-context Runtime Plan",
        "",
        "## What This Adds",
        "",
        "This artifact maps WildClaw field/function sub-contexts into SGLang-facing",
        "runtime inputs. Each sub-context becomes an independent prefill request",
        "with its own `prefill_unit_id`, `cache_key`, and `extra_key`.",
        "",
        "## Runtime Summary",
        "",
        f"- Request id: `{plan['request_id']}`",
        f"- Task id: `{plan['task_id']}`",
        f"- Model: `{plan['model']}`",
        f"- Blocks: {len(plan['prefill_requests'])}",
        f"- Tokens: {isolation['token_count']}",
        f"- Cross-block allowed attention edges: {isolation['cross_block_allowed_edges']}",
        f"- No pairwise attention: {'PASS' if isolation['pairwise_attention_ok'] else 'FAIL'}",
        f"- Validation errors: {len(validation_errors)}",
        "",
        "## Scheduler Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in contract["scheduler_requirements"])
    lines.extend(
        [
            "",
            "## KV Cache Requirements",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in contract["kv_cache_requirements"])
    lines.extend(
        [
            "",
            "## Attention Mask Contract",
            "",
            f"- Same block: `{contract['attention_mask_contract']['same_block']}`",
            f"- Different block: `{contract['attention_mask_contract']['different_block']}`",
            f"- Validation metric: `{contract['attention_mask_contract']['validation_metric']}`",
            "",
            "## Implementation Boundary",
            "",
            "This is the concrete SGLang integration contract and request metadata",
            "producer. It is still not the final SGLang scheduler/KV-cache patch.",
            "The next patch should consume `openclaw_sub_context` and `extra_key`",
            "inside SGLang request objects, then store/reuse KV by `cache_key`.",
        ]
    )
    if validation_errors:
        lines.extend(["", "## Validation Errors", ""])
        lines.extend(f"- {error}" for error in validation_errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/sglang_block_runtime_plan"))
    parser.add_argument("--task-id", default="04_Search_Retrieval_task_2_conflicting_handling")
    parser.add_argument("--request-id", default="openclaw_block_isolated_demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--payload-max-tokens", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input_jsonl)
    selected_rows = select_rows(rows, args.task_id, args.max_blocks)
    blocks = build_blocks(selected_rows)
    plan = build_sglang_block_runtime_plan(
        blocks,
        request_id=args.request_id,
        task_id=args.task_id,
        model=args.model,
    )
    errors = validate_sglang_block_runtime_plan(plan)
    plan_dict = plan.to_dict()

    (args.output_dir / "sglang_block_runtime_plan.json").write_text(
        json.dumps(plan_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(
        args.output_dir / "sglang_prefill_request_payloads.jsonl",
        plan.chat_completion_payloads(max_tokens=args.payload_max_tokens),
    )
    write_csv(
        args.output_dir / "sglang_attention_ranges.csv",
        [row for row in plan_dict["attention_ranges"]],
    )
    write_contract_markdown(
        args.output_dir / "sglang_scheduler_kv_contract.md",
        plan_dict,
        errors,
    )

    print("SGLang block runtime plan")
    print("=========================")
    print(f"Blocks: {len(plan.prefill_requests)}")
    print(f"Tokens: {plan.isolation_report.token_count}")
    print(f"Cross-block allowed edges: {plan.isolation_report.cross_block_allowed_edges}")
    print(f"Validation errors: {len(errors)}")
    print(f"Output: {args.output_dir}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
