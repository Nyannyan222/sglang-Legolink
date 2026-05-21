#!/usr/bin/env python3
"""Prepare a semantic-positive canonicalization benchmark for SGLang replay.

The selected rows are different raw texts that share the same canonical text.
The benchmark compares raw semantic equivalents, exact raw repeats, and
OpenClaw canonical text repeats under the same answer-intent validation prompt.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:140] or "prompt"


def expected_intent(canonical_text: str) -> str:
    lowered = canonical_text.lower()
    if "save the results" in lowered:
        return "save_results"
    if "number of searches" in lowered or "searches must not exceed" in lowered:
        return "search_budget_limit"
    return "generic_instruction"


def comparison_prompt(sub_context: str) -> str:
    return f"""You are validating whether a semantic sub-context preserves answer utility.

Return JSON only with this schema:
{{"intent":"<one label>","key_facts":["short phrases from the sub-context"],"adds_new_facts":false}}

Choose exactly one intent label:
- save_results: the sub-context instructs where/how to save results or output files.
- search_budget_limit: the sub-context limits the number of searches.
- generic_instruction: any other general instruction.

Rules:
- Do not copy the list of labels into the intent field.
- If the sub-context says "save", choose save_results.
- If the sub-context says "searches must not exceed", choose search_budget_limit.
- Use only facts present in the sub-context.

Classify the instruction intent using only the sub-context below.

Sub-context:
{sub_context.strip()}
"""


def write_prompt(prompts_dir: Path, row_id: str, sub_context: str) -> Path:
    path = prompts_dir / f"{safe_name(row_id)}.txt"
    path.write_text(comparison_prompt(sub_context), encoding="utf-8", newline="\n")
    return path


def add_row(
    rows: list[dict[str, Any]],
    prompts_dir: Path,
    source_pair_id: str,
    condition: str,
    task_id: str,
    sub_context: str,
    canonical_key: str,
    canonical_text: str,
    intent: str,
) -> None:
    row_id = f"{source_pair_id}::{condition}"
    prompt_path = write_prompt(prompts_dir, row_id, sub_context)
    rows.append(
        {
            "id": row_id,
            "task_id": task_id,
            "condition": condition,
            "prompt_path": prompt_path.as_posix(),
            "source_pair_id": source_pair_id,
            "canonical_key": canonical_key,
            "canonical_text": canonical_text,
            "expected_intent": intent,
        }
    )


def select_pairs(rows: list[dict[str, str]], max_pairs: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = row.get("pair_canonical_key", "")
        if not key or key in seen_keys:
            continue
        selected.append(row)
        seen_keys.add(key)
        if len(selected) >= max_pairs:
            break
    return selected


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    candidates = select_pairs(read_csv(args.review_candidates), args.max_pairs)
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(candidates, start=1):
        source_pair_id = pair.get("prompt_internal_pair_id") or f"semantic_positive_pair_{index:03d}"
        canonical_text = pair.get("pair_canonical_text", "")
        canonical_key = pair.get("pair_canonical_key", "")
        intent = expected_intent(canonical_text)
        left = pair.get("left_preview", "")
        right = pair.get("right_preview", "")

        add_row(rows, prompts_dir, source_pair_id, "original_raw_text_left", pair.get("left_task_id", ""), left, canonical_key, canonical_text, intent)
        add_row(rows, prompts_dir, source_pair_id, "original_raw_text_right", pair.get("right_task_id", ""), right, canonical_key, canonical_text, intent)
        add_row(rows, prompts_dir, source_pair_id, "sglang_exact_repeat", pair.get("left_task_id", ""), left, canonical_key, canonical_text, intent)
        add_row(rows, prompts_dir, source_pair_id, "sglang_exact_repeat_repeat", pair.get("left_task_id", ""), left, canonical_key, canonical_text, intent)
        add_row(rows, prompts_dir, source_pair_id, "openclaw_canonical_text", pair.get("left_task_id", ""), canonical_text, canonical_key, canonical_text, intent)
        add_row(rows, prompts_dir, source_pair_id, "openclaw_canonical_text_repeat", pair.get("right_task_id", ""), canonical_text, canonical_key, canonical_text, intent)

    summary = {
        "created_at": now_iso(),
        "source": args.review_candidates.as_posix(),
        "semantic_positive_pairs": len(candidates),
        "rows": len(rows),
        "conditions": sorted({row["condition"] for row in rows}),
        "expected_intents": sorted({row["expected_intent"] for row in rows}),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-candidates",
        type=Path,
        default=Path("benchmark_results/wildclaw_prompt_internal_segmentation/wildclaw_prompt_internal_review_candidates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/canonical_semantic_positive_eval"),
    )
    parser.add_argument("--max-pairs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build(args)
    manifest_path = args.output_dir / "canonical_semantic_positive_runtime_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    (args.output_dir / "canonical_semantic_positive_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Canonical semantic-positive eval manifest")
    print("========================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Pairs: {summary['semantic_positive_pairs']}")
    print(f"Rows: {summary['rows']}")


if __name__ == "__main__":
    main()
