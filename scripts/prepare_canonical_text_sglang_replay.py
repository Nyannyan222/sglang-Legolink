#!/usr/bin/env python3
"""Prepare an SGLang replay manifest from canonical semantic sub-context text.

The manifest is intentionally small and ordered. It contains raw parameterized
blocks, their shared canonical text, and exact reusable canonical blocks. This
lets SGLang replay measure ordinary prefix-cache behavior on canonicalized
semantic sub-contexts without claiming runtime-level semantic KV reuse.
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
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_")[:120] or "prompt"


def prompt_text(title: str, text: str) -> str:
    return "\n".join(
        [
            "You are replaying a WildClaw semantic sub-context for cache measurement.",
            f"Sub-context type: {title}",
            "",
            text.strip(),
            "",
            "Acknowledge in one short sentence.",
        ]
    )


def load_subcontexts(path: Path) -> dict[str, dict[str, str]]:
    return {row["id"]: row for row in read_csv(path)}


def write_prompt(prompts_dir: Path, prompt_id: str, text: str) -> Path:
    path = prompts_dir / f"{safe_name(prompt_id)}.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def add_manifest_row(
    rows: list[dict[str, Any]],
    prompts_dir: Path,
    prompt_id: str,
    condition: str,
    task_id: str,
    text: str,
    canonical_key: str,
    canonical_text: str,
    source_pair_id: str,
) -> None:
    prompt_path = write_prompt(prompts_dir, prompt_id, text)
    rows.append(
        {
            "id": prompt_id,
            "task_id": task_id,
            "condition": condition,
            "prompt_path": prompt_path.as_posix(),
            "canonical_key": canonical_key,
            "canonical_text": canonical_text,
            "source_pair_id": source_pair_id,
        }
    )


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    subcontexts = load_subcontexts(args.prompt_internal_subcontexts)
    review_pairs = read_csv(args.prompt_internal_review_candidates)[: args.max_review_pairs]
    safe_pairs = read_csv(args.prompt_internal_safe_pairs)[: args.max_safe_pairs]
    block_safe_pairs = read_csv(args.block_safe_pairs)[: args.max_block_pairs]

    rows: list[dict[str, Any]] = []

    for index, pair in enumerate(review_pairs, start=1):
        pair_id = pair.get("prompt_internal_pair_id") or f"parameterized_pair_{index:03d}"
        left = subcontexts.get(pair["left_id"], {})
        right = subcontexts.get(pair["right_id"], {})
        canonical_key = pair.get("pair_canonical_key", "")
        canonical_text = pair.get("pair_canonical_text", "")
        if left:
            add_manifest_row(
                rows,
                prompts_dir,
                f"{pair_id}::raw_left",
                "raw_parameterized_left",
                left.get("task_id", ""),
                prompt_text("raw parameterized left", left.get("sub_context", "")),
                canonical_key,
                canonical_text,
                pair_id,
            )
        if right:
            add_manifest_row(
                rows,
                prompts_dir,
                f"{pair_id}::raw_right",
                "raw_parameterized_right",
                right.get("task_id", ""),
                prompt_text("raw parameterized right", right.get("sub_context", "")),
                canonical_key,
                canonical_text,
                pair_id,
            )
        if canonical_text:
            canonical_prompt = prompt_text("canonical template text", canonical_text)
            add_manifest_row(
                rows,
                prompts_dir,
                f"{pair_id}::canonical_text",
                "canonical_text",
                pair.get("left_task_id", ""),
                canonical_prompt,
                canonical_key,
                canonical_text,
                pair_id,
            )
            add_manifest_row(
                rows,
                prompts_dir,
                f"{pair_id}::canonical_text_repeat",
                "canonical_text_repeat",
                pair.get("right_task_id", ""),
                canonical_prompt,
                canonical_key,
                canonical_text,
                pair_id,
            )

    seen_exact: set[str] = set()
    for index, pair in enumerate(safe_pairs, start=1):
        canonical_key = pair.get("pair_canonical_key", "")
        canonical_text = pair.get("pair_canonical_text", "")
        if not canonical_key or not canonical_text or canonical_key in seen_exact:
            continue
        seen_exact.add(canonical_key)
        pair_id = pair.get("prompt_internal_pair_id") or f"exact_prompt_pair_{index:03d}"
        text = prompt_text("exact canonical prompt-internal text", canonical_text)
        add_manifest_row(
            rows,
            prompts_dir,
            f"{pair_id}::exact_canonical",
            "exact_canonical_text",
            pair.get("left_task_id", ""),
            text,
            canonical_key,
            canonical_text,
            pair_id,
        )
        add_manifest_row(
            rows,
            prompts_dir,
            f"{pair_id}::exact_canonical_repeat",
            "exact_canonical_text_repeat",
            pair.get("right_task_id", ""),
            text,
            canonical_key,
            canonical_text,
            pair_id,
        )

    seen_block: set[str] = set()
    for index, pair in enumerate(block_safe_pairs, start=1):
        canonical_key = pair.get("pair_canonical_key", "")
        canonical_text = pair.get("pair_canonical_text", "")
        if not canonical_key or not canonical_text or canonical_key in seen_block:
            continue
        seen_block.add(canonical_key)
        pair_id = f"block_exact_pair_{index:03d}"
        text = prompt_text("exact canonical block text", canonical_text)
        add_manifest_row(
            rows,
            prompts_dir,
            f"{pair_id}::block_canonical",
            "block_canonical_text",
            pair.get("left_task_id", ""),
            text,
            canonical_key,
            canonical_text,
            pair_id,
        )
        add_manifest_row(
            rows,
            prompts_dir,
            f"{pair_id}::block_canonical_repeat",
            "block_canonical_text_repeat",
            pair.get("right_task_id", ""),
            text,
            canonical_key,
            canonical_text,
            pair_id,
        )

    summary = {
        "created_at": now_iso(),
        "rows": len(rows),
        "review_pairs_used": len(review_pairs),
        "safe_pairs_read": len(safe_pairs),
        "block_safe_pairs_read": len(block_safe_pairs),
        "conditions": sorted({row["condition"] for row in rows}),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-internal-subcontexts",
        type=Path,
        default=Path("benchmark_results/wildclaw_prompt_internal_segmentation/wildclaw_prompt_internal_subcontexts.csv"),
    )
    parser.add_argument(
        "--prompt-internal-review-candidates",
        type=Path,
        default=Path("benchmark_results/wildclaw_prompt_internal_segmentation/wildclaw_prompt_internal_review_candidates.csv"),
    )
    parser.add_argument(
        "--prompt-internal-safe-pairs",
        type=Path,
        default=Path("benchmark_results/wildclaw_prompt_internal_segmentation/wildclaw_prompt_internal_safe_pairs.csv"),
    )
    parser.add_argument(
        "--block-safe-pairs",
        type=Path,
        default=Path("benchmark_results/wildclaw_reusable_evidence_mining/wildclaw_safe_reuse_positive_pairs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/canonical_text_sglang_replay"),
    )
    parser.add_argument("--max-review-pairs", type=int, default=3)
    parser.add_argument("--max-safe-pairs", type=int, default=3)
    parser.add_argument("--max-block-pairs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_manifest(args)
    manifest_path = args.output_dir / "canonical_text_sglang_runtime_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    (args.output_dir / "canonical_text_sglang_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Canonical text SGLang replay manifest")
    print("====================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
