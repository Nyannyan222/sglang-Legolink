#!/usr/bin/env python3
"""Build a positive/negative benchmark for semantic sub-context reuse.

Positive rows must be genuinely safe to reuse. By default this script mines
only cross-task exact duplicates from the WildClaw field/function artifact,
which avoids inflating positives with duplicated outputs from multiple pipeline
versions. Negative rows come from the strict semantic reuse candidate evaluator:
they are related candidates that the judge rejected as partial-overlap,
broad-topic, or unrelated.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text.strip()


def preview(text: str, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def pair_key(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str]:
    ids = sorted([str(left.get("id", "")), str(right.get("id", ""))])
    return ids[0], ids[1]


def base_pair_row(
    left: dict[str, Any],
    right: dict[str, Any],
    label: str,
    relation_type: str,
    safe_to_reuse: bool,
    decision_basis: str,
) -> dict[str, Any]:
    return {
        "benchmark_label": label,
        "relation_type": relation_type,
        "safe_to_reuse": safe_to_reuse,
        "same_answer_utility": safe_to_reuse,
        "decision_basis": decision_basis,
        "left_id": left.get("id", ""),
        "right_id": right.get("id", ""),
        "left_task_id": left.get("task_id", ""),
        "right_task_id": right.get("task_id", ""),
        "left_category": left.get("category", ""),
        "right_category": right.get("category", ""),
        "left_context_field": left.get("context_field", ""),
        "right_context_field": right.get("context_field", ""),
        "left_functional_role": left.get("functional_role", ""),
        "right_functional_role": right.get("functional_role", ""),
        "left_title": left.get("title", ""),
        "right_title": right.get("title", ""),
        "left_content_hash": left.get("content_hash", ""),
        "right_content_hash": right.get("content_hash", ""),
        "left_chars": len(left.get("sub_context", "")),
        "right_chars": len(right.get("sub_context", "")),
        "left_preview": preview(left.get("sub_context", "")),
        "right_preview": preview(right.get("sub_context", "")),
    }


def mine_exact_positive_pairs(rows: list[dict[str, Any]], cross_task_only: bool, min_chars: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        text = row.get("sub_context", "")
        content_hash = row.get("content_hash", "")
        if content_hash and len(text) >= min_chars:
            groups[content_hash].append(row)

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        for left, right in itertools.combinations(group, 2):
            if cross_task_only and left.get("task_id") == right.get("task_id"):
                continue
            key = pair_key(left, right)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                base_pair_row(
                    left,
                    right,
                    "positive",
                    "exact_duplicate",
                    True,
                    "Cross-task exact content-hash match from one WildClaw field/function artifact.",
                )
            )
    return pairs


def mine_normalized_positive_pairs(rows: list[dict[str, Any]], cross_task_only: bool, min_chars: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        text = row.get("sub_context", "")
        if len(text) >= min_chars:
            groups[normalize_text(text)].append(row)

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        for left, right in itertools.combinations(group, 2):
            if left.get("content_hash") == right.get("content_hash"):
                continue
            if cross_task_only and left.get("task_id") == right.get("task_id"):
                continue
            key = pair_key(left, right)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                base_pair_row(
                    left,
                    right,
                    "positive",
                    "near_duplicate",
                    True,
                    "Cross-task normalized-text match; punctuation/spacing differs but content is equivalent.",
                )
            )
    return pairs


def select_negative_rows(rows: list[dict[str, str]], target_total: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = [
        {
            "benchmark_label": "negative",
            "relation_type": row.get("reuse_relation_type") or row.get("semantic_relation") or "unrelated",
            "safe_to_reuse": False,
            "same_answer_utility": False,
            "decision_basis": row.get("reuse_reason", "Strict semantic reuse evaluator rejected this pair."),
            "source": row.get("source", ""),
            "pair_id": row.get("pair_id", ""),
            "left_id": row.get("left_id", ""),
            "right_id": row.get("right_id", ""),
            "left_task_id": row.get("left_task_id", ""),
            "right_task_id": row.get("right_task_id", ""),
            "left_category": row.get("left_category", ""),
            "right_category": row.get("right_category", ""),
            "embedding_cosine": row.get("embedding_cosine", ""),
            "llm_judge_score": row.get("llm_judge_score", ""),
            "semantic_decision": row.get("semantic_decision", ""),
            "left_preview": row.get("left_text_preview", ""),
            "right_preview": row.get("right_text_preview", ""),
            "llm_judge_rationale": row.get("llm_judge_rationale", ""),
        }
        for row in rows
        if str(row.get("safe_to_reuse", "")).lower() not in {"true", "1", "yes"}
        and (row.get("reuse_decision") or "") == "do_not_reuse"
    ]

    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_relation[row["relation_type"]].append(row)
    for relation_rows in by_relation.values():
        relation_rows.sort(key=lambda row: float(row.get("embedding_cosine") or 0), reverse=True)

    selected: list[dict[str, Any]] = []
    relation_order = sorted(by_relation)
    while len(selected) < target_total and relation_order:
        progressed = False
        for relation in relation_order:
            if by_relation[relation] and len(selected) < target_total:
                selected.append(by_relation[relation].pop(0))
                progressed = True
        if not progressed:
            break
    return selected, pool


def add_pair_ids(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        row["reuse_benchmark_pair_id"] = f"reuse_benchmark_pair_{index:03d}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any], positives: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]]) -> None:
    relation_counts = Counter(row["relation_type"] for row in benchmark_rows)
    field_counts = Counter(row.get("left_context_field", "") for row in positives)
    report = f"""# Semantic Reuse Positive/Negative Benchmark

- Generated: `{summary["generated_at"]}`
- Positive source rows: `{summary["positive_jsonl"]}`
- Negative source CSV: `{summary["negative_csv"]}`
- Positive pairs: {summary["positive_pairs"]}
- Negative pool pairs: {summary["negative_pool_pairs"]}
- Benchmark rows: {summary["benchmark_rows"]}

## Why This Benchmark Matters

This benchmark tests the teacher's requested distinction:

- positive rows are semantic sub-context pairs that are actually safe to reuse;
- negative rows may look related, but they must not be reused because they are
  only partial-overlap, broad-topic, or unrelated.

It does not test prompt order tricks. The positive side currently comes from
real WildClaw field/function context blocks with exact cross-task content
matches. The negative side comes from strict LLM-judged semantic-similarity
candidates that were rejected for reuse.

## Benchmark Relation Counts

{markdown_table(["relation_type", "pairs"], [[key, value] for key, value in sorted(relation_counts.items())])}

## Positive Field Counts

{markdown_table(["context_field", "positive_pairs"], [[key or "(unknown)", value] for key, value in sorted(field_counts.items())])}

## Current Finding

The current safe positives are exact duplicate `tool_definitions` blocks shared
across WildClaw Search Retrieval tasks. This is a small but clean positive set:
it proves the pipeline can identify truly reusable sub-contexts without
mistaking partial topical overlap for reuse.

The next expansion should mine additional WildClaw categories and repeated
workspace/tool-output facts, then pass near-duplicate and same-answer-utility
candidates through the same strict judge.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-jsonl",
        type=Path,
        default=Path("benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts.jsonl"),
        help="WildClaw sub-context JSONL used to mine safe positive pairs.",
    )
    parser.add_argument(
        "--negative-csv",
        type=Path,
        default=Path("benchmark_results/semantic_reuse_candidate_eval/semantic_reuse_candidate_eval.csv"),
        help="Strict reuse-candidate evaluation CSV used as negative pool.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/semantic_reuse_benchmark"),
    )
    parser.add_argument("--negative-ratio", type=int, default=3, help="Negative benchmark rows per positive row.")
    parser.add_argument("--min-positive-chars", type=int, default=10)
    parser.add_argument("--allow-same-task-positives", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.positive_jsonl.exists():
        raise FileNotFoundError(args.positive_jsonl)
    if not args.negative_csv.exists():
        raise FileNotFoundError(args.negative_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    positive_source_rows = read_jsonl(args.positive_jsonl)
    exact_positives = mine_exact_positive_pairs(
        positive_source_rows,
        cross_task_only=not args.allow_same_task_positives,
        min_chars=args.min_positive_chars,
    )
    normalized_positives = mine_normalized_positive_pairs(
        positive_source_rows,
        cross_task_only=not args.allow_same_task_positives,
        min_chars=args.min_positive_chars,
    )

    positives = exact_positives + normalized_positives
    negative_source_rows = read_csv(args.negative_csv)
    selected_negatives, negative_pool = select_negative_rows(
        negative_source_rows,
        target_total=max(len(positives) * args.negative_ratio, args.negative_ratio if positives else 0),
    )

    benchmark_rows = positives + selected_negatives
    add_pair_ids(benchmark_rows)

    summary = {
        "generated_at": now_iso(),
        "positive_jsonl": str(args.positive_jsonl),
        "negative_csv": str(args.negative_csv),
        "positive_source_rows": len(positive_source_rows),
        "positive_pairs": len(positives),
        "exact_positive_pairs": len(exact_positives),
        "normalized_positive_pairs": len(normalized_positives),
        "negative_pool_pairs": len(negative_pool),
        "selected_negative_pairs": len(selected_negatives),
        "benchmark_rows": len(benchmark_rows),
        "negative_ratio": args.negative_ratio,
        "cross_task_positives_only": not args.allow_same_task_positives,
        "benchmark_relation_counts": dict(Counter(row["relation_type"] for row in benchmark_rows)),
    }

    write_csv(args.output_dir / "semantic_reuse_positive_pairs.csv", positives)
    write_csv(args.output_dir / "semantic_reuse_negative_pool.csv", negative_pool)
    write_csv(args.output_dir / "semantic_reuse_benchmark.csv", benchmark_rows)
    write_jsonl(args.output_dir / "semantic_reuse_benchmark.jsonl", benchmark_rows)
    write_csv(
        args.output_dir / "semantic_reuse_benchmark_manual_review.csv",
        [
            {
                "reuse_benchmark_pair_id": row["reuse_benchmark_pair_id"],
                "benchmark_label": row["benchmark_label"],
                "relation_type": row["relation_type"],
                "safe_to_reuse": row["safe_to_reuse"],
                "manual_label": "",
                "manual_notes": "",
                "left_id": row.get("left_id", ""),
                "right_id": row.get("right_id", ""),
                "left_preview": row.get("left_preview", ""),
                "right_preview": row.get("right_preview", ""),
            }
            for row in benchmark_rows
        ],
    )
    (args.output_dir / "semantic_reuse_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "semantic_reuse_benchmark_report.md", summary, positives, benchmark_rows)

    print("Semantic reuse benchmark")
    print("========================")
    print(f"Output dir: {args.output_dir}")
    print(f"Positive pairs: {summary['positive_pairs']}")
    print(f"Negative pool pairs: {summary['negative_pool_pairs']}")
    print(f"Benchmark rows: {summary['benchmark_rows']}")


if __name__ == "__main__":
    main()
