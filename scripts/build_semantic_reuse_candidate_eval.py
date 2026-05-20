#!/usr/bin/env python3
"""Evaluate whether semantic-similarity candidates are safe to reuse.

The semantic-similarity discovery stage intentionally casts a wide net:
embeddings and lexical overlap find related sub-context pairs. This script is
the next filter. It separates "related/relevant" from "safe to reuse" by using
the relation type, same-answer utility judgment, judge score, and exact content
hash matches.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REUSE_RELATIONS = {
    "exact_duplicate",
    "near_duplicate",
    "near_equivalent",
    "same_answer_utility",
    "high_similarity",
}

REJECT_RELATIONS = {
    "partial_overlap",
    "broad_topic",
    "unrelated",
    "parse_error",
}


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_relation(row: dict[str, str]) -> str:
    relation = (row.get("semantic_relation") or "").strip().lower()
    if row.get("left_content_hash") and row.get("left_content_hash") == row.get("right_content_hash"):
        return "exact_duplicate"
    if relation == "near_equivalent":
        return "near_duplicate"
    if relation:
        return relation
    return "unrelated"


def classify_reuse(row: dict[str, str], judge_threshold: float, embedding_threshold: float) -> dict[str, Any]:
    relation = normalize_relation(row)
    judge_score = parse_float(row.get("llm_judge_score"))
    embedding_cosine = parse_float(row.get("embedding_cosine"))
    same_answer_utility = parse_bool(row.get("same_answer_utility"))
    semantic_decision = (row.get("semantic_decision") or "").strip().lower()
    exact_hash = relation == "exact_duplicate"

    if exact_hash:
        safe_to_reuse = True
        reuse_decision = "reuse_allowed"
        reason = "Exact normalized content hash match."
    elif relation in {"near_duplicate", "near_equivalent"} and same_answer_utility and judge_score >= judge_threshold:
        safe_to_reuse = True
        reuse_decision = "reuse_allowed"
        reason = "Judge marked the pair as near-equivalent with same-answer utility."
    elif relation in {"same_answer_utility", "high_similarity"} and same_answer_utility and judge_score >= judge_threshold:
        safe_to_reuse = embedding_cosine >= embedding_threshold or semantic_decision == "semantic_match"
        reuse_decision = "reuse_allowed" if safe_to_reuse else "judge_required"
        reason = (
            "Same-answer utility is present; embedding/decision threshold confirms reuse."
            if safe_to_reuse
            else "Same-answer utility is present, but embedding/decision threshold is not strong enough."
        )
    elif relation in REJECT_RELATIONS:
        safe_to_reuse = False
        reuse_decision = "do_not_reuse"
        reason = f"{relation} is related evidence, not a safe reusable sub-context."
    else:
        safe_to_reuse = False
        reuse_decision = "judge_required"
        reason = "Relation is not in the reusable taxonomy; keep for manual review."

    return {
        "reuse_relation_type": relation,
        "safe_to_reuse": safe_to_reuse,
        "reuse_decision": reuse_decision,
        "reuse_reason": reason,
    }


def evaluate_sources(pairs_csvs: list[Path], judge_threshold: float, embedding_threshold: float) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for path in pairs_csvs:
        source = path.parent.name
        for row in read_csv(path):
            classification = classify_reuse(row, judge_threshold, embedding_threshold)
            evaluated.append(
                {
                    "source": source,
                    "pair_id": row.get("pair_id") or f"{row.get('left_id','')}__{row.get('right_id','')}",
                    "left_id": row.get("left_id", ""),
                    "right_id": row.get("right_id", ""),
                    "left_task_id": row.get("left_task_id", ""),
                    "right_task_id": row.get("right_task_id", ""),
                    "left_category": row.get("left_category", ""),
                    "right_category": row.get("right_category", ""),
                    "embedding_cosine": parse_float(row.get("embedding_cosine")),
                    "lexical_score": parse_float(row.get("lexical_score")),
                    "llm_judge_score": parse_float(row.get("llm_judge_score")),
                    "semantic_relation": row.get("semantic_relation", ""),
                    "same_answer_utility": parse_bool(row.get("same_answer_utility")),
                    "semantic_decision": row.get("semantic_decision", ""),
                    **classification,
                    "left_text_preview": row.get("left_text_preview", ""),
                    "right_text_preview": row.get("right_text_preview", ""),
                    "llm_judge_rationale": row.get("llm_judge_rationale", ""),
                }
            )
    evaluated.sort(
        key=lambda item: (
            not item["safe_to_reuse"],
            item["reuse_decision"] != "judge_required",
            -item["embedding_cosine"],
            -item["llm_judge_score"],
        )
    )
    return evaluated


def summarize(rows: list[dict[str, Any]], inputs: list[Path], judge_threshold: float, embedding_threshold: float) -> dict[str, Any]:
    relation_counts = Counter(row["reuse_relation_type"] for row in rows)
    decision_counts = Counter(row["reuse_decision"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    safe_rows = [row for row in rows if row["safe_to_reuse"]]
    rejected_rows = [row for row in rows if row["reuse_decision"] == "do_not_reuse"]
    by_source: dict[str, dict[str, Any]] = {}
    for source, count in source_counts.items():
        source_rows = [row for row in rows if row["source"] == source]
        by_source[source] = {
            "pairs": count,
            "safe_to_reuse": sum(1 for row in source_rows if row["safe_to_reuse"]),
            "judge_required": sum(1 for row in source_rows if row["reuse_decision"] == "judge_required"),
            "do_not_reuse": sum(1 for row in source_rows if row["reuse_decision"] == "do_not_reuse"),
            "relation_counts": dict(Counter(row["reuse_relation_type"] for row in source_rows)),
        }
    return {
        "generated_at": now_iso(),
        "inputs": [str(path) for path in inputs],
        "judge_threshold": judge_threshold,
        "embedding_threshold": embedding_threshold,
        "total_pairs": len(rows),
        "safe_to_reuse_pairs": len(safe_rows),
        "judge_required_pairs": decision_counts.get("judge_required", 0),
        "do_not_reuse_pairs": len(rejected_rows),
        "relation_counts": dict(relation_counts),
        "decision_counts": dict(decision_counts),
        "by_source": by_source,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    relation_rows = [[key, value] for key, value in sorted(summary["relation_counts"].items())]
    decision_rows = [[key, value] for key, value in sorted(summary["decision_counts"].items())]
    source_rows = [
        [
            source,
            stats["pairs"],
            stats["safe_to_reuse"],
            stats["judge_required"],
            stats["do_not_reuse"],
        ]
        for source, stats in sorted(summary["by_source"].items())
    ]
    examples = rows[:10]
    example_rows = [
        [
            row["source"],
            row["reuse_relation_type"],
            row["reuse_decision"],
            f'{row["embedding_cosine"]:.3f}',
            f'{row["llm_judge_score"]:.1f}',
            row["same_answer_utility"],
            row["left_task_id"],
            row["right_task_id"],
        ]
        for row in examples
    ]
    verdict = (
        "No pair is currently safe to reuse under the strict semantic policy. "
        "This is a useful negative result: embedding similarity found related candidates, "
        "but the judge separated reusable evidence from partial overlap or broad topic matches."
        if summary["safe_to_reuse_pairs"] == 0
        else f'{summary["safe_to_reuse_pairs"]} pair(s) are safe to reuse under the strict semantic policy.'
    )
    report = f"""# Semantic Reuse Candidate Evaluation

- Generated: `{summary["generated_at"]}`
- Input CSVs: {len(summary["inputs"])}
- Total candidate pairs: {summary["total_pairs"]}
- Safe-to-reuse pairs: {summary["safe_to_reuse_pairs"]}
- Judge-required pairs: {summary["judge_required_pairs"]}
- Do-not-reuse pairs: {summary["do_not_reuse_pairs"]}
- Judge threshold: {summary["judge_threshold"]}
- Embedding threshold: {summary["embedding_threshold"]}

## Main Verdict

{verdict}

This evaluation is intentionally different from a prefix-order experiment. The
goal is to test whether two WildClaw semantic sub-contexts have equivalent
answer utility, not whether prompt ordering makes SGLang's native prefix cache
hit.

## Decision Counts

{markdown_table(["reuse_decision", "pairs"], decision_rows)}

## Relation Counts

{markdown_table(["relation_type", "pairs"], relation_rows)}

## Source Summary

{markdown_table(["source", "pairs", "safe_to_reuse", "judge_required", "do_not_reuse"], source_rows)}

## Top Review Rows

{markdown_table(["source", "relation", "decision", "embedding", "judge", "same_answer", "left_task", "right_task"], example_rows)}

## Interpretation

Embedding similarity is used only to find candidates. The reuse decision is made
after checking relation type and same-answer utility:

- `exact_duplicate`, `near_duplicate`, and `same_answer_utility` can be reused
  only when the judge supports equivalent answer utility.
- `partial_overlap`, `broad_topic`, and `unrelated` are rejected because they may
  be relevant but are not safe replacements.
- If no safe pair is found, the correct action is not to force reuse. The next
  step is to mine more WildClaw tasks/workspaces for repeated tool outputs,
  duplicated workspace evidence, and near-equivalent facts, while keeping this
  reuse filter strict.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-csv",
        action="append",
        type=Path,
        required=True,
        help="Input semantic_similar_subcontext_pairs.csv. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/semantic_reuse_candidate_eval"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument("--judge-threshold", type=float, default=3.0)
    parser.add_argument("--embedding-threshold", type=float, default=0.70)
    parser.add_argument("--manual-review-limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [path for path in args.pairs_csv if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input CSV(s): {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate_sources(args.pairs_csv, args.judge_threshold, args.embedding_threshold)
    summary = summarize(rows, args.pairs_csv, args.judge_threshold, args.embedding_threshold)

    write_csv(args.output_dir / "semantic_reuse_candidate_eval.csv", rows)
    write_csv(args.output_dir / "semantic_reuse_candidate_manual_review.csv", rows[: args.manual_review_limit])
    (args.output_dir / "semantic_reuse_candidate_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "semantic_reuse_candidate_eval_report.md", summary, rows)

    print("Semantic reuse candidate evaluation")
    print("===================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Pairs: {summary['total_pairs']}")
    print(f"Safe to reuse: {summary['safe_to_reuse_pairs']}")
    print(f"Judge required: {summary['judge_required_pairs']}")
    print(f"Do not reuse: {summary['do_not_reuse_pairs']}")


if __name__ == "__main__":
    main()
