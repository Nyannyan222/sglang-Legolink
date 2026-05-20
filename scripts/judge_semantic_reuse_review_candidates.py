#!/usr/bin/env python3
"""Judge review candidates and promote only safe semantic reuse pairs.

This script is intentionally conservative. Review candidates discovered by
lexical/embedding overlap are not reusable until a judge says they have the
same answer utility. When no manual labels are provided, the built-in fallback
marks cross-task user prompts as partial-overlap/task-template matches rather
than safe reusable evidence.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_bool(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def load_manual_labels(path: Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    labels = {}
    for row in read_csv(path):
        key = row.get("review_pair_id") or row.get("pair_id") or row.get("left_id", "") + "__" + row.get("right_id", "")
        labels[key] = row
    return labels


def candidate_key(row: dict[str, str], index: int) -> str:
    return row.get("review_pair_id") or f"review_candidate_{index:04d}"


def fallback_judge(row: dict[str, str]) -> dict[str, Any]:
    left_field = row.get("left_context_field", "")
    right_field = row.get("right_context_field", "")
    left_task = row.get("left_task_id", "")
    right_task = row.get("right_task_id", "")
    left_preview = row.get("left_preview", "")
    right_preview = row.get("right_preview", "")

    if row.get("left_normalized_hash") and row.get("left_normalized_hash") == row.get("right_normalized_hash"):
        return {
            "judge_relation": "exact_duplicate",
            "same_answer_utility": True,
            "safe_to_reuse": True,
            "judge_notes": "Exact normalized duplicate.",
            "judge_source": "fallback_exact_hash",
        }

    if left_field == right_field == "user_prompt" and left_task != right_task:
        if "Question:" in left_preview and "Question:" in right_preview:
            notes = (
                "The prompts share an answer-format/template pattern, but ask about different target facts; "
                "the whole sub-context cannot replace the other."
            )
        elif "/tmp_workspace/input/" in left_preview and "/tmp_workspace/input/" in right_preview:
            notes = (
                "The prompts share a puzzle-task template, but task-specific dimensions, piece counts, "
                "and constraints differ; the whole sub-context is not a safe replacement."
            )
        else:
            notes = (
                "Cross-task prompts share some instruction structure, but task-specific objectives differ; "
                "requires finer segmentation before any reusable part can be claimed."
            )
        return {
            "judge_relation": "partial_overlap",
            "same_answer_utility": False,
            "safe_to_reuse": False,
            "judge_notes": notes,
            "judge_source": "fallback_conservative_rule",
        }

    return {
        "judge_relation": "needs_manual_judge",
        "same_answer_utility": False,
        "safe_to_reuse": False,
        "judge_notes": "No explicit manual label and no safe fallback rule matched.",
        "judge_source": "fallback_unresolved",
    }


def apply_judgment(row: dict[str, str], index: int, manual_labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    key = candidate_key(row, index)
    label = manual_labels.get(key) or manual_labels.get(row.get("left_id", "") + "__" + row.get("right_id", ""))
    if label:
        same_answer = parse_bool(label.get("same_answer_utility"))
        safe = parse_bool(label.get("safe_to_reuse"))
        relation = label.get("manual_relation") or label.get("relation_type") or row.get("relation_type", "")
        return {
            **row,
            "review_pair_id": key,
            "judge_relation": relation,
            "same_answer_utility": bool(same_answer),
            "safe_to_reuse": bool(safe and same_answer),
            "judge_notes": label.get("manual_notes", ""),
            "judge_source": "manual_label",
        }

    judgment = fallback_judge(row)
    return {
        **row,
        "review_pair_id": key,
        **judgment,
    }


def normalize_existing_positive(row: dict[str, str]) -> dict[str, Any]:
    output = dict(row)
    output["benchmark_label"] = "positive"
    output["safe_to_reuse"] = True
    output["same_answer_utility"] = True
    output["judge_relation"] = row.get("relation_type", "exact_duplicate")
    output["judge_source"] = row.get("judge_source", "existing_safe_positive")
    return output


def select_negatives(rows: list[dict[str, str]], target_total: int) -> list[dict[str, Any]]:
    if target_total <= 0:
        return []
    by_relation: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        relation = row.get("relation_type") or row.get("reuse_relation_type") or "negative"
        by_relation[relation].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_total:
        progressed = False
        for relation in sorted(by_relation):
            if by_relation[relation] and len(selected) < target_total:
                row = dict(by_relation[relation].pop(0))
                row["benchmark_label"] = "negative"
                row["safe_to_reuse"] = False
                row["same_answer_utility"] = False
                selected.append(row)
                progressed = True
        if not progressed:
            break
    return selected


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    relation_rows = [[key, value] for key, value in sorted(summary["judged_relation_counts"].items())]
    report = f"""# Semantic Reuse Review Candidate Judging

- Generated: `{summary["generated_at"]}`
- Review candidates: {summary["review_candidates"]}
- Promoted positives: {summary["promoted_positive_pairs"]}
- Rejected candidates: {summary["rejected_candidates"]}
- Existing safe positives: {summary["existing_safe_positive_pairs"]}
- Promoted benchmark rows: {summary["promoted_benchmark_rows"]}

## Judgment Counts

{markdown_table(["judge_relation", "rows"], relation_rows)}

## Interpretation

The current near-duplicate candidates are not promoted. They share task
templates, output requirements, or problem structure, but their task-specific
facts differ. Under the semantic reuse policy, that is `partial_overlap`, not
`same_answer_utility`.

This is a useful negative result: it shows the pipeline can find candidates
without over-claiming reuse. The next technical improvement is finer
field-internal segmentation so repeated instruction templates and task-specific
parameters can be separated before judging.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-candidates",
        type=Path,
        default=Path("benchmark_results/wildclaw_reusable_evidence_mining/wildclaw_reuse_review_candidates.csv"),
    )
    parser.add_argument("--manual-labels", type=Path, default=None)
    parser.add_argument(
        "--existing-positive-pairs",
        type=Path,
        default=Path("benchmark_results/wildclaw_reusable_evidence_mining/wildclaw_safe_reuse_positive_pairs.csv"),
    )
    parser.add_argument(
        "--negative-pool",
        type=Path,
        default=Path("benchmark_results/wildclaw_reusable_evidence_mining/wildclaw_expanded_semantic_reuse_benchmark.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/semantic_reuse_candidate_judging"),
    )
    parser.add_argument("--negative-ratio", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_csv(args.review_candidates)
    manual_labels = load_manual_labels(args.manual_labels)
    judged = [apply_judgment(row, index, manual_labels) for index, row in enumerate(candidates, start=1)]
    promoted = [row for row in judged if row.get("safe_to_reuse") is True and row.get("same_answer_utility") is True]
    rejected = [row for row in judged if row not in promoted]

    existing_positive = [normalize_existing_positive(row) for row in read_csv(args.existing_positive_pairs)]
    negative_pool = [
        row
        for row in read_csv(args.negative_pool)
        if str(row.get("safe_to_reuse", "")).lower() in {"false", "0", "no"}
        or row.get("benchmark_label") == "negative"
    ]
    negatives = select_negatives(negative_pool, (len(existing_positive) + len(promoted)) * args.negative_ratio)
    promoted_benchmark = existing_positive + promoted + negatives
    for index, row in enumerate(promoted_benchmark, start=1):
        row["promoted_benchmark_pair_id"] = f"promoted_reuse_pair_{index:04d}"

    summary = {
        "generated_at": now_iso(),
        "review_candidates": len(candidates),
        "promoted_positive_pairs": len(promoted),
        "rejected_candidates": len(rejected),
        "existing_safe_positive_pairs": len(existing_positive),
        "selected_negative_pairs": len(negatives),
        "promoted_benchmark_rows": len(promoted_benchmark),
        "judged_relation_counts": dict(Counter(row.get("judge_relation", "") for row in judged)),
        "outputs": {
            "judged_candidates": str(args.output_dir / "semantic_reuse_judged_candidates.csv"),
            "promoted_positive_pairs": str(args.output_dir / "semantic_reuse_promoted_positive_pairs.csv"),
            "rejected_candidates": str(args.output_dir / "semantic_reuse_rejected_candidates.csv"),
            "promoted_benchmark": str(args.output_dir / "semantic_reuse_promoted_benchmark.csv"),
        },
    }

    write_csv(args.output_dir / "semantic_reuse_judged_candidates.csv", judged)
    write_csv(args.output_dir / "semantic_reuse_promoted_positive_pairs.csv", promoted)
    write_csv(args.output_dir / "semantic_reuse_rejected_candidates.csv", rejected)
    write_csv(args.output_dir / "semantic_reuse_promoted_benchmark.csv", promoted_benchmark)
    write_jsonl(args.output_dir / "semantic_reuse_promoted_benchmark.jsonl", promoted_benchmark)
    (args.output_dir / "semantic_reuse_candidate_judging_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "semantic_reuse_candidate_judging_report.md", summary)

    print("Semantic reuse candidate judging")
    print("================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Review candidates: {summary['review_candidates']}")
    print(f"Promoted positives: {summary['promoted_positive_pairs']}")
    print(f"Rejected candidates: {summary['rejected_candidates']}")
    print(f"Promoted benchmark rows: {summary['promoted_benchmark_rows']}")


if __name__ == "__main__":
    main()
