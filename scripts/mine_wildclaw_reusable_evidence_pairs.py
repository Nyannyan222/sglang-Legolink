#!/usr/bin/env python3
"""Mine safe and review-needed reusable evidence pairs from WildClawBench.

This expands the positive side of the semantic reuse benchmark beyond the
initial three-task pilot. Exact cross-task duplicates are treated as safe
positive reuse pairs. High-overlap non-identical pairs are exported for manual
or LLM judging; they are not marked safe until judged.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real_context_extractor import parse_task_markdown, read_source_file, workspace_path_for


@dataclass(frozen=True)
class Block:
    id: str
    task_id: str
    category: str
    task_name: str
    context_field: str
    functional_role: str
    source_path: str
    char_start: int
    char_end: int
    text: str
    content_hash: str
    normalized_hash: str
    canonical_key: str
    canonical_text: str
    shingles: frozenset[str]


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def compact_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalize_text(text))


def canonical_key(context_field: str, functional_role: str, normalized_digest: str) -> str:
    return f"wildclaw_block:{context_field}:{functional_role}:exact:{normalized_digest[:24]}"


def shingles(text: str, n: int = 7, max_items: int = 1200) -> frozenset[str]:
    compact = compact_text(text)
    if not compact:
        return frozenset()
    if len(compact) <= n:
        return frozenset({compact})
    items = {compact[index : index + n] for index in range(0, len(compact) - n + 1)}
    if len(items) <= max_items:
        return frozenset(items)
    stride = max(1, math.ceil(len(items) / max_items))
    return frozenset(sorted(items)[::stride][:max_items])


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return overlap / len(left | right) if overlap else 0.0


def preview(text: str, limit: int = 280) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def discover_task_files(tasks_root: Path, categories: list[str]) -> list[Path]:
    if categories:
        files: list[Path] = []
        for category in categories:
            files.extend(sorted((tasks_root / category).glob("*.md")))
        return [path for path in files if path.name != "task0_template.md"]
    return sorted(path for path in tasks_root.rglob("*.md") if path.name != "task0_template.md")


def split_segments(text: str, max_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces = [piece for piece in re.split(r"\n\s*\n+", normalized) if piece.strip()]
    raw: list[tuple[int, int, str]] = []
    cursor = 0
    for piece in pieces or [normalized]:
        start = normalized.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        cursor = end
        stripped = piece.strip()
        if len(stripped) <= max_chars:
            raw.append((start, end, stripped))
            continue
        window_start = 0
        while window_start < len(stripped):
            window_end = min(len(stripped), window_start + max_chars)
            chunk = stripped[window_start:window_end].strip()
            if chunk:
                raw.append((start + window_start, start + window_end, chunk))
            if window_end == len(stripped):
                break
            window_start = max(window_start + 1, window_end - overlap_chars)
    return raw


def make_block(
    task: dict[str, Any],
    context_field: str,
    functional_role: str,
    source_path: str,
    text: str,
    char_start: int,
    char_end: int,
    suffix: str,
) -> Block:
    normalized = normalize_text(text)
    digest = stable_hash(text)
    normalized_digest = stable_hash(normalized)
    task_id = task["task_id"]
    block_id = f"{task_id}__{context_field}__{stable_hash(source_path)[:10]}__{suffix}"
    return Block(
        id=block_id,
        task_id=task_id,
        category=task["category"],
        task_name=task.get("name", ""),
        context_field=context_field,
        functional_role=functional_role,
        source_path=source_path,
        char_start=char_start,
        char_end=char_end,
        text=text,
        content_hash=digest,
        normalized_hash=normalized_digest,
        canonical_key=canonical_key(context_field, functional_role, normalized_digest),
        canonical_text=normalized,
        shingles=shingles(text),
    )


def build_blocks(args: argparse.Namespace) -> tuple[list[Block], dict[str, Any]]:
    wildclaw_root = args.wildclaw_root.resolve()
    tasks_root = args.tasks_root.resolve() if args.tasks_root else wildclaw_root / "tasks"
    workspace_root = args.workspace_root.resolve() if args.workspace_root else wildclaw_root / "workspace"
    task_files = discover_task_files(tasks_root, args.category)
    if args.max_tasks:
        task_files = task_files[: args.max_tasks]

    blocks: list[Block] = []
    skipped_workspaces = 0
    included_files = 0
    skipped_files = 0

    for task_file in task_files:
        task = parse_task_markdown(task_file)
        prompt = task.get("prompt") or ""
        if len(normalize_text(prompt)) >= args.min_chars:
            blocks.append(
                make_block(
                    task,
                    "user_prompt",
                    "request_objective_and_constraints",
                    task_file.relative_to(wildclaw_root).as_posix() + "#Prompt",
                    prompt,
                    0,
                    len(prompt),
                    "prompt",
                )
            )

        skills = "\n".join(task.get("skills") or [])
        if len(normalize_text(skills)) >= args.min_chars:
            blocks.append(
                make_block(
                    task,
                    "tool_definitions",
                    "allowed_actions_and_interfaces",
                    task_file.relative_to(wildclaw_root).as_posix() + "#Skills",
                    skills,
                    0,
                    len(skills),
                    "skills",
                )
            )

        workspace_dir = workspace_path_for(workspace_root, task)
        if not workspace_dir.exists():
            skipped_workspaces += 1
            continue
        for path in sorted(workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            source = read_source_file(path, workspace_dir, args.max_file_bytes, args.max_chars_per_file)
            if not source.included_in_context or source.content is None:
                skipped_files += 1
                continue
            included_files += 1
            rel_path = f"{workspace_dir.relative_to(wildclaw_root).as_posix()}/{source.path}"
            content = source.content
            if len(normalize_text(content)) >= args.min_chars and len(content) <= args.max_whole_file_chars:
                blocks.append(
                    make_block(
                        task,
                        "workspace_file",
                        "workspace_document_or_evidence",
                        rel_path,
                        content,
                        0,
                        len(content),
                        "file",
                    )
                )
            for index, (start, end, segment) in enumerate(split_segments(content, args.max_segment_chars, args.segment_overlap_chars), start=1):
                if len(normalize_text(segment)) < args.min_chars:
                    continue
                blocks.append(
                    make_block(
                        task,
                        "workspace_segment",
                        "workspace_document_or_evidence",
                        f"{rel_path}::segment_{index:04d}",
                        segment,
                        start,
                        end,
                        f"segment_{index:04d}",
                    )
                )

    inspection = {
        "generated_at": now_iso(),
        "wildclaw_root": str(wildclaw_root),
        "tasks_scanned": len(task_files),
        "blocks": len(blocks),
        "included_workspace_files": included_files,
        "skipped_workspace_files": skipped_files,
        "skipped_workspaces": skipped_workspaces,
        "categories": args.category or "all",
        "block_field_counts": dict(Counter(block.context_field for block in blocks)),
    }
    return blocks, inspection


def block_to_ref(block: Block, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_id": block.id,
        f"{prefix}_task_id": block.task_id,
        f"{prefix}_category": block.category,
        f"{prefix}_context_field": block.context_field,
        f"{prefix}_functional_role": block.functional_role,
        f"{prefix}_source_path": block.source_path,
        f"{prefix}_chars": len(block.text),
        f"{prefix}_content_hash": block.content_hash,
        f"{prefix}_normalized_hash": block.normalized_hash,
        f"{prefix}_canonical_key": block.canonical_key,
        f"{prefix}_canonical_text": block.canonical_text,
        f"{prefix}_preview": preview(block.text),
    }


def pair_row(left: Block, right: Block, relation_type: str, label: str, decision_basis: str, score: float | None = None) -> dict[str, Any]:
    row = {
        "benchmark_label": label,
        "relation_type": relation_type,
        "safe_to_reuse": label == "positive",
        "same_answer_utility": label == "positive",
        "needs_judge": label != "positive",
        "similarity_score": "" if score is None else round(score, 4),
        "decision_basis": decision_basis,
        "pair_canonical_key": left.canonical_key if left.canonical_key == right.canonical_key else "",
        "pair_canonical_text": left.canonical_text if left.canonical_key == right.canonical_key else "",
        **block_to_ref(left, "left"),
        **block_to_ref(right, "right"),
    }
    return row


def mine_exact_pairs(blocks: list[Block], max_pairs_per_group: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Block]] = defaultdict(list)
    for block in blocks:
        grouped[block.normalized_hash].append(block)

    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        tasks = {block.task_id for block in group}
        if len(group) < 2 or len(tasks) < 2:
            continue
        emitted = 0
        for left, right in itertools.combinations(sorted(group, key=lambda item: item.id), 2):
            if left.task_id == right.task_id:
                continue
            rows.append(
                pair_row(
                    left,
                    right,
                    "exact_duplicate",
                    "positive",
                    "Cross-task normalized exact duplicate from WildClawBench task/workspace context.",
                    1.0,
                )
            )
            emitted += 1
            if emitted >= max_pairs_per_group:
                break
    return rows


def mine_near_pairs(blocks: list[Block], threshold: float, max_pairs: int) -> list[dict[str, Any]]:
    candidates = [
        block
        for block in blocks
        if block.context_field in {"user_prompt", "workspace_file", "workspace_segment"}
        and 40 <= len(block.text) <= 6000
    ]
    buckets: dict[tuple[str, str], list[Block]] = defaultdict(list)
    for block in candidates:
        buckets[(block.category, block.context_field)].append(block)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket in buckets.values():
        bucket = sorted(bucket, key=lambda item: len(item.text), reverse=True)[:500]
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                if left.task_id == right.task_id:
                    continue
                if left.normalized_hash == right.normalized_hash:
                    continue
                key = tuple(sorted([left.id, right.id]))
                if key in seen:
                    continue
                score = jaccard(left.shingles, right.shingles)
                if score < threshold:
                    continue
                seen.add(key)
                rows.append(
                    pair_row(
                        left,
                        right,
                        "near_duplicate_candidate",
                        "review_candidate",
                        "High lexical/semantic overlap, but not marked safe until manual or LLM same-answer-utility judging.",
                        score,
                    )
                )
    rows.sort(key=lambda row: row["similarity_score"], reverse=True)
    return rows[:max_pairs]


def read_negative_pool(path: Path, target_total: int) -> list[dict[str, Any]]:
    if not path.exists() or target_total <= 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        pool = [
            row
            for row in csv.DictReader(fh)
            if row.get("benchmark_label") == "negative" or row.get("safe_to_reuse") in {"False", "false", "0"}
        ]
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_relation[row.get("relation_type") or row.get("reuse_relation_type") or "negative"].append(row)
    for relation_rows in by_relation.values():
        relation_rows.sort(key=lambda item: float(item.get("embedding_cosine") or item.get("similarity_score") or 0), reverse=True)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_total:
        progressed = False
        for relation in sorted(by_relation):
            if by_relation[relation] and len(selected) < target_total:
                selected.append(by_relation[relation].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    field_rows = [[field, count] for field, count in sorted(summary["positive_field_counts"].items())]
    relation_rows = [[relation, count] for relation, count in sorted(summary["benchmark_relation_counts"].items())]
    report = f"""# WildClaw Reusable Evidence Pair Mining

- Generated: `{summary["generated_at"]}`
- Tasks scanned: {summary["tasks_scanned"]}
- Blocks built: {summary["blocks"]}
- Safe exact positive pairs: {summary["safe_exact_positive_pairs"]}
- Near-duplicate review candidates: {summary["near_duplicate_review_candidates"]}
- Benchmark rows: {summary["benchmark_rows"]}

## What Changed

This step expands positive mining from the small pilot artifacts to the local
WildClawBench task/workspace tree. Exact cross-task duplicate sub-contexts are
safe positive pairs. High-overlap non-identical pairs are kept as review
candidates rather than being treated as reusable.

## Positive Field Counts

{markdown_table(["context_field", "positive_pairs"], field_rows)}

## Benchmark Relation Counts

{markdown_table(["relation_type", "pairs"], relation_rows)}

## Interpretation

The current safe positives are still mostly reusable interface/context blocks,
especially repeated tool definitions. This is valid but not yet enough for the
final claim. The useful next step is to run LLM or manual judging over the
near-duplicate review candidates and promote only pairs with same-answer utility
into the positive set.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wildclaw-root", type=Path, default=Path("external/WildClawBench"))
    parser.add_argument("--tasks-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--category", action="append", default=[], help="WildClaw category to scan. Repeatable; defaults to all.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_reusable_evidence_mining"))
    parser.add_argument("--negative-pool-csv", type=Path, default=Path("benchmark_results/semantic_reuse_benchmark/semantic_reuse_negative_pool.csv"))
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-chars-per-file", type=int, default=80_000)
    parser.add_argument("--max-whole-file-chars", type=int, default=6_000)
    parser.add_argument("--max-segment-chars", type=int, default=1_500)
    parser.add_argument("--segment-overlap-chars", type=int, default=120)
    parser.add_argument("--near-threshold", type=float, default=0.82)
    parser.add_argument("--max-near-pairs", type=int, default=200)
    parser.add_argument("--max-pairs-per-exact-group", type=int, default=30)
    parser.add_argument("--negative-ratio", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blocks, inspection = build_blocks(args)
    exact_positive_pairs = mine_exact_pairs(blocks, args.max_pairs_per_exact_group)
    near_review_candidates = mine_near_pairs(blocks, args.near_threshold, args.max_near_pairs)
    negatives = read_negative_pool(args.negative_pool_csv, len(exact_positive_pairs) * args.negative_ratio)

    benchmark_rows = exact_positive_pairs + negatives
    for index, row in enumerate(benchmark_rows, start=1):
        row["expanded_benchmark_pair_id"] = f"expanded_reuse_pair_{index:04d}"

    summary = {
        **inspection,
        "safe_exact_positive_pairs": len(exact_positive_pairs),
        "near_duplicate_review_candidates": len(near_review_candidates),
        "negative_pairs_selected": len(negatives),
        "benchmark_rows": len(benchmark_rows),
        "positive_field_counts": dict(Counter(row["left_context_field"] for row in exact_positive_pairs)),
        "benchmark_relation_counts": dict(Counter(row.get("relation_type", "negative") for row in benchmark_rows)),
        "near_threshold": args.near_threshold,
        "outputs": {
            "safe_positive_pairs": str(args.output_dir / "wildclaw_safe_reuse_positive_pairs.csv"),
            "review_candidates": str(args.output_dir / "wildclaw_reuse_review_candidates.csv"),
            "benchmark": str(args.output_dir / "wildclaw_expanded_semantic_reuse_benchmark.csv"),
        },
    }

    write_csv(args.output_dir / "wildclaw_safe_reuse_positive_pairs.csv", exact_positive_pairs)
    write_csv(args.output_dir / "wildclaw_reuse_review_candidates.csv", near_review_candidates)
    write_csv(args.output_dir / "wildclaw_expanded_semantic_reuse_benchmark.csv", benchmark_rows)
    write_jsonl(args.output_dir / "wildclaw_expanded_semantic_reuse_benchmark.jsonl", benchmark_rows)
    write_csv(
        args.output_dir / "wildclaw_reuse_review_sheet.csv",
        [
            {
                "review_pair_id": f"review_candidate_{index:04d}",
                "relation_type": row["relation_type"],
                "similarity_score": row["similarity_score"],
                "manual_relation": "",
                "same_answer_utility": "",
                "safe_to_reuse": "",
                "manual_notes": "",
                "left_id": row["left_id"],
                "right_id": row["right_id"],
                "left_preview": row["left_preview"],
                "right_preview": row["right_preview"],
            }
            for index, row in enumerate(near_review_candidates, start=1)
        ],
    )
    (args.output_dir / "wildclaw_reusable_evidence_mining_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "wildclaw_reusable_evidence_mining_report.md", summary)

    print("WildClaw reusable evidence mining")
    print("=================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Tasks scanned: {summary['tasks_scanned']}")
    print(f"Blocks built: {summary['blocks']}")
    print(f"Safe exact positive pairs: {summary['safe_exact_positive_pairs']}")
    print(f"Near-duplicate review candidates: {summary['near_duplicate_review_candidates']}")
    print(f"Benchmark rows: {summary['benchmark_rows']}")


if __name__ == "__main__":
    main()
