#!/usr/bin/env python3
"""Split WildClaw prompts into finer field-internal semantic blocks.

Whole prompts can be near-duplicates while still unsafe to reuse, because they
mix reusable instruction templates with task-specific facts. This script splits
prompt fields into smaller blocks, mines exact reusable instruction positives,
and exports parameterized-template candidates for later judge/manual review.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from real_context_extractor import parse_task_markdown
from mine_wildclaw_reusable_evidence_pairs import discover_task_files


REUSABLE_HINTS = (
    "please provide",
    "provide the final answer",
    "evidence chain",
    "must explicitly state",
    "must include",
    "must not",
    "do not",
    "do not guess",
    "write your results",
    "write the",
    "output format",
    "strictly follow",
    "save",
    "保存",
    "写入",
    "格式如下",
    "要求",
    "必须",
    "不要",
    "请",
)

TASK_SPECIFIC_HINTS = (
    "question:",
    "from which python version",
    "in which python version",
    "目录下有",
    "这些碎片来自",
    "原图",
    "standard library",
    "cpthon github pull request",
)


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip().lower()


def canonical_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-")
    return text


def template_text(text: str) -> str:
    text = canonical_text(text)
    text = re.sub(r"`[^`]+`", "<CODE>", text)
    text = re.sub(r"/tmp_workspace/[^\s,，。)）]+", "<PATH>", text)
    text = re.sub(r"piece_\d+\.png", "piece_<N>.png", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", text)
    text = re.sub(r"\d+\s*[×x]\s*\d+", "<GRID>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_hash_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff<>]+", "", text)


def canonical_reuse_fields(block_type: str, functional_role: str, canonical: str, template: str, requires_delta: bool) -> dict[str, str]:
    canonical_kind = "template" if requires_delta else "exact"
    canonical_value = template if requires_delta else canonical
    digest = stable_hash(compact_hash_text(canonical_value))
    key = f"prompt_internal:{functional_role}:{canonical_kind}:{digest[:24]}"
    return {
        "canonical_key": key,
        "canonical_text": canonical_value,
        "canonical_kind": canonical_kind,
        "canonical_hash": digest,
        "exact_text_hash": stable_hash(compact_hash_text(canonical)),
        "template_hash": stable_hash(compact_hash_text(template)),
    }


def split_fenced_blocks(prompt: str) -> list[tuple[str, bool, int, int]]:
    parts: list[tuple[str, bool, int, int]] = []
    cursor = 0
    for match in re.finditer(r"```.*?```", prompt, flags=re.DOTALL):
        if match.start() > cursor:
            parts.append((prompt[cursor : match.start()], False, cursor, match.start()))
        parts.append((match.group(0), True, match.start(), match.end()))
        cursor = match.end()
    if cursor < len(prompt):
        parts.append((prompt[cursor:], False, cursor, len(prompt)))
    return parts


def split_plain_text(text: str, base_start: int, max_chars: int) -> list[tuple[str, int, int]]:
    pieces: list[tuple[str, int, int]] = []
    paragraphs = [part for part in re.split(r"\n\s*\n+", text) if part.strip()]
    cursor = 0
    for paragraph in paragraphs:
        start = text.find(paragraph, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(paragraph)
        paragraph = paragraph.strip()
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1:
            for line in lines:
                line_start = text.find(line, start)
                pieces.extend(split_long_clause(line, base_start + line_start, max_chars))
            continue
        pieces.extend(split_long_clause(paragraph, base_start + start, max_chars))
    return pieces


def split_long_clause(text: str, base_start: int, max_chars: int) -> list[tuple[str, int, int]]:
    text = text.strip()
    if len(text) <= max_chars:
        return [(text, base_start, base_start + len(text))]
    clauses = [part.strip() for part in re.split(r"(?<=[。.!?])\s+|(?<=；)\s*|(?<=;)\s+", text) if part.strip()]
    if len(clauses) == 1:
        clauses = [part.strip() for part in re.split(r"(?<=。)|(?<=；)|(?<=;)", text) if part.strip()]
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for clause in clauses:
        start = text.find(clause, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(clause)
        if len(clause) <= max_chars:
            out.append((clause, base_start + start, base_start + start + len(clause)))
        else:
            for index in range(0, len(clause), max_chars):
                chunk = clause[index : index + max_chars].strip()
                if chunk:
                    out.append((chunk, base_start + start + index, base_start + start + index + len(chunk)))
    return out


def classify_block(text: str, is_code: bool) -> tuple[str, str, bool]:
    lowered = canonical_text(text)
    stripped = text.strip().strip("*# ")
    if is_code:
        return "output_format_template", "structured_output_contract", True
    if len(stripped) < 30 and stripped.endswith((":", "：")):
        return "section_heading_or_short_label", "prompt_structure", False
    if any(hint in lowered for hint in TASK_SPECIFIC_HINTS):
        if any(hint in lowered for hint in REUSABLE_HINTS):
            return "mixed_instruction_with_parameters", "task_specific_instruction_mixed_with_reusable_template", False
        return "task_specific_parameter", "task_objective_or_parameter", False
    if any(hint in lowered for hint in REUSABLE_HINTS):
        if re.search(r"\b\d+\b|`[^`]+`|/tmp_workspace/", text):
            return "parameterized_instruction_template", "instruction_template_requires_delta", False
        return "reusable_instruction_template", "generic_instruction_or_constraint", True
    if len(text) < 80:
        return "section_heading_or_short_label", "prompt_structure", False
    return "task_specific_parameter", "task_objective_or_parameter", False


def prompt_blocks(task: dict[str, Any], task_file: Path, wildclaw_root: Path, max_chars: int, min_chars: int) -> list[dict[str, Any]]:
    prompt = task.get("prompt") or ""
    rows: list[dict[str, Any]] = []
    block_index = 0
    for part, is_code, start, end in split_fenced_blocks(prompt):
        if is_code:
            candidates = [(part.strip(), start, end)]
        else:
            candidates = split_plain_text(part, start, max_chars)
        for text, char_start, char_end in candidates:
            if len(normalize_text(text)) < min_chars:
                continue
            block_index += 1
            block_type, functional_role, exact_safe = classify_block(text, is_code)
            canonical = canonical_text(text)
            template = template_text(text)
            requires_delta = block_type in {"parameterized_instruction_template", "mixed_instruction_with_parameters"}
            canonical_fields = canonical_reuse_fields(block_type, functional_role, canonical, template, requires_delta)
            rows.append(
                {
                    "id": f"{task['task_id']}__prompt_internal__{block_index:04d}",
                    "source": "WildClawBench",
                    "method": "field_internal_prompt_segmentation",
                    "category": task["category"],
                    "task_id": task["task_id"],
                    "task_name": task.get("name", ""),
                    "context_field": "user_prompt",
                    "block_type": block_type,
                    "functional_role": functional_role,
                    "source_path": task_file.relative_to(wildclaw_root).as_posix() + "#Prompt",
                    "char_start": char_start,
                    "char_end": char_end,
                    "sub_context": text,
                    "sub_context_chars": len(text),
                    **canonical_fields,
                    "template_text": template,
                    "safe_exact_reuse_candidate": exact_safe,
                    "requires_parameter_delta": requires_delta,
                    "attention_policy_target": "independent_prompt_internal_subcontext",
                }
            )
    return rows


def mine_exact_safe_pairs(blocks: list[dict[str, Any]], max_pairs_per_group: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        if block["safe_exact_reuse_candidate"]:
            groups[block["canonical_key"]].append(block)
    pairs: list[dict[str, Any]] = []
    for group in groups.values():
        if len({row["task_id"] for row in group}) < 2:
            continue
        emitted = 0
        for left, right in itertools.combinations(sorted(group, key=lambda row: row["id"]), 2):
            if left["task_id"] == right["task_id"]:
                continue
            pairs.append(pair_row(left, right, "exact_instruction_template", "positive", True, "Exact reusable prompt-internal instruction block."))
            emitted += 1
            if emitted >= max_pairs_per_group:
                break
    return pairs


def mine_parameterized_candidates(blocks: list[dict[str, Any]], max_pairs_per_group: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        if block["requires_parameter_delta"] or block["block_type"] in {"output_format_template", "reusable_instruction_template"}:
            groups[block["canonical_key"]].append(block)
    pairs: list[dict[str, Any]] = []
    for group in groups.values():
        tasks = {row["task_id"] for row in group}
        exact_hashes = {row["exact_text_hash"] for row in group}
        if len(group) < 2 or len(tasks) < 2 or len(exact_hashes) < 2:
            continue
        emitted = 0
        for left, right in itertools.combinations(sorted(group, key=lambda row: row["id"]), 2):
            if left["task_id"] == right["task_id"] or left["exact_text_hash"] == right["exact_text_hash"]:
                continue
            pairs.append(
                pair_row(
                    left,
                    right,
                    "parameterized_template_candidate",
                    "review_candidate",
                    False,
                    "Template matches after placeholder normalization, but task-specific parameters differ; requires parameter delta and judge.",
                )
            )
            emitted += 1
            if emitted >= max_pairs_per_group:
                break
    return pairs


def pair_row(
    left: dict[str, Any],
    right: dict[str, Any],
    relation_type: str,
    label: str,
    safe_to_reuse: bool,
    decision_basis: str,
) -> dict[str, Any]:
    return {
        "benchmark_label": label,
        "relation_type": relation_type,
        "safe_to_reuse": safe_to_reuse,
        "same_answer_utility": safe_to_reuse,
        "requires_parameter_delta": left["requires_parameter_delta"] or right["requires_parameter_delta"],
        "decision_basis": decision_basis,
        "left_id": left["id"],
        "right_id": right["id"],
        "left_task_id": left["task_id"],
        "right_task_id": right["task_id"],
        "left_category": left["category"],
        "right_category": right["category"],
        "left_block_type": left["block_type"],
        "right_block_type": right["block_type"],
        "left_functional_role": left["functional_role"],
        "right_functional_role": right["functional_role"],
        "left_template_hash": left["template_hash"],
        "right_template_hash": right["template_hash"],
        "left_canonical_hash": left["canonical_hash"],
        "right_canonical_hash": right["canonical_hash"],
        "left_canonical_key": left["canonical_key"],
        "right_canonical_key": right["canonical_key"],
        "pair_canonical_key": left["canonical_key"] if left["canonical_key"] == right["canonical_key"] else "",
        "pair_canonical_text": left["canonical_text"] if left["canonical_key"] == right["canonical_key"] else "",
        "left_canonical_text": left["canonical_text"],
        "right_canonical_text": right["canonical_text"],
        "left_preview": preview(left["sub_context"]),
        "right_preview": preview(right["sub_context"]),
    }


def preview(text: str, limit: int = 260) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    block_rows = [[key, value] for key, value in sorted(summary["block_type_counts"].items())]
    relation_rows = [[key, value] for key, value in sorted(summary["pair_relation_counts"].items())]
    report = f"""# WildClaw Prompt Internal Segmentation

- Generated: `{summary["generated_at"]}`
- Tasks scanned: {summary["tasks_scanned"]}
- Prompt-internal blocks: {summary["prompt_internal_blocks"]}
- Safe exact prompt-template positive pairs: {summary["safe_exact_pairs"]}
- Parameterized template review candidates: {summary["parameterized_review_candidates"]}

## Why This Step Matters

The previous near-duplicate whole prompts were rejected because they mixed
shared instruction templates with task-specific facts. This step splits the
`user_prompt` field into smaller semantic blocks before reuse mining.

## Block Types

{markdown_table(["block_type", "rows"], block_rows)}

## Pair Types

{markdown_table(["relation_type", "pairs"], relation_rows)}

## Interpretation

Exact repeated prompt-internal instruction blocks are safe reusable positives.
Parameterized template matches are not promoted yet: they need a task-specific
delta and manual/LLM judging before they can be used for reuse.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wildclaw-root", type=Path, default=Path("external/WildClawBench"))
    parser.add_argument("--tasks-root", type=Path, default=None)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_prompt_internal_segmentation"))
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-clause-chars", type=int, default=650)
    parser.add_argument("--max-pairs-per-group", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wildclaw_root = args.wildclaw_root.resolve()
    tasks_root = args.tasks_root.resolve() if args.tasks_root else wildclaw_root / "tasks"
    task_files = discover_task_files(tasks_root, args.category)
    if args.max_tasks:
        task_files = task_files[: args.max_tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[dict[str, Any]] = []
    for task_file in task_files:
        task = parse_task_markdown(task_file)
        blocks.extend(prompt_blocks(task, task_file, wildclaw_root, args.max_clause_chars, args.min_chars))

    safe_pairs = mine_exact_safe_pairs(blocks, args.max_pairs_per_group)
    review_pairs = mine_parameterized_candidates(blocks, args.max_pairs_per_group)
    all_pairs = safe_pairs + review_pairs
    for index, row in enumerate(all_pairs, start=1):
        row["prompt_internal_pair_id"] = f"prompt_internal_pair_{index:04d}"

    summary = {
        "generated_at": now_iso(),
        "wildclaw_root": str(wildclaw_root),
        "tasks_scanned": len(task_files),
        "prompt_internal_blocks": len(blocks),
        "safe_exact_pairs": len(safe_pairs),
        "parameterized_review_candidates": len(review_pairs),
        "block_type_counts": dict(Counter(row["block_type"] for row in blocks)),
        "pair_relation_counts": dict(Counter(row["relation_type"] for row in all_pairs)),
        "outputs": {
            "subcontexts": str(args.output_dir / "wildclaw_prompt_internal_subcontexts.jsonl"),
            "safe_pairs": str(args.output_dir / "wildclaw_prompt_internal_safe_pairs.csv"),
            "review_candidates": str(args.output_dir / "wildclaw_prompt_internal_review_candidates.csv"),
        },
    }

    write_jsonl(args.output_dir / "wildclaw_prompt_internal_subcontexts.jsonl", blocks)
    write_csv(args.output_dir / "wildclaw_prompt_internal_subcontexts.csv", blocks)
    write_csv(args.output_dir / "wildclaw_prompt_internal_safe_pairs.csv", safe_pairs)
    write_csv(args.output_dir / "wildclaw_prompt_internal_review_candidates.csv", review_pairs)
    write_csv(args.output_dir / "wildclaw_prompt_internal_pairs.csv", all_pairs)
    (args.output_dir / "wildclaw_prompt_internal_segmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "wildclaw_prompt_internal_segmentation_report.md", summary)

    print("WildClaw prompt internal segmentation")
    print("====================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Tasks scanned: {summary['tasks_scanned']}")
    print(f"Prompt-internal blocks: {summary['prompt_internal_blocks']}")
    print(f"Safe exact pairs: {summary['safe_exact_pairs']}")
    print(f"Parameterized review candidates: {summary['parameterized_review_candidates']}")


if __name__ == "__main__":
    main()
