#!/usr/bin/env python3
"""Prepare field/function-first semantic sub-contexts.

This script does not segment by timeline first. It starts from explicit fields
and functional blocks in a WildClaw/OpenClaw context object, then only splits
overlong fields into stable internal chunks. The output metadata is shaped so
each sub-context can later become an independently stored SGLang prefill/cache
unit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl")

FIELD_ORDER = [
    "user_prompt",
    "tool_definitions",
    "tool_history_output",
    "conversation_history",
    "workspace_evidence",
    "task_metadata",
]


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def stringify(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def nested_get_any(row: dict[str, Any], names: Iterable[str]) -> Any:
    queue: list[Any] = [row]
    seen = set()
    target_names = {name.lower() for name in names}
    while queue:
        current = queue.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, dict):
            for key, value in current.items():
                if key.lower() in target_names:
                    return value
                if isinstance(value, dict | list):
                    queue.append(value)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, dict | list):
                    queue.append(item)
    return None


def split_overlong_text(text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, dict[str, Any]]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [(text, {"split_strategy": "none", "chunk_index": 1, "chunk_count": 1, "char_start": 0, "char_end": len(text)})]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        projected = current_len + len(paragraph) + (2 if current else 0)
        if current and projected > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if len(paragraph) > max_chars:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end])
                if end == len(paragraph):
                    break
                start = max(start + 1, end - overlap_chars)
            continue
        current.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)
    if current:
        chunks.append("\n\n".join(current))

    if not chunks:
        chunks = [text[:max_chars]]
    result = []
    cursor = 0
    for index, chunk in enumerate(chunks, start=1):
        found = text.find(chunk[: min(80, len(chunk))], cursor)
        start = found if found >= 0 else cursor
        end = start + len(chunk)
        cursor = max(cursor, end - overlap_chars)
        result.append(
            (
                chunk,
                {
                    "split_strategy": "field_internal_paragraph_or_window",
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "char_start": start,
                    "char_end": end,
                },
            )
        )
    return result


def task_id_of(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    return str(task.get("task_id") or row.get("task_id") or "unknown_task")


def category_of(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    return str(task.get("category") or row.get("category") or task_id_of(row).split("_task_", 1)[0])


def task_name_of(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    return str(task.get("name") or row.get("task_name") or "")


def build_task_metadata(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    fields = {
        "task_id": task_id_of(row),
        "category": category_of(row),
        "task_name": task_name_of(row),
        "timeout_seconds": task.get("timeout_seconds", ""),
        "workspace_path": task.get("workspace_path", row.get("workspace_dir", "")),
        "skills": task.get("skills", []),
        "env": task.get("env", []),
    }
    return json.dumps(fields, ensure_ascii=False, indent=2)


def collect_functional_fields(row: dict[str, Any], max_source_files: int) -> list[dict[str, Any]]:
    task = row.get("task") or {}
    fields: list[dict[str, Any]] = []

    prompt = stringify(task.get("prompt") or row.get("user_prompt") or row.get("prompt"))
    if prompt:
        fields.append(
            {
                "context_field": "user_prompt",
                "functional_role": "request_objective_and_constraints",
                "title": "User prompt / task objective",
                "source": "task.prompt",
                "text": prompt,
            }
        )

    tool_defs = nested_get_any(
        row,
        [
            "tool_definitions",
            "tools",
            "available_tools",
            "tool_schema",
            "tool_schemas",
            "skills",
        ],
    )
    tool_defs_text = stringify(tool_defs or task.get("skills"))
    if tool_defs_text:
        fields.append(
            {
                "context_field": "tool_definitions",
                "functional_role": "allowed_actions_and_interfaces",
                "title": "Tool definitions / available skills",
                "source": "tool_definitions_or_task.skills",
                "text": tool_defs_text,
            }
        )

    tool_history = nested_get_any(
        row,
        [
            "tool_history",
            "tool_outputs",
            "tool_results",
            "observations",
            "execution_trace",
        ],
    )
    tool_history_text = stringify(tool_history)
    if tool_history_text:
        fields.append(
            {
                "context_field": "tool_history_output",
                "functional_role": "external_observations_and_tool_outputs",
                "title": "Tool history output",
                "source": "tool_history_or_outputs",
                "text": tool_history_text,
            }
        )

    conversation = nested_get_any(
        row,
        [
            "conversation_history",
            "conversation",
            "messages",
            "chat_history",
            "dialogue",
        ],
    )
    conversation_text = stringify(conversation)
    if conversation_text:
        fields.append(
            {
                "context_field": "conversation_history",
                "functional_role": "prior_user_assistant_state",
                "title": "Conversation history",
                "source": "conversation_history_or_messages",
                "text": conversation_text,
            }
        )

    source_files = [
        source
        for source in row.get("source_files", [])
        if source.get("included_in_context") and source.get("content")
    ][:max_source_files]
    for source in source_files:
        fields.append(
            {
                "context_field": "workspace_evidence",
                "functional_role": "workspace_document_or_evidence",
                "title": f"Workspace evidence: {source.get('path')}",
                "source": source.get("path", "workspace_file"),
                "text": stringify(source.get("content")),
                "source_file_sha256": source.get("sha256", ""),
                "line_count": source.get("line_count", ""),
            }
        )

    metadata = build_task_metadata(row)
    if metadata:
        fields.append(
            {
                "context_field": "task_metadata",
                "functional_role": "routing_and_selection_metadata",
                "title": "Task metadata",
                "source": "task_metadata",
                "text": metadata,
            }
        )

    return sorted(fields, key=lambda item: FIELD_ORDER.index(item["context_field"]))


def make_subcontext(
    *,
    raw: dict[str, Any],
    field: dict[str, Any],
    text: str,
    split_meta: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    task_id = task_id_of(raw)
    context_field = field["context_field"]
    chunk_index = split_meta["chunk_index"]
    field_instance = re.sub(r"[^A-Za-z0-9_]+", "_", field["source"])[:48].strip("_").lower() or context_field
    source_hash = hashlib.sha256(str(field["source"]).encode("utf-8")).hexdigest()[:10]
    sub_id = f"{task_id}__{context_field}__{field_instance}_{source_hash}__part_{chunk_index:02d}"
    hash_value = content_hash(text)
    isolation_group_id = f"{task_id}:{context_field}:{field_instance}:{source_hash}:{chunk_index}"
    return {
        "schema_version": "field-function-subcontext-v1",
        "id": sub_id,
        "source": raw.get("source", "WildClawBench/OpenClaw"),
        "method": "field_function_semantic",
        "category": category_of(raw),
        "task_id": task_id,
        "task_name": task_name_of(raw),
        "title": field["title"],
        "context_field": context_field,
        "functional_role": field["functional_role"],
        "sub_context": text,
        "source_spans": [
            {
                "source": field["source"],
                "char_start": split_meta["char_start"],
                "char_end": split_meta["char_end"],
                "chunk_index": chunk_index,
                "chunk_count": split_meta["chunk_count"],
            }
        ],
        "split_strategy": split_meta["split_strategy"],
        "sub_context_chars": len(text),
        "content_hash": hash_value,
        "semantic_independence_target": "field_or_function_isolated",
        "attention_policy_target": "no_pairwise_attention_between_subcontexts",
        "sglang_storage_plan": {
            "prefill_unit_id": isolation_group_id,
            "cache_key": hash_value,
            "cache_scope": "sub_context_block",
            "can_prefill_independently": True,
            "requires_block_isolation_attention_mask": True,
            "kv_reuse_granularity": "sub_context",
        },
        "selection_metadata": {
            "field_order": FIELD_ORDER.index(context_field),
            "functional_role": field["functional_role"],
            "source_file_sha256": field.get("source_file_sha256", ""),
            "line_count": field.get("line_count", ""),
        },
        "leakage_checked": bool(raw.get("leakage_policy")),
        "manual_review_status": "pending",
        "created_at": now_iso(),
        "row_index": index,
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    raw_rows = read_jsonl(args.input_jsonl)
    row_index = 1
    for raw in raw_rows:
        for field in collect_functional_fields(raw, args.max_source_files_per_task):
            for text, split_meta in split_overlong_text(field["text"], args.max_field_chars, args.overlap_chars):
                if len(text.strip()) < args.min_chars:
                    continue
                rows.append(make_subcontext(raw=raw, field=field, text=text, split_meta=split_meta, index=row_index))
                row_index += 1
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["task_id"], row["context_field"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (task_id, context_field), items in sorted(grouped.items()):
        summary.append(
            {
                "task_id": task_id,
                "context_field": context_field,
                "sub_context_count": len(items),
                "total_chars": sum(int(item["sub_context_chars"]) for item in items),
                "avg_chars": round(sum(int(item["sub_context_chars"]) for item in items) / len(items), 1),
                "attention_policy_target": "no_pairwise_attention_between_subcontexts",
                "sglang_storage_ready": all(item["sglang_storage_plan"]["can_prefill_independently"] for item in items),
            }
        )
    return summary


def write_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Field/Function Semantic Sub-contexts",
        "",
        "This artifact segments WildClaw/OpenClaw context by fields and functional blocks first, not by timeline trace order.",
        "",
        "## Policy",
        "",
        "- First split by `user_prompt`, `tool_definitions`, `tool_history_output`, `conversation_history`, `workspace_evidence`, and `task_metadata`.",
        "- Only overlong fields are split internally.",
        "- Each output row carries SGLang-oriented metadata for independent prefill/cache/reuse.",
        "- The target runtime policy is no pairwise attention between sub-context blocks.",
        "",
        "## Inputs",
        "",
        f"- Input JSONL: `{args.input_jsonl.as_posix()}`",
        f"- Max field chars before internal split: {args.max_field_chars}",
        f"- Max source files per task: {args.max_source_files_per_task}",
        "",
        "## Summary",
        "",
        "| task_id | context_field | sub-contexts | total chars | avg chars |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['task_id']}` | `{row['context_field']}` | {row['sub_context_count']} | {row['total_chars']} | {row['avg_chars']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime Interpretation",
            "",
            "Each row is intended to become an SGLang sub-context block with its own `prefill_unit_id` and `cache_key`.",
            "A future runtime implementation should prefill these blocks independently and use a block-isolation attention mask so tokens inside one sub-context do not attend to tokens inside another sub-context.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_field_function_subcontexts"))
    parser.add_argument("--max-field-chars", type=int, default=4000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-source-files-per-task", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    summary = summarize(rows)
    output_jsonl = args.output_dir / "wildclaw_field_function_subcontexts.jsonl"
    summary_csv = args.output_dir / "wildclaw_field_function_subcontexts_summary.csv"
    report = args.output_dir / "wildclaw_field_function_subcontexts_report.md"
    write_jsonl(output_jsonl, rows)
    write_csv(summary_csv, summary)
    write_report(report, rows, summary, args)

    print("Field/function semantic sub-contexts")
    print("====================================")
    print(f"Rows: {len(rows)}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
