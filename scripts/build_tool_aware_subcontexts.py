#!/usr/bin/env python3
"""Expand tool_history_output sub-contexts into tool-aware sub-context rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SEGMENTATION = Path("benchmark_results/wildclaw_tool_history_trace_segmentation/wildclaw_llm_trace_field_subcontexts.jsonl")
DEFAULT_TRACES = Path("benchmark_results/wildclaw_tool_history_trace_fixture/wildclaw_tool_history_trace_contexts.jsonl")
DEFAULT_OUTPUT_DIR = Path("benchmark_results/wildclaw_tool_aware_trace_segmentation")

READ_ONLY_TOOLS = {"read_file", "web_search", "search", "browser_search", "fetch_url", "list_files", "grep", "rg"}
MUTATING_TOOLS = {"write_file", "edit_file", "delete_file", "shell", "bash", "python", "browser_click", "submit"}
TOOL_TYPES = {
    "read_file": "filesystem_read",
    "list_files": "filesystem_read",
    "grep": "filesystem_read",
    "rg": "filesystem_read",
    "web_search": "retrieval",
    "search": "retrieval",
    "browser_search": "retrieval",
    "fetch_url": "retrieval",
    "write_file": "filesystem_write",
    "edit_file": "filesystem_write",
    "delete_file": "filesystem_write",
    "shell": "execution",
    "bash": "execution",
    "python": "execution",
    "browser_click": "browser_action",
    "submit": "external_action",
}


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def short_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def task_id_of(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    return str(task.get("task_id") or row.get("task_id") or "unknown_task")


def infer_tool_type(tool_name: str) -> str:
    return TOOL_TYPES.get(tool_name, "unknown")


def is_mutating(tool_name: str, tool_type: str) -> bool:
    return tool_name in MUTATING_TOOLS or tool_type in {"filesystem_write", "execution", "browser_action", "external_action"}


def tool_reuse_policy(tool_name: str, tool_type: str, mutating: bool) -> str:
    if mutating:
        return "no_reuse_mutating_tool"
    if tool_name in READ_ONLY_TOOLS or tool_type in {"filesystem_read", "retrieval"}:
        return "candidate_reuse_after_content_or_same_answer_judge"
    return "no_reuse_unknown_tool"


def render_tool_subcontext(task_id: str, event: dict[str, Any]) -> str:
    tool_name = str(event.get("tool_name") or "unknown_tool")
    tool_type = str(event.get("tool_type") or infer_tool_type(tool_name))
    step = event.get("step", "")
    tool_call = event.get("tool_call", {})
    tool_output = event.get("tool_output", "")
    semantic_role = event.get("semantic_role", "")
    return "\n".join(
        [
            f"task_id: {task_id}",
            f"tool_step: {step}",
            f"tool_name: {tool_name}",
            f"tool_type: {tool_type}",
            f"semantic_role: {semantic_role}",
            "tool_call:",
            json.dumps(tool_call, ensure_ascii=False, sort_keys=True),
            "tool_output:",
            str(tool_output).strip(),
        ]
    ).strip()


def make_tool_row(template: dict[str, Any], event: dict[str, Any], task_id: str, event_index: int) -> dict[str, Any]:
    tool_name = str(event.get("tool_name") or "unknown_tool")
    tool_type = str(event.get("tool_type") or infer_tool_type(tool_name))
    tool_call_id = str(event.get("tool_call_id") or f"{task_id}:tool:{event_index:02d}:{tool_name}")
    tool_step = int(event.get("step") or event_index)
    tool_input = event.get("tool_call", {})
    tool_output = str(event.get("tool_output", ""))
    input_hash = short_hash(tool_input)
    output_hash = hashlib.sha256(normalize_text(tool_output).encode("utf-8")).hexdigest()[:24]
    mutating = is_mutating(tool_name, tool_type)
    sub_context = render_tool_subcontext(task_id, event)
    hash_value = content_hash(sub_context)
    safe_tool_name = re.sub(r"[^A-Za-z0-9_]+", "_", tool_name).strip("_").lower() or "unknown_tool"
    row = dict(template)
    row.update(
        {
            "schema_version": "tool-aware-field-subcontext-v1",
            "id": f"{task_id}__tool_history_output__tool_{tool_step:02d}_{safe_tool_name}__part_01",
            "method": "tool_aware_field_segmentation",
            "title": f"Tool output: {tool_name}",
            "context_field": "tool_history_output",
            "functional_role": "external_observations_and_tool_outputs",
            "sub_context": sub_context,
            "source_spans": [
                {
                    "source": f"tool_history[{event_index - 1}].{tool_name}",
                    "char_start": 0,
                    "char_end": len(sub_context),
                    "chunk_index": 1,
                    "chunk_count": 1,
                }
            ],
            "split_strategy": "tool_history_event_split_before_field_internal_window",
            "sub_context_chars": len(sub_context),
            "content_hash": hash_value,
            "tool_name": tool_name,
            "tool_type": tool_type,
            "tool_call_id": tool_call_id,
            "tool_step": tool_step,
            "tool_input": json.dumps(tool_input, ensure_ascii=False, sort_keys=True),
            "tool_input_hash": input_hash,
            "tool_output_hash": output_hash,
            "is_mutating_tool": mutating,
            "tool_reuse_policy_hint": tool_reuse_policy(tool_name, tool_type, mutating),
            "manual_review_status": "pending",
            "created_at": now_iso(),
        }
    )
    row["sglang_storage_plan"] = {
        "prefill_unit_id": f"{task_id}:tool_history_output:{tool_name}:{tool_call_id}",
        "cache_key": hash_value,
        "cache_scope": "sub_context_block",
        "can_prefill_independently": True,
        "requires_block_isolation_attention_mask": True,
        "kv_reuse_granularity": "sub_context",
        "tool_identity": {
            "tool_name": tool_name,
            "tool_type": tool_type,
            "tool_call_id": tool_call_id,
            "is_mutating_tool": mutating,
        },
    }
    row["tool_metadata"] = {
        "tool_name": tool_name,
        "tool_type": tool_type,
        "tool_call_id": tool_call_id,
        "tool_step": tool_step,
        "tool_input_hash": input_hash,
        "tool_output_hash": output_hash,
        "is_mutating_tool": mutating,
        "reuse_policy_hint": row["tool_reuse_policy_hint"],
    }
    row["selection_metadata"] = dict(row.get("selection_metadata") or {})
    row["selection_metadata"].update({"tool_name": tool_name, "tool_type": tool_type, "is_mutating_tool": mutating})
    return row


def blank_tool_fields(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("tool_name", "")
    row.setdefault("tool_type", "")
    row.setdefault("tool_call_id", "")
    row.setdefault("tool_step", "")
    row.setdefault("tool_input", "")
    row.setdefault("tool_input_hash", "")
    row.setdefault("tool_output_hash", "")
    row.setdefault("is_mutating_tool", "")
    row.setdefault("tool_reuse_policy_hint", "")
    row.setdefault("tool_metadata", {})
    return row


def build_tool_aware_rows(segmentation_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces_by_task = {task_id_of(row): row for row in trace_rows}
    output: list[dict[str, Any]] = []
    for row in segmentation_rows:
        if row.get("context_field") != "tool_history_output":
            output.append(blank_tool_fields(row))
            continue
        task_id = str(row.get("task_id") or "unknown_task")
        events = traces_by_task.get(task_id, {}).get("tool_history", [])
        if not isinstance(events, list) or not events:
            output.append(blank_tool_fields(row))
            continue
        for index, event in enumerate(events, start=1):
            if isinstance(event, dict):
                output.append(make_tool_row(row, event, task_id, index))
    return output


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = (
            str(row.get("context_field", "")),
            str(row.get("tool_name", "")),
            str(row.get("tool_type", "")),
            str(row.get("tool_reuse_policy_hint", "")),
        )
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "context_field": field,
            "tool_name": tool_name,
            "tool_type": tool_type,
            "tool_reuse_policy_hint": hint,
            "rows": count,
        }
        for (field, tool_name, tool_type, hint), count in sorted(counts.items())
    ]


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    tool_rows = [row for row in rows if row.get("context_field") == "tool_history_output"]
    by_tool: dict[tuple[str, str, str], int] = {}
    for row in tool_rows:
        key = (str(row.get("tool_name", "")), str(row.get("tool_type", "")), str(row.get("tool_reuse_policy_hint", "")))
        by_tool[key] = by_tool.get(key, 0) + 1
    lines = [
        "# Tool-aware Field Segmentation",
        "",
        f"- Input segmentation JSONL: `{args.input_jsonl.as_posix()}`",
        f"- Input trace JSONL: `{args.trace_jsonl.as_posix()}`",
        f"- Output rows: {len(rows)}",
        f"- Tool-history output rows: {len(tool_rows)}",
        "",
        "## Tool Counts",
        "",
        "| tool_name | tool_type | reuse policy hint | rows |",
        "| --- | --- | --- | ---: |",
    ]
    for (tool_name, tool_type, hint), count in sorted(by_tool.items()):
        lines.append(f"| `{tool_name}` | `{tool_type}` | `{hint}` | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`tool_history_output` is now split by individual tool identity instead of one aggregate field block.",
            "Read-only retrieval/file-read tools can become semantic reuse candidates after content or same-answer judging; mutating tools are marked `no_reuse_mutating_tool`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--trace-jsonl", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_tool_aware_rows(read_jsonl(args.input_jsonl), read_jsonl(args.trace_jsonl))
    output_jsonl = args.output_dir / "wildclaw_tool_aware_field_subcontexts.jsonl"
    summary_csv = args.output_dir / "wildclaw_tool_aware_field_summary.csv"
    report = args.output_dir / "wildclaw_tool_aware_field_report.md"
    write_jsonl(output_jsonl, rows)
    write_csv(summary_csv, summarize(rows))
    write_report(report, rows, args)
    print("Tool-aware field segmentation")
    print("=============================")
    print(f"Rows: {len(rows)}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
