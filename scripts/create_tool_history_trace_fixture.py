#!/usr/bin/env python3
"""Create WildClaw/OpenClaw-style trace rows that contain tool outputs.

The original WildClaw real-context pilot rows mostly contain task prompts and
workspace files, but not full agent execution traces. This fixture keeps the
same task/workspace source and adds lightweight tool-call/tool-output history
so field-first segmentation can exercise the `tool_history_output` field.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl")
DEFAULT_OUTPUT = Path("benchmark_results/wildclaw_tool_history_trace_fixture/wildclaw_tool_history_trace_contexts.jsonl")


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


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


def first_included_file(row: dict[str, Any]) -> dict[str, Any]:
    for source in row.get("source_files", []):
        if source.get("included_in_context") and source.get("content"):
            return source
    return {}


def excerpt(text: str, chars: int = 900) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text[:chars] + ("\n[TRUNCATED_TOOL_OUTPUT]" if len(text) > chars else "")


def build_tool_history(row: dict[str, Any]) -> list[dict[str, Any]]:
    task = row.get("task") or {}
    task_id = str(task.get("task_id") or row.get("task_id") or "unknown_task")
    first_file = first_included_file(row)
    file_path = first_file.get("path", "workspace_file")
    file_text = excerpt(str(first_file.get("content", "")), 900)
    prompt = str(task.get("prompt", ""))
    query = " ".join(prompt.split()[:24]) or task_id
    return [
        {
            "step": 1,
            "tool_name": "read_file",
            "tool_type": "filesystem_read",
            "tool_call_id": f"{task_id}:tool:01:read_file",
            "is_mutating_tool": False,
            "tool_call": {"path": file_path, "max_chars": 900},
            "tool_output": file_text,
            "semantic_role": "workspace evidence read from local files",
        },
        {
            "step": 2,
            "tool_name": "web_search",
            "tool_type": "retrieval",
            "tool_call_id": f"{task_id}:tool:02:web_search",
            "is_mutating_tool": False,
            "tool_call": {"query": query},
            "tool_output": (
                "Search results indicate that the local evidence should be checked "
                "against current public sources when the task mentions outdated, "
                "conflicting, or fuzzy retrieval requirements."
            ),
            "semantic_role": "external retrieval observation",
        },
        {
            "step": 3,
            "tool_name": "write_file",
            "tool_type": "filesystem_write",
            "tool_call_id": f"{task_id}:tool:03:write_file",
            "is_mutating_tool": True,
            "tool_call": {"path": "/tmp_workspace/results/results.md"},
            "tool_output": "Write operation prepared for the requested result path.",
            "semantic_role": "output artifact operation",
        },
    ]


def build_conversation_history(row: dict[str, Any]) -> list[dict[str, str]]:
    task = row.get("task") or {}
    return [
        {
            "role": "user",
            "content": str(task.get("prompt", "")),
        },
        {
            "role": "assistant",
            "content": "I will inspect local workspace files, use search when needed, and write the final result to the requested path.",
        },
    ]


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(args.input_jsonl)
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        row["schema_version"] = "wildclaw-tool-history-trace-fixture-v1"
        row["source"] = "WildClawBench/OpenClaw tool-history fixture"
        row["created_at"] = now_iso()
        row["conversation_history"] = build_conversation_history(raw)
        row["tool_history"] = build_tool_history(raw)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    write_jsonl(args.output_jsonl, rows)
    print("Tool-history trace fixture")
    print("==========================")
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
