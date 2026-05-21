#!/usr/bin/env python3
"""Use an LLM to split complete agent traces into field/function sub-contexts.

The goal is field-first segmentation, not timeline segmentation. The script
renders each WildClaw/OpenClaw task row as a marked trace dump, asks an
OpenAI-compatible LLM to classify blocks into stable context fields, then
writes SGLang-facing sub-context rows:

- user_prompt
- tool_definitions
- tool_history_output
- conversation_history
- workspace_evidence
- task_metadata

If no API key is available in auto mode, it falls back to the deterministic
field/function extractor so the pipeline remains testable offline.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl")
DEFAULT_OUTPUT_DIR = Path("benchmark_results/wildclaw_llm_trace_field_segmentation")

FIELD_ORDER = [
    "user_prompt",
    "tool_definitions",
    "tool_history_output",
    "conversation_history",
    "workspace_evidence",
    "task_metadata",
]

FIELD_ROLES = {
    "user_prompt": "request_objective_and_constraints",
    "tool_definitions": "allowed_actions_and_interfaces",
    "tool_history_output": "external_observations_and_tool_outputs",
    "conversation_history": "prior_user_assistant_state",
    "workspace_evidence": "workspace_document_or_evidence",
    "task_metadata": "routing_and_selection_metadata",
}

FIELD_TITLES = {
    "user_prompt": "User prompt / task objective",
    "tool_definitions": "Tool definitions / available skills",
    "tool_history_output": "Tool history output",
    "conversation_history": "Conversation history",
    "workspace_evidence": "Workspace evidence",
    "task_metadata": "Task metadata",
}

SYSTEM_PROMPT = """You split complete agent traces into semantic-independent context fields.

Do not solve the task. Do not summarize the final answer. Do not use ground
truth, grading scripts, or answer files.

Classify trace material into these context fields:
- user_prompt: user request, task objective, output constraints.
- tool_definitions: available tools, tool schemas, skills, allowed interfaces.
- tool_history_output: tool calls, observations, search/browser/file outputs,
  execution results, stdout/stderr, retrieved pages.
- conversation_history: previous user/assistant messages and reasoning state
  that are not tool outputs.
- workspace_evidence: repository/workspace files, documents, code, logs, data.
- task_metadata: routing metadata such as task id/category/workspace path/env.

Important:
- Segment by field/function first, not by chronological order.
- If the same field appears at multiple points in the trace, group it under the
  same context_field when it can be stored as one independent block.
- Preserve evidence-bearing text. Do not invent facts.
- Each returned field block should be independently understandable.
- Return strict JSON only.
"""

USER_PROMPT_TEMPLATE = """Task:
task_id: {task_id}
category: {category}
name: {task_name}

Marked full agent trace / task context:
{trace_text}

Return JSON with this exact shape:
{{
  "fields": [
    {{
      "context_field": "user_prompt | tool_definitions | tool_history_output | conversation_history | workspace_evidence | task_metadata",
      "title": "short title",
      "source_ids": ["TRACE_BLOCK id values used"],
      "sub_context": "verbatim or lightly cleaned field text, preserving facts",
      "confidence": 0.0,
      "notes": "short reason why this field is independent"
    }}
  ]
}}
"""


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


def task_id_of(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    return str(task.get("task_id") or row.get("task_id") or "unknown_task")


def category_of(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    return str(task.get("category") or row.get("category") or task_id_of(row).split("_task_", 1)[0])


def task_name_of(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    return str(task.get("name") or row.get("task_name") or "")


def nested_values_by_name(row: Any, names: set[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    queue: list[tuple[str, Any]] = [("$", row)]
    seen: set[int] = set()
    while queue:
        path, current = queue.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, dict):
            for key, value in current.items():
                child_path = f"{path}.{key}"
                if key.lower() in names:
                    found.append((child_path, value))
                if isinstance(value, dict | list):
                    queue.append((child_path, value))
        elif isinstance(current, list):
            for index, value in enumerate(current):
                child_path = f"{path}[{index}]"
                if isinstance(value, dict | list):
                    queue.append((child_path, value))
    return found


def add_block(blocks: list[dict[str, str]], block_id: str, suggested_field: str, label: str, text: str, max_chars: int) -> None:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED_TRACE_BLOCK]"
    blocks.append(
        {
            "id": block_id,
            "suggested_field": suggested_field,
            "label": label,
            "text": text,
        }
    )


def build_trace_blocks(row: dict[str, Any], max_block_chars: int, max_workspace_files: int) -> list[dict[str, str]]:
    task = row.get("task") or {}
    blocks: list[dict[str, str]] = []

    add_block(blocks, "task.prompt", "user_prompt", "Task prompt", stringify(task.get("prompt") or row.get("prompt")), max_block_chars)
    add_block(
        blocks,
        "task.expected_behavior",
        "conversation_history",
        "Expected behavior / task policy",
        stringify(task.get("expected_behavior")),
        max_block_chars,
    )
    add_block(blocks, "task.skills", "tool_definitions", "Task skills", stringify(task.get("skills")), max_block_chars)
    add_block(
        blocks,
        "task.metadata",
        "task_metadata",
        "Task metadata",
        stringify(
            {
                "task_id": task_id_of(row),
                "category": category_of(row),
                "task_name": task_name_of(row),
                "timeout_seconds": task.get("timeout_seconds", ""),
                "workspace_path": task.get("workspace_path", row.get("workspace_dir", "")),
                "env": task.get("env", []),
            }
        ),
        max_block_chars,
    )

    name_groups = [
        ("tool_definitions", {"tool_definitions", "tools", "available_tools", "tool_schema", "tool_schemas"}),
        ("tool_history_output", {"tool_history", "tool_outputs", "tool_results", "observations", "execution_trace"}),
        ("conversation_history", {"conversation_history", "conversation", "messages", "chat_history", "dialogue"}),
    ]
    for suggested_field, names in name_groups:
        for path, value in nested_values_by_name(row, names):
            add_block(blocks, path, suggested_field, path, stringify(value), max_block_chars)

    for index, source in enumerate(row.get("source_files", []), start=1):
        if index > max_workspace_files:
            break
        if not source.get("included_in_context") or not source.get("content"):
            continue
        add_block(
            blocks,
            f"source_files[{index}]:{source.get('path', 'workspace_file')}",
            "workspace_evidence",
            str(source.get("path", "workspace_file")),
            stringify(source.get("content")),
            max_block_chars,
        )

    if not blocks and row.get("long_context"):
        add_block(blocks, "long_context", "workspace_evidence", "Long context", stringify(row.get("long_context")), max_block_chars)
    return blocks


def render_trace(blocks: list[dict[str, str]], max_trace_chars: int) -> str:
    rendered_parts = []
    total = 0
    for block in blocks:
        part = (
            f'<<<TRACE_BLOCK id="{block["id"]}" suggested_field="{block["suggested_field"]}" label="{block["label"]}">>>\n'
            f"{block['text']}\n"
            "<<<END_TRACE_BLOCK>>>\n"
        )
        if total + len(part) > max_trace_chars:
            remaining = max_trace_chars - total
            if remaining > 300:
                rendered_parts.append(part[:remaining] + "\n[TRUNCATED_FULL_TRACE_FOR_LLM]\n")
            break
        rendered_parts.append(part)
        total += len(part)
    return "\n".join(rendered_parts)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_completion(args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
    }
    if not args.disable_response_format:
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Nyannyan222/openclaw-sglang-prefix-cache",
            "X-Title": "OpenClaw LLM trace field segmentation",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def split_overlong_text(text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, dict[str, Any]]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [(text, {"split_strategy": "none", "chunk_index": 1, "chunk_count": 1, "char_start": 0, "char_end": len(text)})]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return [
        (
            chunk,
            {
                "split_strategy": "field_internal_window_after_llm_field_classification",
                "chunk_index": index,
                "chunk_count": len(chunks),
                "char_start": max(0, (index - 1) * (max_chars - overlap_chars)),
                "char_end": max(0, (index - 1) * (max_chars - overlap_chars)) + len(chunk),
            },
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def make_subcontext(
    *,
    raw: dict[str, Any],
    context_field: str,
    title: str,
    source_ids: list[str],
    text: str,
    split_meta: dict[str, Any],
    index: int,
    method: str,
    confidence: float | None,
    notes: str,
) -> dict[str, Any]:
    task_id = task_id_of(raw)
    context_field = context_field if context_field in FIELD_ORDER else "workspace_evidence"
    field_instance = re.sub(r"[^A-Za-z0-9_]+", "_", "_".join(source_ids)[:80]).strip("_").lower() or context_field
    source_hash = hashlib.sha256("|".join(source_ids).encode("utf-8")).hexdigest()[:10]
    chunk_index = int(split_meta["chunk_index"])
    sub_id = f"{task_id}__{context_field}__{field_instance}_{source_hash}__part_{chunk_index:02d}"
    hash_value = content_hash(text)
    prefill_unit_id = f"{task_id}:{context_field}:{field_instance}:{source_hash}:{chunk_index}"
    return {
        "schema_version": "llm-trace-field-subcontext-v1",
        "id": sub_id,
        "source": raw.get("source", "WildClawBench/OpenClaw"),
        "method": method,
        "category": category_of(raw),
        "task_id": task_id,
        "task_name": task_name_of(raw),
        "title": title or FIELD_TITLES[context_field],
        "context_field": context_field,
        "functional_role": FIELD_ROLES[context_field],
        "sub_context": text,
        "source_spans": [
            {
                "source": source_id,
                "char_start": split_meta["char_start"],
                "char_end": split_meta["char_end"],
                "chunk_index": chunk_index,
                "chunk_count": split_meta["chunk_count"],
            }
            for source_id in source_ids
        ],
        "split_strategy": split_meta["split_strategy"],
        "sub_context_chars": len(text),
        "content_hash": hash_value,
        "semantic_independence_target": "field_or_function_isolated",
        "attention_policy_target": "no_pairwise_attention_between_subcontexts",
        "sglang_storage_plan": {
            "prefill_unit_id": prefill_unit_id,
            "cache_key": hash_value,
            "cache_scope": "sub_context_block",
            "can_prefill_independently": True,
            "requires_block_isolation_attention_mask": True,
            "kv_reuse_granularity": "sub_context",
        },
        "llm_field_segmentation": {
            "confidence": confidence,
            "notes": notes,
            "source_ids": source_ids,
        },
        "selection_metadata": {
            "field_order": FIELD_ORDER.index(context_field),
            "functional_role": FIELD_ROLES[context_field],
        },
        "leakage_checked": bool(raw.get("leakage_policy", True)),
        "manual_review_status": "pending",
        "created_at": now_iso(),
        "row_index": index,
    }


def clamp_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def heuristic_fields(blocks: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for block in blocks:
        field = block["suggested_field"] if block["suggested_field"] in FIELD_ORDER else "workspace_evidence"
        grouped.setdefault(field, []).append(block)
    fields = []
    for field in FIELD_ORDER:
        for block in grouped.get(field, []):
            fields.append(
                {
                    "context_field": field,
                    "title": FIELD_TITLES[field] if field != "workspace_evidence" else f"Workspace evidence: {block['label']}",
                    "source_ids": [block["id"]],
                    "sub_context": block["text"],
                    "confidence": 0.65,
                    "notes": "Heuristic fallback from marked trace block metadata.",
                }
            )
    return fields


def llm_fields(raw: dict[str, Any], blocks: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    trace_text = render_trace(blocks, args.max_trace_chars)
    message = USER_PROMPT_TEMPLATE.format(
        task_id=task_id_of(raw),
        category=category_of(raw),
        task_name=task_name_of(raw),
        trace_text=trace_text,
    )
    content = chat_completion(
        args,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    parsed = extract_json_object(content)
    result = []
    for item in parsed.get("fields", []):
        field = str(item.get("context_field", "")).strip()
        text = str(item.get("sub_context", "")).strip()
        if field not in FIELD_ORDER or not text:
            continue
        source_ids = item.get("source_ids") or []
        if not isinstance(source_ids, list):
            source_ids = [str(source_ids)]
        result.append(
            {
                "context_field": field,
                "title": str(item.get("title") or FIELD_TITLES[field]),
                "source_ids": [str(value) for value in source_ids if str(value).strip()] or [field],
                "sub_context": text,
                "confidence": clamp_confidence(item.get("confidence")),
                "notes": str(item.get("notes") or ""),
            }
        )
    return result


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_jsonl(args.input_jsonl)
    if args.limit:
        raw_rows = raw_rows[: args.limit]

    use_llm = args.mode == "llm" or (args.mode == "auto" and bool(args.api_key))
    if args.mode == "llm" and not args.api_key:
        raise SystemExit("LLM mode requires --api-key or OPENAI_API_KEY/OPENROUTER_API_KEY.")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    row_index = 1
    for raw in raw_rows:
        blocks = build_trace_blocks(raw, args.max_trace_block_chars, args.max_workspace_files_per_task)
        if use_llm:
            try:
                fields = llm_fields(raw, blocks, args)
                method = "llm_trace_field_segmentation"
                if not fields:
                    raise RuntimeError("LLM returned no valid field blocks")
            except Exception as exc:
                if args.mode == "llm":
                    raise
                failures.append({"task_id": task_id_of(raw), "error": str(exc)})
                fields = heuristic_fields(blocks)
                method = "heuristic_trace_field_fallback"
        else:
            fields = heuristic_fields(blocks)
            method = "heuristic_trace_field_fallback"

        for field in fields:
            for text, split_meta in split_overlong_text(field["sub_context"], args.max_field_chars, args.overlap_chars):
                if len(text.strip()) < args.min_chars:
                    continue
                rows.append(
                    make_subcontext(
                        raw=raw,
                        context_field=field["context_field"],
                        title=field.get("title", ""),
                        source_ids=list(field.get("source_ids") or [field["context_field"]]),
                        text=text,
                        split_meta=split_meta,
                        index=row_index,
                        method=method,
                        confidence=field.get("confidence"),
                        notes=str(field.get("notes") or ""),
                    )
                )
                row_index += 1

    run_info = {
        "created_at": now_iso(),
        "input_jsonl": args.input_jsonl.as_posix(),
        "mode": args.mode,
        "used_llm": use_llm,
        "model": args.model if use_llm else "",
        "row_count": len(rows),
        "task_count": len(raw_rows),
        "llm_failures": failures,
    }
    return rows, run_info


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["task_id"], row["context_field"], row["method"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (task_id, context_field, method), items in sorted(grouped.items()):
        summary.append(
            {
                "task_id": task_id,
                "context_field": context_field,
                "method": method,
                "sub_context_count": len(items),
                "total_chars": sum(int(item["sub_context_chars"]) for item in items),
                "avg_chars": round(sum(int(item["sub_context_chars"]) for item in items) / len(items), 1),
                "attention_policy_target": "no_pairwise_attention_between_subcontexts",
                "sglang_storage_ready": all(item["sglang_storage_plan"]["can_prefill_independently"] for item in items),
            }
        )
    return summary


def write_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], run_info: dict[str, Any]) -> None:
    field_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    for row in rows:
        field_counts[row["context_field"]] = field_counts.get(row["context_field"], 0) + 1
        method_counts[row["method"]] = method_counts.get(row["method"], 0) + 1

    lines = [
        "# LLM Trace Field Segmentation",
        "",
        "This artifact uses an LLM-style field classifier to split complete agent trace/task context into field/function sub-contexts before any timeline slicing.",
        "",
        "## Run",
        "",
        f"- Input JSONL: `{run_info['input_jsonl']}`",
        f"- Mode: `{run_info['mode']}`",
        f"- Used LLM: `{run_info['used_llm']}`",
        f"- Model: `{run_info['model']}`",
        f"- Tasks: {run_info['task_count']}",
        f"- Output rows: {run_info['row_count']}",
        "",
        "## Field Counts",
        "",
        "| context_field | rows |",
        "| --- | ---: |",
    ]
    for field in FIELD_ORDER:
        lines.append(f"| `{field}` | {field_counts.get(field, 0)} |")

    lines.extend(
        [
            "",
            "## Method Counts",
            "",
            "| method | rows |",
            "| --- | ---: |",
        ]
    )
    for method, count in sorted(method_counts.items()):
        lines.append(f"| `{method}` | {count} |")

    lines.extend(
        [
            "",
            "## Task / Field Summary",
            "",
            "| task_id | context_field | method | sub-contexts | total chars | avg chars |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            f"| `{row['task_id']}` | `{row['context_field']}` | `{row['method']}` | {row['sub_context_count']} | {row['total_chars']} | {row['avg_chars']} |"
        )

    if run_info["llm_failures"]:
        lines.extend(["", "## LLM Fallbacks", ""])
        for failure in run_info["llm_failures"]:
            lines.append(f"- `{failure['task_id']}`: {failure['error']}")

    lines.extend(
        [
            "",
            "## Runtime Interpretation",
            "",
            "Each output row is intended to be stored as an independent SGLang prefill/cache block.",
            "The target attention policy remains `no_pairwise_attention_between_subcontexts`; long fields are split only after field classification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="auto")
    parser.add_argument("--model", default=os.environ.get("DEFAULT_MODEL") or os.environ.get("OPENAI_MODEL") or "openai/gpt-5.4")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-trace-chars", type=int, default=18000)
    parser.add_argument("--max-trace-block-chars", type=int, default=4000)
    parser.add_argument("--max-field-chars", type=int, default=4000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-workspace-files-per-task", type=int, default=8)
    parser.add_argument("--disable-response-format", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, run_info = build_rows(args)
    summary = summarize(rows)

    output_jsonl = args.output_dir / "wildclaw_llm_trace_field_subcontexts.jsonl"
    summary_csv = args.output_dir / "wildclaw_llm_trace_field_summary.csv"
    run_info_json = args.output_dir / "wildclaw_llm_trace_field_run.json"
    report = args.output_dir / "wildclaw_llm_trace_field_report.md"

    write_jsonl(output_jsonl, rows)
    write_csv(summary_csv, summary)
    run_info_json.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(report, rows, summary, run_info)

    print("LLM trace field segmentation")
    print("============================")
    print(f"Rows: {len(rows)}")
    print(f"Used LLM: {run_info['used_llm']}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
