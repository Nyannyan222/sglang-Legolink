#!/usr/bin/env python3
"""Convert field sub-context rows into SGLang request metadata manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("benchmark_results/wildclaw_tool_history_trace_segmentation/wildclaw_llm_trace_field_subcontexts.jsonl")
DEFAULT_OUTPUT_DIR = Path("benchmark_results/wildclaw_field_sglang_metadata")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_trailing_line_spaces(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


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
    clean_rows = [
        {key: strip_trailing_line_spaces(value) if isinstance(value, str) else value for key, value in row.items()}
        for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def canonical_text(row: dict[str, Any]) -> str:
    existing = row.get("canonical_text")
    if isinstance(existing, str) and existing.strip():
        return normalize_text(existing)
    return normalize_text(str(row.get("sub_context", "")))


def canonical_key(row: dict[str, Any], canon: str) -> str:
    existing = row.get("canonical_key")
    if isinstance(existing, str) and existing.strip():
        return existing
    field = str(row.get("context_field", "unknown_field"))
    role = str(row.get("functional_role", "unknown_role"))
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:24]
    if field == "tool_history_output" and str(row.get("tool_name", "")).strip():
        tool_type = str(row.get("tool_type") or "unknown_tool_type")
        tool_name = str(row.get("tool_name") or "unknown_tool")
        return f"field:{field}:tool:{tool_type}:{tool_name}:{digest}"
    return f"field:{field}:{role}:{digest}"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def reuse_policy(row: dict[str, Any], canon: str) -> str:
    if row.get("context_field") == "tool_history_output" and as_bool(row.get("is_mutating_tool")):
        return "no_reuse"
    if not canon:
        return "no_reuse"
    if normalize_text(str(row.get("sub_context", ""))) == canon:
        return "exact"
    return "canonical"


def prompt_text(metadata: dict[str, Any]) -> str:
    text = metadata["canonical_text"] if metadata["reuse_policy"] == "canonical" else metadata["raw_text"]
    text = strip_trailing_line_spaces(str(text)).strip()
    header_lines = [
        "OpenClaw field sub-context prefill request.",
        "Use this block as an independent context unit.",
        "",
        f"sub_context_id: {metadata['sub_context_id']}",
        f"context_field: {metadata['context_field']}",
    ]
    if metadata.get("tool_name"):
        header_lines.append(f"tool_name: {metadata['tool_name']}")
    if metadata.get("tool_type"):
        header_lines.append(f"tool_type: {metadata['tool_type']}")
    header_lines.extend(
        [
            f"reuse_policy: {metadata['reuse_policy']}",
            f"canonical_key: {metadata['canonical_key']}",
            "",
            "TEXT:",
            text,
        ]
    )
    return "\n".join(header_lines)


def build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(row.get("sub_context", ""))
    canon = canonical_text(row)
    ckey = canonical_key(row, canon)
    plan = row.get("sglang_storage_plan") if isinstance(row.get("sglang_storage_plan"), dict) else {}
    sub_context_id = str(row.get("id") or plan.get("prefill_unit_id") or ckey)
    return {
        "sub_context_id": sub_context_id,
        "context_field": str(row.get("context_field", "")),
        "raw_text": raw_text,
        "canonical_text": canon,
        "canonical_key": ckey,
        "attention_policy": "no_pairwise_attention_between_subcontexts",
        "reuse_policy": reuse_policy(row, canon),
        "task_id": str(row.get("task_id", "")),
        "functional_role": str(row.get("functional_role", "")),
        "source_method": str(row.get("method", "")),
        "content_hash": str(row.get("content_hash") or content_hash(raw_text)),
        "tool_name": str(row.get("tool_name", "")),
        "tool_type": str(row.get("tool_type", "")),
        "tool_call_id": str(row.get("tool_call_id", "")),
        "tool_step": str(row.get("tool_step", "")),
        "tool_input_hash": str(row.get("tool_input_hash", "")),
        "tool_output_hash": str(row.get("tool_output_hash", "")),
        "is_mutating_tool": as_bool(row.get("is_mutating_tool")) if str(row.get("tool_name", "")).strip() else "",
        "tool_reuse_policy_hint": str(row.get("tool_reuse_policy_hint", "")),
    }


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(args.input_jsonl)
    metadata_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, start=1):
        meta = build_metadata(row)
        prompt_digest = hashlib.sha256(meta["sub_context_id"].encode("utf-8")).hexdigest()[:16]
        prompt_field = re.sub(r"[^A-Za-z0-9_.-]+", "_", meta["context_field"])[:40]
        prompt_path = prompts_dir / f"{index:03d}_{prompt_field}_{prompt_digest}.txt"
        prompt_path.write_text(prompt_text(meta), encoding="utf-8", newline="\n")
        metadata_rows.append(meta)
        manifest_rows.append(
            {
                "id": f"{meta['sub_context_id']}::field_prefill",
                "task_id": meta["task_id"],
                "condition": "field_prefill",
                "prompt_path": prompt_path.as_posix(),
                "sub_context_id": meta["sub_context_id"],
                "context_field": meta["context_field"],
                "canonical_key": meta["canonical_key"],
                "canonical_text": meta["canonical_text"],
                "reuse_policy": meta["reuse_policy"],
                "tool_name": meta.get("tool_name", ""),
                "tool_type": meta.get("tool_type", ""),
                "tool_call_id": meta.get("tool_call_id", ""),
                "is_mutating_tool": meta.get("is_mutating_tool", ""),
                "extra_key": meta["canonical_key"],
                "openclaw_metadata": meta,
            }
        )
    return metadata_rows, manifest_rows


def write_report(path: Path, metadata_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    tool_counts: dict[tuple[str, str, str], int] = {}
    for row in metadata_rows:
        counts[row["context_field"]] = counts.get(row["context_field"], 0) + 1
        if row.get("context_field") == "tool_history_output" and row.get("tool_name"):
            key = (str(row.get("tool_name", "")), str(row.get("tool_type", "")), str(row.get("reuse_policy", "")))
            tool_counts[key] = tool_counts.get(key, 0) + 1
    lines = [
        "# SGLang Field Metadata Manifest",
        "",
        f"- Input JSONL: `{args.input_jsonl.as_posix()}`",
        f"- Metadata rows: {len(metadata_rows)}",
        f"- Runtime manifest rows: {len(manifest_rows)}",
        "",
        "## Field Counts",
        "",
        "| context_field | rows |",
        "| --- | ---: |",
    ]
    for field, count in sorted(counts.items()):
        lines.append(f"| `{field}` | {count} |")
    if tool_counts:
        lines.extend(["", "## Tool Counts", "", "| tool_name | tool_type | reuse_policy | rows |", "| --- | --- | --- | ---: |"])
        for (tool_name, tool_type, policy), count in sorted(tool_counts.items()):
            lines.append(f"| `{tool_name}` | `{tool_type}` | `{policy}` | {count} |")

    lines.extend(
        [
            "",
            "## Metadata Contract",
            "",
            "Each manifest row carries `openclaw_metadata` with `sub_context_id`, `context_field`, `raw_text`, `canonical_text`, `canonical_key`, `attention_policy`, `reuse_policy`, and optional tool identity fields.",
            "The runtime replay runner forwards this metadata into the SGLang OpenAI-compatible request body.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_rows, manifest_rows = build(args)
    metadata_jsonl = args.output_dir / "field_sglang_metadata.jsonl"
    metadata_csv = args.output_dir / "field_sglang_metadata.csv"
    manifest_jsonl = args.output_dir / "field_sglang_runtime_manifest.jsonl"
    report = args.output_dir / "field_sglang_metadata_report.md"
    write_jsonl(metadata_jsonl, metadata_rows)
    write_csv(metadata_csv, metadata_rows)
    write_jsonl(manifest_jsonl, manifest_rows)
    write_report(report, metadata_rows, manifest_rows, args)
    print("SGLang field metadata manifest")
    print("==============================")
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Runtime manifest: {manifest_jsonl}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
