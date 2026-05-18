#!/usr/bin/env python3
"""Validate field/function semantic sub-context artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "method",
    "task_id",
    "context_field",
    "functional_role",
    "sub_context",
    "source_spans",
    "split_strategy",
    "sub_context_chars",
    "content_hash",
    "semantic_independence_target",
    "attention_policy_target",
    "sglang_storage_plan",
}

ALLOWED_CONTEXT_FIELDS = {
    "user_prompt",
    "tool_definitions",
    "tool_history_output",
    "conversation_history",
    "workspace_evidence",
    "task_metadata",
}

REQUIRED_SGLANG_FIELDS = {
    "prefill_unit_id",
    "cache_key",
    "cache_scope",
    "can_prefill_independently",
    "requires_block_isolation_attention_mask",
    "kv_reuse_granularity",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at line {line_no}: {exc}") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


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


def fail(checks: list[dict[str, Any]], name: str, reason: str, row_id: str = "", line_no: int | str = "") -> None:
    checks.append(
        {
            "status": "fail",
            "check": name,
            "row_id": row_id,
            "line_no": line_no,
            "reason": reason,
        }
    )


def pass_check(checks: list[dict[str, Any]], name: str, reason: str = "") -> None:
    checks.append({"status": "pass", "check": name, "row_id": "", "line_no": "", "reason": reason})


def validate_rows(rows: list[dict[str, Any]], max_unsplit_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    ids = [str(row.get("id", "")) for row in rows]
    prefill_ids = [str((row.get("sglang_storage_plan") or {}).get("prefill_unit_id", "")) for row in rows]
    cache_keys = [str((row.get("sglang_storage_plan") or {}).get("cache_key", "")) for row in rows]

    if rows:
        pass_check(checks, "non_empty_artifact", f"{len(rows)} rows")
    else:
        fail(checks, "non_empty_artifact", "No rows found")

    id_counts = Counter(ids)
    duplicate_ids = [value for value, count in id_counts.items() if count > 1]
    if duplicate_ids:
        for duplicate in duplicate_ids:
            fail(checks, "unique_ids", "Duplicate id", duplicate)
    else:
        pass_check(checks, "unique_ids", f"{len(ids)} unique ids")

    prefill_counts = Counter(prefill_ids)
    duplicate_prefill = [value for value, count in prefill_counts.items() if count > 1]
    if duplicate_prefill:
        for duplicate in duplicate_prefill:
            fail(checks, "unique_prefill_unit_ids", "Duplicate prefill_unit_id", duplicate)
    else:
        pass_check(checks, "unique_prefill_unit_ids", f"{len(prefill_ids)} unique prefill units")

    for row in rows:
        row_id = str(row.get("id", ""))
        line_no = row.get("_line_no", "")
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            fail(checks, "required_fields", f"Missing fields: {', '.join(missing)}", row_id, line_no)

        context_field = row.get("context_field")
        if context_field not in ALLOWED_CONTEXT_FIELDS:
            fail(checks, "allowed_context_field", f"Unexpected context_field: {context_field}", row_id, line_no)

        text = str(row.get("sub_context", ""))
        if not text.strip():
            fail(checks, "non_empty_sub_context", "sub_context is empty", row_id, line_no)

        if int(row.get("sub_context_chars") or -1) != len(text):
            fail(checks, "sub_context_chars_match", "sub_context_chars does not match text length", row_id, line_no)

        if row.get("content_hash") != content_hash(text):
            fail(checks, "content_hash_match", "content_hash does not match normalized sub_context", row_id, line_no)

        if row.get("semantic_independence_target") != "field_or_function_isolated":
            fail(checks, "semantic_independence_target", "Unexpected semantic_independence_target", row_id, line_no)

        if row.get("attention_policy_target") != "no_pairwise_attention_between_subcontexts":
            fail(checks, "attention_policy_target", "Unexpected attention_policy_target", row_id, line_no)

        source_spans = row.get("source_spans")
        if not isinstance(source_spans, list) or not source_spans:
            fail(checks, "source_spans_present", "source_spans is missing or empty", row_id, line_no)

        sglang_plan = row.get("sglang_storage_plan")
        if not isinstance(sglang_plan, dict):
            fail(checks, "sglang_storage_plan_shape", "sglang_storage_plan is not an object", row_id, line_no)
            continue

        missing_sglang = sorted(REQUIRED_SGLANG_FIELDS - set(sglang_plan))
        if missing_sglang:
            fail(checks, "sglang_required_fields", f"Missing SGLang fields: {', '.join(missing_sglang)}", row_id, line_no)

        if sglang_plan.get("cache_key") != row.get("content_hash"):
            fail(checks, "cache_key_matches_content_hash", "cache_key differs from content_hash", row_id, line_no)

        if sglang_plan.get("cache_scope") != "sub_context_block":
            fail(checks, "cache_scope", "cache_scope is not sub_context_block", row_id, line_no)

        if sglang_plan.get("can_prefill_independently") is not True:
            fail(checks, "can_prefill_independently", "can_prefill_independently is not true", row_id, line_no)

        if sglang_plan.get("requires_block_isolation_attention_mask") is not True:
            fail(checks, "requires_block_isolation_attention_mask", "requires_block_isolation_attention_mask is not true", row_id, line_no)

        if sglang_plan.get("kv_reuse_granularity") != "sub_context":
            fail(checks, "kv_reuse_granularity", "kv_reuse_granularity is not sub_context", row_id, line_no)

        split_strategy = row.get("split_strategy")
        if len(text) > max_unsplit_chars and split_strategy == "none":
            fail(checks, "overlong_fields_split", "Long sub_context was not split", row_id, line_no)

    already_failed = {item["check"] for item in checks if item["status"] == "fail"}
    aggregate_checks = [
        "required_fields",
        "allowed_context_field",
        "non_empty_sub_context",
        "sub_context_chars_match",
        "content_hash_match",
        "semantic_independence_target",
        "attention_policy_target",
        "source_spans_present",
        "sglang_storage_plan_shape",
        "sglang_required_fields",
        "cache_key_matches_content_hash",
        "cache_scope",
        "can_prefill_independently",
        "requires_block_isolation_attention_mask",
        "kv_reuse_granularity",
        "overlong_fields_split",
    ]
    for name in aggregate_checks:
        if name not in already_failed:
            pass_check(checks, name)

    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    by_field = Counter()
    split_count = 0
    for row in rows:
        by_task[str(row.get("task_id", ""))][str(row.get("context_field", ""))] += 1
        by_field[str(row.get("context_field", ""))] += 1
        if row.get("split_strategy") != "none":
            split_count += 1

    summary = {
        "row_count": len(rows),
        "unique_ids": len(set(ids)),
        "unique_prefill_unit_ids": len(set(prefill_ids)),
        "unique_cache_keys": len(set(cache_keys)),
        "field_counts": dict(sorted(by_field.items())),
        "task_field_counts": {task: dict(sorted(counter.items())) for task, counter in sorted(by_task.items())},
        "internally_split_rows": split_count,
        "failed_checks": sum(1 for item in checks if item["status"] == "fail"),
        "passed_checks": sum(1 for item in checks if item["status"] == "pass"),
    }
    return checks, summary


def write_report(path: Path, source_path: Path, checks: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    failed = [item for item in checks if item["status"] == "fail"]
    lines = [
        "# Field/Function Sub-context Validation Report",
        "",
        f"- Source JSONL: `{source_path.as_posix()}`",
        f"- Rows: {summary['row_count']}",
        f"- Unique ids: {summary['unique_ids']}",
        f"- Unique prefill units: {summary['unique_prefill_unit_ids']}",
        f"- Unique cache keys: {summary['unique_cache_keys']}",
        f"- Internally split rows: {summary['internally_split_rows']}",
        f"- Failed checks: {summary['failed_checks']}",
        "",
        "## Field Counts",
        "",
        "| context_field | rows |",
        "| --- | ---: |",
    ]
    for field_name, count in summary["field_counts"].items():
        lines.append(f"| `{field_name}` | {count} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Passing this validation means the artifact is field/function-first and SGLang-metadata-ready.",
            "- It does not prove SGLang already enforces block-isolated attention at runtime.",
            "- Runtime no-pairwise attention still requires an SGLang attention-mask / independent-prefill implementation.",
        ]
    )

    if failed:
        lines.extend(["", "## Failures", "", "| check | row_id | reason |", "| --- | --- | --- |"])
        for item in failed[:100]:
            lines.append(f"| `{item['check']}` | `{item['row_id']}` | {item['reason']} |")
    else:
        lines.extend(["", "## Result", "", "All validation checks passed."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/wildclaw_field_function_subcontexts_validation"),
    )
    parser.add_argument("--max-unsplit-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input_jsonl)
    checks, summary = validate_rows(rows, args.max_unsplit_chars)
    write_csv(args.output_dir / "field_function_subcontext_validation_checks.csv", checks)
    (args.output_dir / "field_function_subcontext_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "field_function_subcontext_validation_report.md", args.input_jsonl, checks, summary)

    print("Field/function sub-context validation")
    print("=====================================")
    print(f"Rows: {summary['row_count']}")
    print(f"Failed checks: {summary['failed_checks']}")
    print(f"Report: {args.output_dir / 'field_function_subcontext_validation_report.md'}")
    return 0 if summary["failed_checks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
