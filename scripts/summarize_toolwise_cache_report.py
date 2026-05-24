#!/usr/bin/env python3
"""Summarize runtime replay cache metrics by context field and tool identity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_group(rows: list[dict[str, str]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = tuple(row.get(field, "") for field in key_fields)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group_rows in sorted(groups.items()):
        prompt_values = [value for row in group_rows if (value := parse_float(row.get("prompt_tokens"))) is not None]
        cached_values = [value for row in group_rows if (value := parse_float(row.get("cached_tokens"))) is not None]
        prefill_values = [value for row in group_rows if (value := parse_float(row.get("estimated_prefill_tokens"))) is not None]
        latency_values = [value for row in group_rows if (value := parse_float(row.get("latency_s"))) is not None]
        prompt_sum = sum(prompt_values)
        cached_sum = sum(cached_values)
        row = {field: value for field, value in zip(key_fields, key, strict=True)}
        row.update(
            {
                "rows": len(group_rows),
                "avg_tokens": round(avg(prompt_values) or 0.0, 2),
                "avg_cached": round(avg(cached_values) or 0.0, 2),
                "avg_prefill": round(avg(prefill_values) or 0.0, 2),
                "cached_ratio": round(cached_sum / prompt_sum, 4) if prompt_sum else "",
                "avg_latency_s": round(avg(latency_values) or 0.0, 4),
            }
        )
        summary.append(row)
    return summary


def write_report(path: Path, source_csv: Path, field_summary: list[dict[str, Any]], tool_summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Tool-wise Cache Report",
        "",
        f"- Source CSV: `{source_csv.as_posix()}`",
        "",
        "## Field Summary",
        "",
        "| field | rows | avg tokens | avg cached | avg prefill | cached ratio | avg latency s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in field_summary:
        ratio = row["cached_ratio"]
        ratio_text = f"{ratio:.1%}" if isinstance(ratio, float) else ""
        lines.append(
            f"| `{row['context_field']}` | {row['rows']} | {row['avg_tokens']} | {row['avg_cached']} | {row['avg_prefill']} | {ratio_text} | {row['avg_latency_s']} |"
        )
    lines.extend(
        [
            "",
            "## Tool-history Output Summary",
            "",
            "| tool_name | tool_type | is_mutating_tool | rows | avg tokens | avg cached | avg prefill | cached ratio | avg latency s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tool_summary:
        if row.get("context_field") != "tool_history_output":
            continue
        ratio = row["cached_ratio"]
        ratio_text = f"{ratio:.1%}" if isinstance(ratio, float) else ""
        lines.append(
            f"| `{row['tool_name']}` | `{row['tool_type']}` | `{row['is_mutating_tool']}` | {row['rows']} | {row['avg_tokens']} | {row['avg_cached']} | {row['avg_prefill']} | {ratio_text} | {row['avg_latency_s']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_tool_aware_sglang_metadata"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.runtime_csv)
    field_summary = summarize_group(rows, ["context_field"])
    tool_summary = summarize_group(rows, ["context_field", "tool_name", "tool_type", "is_mutating_tool"])
    write_csv(args.output_dir / "toolwise_field_cache_summary.csv", field_summary)
    write_csv(args.output_dir / "toolwise_cache_summary.csv", tool_summary)
    write_report(args.output_dir / "toolwise_cache_report.md", args.runtime_csv, field_summary, tool_summary)
    print("Tool-wise cache report")
    print("======================")
    print(f"Rows: {len(rows)}")
    print(f"Report: {args.output_dir / 'toolwise_cache_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
