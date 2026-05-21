#!/usr/bin/env python3
"""Summarize runtime replay cache metrics by OpenClaw context field."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "user_prompt",
    "tool_definitions",
    "conversation_history",
    "tool_history_output",
    "workspace_evidence",
    "task_metadata",
]


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


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary = []
    for field in FIELDS:
        field_rows = [row for row in rows if row.get("context_field") == field and row.get("status") == "completed"]
        prompt_values = [value for row in field_rows if (value := parse_float(row.get("prompt_tokens"))) is not None]
        cached_values = [value for row in field_rows if (value := parse_float(row.get("cached_tokens"))) is not None]
        prefill_values = [value for row in field_rows if (value := parse_float(row.get("estimated_prefill_tokens"))) is not None]
        latency_values = [value for row in field_rows if (value := parse_float(row.get("latency_s"))) is not None]
        prompt_sum = sum(prompt_values)
        cached_sum = sum(cached_values)
        summary.append(
            {
                "field": field,
                "rows": len(field_rows),
                "avg_tokens": round(avg(prompt_values) or 0.0, 2) if field_rows else "",
                "avg_cached": round(avg(cached_values) or 0.0, 2) if field_rows else "",
                "avg_prefill": round(avg(prefill_values) or 0.0, 2) if field_rows else "",
                "cached_ratio": round(cached_sum / prompt_sum, 4) if prompt_sum else "",
                "avg_latency_s": round(avg(latency_values) or 0.0, 4) if field_rows else "",
            }
        )
    return summary


def write_report(path: Path, source_csv: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Field-wise Cache Report",
        "",
        f"- Source CSV: `{source_csv.as_posix()}`",
        "",
        "| field | rows | avg tokens | avg cached | avg prefill | cached ratio | avg latency s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        ratio = row["cached_ratio"]
        ratio_text = f"{ratio:.1%}" if isinstance(ratio, float) else ""
        lines.append(
            f"| `{row['field']}` | {row['rows']} | {row['avg_tokens']} | {row['avg_cached']} | {row['avg_prefill']} | {ratio_text} | {row['avg_latency_s']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_field_sglang_metadata"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.runtime_csv)
    summary = build_summary(rows)
    output_csv = args.output_dir / "fieldwise_cache_summary.csv"
    output_report = args.output_dir / "fieldwise_cache_report.md"
    write_csv(output_csv, summary)
    write_report(output_report, args.runtime_csv, summary)
    print("Field-wise cache report")
    print("=======================")
    print(f"Rows: {len(rows)}")
    print(f"Summary CSV: {output_csv}")
    print(f"Report: {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
