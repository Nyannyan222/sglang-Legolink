#!/usr/bin/env python3
"""Verify OpenClaw block metadata in SGLang cache-lookup logs.

This reads an SGLang server log after applying
`scripts/patch_sglang_cache_lookup_logging.py` and running block-prefill
payloads. It checks whether cache-lookup events include block metadata:

- `openclaw_prefill_unit_id`
- `openclaw_cache_key`
- `openclaw_context_field`
- `openclaw_extra_key_matches_cache_key`
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_json_object(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    start = line.find("{")
    if start < 0:
        return None
    try:
        obj = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def read_cache_lookup_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            obj = parse_json_object(line)
            if obj and obj.get("event") == "cache_lookup":
                events.append(obj)
    return events


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


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    with_block_id = [event for event in events if event.get("openclaw_prefill_unit_id")]
    with_cache_key = [event for event in events if event.get("openclaw_cache_key")]
    matching = [
        event
        for event in events
        if event.get("openclaw_extra_key_matches_cache_key") is True
    ]
    mismatching = [
        event
        for event in events
        if event.get("openclaw_extra_key_matches_cache_key") is False
    ]
    block_ids = sorted({str(event.get("openclaw_prefill_unit_id")) for event in with_block_id})
    return {
        "cache_lookup_events": len(events),
        "events_with_openclaw_prefill_unit_id": len(with_block_id),
        "events_with_openclaw_cache_key": len(with_cache_key),
        "events_extra_key_matches_cache_key": len(matching),
        "events_extra_key_mismatch_cache_key": len(mismatching),
        "unique_openclaw_blocks": len(block_ids),
        "metadata_complete": bool(events) and len(with_block_id) == len(events) and len(with_cache_key) == len(events),
        "extra_key_consistent": not mismatching,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/sglang_openclaw_block_log_verification"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = read_cache_lookup_events(args.log_path)
    summary = summarize(events)
    write_csv(args.output_dir / "openclaw_cache_lookup_events.csv", events)
    (args.output_dir / "openclaw_cache_lookup_summary.json").write_text(
        json.dumps(
            {
                "log_path": args.log_path.as_posix(),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("SGLang OpenClaw block metadata log verification")
    print("==============================================")
    print(f"Cache lookup events: {summary['cache_lookup_events']}")
    print(f"Events with block id: {summary['events_with_openclaw_prefill_unit_id']}")
    print(f"Events with cache key: {summary['events_with_openclaw_cache_key']}")
    print(f"Extra key mismatches: {summary['events_extra_key_mismatch_cache_key']}")
    print(f"Unique OpenClaw blocks: {summary['unique_openclaw_blocks']}")
    print(f"Output: {args.output_dir}")
    return 0 if summary["metadata_complete"] and summary["extra_key_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
