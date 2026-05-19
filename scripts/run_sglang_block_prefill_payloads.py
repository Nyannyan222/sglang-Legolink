#!/usr/bin/env python3
"""Replay SGLang block-prefill payloads through an OpenAI-compatible endpoint.

The input is produced by `scripts/prepare_sglang_block_runtime_plan.py`.
This runner is the plumbing test before a deeper SGLang scheduler/KV-cache
patch: it verifies that block payloads can be sent independently and records
whether the runtime reports cached tokens on repeated block requests.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PAYLOADS = Path("benchmark_results/sglang_block_runtime_plan/sglang_prefill_request_payloads.jsonl")


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
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


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def usage_int(data: dict[str, Any], name: str) -> int | None:
    value = (data.get("usage") or {}).get(name)
    return int(value) if isinstance(value, int | float) else None


def metadata(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("openclaw_sub_context")
    return value if isinstance(value, dict) else {}


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    payloads = read_jsonl(args.payloads)
    if args.limit:
        payloads = payloads[: args.limit]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"sglang_block_prefill_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = normalize_base_url(args.base_url) + "/chat/completions"

    rows: list[dict[str, Any]] = []
    for replay_index in range(1, args.repeat + 1):
        for index, payload in enumerate(payloads, start=1):
            block_meta = metadata(payload)
            status = "completed"
            error = ""
            usage: dict[str, Any] = {}
            started = time.perf_counter()
            try:
                response = post_json(endpoint, payload, args.timeout, args.api_key)
                usage = response.get("usage") or {}
            except Exception as exc:
                status = "error"
                error = str(exc)
            elapsed = time.perf_counter() - started
            prompt_tokens = usage_int({"usage": usage}, "prompt_tokens")
            cached_tokens = usage_int({"usage": usage}, "cached_tokens")
            rows.append(
                {
                    "created_at": now_iso(),
                    "replay_index": replay_index,
                    "payload_index": index,
                    "status": status,
                    "block_id": block_meta.get("prefill_unit_id", ""),
                    "cache_key": block_meta.get("cache_key", ""),
                    "context_field": block_meta.get("context_field", ""),
                    "functional_role": block_meta.get("functional_role", ""),
                    "payload_extra_key": payload.get("extra_key", ""),
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "estimated_prefill_tokens": (
                        prompt_tokens - cached_tokens
                        if prompt_tokens is not None and cached_tokens is not None
                        else None
                    ),
                    "latency_s": round(elapsed, 4),
                    "error": error,
                }
            )

    write_csv(output_dir / "sglang_block_prefill_results.csv", rows)
    (output_dir / "sglang_block_prefill_results.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "payloads": args.payloads.as_posix(),
                "base_url": args.base_url,
                "repeat": args.repeat,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/sglang_block_prefill_runs"))
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, output_dir = run(args)
    errors = sum(1 for row in rows if row["status"] != "completed")
    print("SGLang block prefill payload replay")
    print("===================================")
    print(f"Rows: {len(rows)}")
    print(f"Errors: {errors}")
    print(f"Output: {output_dir}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
