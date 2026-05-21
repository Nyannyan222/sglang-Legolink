#!/usr/bin/env python3
"""Replay WildClaw framework prompts through an SGLang OpenAI-compatible server.

The runner reads `wildclaw_phase2_sglang_runtime_manifest.jsonl`, sends each
prompt to `/v1/chat/completions`, scrapes SGLang `/metrics` before and after
each request, and writes token/cache/latency measurements.

This measures runtime behavior; it does not yet implement semantic KV reuse
inside SGLang. Use `--repeat 2` to observe ordinary prefix-cache behavior across
identical prompt replays.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


METRICS = (
    "sglang:cached_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:num_requests_total",
    "sglang:cache_hit_rate",
    "sglang:new_token_ratio",
    "sglang:num_used_tokens",
    "sglang:time_to_first_token_seconds",
    "sglang:time_to_first_token_seconds_sum",
    "sglang:time_to_first_token_seconds_count",
    "sglang:e2e_request_latency_seconds",
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:e2e_request_latency_seconds_count",
)


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


def metrics_url(base_url: str) -> str:
    base = normalize_base_url(base_url)
    if base.endswith("/v1"):
        return base[:-3] + "/metrics"
    return base + "/metrics"


def parse_prometheus(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def http_text(url: str, timeout: float) -> str:
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def scrape(url: str, timeout: float) -> dict[str, float]:
    return parse_prometheus(http_text(url, timeout))


def post_text(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, data=b"", headers={"Accept": "text/plain"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float | None:
    if name not in before and name not in after:
        return None
    return after.get(name, 0.0) - before.get(name, 0.0)


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    endpoint = normalize_base_url(base_url) + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def usage_value(data: dict[str, Any], key: str) -> int | None:
    usage = data.get("usage") or {}
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) else None


def safe_read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def cache_group_key(row: dict[str, Any], mode: str) -> str:
    if mode == "pair":
        manifest_id = row.get("id", "")
        if "::" in manifest_id:
            return manifest_id.split("::", 1)[0]
    return row.get("task_id", "") or row.get("id", "")


def should_flush(args: argparse.Namespace, row: dict[str, Any], replay_index: int, last_group: str | None) -> tuple[bool, str | None]:
    if not args.flush_cache_url or args.flush_before == "none":
        return False, last_group
    if args.flush_before == "row":
        return True, last_group
    if args.flush_before == "replay":
        return True, last_group
    if args.flush_before == "pair":
        group = cache_group_key(row, "pair")
        if replay_index == 1 and group != last_group:
            return True, group
        return False, last_group
    return False, last_group


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"wildclaw_sglang_runtime_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    answers_dir = run_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    murl = args.metrics_url or metrics_url(args.base_url)

    results: list[dict[str, Any]] = []
    last_flush_group: str | None = None
    for row in rows:
        prompt = safe_read(row["prompt_path"])
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        for replay_index in range(1, args.repeat + 1):
            status = "completed"
            error = ""
            usage: dict[str, Any] = {}
            answer = ""
            flushed_cache = False
            flush_error = ""
            do_flush, next_group = should_flush(args, row, replay_index, last_flush_group)
            if do_flush:
                try:
                    post_text(args.flush_cache_url, args.metrics_timeout)
                    flushed_cache = True
                    last_flush_group = next_group
                    time.sleep(args.flush_settle_seconds)
                except Exception as exc:
                    flushed_cache = False
                    flush_error = str(exc)
            before = scrape(murl, args.metrics_timeout)
            started = time.perf_counter()
            try:
                data = chat_completion(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    api_key=args.api_key,
                )
                usage = data.get("usage") or {}
                answer = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
            except Exception as exc:
                status = "error"
                error = str(exc)
            elapsed = time.perf_counter() - started
            after = scrape(murl, args.metrics_timeout)

            prompt_tokens = usage_value({"usage": usage}, "prompt_tokens")
            completion_tokens = usage_value({"usage": usage}, "completion_tokens")
            cached_tokens = usage_value({"usage": usage}, "cached_tokens")
            if cached_tokens is None:
                delta_cached = metric_delta(before, after, "sglang:cached_tokens_total")
                cached_tokens = int(delta_cached) if delta_cached is not None else None
            estimated_prefill_tokens = (
                prompt_tokens - cached_tokens
                if prompt_tokens is not None and cached_tokens is not None
                else None
            )
            answer_path = answers_dir / f"{hashlib.sha256((row.get('id', '') + str(replay_index)).encode('utf-8')).hexdigest()[:16]}.txt"
            answer_path.write_text(answer if answer else error, encoding="utf-8", newline="\n")

            record: dict[str, Any] = {
                "created_at": now_iso(),
                "manifest_id": row.get("id", ""),
                "task_id": row.get("task_id", ""),
                "condition": row.get("condition", ""),
                "replay_index": replay_index,
                "prompt_path": row.get("prompt_path", ""),
                "prompt_hash": prompt_hash,
                "prompt_chars": len(prompt),
                "status": status,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "estimated_prefill_tokens": estimated_prefill_tokens,
                "latency_s": round(elapsed, 4),
                "answer_chars": len(answer),
                "answer_path": answer_path.as_posix(),
                "error": error,
                "flushed_cache_before": flushed_cache,
                "flush_error": flush_error,
            }
            for name in METRICS:
                delta = metric_delta(before, after, name)
                record[f"metric_delta__{name}"] = delta if delta is not None else ""
            results.append(record)

    write_csv(run_dir / "wildclaw_sglang_runtime_results.csv", results)
    (run_dir / "wildclaw_sglang_runtime_results.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "manifest": args.manifest.as_posix(),
                "base_url": args.base_url,
                "metrics_url": murl,
                "model": args.model,
                "repeat": args.repeat,
                "flush_cache_url": args.flush_cache_url,
                "flush_before": args.flush_before,
                "rows": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return results, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark_results/wildclaw_phase2/wildclaw_phase2_sglang_runtime_manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_sglang_runtime_runs"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--model", default="default")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--metrics-timeout", type=float, default=5)
    parser.add_argument("--flush-cache-url", default="", help="Optional SGLang cache flush endpoint, for example http://127.0.0.1:30000/flush_cache")
    parser.add_argument(
        "--flush-before",
        choices=["none", "row", "replay", "pair"],
        default="none",
        help="Flush cache before each row/replay/pair. Pair mode uses the manifest id prefix before '::'.",
    )
    parser.add_argument("--flush-settle-seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results, run_dir = run(args)
    print("WildClaw SGLang runtime replay")
    print("==============================")
    print(f"Run dir: {run_dir}")
    print(f"Rows: {len(results)}")
    errors = sum(1 for row in results if row["status"] != "completed")
    print(f"Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
