#!/usr/bin/env python3
"""Minimal Compile -> Reuse/Link -> Decode walk-through of the host pipeline against a LegoLink-only SGLang server.

It drives the real host pipeline (events -> field extraction -> deterministic segmentation -> SemanticSubContext store -> selection ->
namespace lookup -> compile policy -> link feasibility -> Compile + Link requests -> residency recovery) for five turns, then makes the
runtime forget every block and shows the host recovering. For every turn it prints each block's policy action, layout adjustment and
effective action, how many Compile requests the host sent, reused vs compiled tokens, the cache keys the Link named, and a check that the
Link carries none of the runtime-owned fields (block ranges, ``imp_indices``, ``kvlink``, recompute positions, ``mutable_start_token`` ...).

Two modes
---------
* default (no ``--api-base``): a strict *fake* runtime in this process. CPU only, no network, no model. It shows the structure of the protocol
  and exits 0 when every check holds. It says nothing about performance or about the real runtime.
* ``--api-base http://HOST:PORT/v1``: a real patched server (started with ``scripts/serve_patched.sh``). Compile requests register the blocks in
  the server's RadixCache, Link requests run ``PICRuntime.run_link`` (KV lookup, RoPE position remap, sparse boundary repair) and native decode
  produces the text printed per turn. ``--fake-http`` serves the fake runtime on a local port and points the same code at it, which exercises the
  HTTP transport without a GPU. THE REAL-SERVER MODE WAS NOT EXECUTED WHEN THIS REPOSITORY WAS PUBLISHED (no GPU was used); see docs/VALIDATION.md.

    python examples/compile_link_decode.py
    python examples/compile_link_decode.py --fake-http
    python examples/compile_link_decode.py --api-base http://127.0.0.1:30000/v1 --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from noc_context_manager.pic_link_contract import PIC_LINK_RUNTIME_OWNED_FIELDS  # noqa: E402
from strict_fake_runtime import StrictFakeRuntime, serve  # noqa: E402
from terminal_bench_integration import replay  # noqa: E402

OBJECTIVE = "fix the failing parser test: run pytest, read the tool output, continue the conversation"
SCRIPT = [
    {"prompt": OBJECTIVE, "events": [
        {"event_type": "task_instruction", "source_id": "task", "raw_text": "Fix the failing parser test. " * 40, "metadata": {}},
    ]},
    {"prompt": OBJECTIVE, "events": [
        {"event_type": "conversation_message", "source_id": "assistant-1", "raw_text": "I will run the pytest tests first. " * 40, "metadata": {"message_role": "assistant"}},
    ]},
    {"prompt": OBJECTIVE, "events": [
        {"event_type": "tool_output", "source_id": "pytest-1", "raw_text": "FAILED tests/test_parser.py::test_empty parser error " * 40, "metadata": {}},
    ]},
    # a later entry exists: assistant-1 and pytest-1 are settled and become reusable blocks; the newest entries stay in the mutable suffix
    {"prompt": OBJECTIVE, "events": [
        {"event_type": "conversation_message", "source_id": "assistant-2", "raw_text": "The parser test fails on empty input, patching it now. " * 40, "metadata": {"message_role": "assistant"}},
        {"event_type": "tool_output", "source_id": "pytest-2", "raw_text": "PASSED tests/test_parser.py::test_empty parser fixed " * 40, "metadata": {}},
    ]},
    {"prompt": OBJECTIVE, "events": []},
    {"prompt": OBJECTIVE + " (after the runtime evicted everything)", "events": []},
]


def planning_rows(metrics_row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "block": row["block_id"][-12:], "tokens": row["token_count"], "policy_action": row["policy_action"],
            "layout_adjustment": row["layout_adjustment"], "effective_action": row["effective_action"], "category": row["planner_category"],
        }
        for row in metrics_row.get("block_planning") or []
    ]


def run(args: argparse.Namespace, runtime: StrictFakeRuntime | None, *, in_process: bool) -> dict[str, Any]:
    """Drive the scripted session. ``runtime`` is the fake whose request log is inspected (None for a remote server); ``in_process`` replaces
    the HTTP transport by calling it directly, otherwise the real HTTP transport is used (against the fake served locally, or a real server)."""

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="legolink_example_"))
    workdir.mkdir(parents=True, exist_ok=True)
    replay.ensure_terminal_bench_importable()
    llm = replay.make_replay_class()(
        script=SCRIPT, model_name=args.model, api_base=args.api_base, original_api_base=args.api_base,
        variant="selected_context_radixcache_legolink", operation_mode="strict_experimental", segmentation_mode="deterministic",
        legolink_k=args.legolink_k, max_tokens=args.max_tokens, timeout=args.timeout, artifacts_root=str(workdir),
        tokenizer_mode=args.tokenizer_mode, tokenizer_api_base=args.api_base, semantic_cache_namespace_mode="logical_namespace",
        residency_recovery=True, residency_recovery_attempts=2,
    )
    if in_process:  # replace the HTTP transport, nothing else
        llm._patched_transport = runtime
        llm._original_transport = runtime
    llm.set_task_context("example-task", workdir / "metrics.jsonl")

    outputs: list[str] = []

    def step() -> dict[str, Any]:
        output, _ = llm.replay_step(workdir / "debug" / f"turn{llm.turn + 1}.json")
        outputs.append(output)
        rows = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        return rows[-1]

    report: dict[str, Any] = {"mode": "in-process strict fake runtime" if in_process else f"HTTP server {args.api_base}", "turns": []}
    for index in range(len(SCRIPT) - 1):
        seen = len(runtime.requests) if runtime is not None else 0
        row = step()
        new = runtime.requests[seen:] if runtime is not None else []
        link = next((r["pic_link"] for r in new if r.get("pic_link")), None)
        totals = (row.get("namespace_metrics") or {}).get("totals") or {}
        turn: dict[str, Any] = {
            "turn": index + 1,
            "blocks": planning_rows(row),
            "compiled_tokens": totals.get("compiled_token_count"),
            "reused_tokens": totals.get("reused_token_count"),
            "output": (outputs[-1] or "")[:120],
        }
        if runtime is not None:
            turn.update({
                "compile_requests_sent": len(runtime.compile_requests(seen)),
                "link_requests_sent": len(runtime.link_requests(seen)),
                "link_cache_keys": [key[:12] for key in (link or {}).get("cache_keys", [])],
                "runtime_owned_fields_on_the_wire": sorted(set(link or {}) & PIC_LINK_RUNTIME_OWNED_FIELDS),
            })
        report["turns"].append(turn)

    if runtime is not None:
        gone = runtime.evict_everything()
        seen = len(runtime.requests)
        row = step()  # the last scripted turn (no new events): every block it needs is gone from the runtime
        recovery = row["residency_recovery"]
        report["after_eviction"] = {
            "blocks_evicted_in_runtime": len(gone), "outcome": recovery["outcome"], "recovered_cache_keys": len(recovery["recovered_cache_keys"]),
            "compile_requests_sent": len(runtime.compile_requests(seen)),
            "plain_generation_requests_ever_sent": len(runtime.plain_generation_requests()),
        }
    return report


def checks_hold(report: dict[str, Any]) -> bool:
    if "after_eviction" not in report:  # a remote server: the wire checks need the fake's request log, so only completion is checked
        return len(report["turns"]) == len(SCRIPT) - 1
    return (
        all(not turn["runtime_owned_fields_on_the_wire"] for turn in report["turns"])
        and report["after_eviction"]["plain_generation_requests_ever_sent"] == 0
        and report["after_eviction"]["outcome"] == "recovered_after_source_missing"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-base", default=None, help="OpenAI-style base URL of a patched server (.../v1). Default: an in-process fake runtime")
    parser.add_argument("--fake-http", action="store_true", help="serve the fake runtime on a local port and use the HTTP transport against it")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="served model name (real server)")
    parser.add_argument("--tokenizer-mode", default=None, choices=("sglang", "regex"), help="sglang = the server's authoritative /v1/tokenize (default for a real server)")
    parser.add_argument("--legolink-k", type=int, default=16, help="repair-window size per block (tokens)")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--workdir", type=Path, default=None, help="where the session artifacts go (default: a temporary directory)")
    args = parser.parse_args()
    if args.fake_http and args.api_base:
        parser.error("--fake-http serves its own fake runtime; do not combine it with --api-base")

    runtime = StrictFakeRuntime() if (args.fake_http or args.api_base is None) else None
    server = None
    if args.fake_http:
        server = serve(runtime)
        args.api_base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    elif runtime is not None:
        args.api_base = "http://fake-patched/v1"
    args.tokenizer_mode = args.tokenizer_mode or ("regex" if runtime is not None else "sglang")
    try:
        report = run(args, runtime, in_process=runtime is not None and not args.fake_http)
        if args.fake_http:
            report["mode"] = f"strict fake runtime over HTTP ({args.api_base})"
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
    print(json.dumps(report, indent=2))
    return 0 if checks_hold(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
