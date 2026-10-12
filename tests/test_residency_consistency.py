"""Semantic reuse selection versus physical residency, after the runtime evicts.

The host catalog records *intent* ("this block was compiled"); the runtime registry is the authority on residency. Eviction after
release is legal, so a later request can select a block whose KV is gone. These tests pin what must then happen: the runtime says so
(strictly, never as plain generation), the host compiles that block again under the same identity, the accounting does not call it
reuse, and the recovery stays visible.

Runtime-side invariants (a pinned or acquired source cannot be evicted; after release it can; strict LegoLink-only refusal) are pinned by
tests/test_radix_cache_legolink.py, tests/pic/test_link_reservation_lifetime.py and tests/test_sglang_pic_runtime.py; the last test here
checks that those files still carry the cases this contract relies on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terminal_bench_integration import replay as runner  # noqa: E402
runner.ensure_terminal_bench_importable()

from noc_context_manager.sglang_adapter import SGLangRequestError  # noqa: E402


class FakeRuntime:
    """A strict LegoLink-only runtime with a residency registry: Compile registers a key, Link needs every key resident."""

    def __init__(self) -> None:
        self.resident: set[str] = set()
        self.payloads: list[dict] = []
        self.always_missing: set[str] = set()

    def __call__(self, payload):
        self.payloads.append(payload)
        link = payload.get("pic_link")
        if link is None:
            if "extra_key" not in payload:  # a plain generation request: the strict deployment refuses it
                raise SGLangRequestError("PIC LegoLink-only deployment refuses plain generation", reason="incompatible", status=400)
            self.resident.add(payload["extra_key"])
            return self._answer("compiled")
        missing = [k for k in link["cache_keys"] if k not in self.resident or k in self.always_missing]
        if missing:
            detail = {"noc_residency_recovery": {"reason": "scheduler_source_preflight_failed", "missing_cache_keys": missing,
                                                 "stale_cache_keys": [], "incompatible_cache_keys": []}}
            raise SGLangRequestError("LegoLink source preflight failed before destination KV allocation: " + ", ".join(missing) + " | " + json.dumps(detail),
                                     reason="missing", status=400)
        return self._answer('{"commands":[],"is_task_complete":true}')

    @staticmethod
    def _answer(content):
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 300, "completion_tokens": 10}}

    # eviction after release is legal: the runtime simply stops holding the block
    def evict_all(self) -> list[str]:
        gone = sorted(self.resident)
        self.resident.clear()
        return gone

    @property
    def link_payloads(self):
        return [p for p in self.payloads if p.get("pic_link")]

    @property
    def compile_payloads(self):
        return [p for p in self.payloads if not p.get("pic_link") and "extra_key" in p]

    @property
    def plain_generation_payloads(self):
        return [p for p in self.payloads if not p.get("pic_link") and "extra_key" not in p]


def corpus_task():
    text = lambda tag: f"{tag} " * 120  # noqa: E731
    return {"task_id": "t", "turns": [
        {"turn": 1, "prompt": "p1", "events": [{"event_type": "task_instruction", "source_id": "s-instruction", "raw_text": text("instruction"), "metadata": {}}]},
        {"turn": 2, "prompt": "p2", "events": [{"event_type": "conversation_message", "source_id": "s-conv", "raw_text": text("conversation"), "metadata": {"message_role": "assistant"}}]},
    ]}


def make_llm(tmp_path, runtime, *, recovery=True, attempts=2):
    replay = runner.make_replay_class()
    llm = replay(script=[{"prompt": "p1", "events": corpus_task()["turns"][0]["events"]}, {"prompt": "p2", "events": []}, {"prompt": "p3", "events": []}],
                 model_name="model", api_base="http://patched/v1", original_api_base="http://original/v1", variant="selected_context_radixcache_legolink",
                 operation_mode="strict_experimental", segmentation_mode="deterministic", legolink_k=16, max_tokens=16, timeout=1,
                 artifacts_root=str(tmp_path), tokenizer_mode="regex", semantic_cache_namespace_mode="logical_namespace",
                 residency_recovery=recovery, residency_recovery_attempts=attempts)
    llm._patched_transport = runtime
    llm._original_transport = runtime
    llm.set_task_context("t", tmp_path / "metrics.jsonl")
    return llm


def rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def step(llm, tmp_path):
    llm.replay_step(tmp_path / "debug" / f"d{llm.turn}.json")
    return rows(tmp_path)[-1]


# --- A: semantic candidate + resident -> normal reuse --------------------------------------------------------------------------------


def test_a_resident_block_is_reused_normally_and_nothing_is_recovered(tmp_path):
    runtime = FakeRuntime()
    llm = make_llm(tmp_path, runtime)
    step(llm, tmp_path)
    second = step(llm, tmp_path)
    totals = second["namespace_metrics"]["totals"]
    assert totals["reused_token_count"] > 0 and totals["compiled_token_count"] == 0 and totals["residency_recovered_token_count"] == 0
    assert second["residency_recovery"]["outcome"] == "reuse_hit" and second["residency_recovery"]["recovered_cache_keys"] == []
    assert len(runtime.link_payloads) == 2 and len(runtime.compile_payloads) == 1


# --- B: semantic candidate + evicted before the request -> explicit recovery, request succeeds, observable ---------------------------


def test_an_evicted_block_is_compiled_again_the_request_succeeds_and_the_recovery_is_visible(tmp_path):
    runtime = FakeRuntime()
    llm = make_llm(tmp_path, runtime)
    step(llm, tmp_path)
    gone = runtime.evict_all()
    assert gone  # after release, eviction is legal
    second = step(llm, tmp_path)

    recovery = second["residency_recovery"]
    assert recovery["outcome"] == "recovered_after_source_missing" and recovery["attempts"] == 1
    assert recovery["recovered_cache_keys"] == gone and recovery["compiles"] == len(gone) and recovery["enabled"] is True
    # the host compiled it again (a real extra Compile request), then the retried Link was served
    assert len(runtime.compile_payloads) == 1 + len(gone) and len(runtime.link_payloads) == 3
    assert second["newly_compiled_blocks"] == len(gone)
    assert set(runtime.resident) == set(gone)


# --- E: a missing source never becomes plain generation ----------------------------------------------------------------------------------


def test_without_recovery_a_missing_source_raises_and_never_executes_plain_generation(tmp_path):
    runtime = FakeRuntime()
    llm = make_llm(tmp_path, runtime, recovery=False)
    step(llm, tmp_path)
    runtime.evict_all()
    with pytest.raises(SGLangRequestError) as caught:
        step(llm, tmp_path)
    assert caught.value.reason == "missing"
    assert runtime.plain_generation_payloads == []


def test_when_recovery_cannot_heal_the_request_fails_after_the_bound_and_still_never_falls_back(tmp_path):
    runtime = FakeRuntime()
    llm = make_llm(tmp_path, runtime, attempts=2)
    step(llm, tmp_path)
    runtime.always_missing = set(runtime.resident)  # every recompile is evicted again before the Link
    with pytest.raises(SGLangRequestError):
        step(llm, tmp_path)
    assert runtime.plain_generation_payloads == []
    # one failed Link, then at most `attempts` recoveries each followed by a Link: bounded, never an endless loop
    assert len(runtime.link_payloads) <= 1 + 1 + 2


# --- F: recovery must not fabricate reuse accounting ----------------------------------------------------------------------------------------


def test_recovered_tokens_count_as_compiled_never_as_reused(tmp_path):
    control_runtime = FakeRuntime()
    (tmp_path / "control").mkdir()
    control = make_llm(tmp_path / "control", control_runtime)
    step(control, tmp_path / "control")
    control_row = step(control, tmp_path / "control")
    control_totals = control_row["namespace_metrics"]["totals"]
    assert control_totals["reused_token_count"] > 0

    runtime = FakeRuntime()
    llm = make_llm(tmp_path, runtime)
    step(llm, tmp_path)
    runtime.evict_all()
    row = step(llm, tmp_path)
    totals = row["namespace_metrics"]["totals"]
    assert totals["reused_token_count"] == 0
    assert totals["compiled_token_count"] == control_totals["reused_token_count"]
    assert totals["residency_recovered_token_count"] == control_totals["reused_token_count"] and totals["residency_recovered_blocks"] == 1
    assert totals["physical_kv_reused"] == 0
    counts = row["reuse_accounting"]["counts"]
    value = lambda name: counts[name]["value"]  # noqa: E731
    assert value("linked_cached_block_count") == 0 and value("compiled_block_count") == 1
    assert value("residency_recovered_block_count") == 1 and value("residency_recovered_tokens") == totals["residency_recovered_token_count"]
    assert value("compile_tokens") == totals["compiled_token_count"]
    # the unrecovered control stays exactly as before: nothing in it is attributed to recovery
    control_counts = control_row["reuse_accounting"]["counts"]
    assert control_counts["residency_recovered_block_count"]["value"] == 0 and control_counts["linked_cached_block_count"]["value"] == 1


def test_without_a_recovered_key_the_accounting_function_is_unchanged():
    from noc_context_manager.cache_namespace import summarize_namespace_request

    assert summarize_namespace_request([], linked_block_ids=[], link_cache_key_by_block={})["totals"]["residency_recovered_token_count"] == 0


# --- the experiment runner: rows, error classes, summary ------------------------------------------------------------------------------------------


def cli_args(**extra):
    base = dict(model="model", api_base="http://patched/v1", native_api_base="http://original/v1", tokenizer_api_base=None, tokenizer_mode="regex", max_tokens=16, timeout=5,
                server_trace_path=None, trace_pull_cmd=None, memory_sample_cmd=None, label="TEST", restart_patched_cmd=None, restart_native_cmd=None, warmup_task=None,
                keep_debug=1, allow_stale=False)
    base.update(extra)
    return argparse.Namespace(**base)


class EvictingRuntime(FakeRuntime):
    """Evicts everything between a session's turns, as a pool-capped server does once the blocks are released."""

    def __call__(self, payload):
        if payload.get("pic_link") and self.link_payloads and not getattr(self, "_evicted", False):
            self._evicted = True
            self.evict_all()
        return super().__call__(payload)


def run_repeated_cell(tmp_path, monkeypatch, **extra):
    runtime = EvictingRuntime()
    original = runner.make_replay_class()

    class Fake(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._patched_transport = runtime
            self._original_transport = runtime

    monkeypatch.setattr(runner, "make_replay_class", lambda: Fake)
    corpus = {"a": corpus_task(), "w": {**corpus_task(), "task_id": "w", "turns": corpus_task()["turns"][:1]}}
    cell = runner.Cell("repeated", "logical_namespace", 16, 3)
    meta = runner.execute_cell(cell=cell, phase="measured", repeat=0, args=cli_args(**extra), corpus=corpus, run_dir=tmp_path, trace=runner.TraceReader(None, None))
    base = tmp_path / "cells" / cell.cell_id / "measured_0"
    return meta, runner.read_jsonl(base / "requests.jsonl"), runner.read_jsonl(base / "errors.jsonl"), runtime


def test_a_non_residency_failure_is_a_runtime_error():
    assert runner.classify_request_error(RuntimeError("CUDA out of memory"))["error_class"] == "runtime_error"
    assert runner.classify_request_error(RuntimeError("HTTP 500"), body='{"message":"boom"}')["error_class"] == "runtime_error"


RECOVERED_ROW = {"arm": "logical_namespace", "residency_outcome": "recovered_after_source_missing", "residency_recovery_enabled": True, "residency_recovery_attempts": 1,
                 "recovered_cache_keys": ["dd88"], "recovered_blocks": 1, "recovered_token_count": 817, "compile_keys": ["dd88"], "compile_count": 1,
                 "compiled_token_count": 817}
PLAIN_ROW = {"arm": "logical_namespace", "residency_outcome": "reuse_hit", "residency_recovery_enabled": True, "residency_recovery_attempts": 0,
             "recovered_cache_keys": [], "recovered_blocks": 0, "recovered_token_count": 0}


def test_the_runtime_tests_this_contract_relies_on_are_still_there():
    """Invariants 2, 3, 6 and the acquire/release balance live in the runtime test files; keep them from being quietly deleted."""

    if not (ROOT / "tests" / "test_radix_cache_legolink.py").is_file():
        pytest.skip("the runtime test files are not part of the reproducibility bundle")
    radix = (ROOT / "tests" / "test_radix_cache_legolink.py").read_text(encoding="utf-8")
    assert "test_eviction_skips_pinned_block_then_frees_once_after_release" in radix
    assert "test_remove_legolink_block_respects_pin_refcount_and_counts_eviction_once" in radix
    lifetime = (ROOT / "tests" / "pic" / "test_link_reservation_lifetime.py").read_text(encoding="utf-8")
    assert "reserve" in lifetime.lower() and "release" in lifetime.lower()
    runtime = (ROOT / "tests" / "test_sglang_pic_runtime.py").read_text(encoding="utf-8")
    assert "legolink_only" in runtime.lower() or "legolink-only" in runtime.lower()
