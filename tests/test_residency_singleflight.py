"""Concurrency-safe residency recovery: one owner per missing physical cache key, everyone else waits.

The host catalog is intent; the runtime is the authority on residency. When several agents of one process find the same planned source gone
(legal eviction after release), exactly one of them recompiles it. These tests run real ``NOCSGLangLLM`` agents on threads against a strict
fake runtime and pin: one recovery Compile per missing key however many requests miss it (A, B), independent recovery of different keys with
no global serialization (C), deterministic failure of every waiter when the owner fails (D), no overhead when nothing is missing (E), no
physical placement in the host (F), recovered work counted as Compile work once (G). The runtime-side invariants (a pinned or acquired source
is never evicted; eviction after release is legal) are pinned by tests/test_radix_cache_legolink.py and tests/pic/test_link_reservation_lifetime.py.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_residency_consistency as base  # noqa: E402
from noc_context_manager.residency_recovery import OWNER, SETTLED, WAITER, ResidencyRecoveryCoordinator, default_coordinator  # noqa: E402
from noc_context_manager.sglang_adapter import NOCSGLangAdapter, SGLangRequestError  # noqa: E402

runner = base.runner


@pytest.fixture(autouse=True)
def fresh_coordinator():
    default_coordinator().reset()
    yield default_coordinator()
    default_coordinator().reset()


class ConcurrentRuntime(base.FakeRuntime):
    """The strict fake runtime made thread-safe, with a Compile that can be held, failed, or made to meet another Compile."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.compile_gate = threading.Event()
        self.compile_gate.set()
        self.compile_barrier: threading.Barrier | None = None
        self.compiles: Counter[str] = Counter()
        self.fail_compile_for: set[str] = set()
        self.compile_started = threading.Event()

    def __call__(self, payload):
        if payload.get("pic_link") is None and "extra_key" in payload:
            key = payload["extra_key"]
            with self.lock:
                self.compiles[key] += 1
                self.payloads.append(payload)
            self.compile_started.set()
            if self.compile_barrier is not None:
                self.compile_barrier.wait(timeout=20)  # two Compiles must be in flight together or this raises BrokenBarrierError
            assert self.compile_gate.wait(timeout=30), "compile gate never opened"
            if key in self.fail_compile_for:
                raise SGLangRequestError("compile failed in the runtime", reason="incompatible", status=500)
            with self.lock:
                self.resident.add(key)
            return self._answer("compiled")
        return super().__call__(payload)


def make_llm_text(tmp_path, runtime, tag, *, attempts=2, catalog=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    events = [{"event_type": "task_instruction", "source_id": f"s-{tag}", "raw_text": f"{tag} " * 120, "metadata": {}}]
    replay = runner.make_replay_class()
    llm = replay(script=[{"prompt": "p1", "events": events}, {"prompt": "p2", "events": []}, {"prompt": "p3", "events": []}], model_name="model", api_base="http://patched/v1",
                 original_api_base="http://original/v1", variant="selected_context_radixcache_legolink", operation_mode="strict_experimental",
                 segmentation_mode="deterministic", legolink_k=16, max_tokens=16, timeout=1, artifacts_root=str(tmp_path), tokenizer_mode="regex",
                 semantic_cache_namespace_mode="logical_namespace", residency_recovery=True, residency_recovery_attempts=attempts,
                 **({"shared_cache_catalog": catalog} if catalog is not None else {}))
    llm._patched_transport = runtime
    llm._original_transport = runtime
    llm.set_task_context("t", tmp_path / "metrics.jsonl")
    return llm


def start_agents(tmp_path, runtime, tags, *, attempts=2):
    """Every agent runs turn 1 (Compile + Link); then the runtime evicts everything, which is legal after release."""

    dirs = [tmp_path / f"agent{index}" for index in range(len(tags))]
    llms = [make_llm_text(path, runtime, tag, attempts=attempts) for path, tag in zip(dirs, tags)]
    for llm, path in zip(llms, dirs):
        base.step(llm, path)
    gone = runtime.evict_all()
    return llms, dirs, gone


def run_turn_two(llms, dirs):
    """Turn 2 of every agent at once; returns the metrics row or the exception of each."""

    results: list = [None] * len(llms)
    barrier = threading.Barrier(len(llms))

    def work(index):
        barrier.wait(timeout=20)
        try:
            results[index] = base.step(llms[index], dirs[index])
        except BaseException as error:  # noqa: BLE001
            results[index] = error

    threads = [threading.Thread(target=work, args=(index,), daemon=True) for index in range(len(llms))]
    for thread in threads:
        thread.start()
    return threads, results


def finish(threads, results):
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a request is still blocked: a waiter outlived its owner"
    return results


def wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached in time")


def recompiles(runtime, before):
    return Counter({key: count - before.get(key, 0) for key, count in runtime.compiles.items() if count - before.get(key, 0)})


# --- A / B: one missing key, N requests -> one recovery Compile, one owner, N-1 waiters -----------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 6])
def test_ab_requests_missing_the_same_key_recompile_it_once_with_one_owner_and_the_rest_waiting(tmp_path, fresh_coordinator, n):
    runtime = ConcurrentRuntime()
    llms, dirs, gone = start_agents(tmp_path, runtime, ["shared"] * n)
    assert len(gone) == 1
    before = Counter(runtime.compiles)
    runtime.compile_gate.clear()  # hold the owner inside its Compile so the other requests reach the key while it is being recovered
    threads, results = run_turn_two(llms, dirs)
    wait_until(lambda: fresh_coordinator.stats()["recovery_waiter_count"] == n - 1)
    assert fresh_coordinator.stats()["in_flight"] == 1
    runtime.compile_gate.set()
    rows = finish(threads, results)

    assert all(isinstance(row, dict) for row in rows), rows
    assert recompiles(runtime, before) == Counter({gone[0]: 1})  # exactly ONE recovery Compile for the key
    outcomes = Counter(row["residency_recovery"]["outcome"] for row in rows)
    assert outcomes == Counter({"recovered_after_source_missing": 1, "recovered_by_peer_after_source_missing": n - 1})
    stats = fresh_coordinator.stats()
    assert stats["recovery_owner_count"] == 1 and stats["recovery_waiter_count"] == n - 1 and stats["recovery_compile_count"] == 1
    assert stats["duplicate_recovery_compile_count"] == 0 and stats["recovery_failure_count"] == 0 and stats["in_flight"] == 0
    owner = next(row for row in rows if row["residency_recovery"]["owner_count"])
    waiters = [row for row in rows if row is not owner]
    assert owner["residency_recovery"]["recovered_cache_keys"] == gone and owner["residency_recovery"]["compiles"] == 1
    for waiter in waiters:
        recovery = waiter["residency_recovery"]
        assert recovery["waiter_count"] == 1 and recovery["owner_count"] == 0 and recovery["compiles"] == 0 and recovery["recovered_cache_keys"] == []
        assert recovery["waited_cache_keys"] == gone and recovery["attempts"] == 1 and recovery["wait_time_s"] > 0 and recovery["duplicate_compile_count"] == 0
    assert runtime.plain_generation_payloads == [] and set(runtime.resident) == set(gone)


# --- C: two different missing keys recover independently, not one behind the other ----------------------------------------------------------------------


def test_c_different_missing_keys_recover_concurrently_each_with_its_own_owner(tmp_path, fresh_coordinator):
    runtime = ConcurrentRuntime()
    llms, dirs, gone = start_agents(tmp_path, runtime, ["alpha", "alpha", "beta", "beta"])
    assert len(gone) == 2
    before = Counter(runtime.compiles)
    # both owners' Compiles must be in the runtime at the same moment: a global lock around recovery would make this barrier break
    runtime.compile_barrier = threading.Barrier(2)
    runtime.compile_gate.clear()
    threads, results = run_turn_two(llms, dirs)
    wait_until(lambda: fresh_coordinator.stats()["recovery_waiter_count"] == 2 and fresh_coordinator.stats()["recovery_owner_count"] == 2)
    runtime.compile_gate.set()
    rows = finish(threads, results)

    assert all(isinstance(row, dict) for row in rows), rows
    assert recompiles(runtime, before) == Counter({key: 1 for key in gone})
    stats = fresh_coordinator.stats()
    assert stats["recovery_owner_count"] == 2 and stats["recovery_waiter_count"] == 2 and stats["recovery_compile_count"] == 2 and stats["duplicate_recovery_compile_count"] == 0
    by_pair = [sorted(row["residency_recovery"]["outcome"] for row in rows[:2]), sorted(row["residency_recovery"]["outcome"] for row in rows[2:])]
    assert by_pair == [["recovered_after_source_missing", "recovered_by_peer_after_source_missing"]] * 2  # one owner per key, never two for one key


# --- D: the owner fails -> every waiter fails the same way, nobody stampedes ---------------------------------------------------------------------------------------


def test_d_when_the_owner_fails_every_waiter_fails_deterministically_and_the_failure_is_not_sticky(tmp_path, fresh_coordinator):
    runtime = ConcurrentRuntime()
    llms, dirs, gone = start_agents(tmp_path, runtime, ["shared"] * 4)
    before = Counter(runtime.compiles)
    runtime.fail_compile_for = set(gone)
    runtime.compile_gate.clear()
    threads, results = run_turn_two(llms, dirs)
    wait_until(lambda: fresh_coordinator.stats()["recovery_waiter_count"] == 3)
    runtime.compile_gate.set()
    outcomes = finish(threads, results)

    assert all(isinstance(item, SGLangRequestError) for item in outcomes), outcomes
    assert recompiles(runtime, before) == Counter({gone[0]: 1})  # the waiters did not each try their own Compile
    failed_for_waiters = [str(item) for item in outcomes if "residency recovery failed" in str(item)]
    assert len(failed_for_waiters) == 3  # the three waiters; the owner raises the original strict abort
    reasons = {message.split(";")[0] for message in failed_for_waiters}
    assert len(reasons) == 1 and gone[0] in reasons.pop()  # one recorded reason, naming the key, for every waiter
    assert all(item.reason == "missing" for item in outcomes) and all("noc_residency_recovery" in str(item) for item in outcomes)
    assert fresh_coordinator.stats()["recovery_failure_count"] == 1 and fresh_coordinator.stats()["in_flight"] == 0
    assert runtime.plain_generation_payloads == []
    # not sticky: the next request that finds the key missing starts a NEW flight and recovers it
    runtime.fail_compile_for = set()
    # a WAITER still believes the key is cached (the failed owner emptied only its own catalog, and would simply plan a fresh Compile)
    waiter = next(index for index, item in enumerate(outcomes) if "residency recovery failed" in str(item))
    row = base.step(llms[waiter], dirs[waiter])
    assert row["residency_recovery"]["outcome"] == "recovered_after_source_missing"
    assert recompiles(runtime, before) == Counter({gone[0]: 2})


# --- E: nothing missing -> no single-flight activity at all ----------------------------------------------------------------------------------------------------


def test_e_resident_keys_cost_no_recovery_machinery(tmp_path, fresh_coordinator):
    runtime = ConcurrentRuntime()
    dirs = [tmp_path / f"agent{index}" for index in range(3)]
    llms = [make_llm_text(path, runtime, "shared") for path in dirs]
    for llm, path in zip(llms, dirs):
        base.step(llm, path)
    threads, results = run_turn_two(llms, dirs)  # no eviction
    rows = finish(threads, results)
    assert all(isinstance(row, dict) and row["residency_recovery"]["outcome"] == "reuse_hit" for row in rows)
    assert all(row["residency_recovery"]["owner_count"] == row["residency_recovery"]["waiter_count"] == row["residency_recovery"]["settled_count"] == 0 for row in rows)
    assert all(row["residency_recovery"]["wait_time_s"] == 0 and row["residency_recovery"]["attempts"] == 0 for row in rows)
    assert {k: v for k, v in fresh_coordinator.stats().items() if k != "recovery_wait_time_s"} == {
        "recovery_owner_count": 0, "recovery_waiter_count": 0, "recovery_settled_count": 0, "recovery_compile_count": 0,
        "duplicate_recovery_compile_count": 0, "recovery_failure_count": 0, "in_flight": 0, "completions": 0}


# --- F: the host moves no physical placement, and the runtime still protects what is acquired ------------------------------------------------------------------


def test_f_recovery_compiles_carry_no_physical_placement_and_the_runtime_pin_tests_still_exist(tmp_path, fresh_coordinator):
    runtime = ConcurrentRuntime()
    llms, dirs, gone = start_agents(tmp_path, runtime, ["shared"] * 3)
    before = len(runtime.compile_payloads)
    threads, results = run_turn_two(llms, dirs)
    finish(threads, results)
    forbidden = ("slot", "position", "imp_indices", "kv_index", "destination", "block_table", "layout")
    recovery_compiles = runtime.compile_payloads[before:]
    assert len(recovery_compiles) == 1

    def keys_of(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key)
                yield from keys_of(item)

    names = list(keys_of(recovery_compiles[0]))
    assert not [n for n in names if any(word in n.lower() for word in forbidden if word != "layout")], names
    # the Compile names a block by content identity only: the runtime alone decides where it lives
    assert recovery_compiles[0]["extra_key"] == gone[0] and recovery_compiles[0]["sub_context"]["cache_key"] == gone[0]
    radix = ROOT / "tests" / "test_radix_cache_legolink.py"
    if not radix.is_file():
        pytest.skip("the runtime test files are not part of the reproducibility bundle")
    text = radix.read_text(encoding="utf-8")
    assert "test_eviction_skips_pinned_block_then_frees_once_after_release" in text
    assert "test_remove_legolink_block_respects_pin_refcount_and_counts_eviction_once" in text


# --- G: recovered work is Compile work, counted once -------------------------------------------------------------------------------------------------------------


def test_g_the_recovered_block_is_counted_as_compile_work_once_not_once_per_request(tmp_path, fresh_coordinator):
    runtime = ConcurrentRuntime()
    llms, dirs, gone = start_agents(tmp_path, runtime, ["shared"] * 4)
    runtime.compile_gate.clear()
    threads, results = run_turn_two(llms, dirs)
    wait_until(lambda: fresh_coordinator.stats()["recovery_waiter_count"] == 3)
    runtime.compile_gate.set()
    rows = finish(threads, results)
    owner = next(row for row in rows if row["residency_recovery"]["owner_count"])
    waiters = [row for row in rows if row is not owner]
    totals = owner["namespace_metrics"]["totals"]
    assert totals["residency_recovered_blocks"] == 1 and totals["residency_recovered_token_count"] > 0 and totals["reused_token_count"] == 0
    assert totals["compiled_token_count"] >= totals["residency_recovered_token_count"] and owner["newly_compiled_blocks"] == 1
    for waiter in waiters:
        wt = waiter["namespace_metrics"]["totals"]
        # a waiter issued no Compile and recovered nothing: it adds no Compile work to the books (the owner's one Compile is the only one)
        assert wt["compiled_token_count"] == 0 and wt["residency_recovered_token_count"] == 0 and wt["residency_recovered_blocks"] == 0
        assert waiter["newly_compiled_blocks"] == 0
    assert sum(row["newly_compiled_blocks"] for row in rows) == 1
    assert sum(row["namespace_metrics"]["totals"]["residency_recovered_token_count"] for row in rows) == totals["residency_recovered_token_count"]


# --- the coordinator itself ----------------------------------------------------------------------------------------------------------------------------------------


def test_a_recovery_that_completed_after_the_link_was_sent_settles_the_late_request_without_a_compile():
    coordinator = ResidencyRecoveryCoordinator()
    observed = coordinator.completions
    owner = coordinator.claim("rt", "K", observed=observed)
    assert owner.role == OWNER
    assert coordinator.note_compile(owner) == 0
    coordinator.complete(owner, ok=True, compile_requests=1)
    late = coordinator.claim("rt", "K", observed=observed)  # its Link was sent before the recovery finished: its abort is stale
    assert late.role == SETTLED and coordinator.stats()["recovery_compile_count"] == 1
    fresh = coordinator.claim("rt", "K", observed=coordinator.completions)  # a Link sent after it, missing again: a genuinely new loss
    assert fresh.role == OWNER
    assert coordinator.claim("other-runtime", "K", observed=observed).role == OWNER  # residency is per runtime


def test_a_second_compile_inside_one_flight_is_counted_as_a_duplicate():
    coordinator = ResidencyRecoveryCoordinator()
    owner = coordinator.claim("rt", "K", observed=0)
    assert coordinator.note_compile(owner) == 0 and coordinator.note_compile(owner) == 1
    assert coordinator.stats()["duplicate_recovery_compile_count"] == 1


def test_a_waiter_never_outlives_its_owner_even_when_the_owner_dies_or_the_wait_times_out():
    coordinator = ResidencyRecoveryCoordinator(wait_timeout_s=0.2)
    owner = coordinator.claim("rt", "K", observed=0)
    waiter = coordinator.claim("rt", "K", observed=0)
    assert waiter.role == WAITER
    ok, reason, waited = coordinator.wait(waiter)  # the owner is still working: the wait is bounded and says why it failed
    assert ok is False and "timed out" in reason and waited >= 0.15  # Event.wait can return a timer tick early
    coordinator.complete(owner, ok=False, error="the owner crashed")
    waiter_two = coordinator.claim("rt", "K", observed=0)
    assert waiter_two.role == OWNER  # the dead flight is gone; a new request starts a new one
    coordinator.complete(waiter_two, ok=True)


class _StubAdapter(NOCSGLangAdapter):
    """Only the parts ``_recover_residency_singleflight`` touches."""

    def __init__(self, coordinator, *, behavior):
        self.residency_coordinator = coordinator
        self.patched_transport = behavior["runtime"]
        self.behavior = behavior
        self.cache_catalog = base_catalog()

    def _normalized_layout(self, _stitched):
        return None

    def compile(self, value, *, physical_segment=None):
        self.behavior["on_compile"](value.cache_key)
        if self.behavior.get("raises"):
            raise RuntimeError(self.behavior["raises"])
        self.cache_catalog.mark_compiled(value.cache_key, token_count=1)
        return {}, 0.0, True


def base_catalog():
    from noc_context_manager.compile_policy import CacheCatalog

    catalog = CacheCatalog()
    for key in ("K1", "K2"):
        catalog.mark_compiled(key, token_count=1)
    return catalog


class _Block:
    def __init__(self, key):
        self.cache_key, self.sub_context_id, self.semantic_identity = key, f"sc-{key}", f"si-{key}"


def detail(*keys):
    return json.dumps({"noc_residency_recovery": {"reason": "scheduler_source_preflight_failed", "missing_cache_keys": list(keys), "stale_cache_keys": [], "incompatible_cache_keys": []}})


def test_two_requests_that_each_own_one_key_and_want_the_other_do_not_deadlock():
    coordinator = ResidencyRecoveryCoordinator(wait_timeout_s=20)
    runtime = object()
    first_compiles = threading.Barrier(2)
    steps: dict[str, object] = {}

    def on_compile(_key):  # both owners are inside their Compile before either finishes: each then waits on the other's key
        first_compiles.wait(timeout=10)

    def request(name, owned_first, linked_keys):
        adapter = _StubAdapter(coordinator, behavior={"runtime": runtime, "on_compile": on_compile})
        steps[name] = adapter._recover_residency_singleflight(detail(*linked_keys), [_Block(k) for k in linked_keys], None, observed=0)

    # make the claim order deterministic: A claims K1 then K2, B claims K2 then K1 -> A owns K1 and waits on K2, B owns K2 and waits on K1
    a_claimed, b_claimed = threading.Event(), threading.Event()
    original = coordinator.claim

    def staged(runtime_id, key, *, observed, owner=""):
        ticket = original(runtime_id, key, observed=observed, owner=owner)
        if threading.current_thread().name == "A" and key == "K1":
            a_claimed.set()
            b_claimed.wait(timeout=10)
        if threading.current_thread().name == "B" and key == "K2":
            b_claimed.set()
        return ticket

    coordinator.claim = staged  # type: ignore[method-assign]
    a = threading.Thread(target=request, args=("A", "K1", ["K1", "K2"]), name="A", daemon=True)
    b = threading.Thread(target=lambda: (a_claimed.wait(timeout=10), request("B", "K2", ["K2", "K1"])), name="B", daemon=True)
    a.start(), b.start()
    a.join(30), b.join(30)
    assert not a.is_alive() and not b.is_alive(), "deadlock"
    # A owns K1 only (B had not claimed K2 yet when A did... whoever owns what, every key was compiled exactly once and nobody failed)
    assert all(step.failure is None for step in steps.values())
    assert coordinator.stats()["recovery_compile_count"] == 2 and coordinator.stats()["duplicate_recovery_compile_count"] == 0


def test_an_owner_that_raises_still_ends_its_flight():
    coordinator = ResidencyRecoveryCoordinator(wait_timeout_s=5)
    adapter = _StubAdapter(coordinator, behavior={"runtime": object(), "on_compile": lambda key: None, "raises": "boom"})
    step = adapter._recover_residency_singleflight(detail("K1"), [_Block("K1")], None, observed=0)
    assert step.recovered == () and step.compiles == 0 and step.failure is None  # the historical outcome of a refused Compile: raise the original abort
    stats = coordinator.stats()
    assert stats["in_flight"] == 0 and stats["recovery_failure_count"] == 1
    assert coordinator.claim(adapter.residency_runtime_id, "K1", observed=0).role == OWNER


# --- the rows and the summary ---------------------------------------------------------------------------------------------------------------------------------------


PEER_ROW = {"arm": "logical_namespace", "residency_outcome": "recovered_by_peer_after_source_missing", "residency_recovery_enabled": True, "residency_recovery_attempts": 1,
            "recovered_cache_keys": [], "recovered_blocks": 0, "recovered_token_count": 0, "recovery_owner_count": 0, "recovery_waiter_count": 1, "recovery_settled_count": 0,
            "recovery_compile_count": 0, "recovery_wait_time_s": 0.4, "recovery_waited_cache_keys": ["K"], "duplicate_recovery_compile_count": 0}
OWNER_ROW = {"arm": "logical_namespace", "residency_outcome": "recovered_after_source_missing", "residency_recovery_enabled": True, "residency_recovery_attempts": 1,
             "recovered_cache_keys": ["K"], "recovered_blocks": 1, "recovered_token_count": 800, "compile_keys": ["K"], "compile_count": 1, "compiled_token_count": 800,
             "recovery_owner_count": 1, "recovery_waiter_count": 0, "recovery_settled_count": 0, "recovery_compile_count": 1, "recovery_wait_time_s": 0.0,
             "recovery_waited_cache_keys": [], "duplicate_recovery_compile_count": 0}


