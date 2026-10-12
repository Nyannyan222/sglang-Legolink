"""The pre-Link host-catalog guard against a peer's residency recovery (the Job768 race).

A request that already holds a plan reaches ``link()``'s strict guard (a COMPILE decision whose cache key is not in the host catalog raises)
while a peer's recovery has emptied that very catalog entry on purpose and is putting the block back. That miss is not an invalid plan: the
guard must join the per-(runtime, cache key) recovery, wait for its owner, and go on to the strict runtime Link, whose source preflight is the
residency check. A miss no recovery accounts for keeps the strict failure. The agents here are real ``NOCSGLangLLM`` objects on threads sharing
one host catalog, against a strict fake runtime; B is held between its compile step and the guard so the interleaving is exact, not lucky.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_residency_singleflight as sf  # noqa: E402
from test_residency_singleflight import fresh_coordinator  # noqa: E402,F401  (autouse fixture)
from noc_context_manager.compile_policy import CacheCatalog  # noqa: E402
from noc_context_manager.residency_recovery import FAILED, OWNER, SETTLED, WAITER, ResidencyRecoveryCoordinator  # noqa: E402
from noc_context_manager.sglang_adapter import NOCSGLangAdapter, SGLangRequestError  # noqa: E402

base, runner = sf.base, sf.runner


class WatchedCatalog(CacheCatalog):
    """The shared host catalog, recording who invalidated what and when."""

    def __init__(self) -> None:
        super().__init__()
        self.invalidations: list[str] = []

    def invalidate(self, cache_key: str) -> None:
        self.invalidations.append(cache_key)
        super().invalidate(cache_key)


def two_agents(tmp_path, runtime, catalog, tags=("shared", "shared")):
    """Agents with one shared catalog; every agent ran turn 1, then the runtime evicts everything (legal after release)."""

    dirs = [tmp_path / f"agent{i}" for i in range(len(tags))]
    llms = [sf.make_llm_text(path, runtime, tag, catalog=catalog) for path, tag in zip(dirs, tags)]
    for llm, path in zip(llms, dirs):
        base.step(llm, path)
    return llms, dirs, runtime.evict_all()


def hold_at_link(llm):
    """Hold the agent between its compile step and ``link()``: its plan was made while the catalog still had the block."""

    at_link, release = threading.Event(), threading.Event()
    original = llm._adapter.link

    def held(*args, **kwargs):
        at_link.set()
        assert release.wait(timeout=30), "the held request was never released"
        return original(*args, **kwargs)

    llm._adapter.link = held
    return at_link, release


def hold_inside_link(llm, *, then):
    """Hold the agent INSIDE ``link()`` (after it read its observation point, before the strict guard); ``then()`` runs when released."""

    inside, release = threading.Event(), threading.Event()
    original = llm._adapter._link_payload

    def held(*args, **kwargs):
        inside.set()
        assert release.wait(timeout=30), "the held request was never released"
        then()
        return original(*args, **kwargs)

    llm._adapter._link_payload = held
    return inside, release


def spawn(llm, path, out, index):
    def work():
        try:
            out[index] = base.step(llm, path)
        except BaseException as error:  # noqa: BLE001
            out[index] = error

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


def join_all(*threads):
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a request is still blocked"


# --- A / B: the owner empties the entry; the request at the guard waits, is not rejected, sends no Compile -----------------------------------------


def run_race(tmp_path, *, fail_owner=False):
    runtime = sf.ConcurrentRuntime()
    catalog = WatchedCatalog()
    (a_llm, b_llm), (a_dir, b_dir), gone = two_agents(tmp_path, runtime, catalog)
    key = gone[0]
    assert catalog.contains(key)
    before = Counter(runtime.compiles)
    catalog.invalidations.clear()
    if fail_owner:
        runtime.fail_compile_for = {key}
    b_at_link, release_b = hold_at_link(b_llm)
    out: list = [None, None]
    b_thread = spawn(b_llm, b_dir, out, 1)
    assert b_at_link.wait(timeout=30)  # B has its plan (the key was in the catalog) and has done its compile step
    runtime.compile_gate.clear()
    runtime.compile_started.clear()  # set by turn 1's Compile; it must mean A's recovery Compile
    a_thread = spawn(a_llm, a_dir, out, 0)
    assert runtime.compile_started.wait(timeout=30)  # A's Link was refused, A invalidated the entry and is inside its recovery Compile
    assert not catalog.contains(key)  # the exact state that used to make B's strict guard raise
    release_b.set()
    sf.wait_until(lambda: sf.default_coordinator().stats()["recovery_waiter_count"] == 1)
    return runtime, catalog, key, before, out, (a_thread, b_thread)


def test_a_a_request_at_the_strict_guard_waits_for_the_owner_instead_of_failing(tmp_path, fresh_coordinator):
    runtime, catalog, key, before, out, (a_thread, b_thread) = run_race(tmp_path)
    assert b_thread.is_alive() and out[1] is None  # blocked in the join, not rejected
    assert fresh_coordinator.stats()["in_flight"] == 1
    runtime.compile_gate.set()
    join_all(a_thread, b_thread)
    assert isinstance(out[0], dict) and isinstance(out[1], dict), out
    assert catalog.contains(key)  # the owner's fresh entry stands


# --- C: the owner fails -> the waiter receives the recorded failure, deterministically ------------------------------------------------------------


def test_c_when_the_owner_fails_the_request_at_the_guard_fails_with_the_recorded_reason(tmp_path, fresh_coordinator):
    runtime, catalog, key, before, out, (a_thread, b_thread) = run_race(tmp_path, fail_owner=True)
    runtime.compile_gate.set()
    join_all(a_thread, b_thread)
    owner_error, waiter_error = out
    assert isinstance(owner_error, SGLangRequestError) and "noc_residency_recovery" in str(owner_error)  # the owner raises the strict runtime abort
    assert isinstance(waiter_error, SGLangRequestError) and waiter_error.reason == "missing"
    assert str(waiter_error).startswith(f"residency recovery failed ({key}: ") and f"strict link rejected missing cache key {key}" in str(waiter_error)
    assert runtime.compiles[key] - before.get(key, 0) == 1  # the waiter did not try its own Compile
    assert fresh_coordinator.stats()["recovery_failure_count"] == 1 and fresh_coordinator.stats()["in_flight"] == 0
    assert runtime.plain_generation_payloads == []
    assert runner.classify_request_error(waiter_error)["error_class"] == "residency_recovery_failed"


# --- D: arriving just after the recovery completed -> settled, no second flight, no Compile ----------------------------------------------------------


def test_d_a_request_that_arrives_just_after_a_completed_recovery_proceeds_without_a_new_flight(tmp_path, fresh_coordinator):
    runtime = sf.ConcurrentRuntime()
    catalog = WatchedCatalog()
    (a_llm, b_llm), (a_dir, b_dir), gone = two_agents(tmp_path, runtime, catalog)
    key = gone[0]
    before = Counter(runtime.compiles)
    # B reads its observation point at the top of link(), then spends time building its payload; the recovery of the key completes in that window.
    # B's catalog view then lacks the entry (modelled by dropping it, as a stale view would): it must not start a recovery of its own.
    inside, release_b = hold_inside_link(b_llm, then=lambda: CacheCatalog.invalidate(catalog, key))  # bypasses the recorder: only the code under test is recorded
    out: list = [None, None]
    b_thread = spawn(b_llm, b_dir, out, 1)
    assert inside.wait(timeout=30)
    base.step(a_llm, a_dir)  # A runs its whole turn: Link refused, owner, Compile, Link served
    assert fresh_coordinator.stats()["recovery_owner_count"] == 1 and fresh_coordinator.stats()["in_flight"] == 0
    catalog.invalidations.clear()
    release_b.set()
    join_all(b_thread)
    row = out[1]
    assert isinstance(row, dict), row
    recovery = row["residency_recovery"]
    assert recovery["outcome"] == "recovered_by_peer_after_source_missing" and recovery["settled_count"] == 1 and recovery["waiter_count"] == 0
    assert recovery["compiles"] == 0 and recovery["wait_time_s"] == 0
    assert runtime.compiles[key] - before.get(key, 0) == 1  # still the one Compile, A's
    assert fresh_coordinator.stats()["recovery_owner_count"] == 1  # B started no flight
    assert catalog.invalidations == []  # and B erased nothing


# --- E: a miss no recovery explains keeps the strict contract ------------------------------------------------------------------------------------


def test_e_a_genuine_catalog_miss_without_recovery_provenance_still_fails_strictly(tmp_path, fresh_coordinator):
    runtime = sf.ConcurrentRuntime()
    catalog = WatchedCatalog()
    (_a, b_llm), (_ad, b_dir), gone = two_agents(tmp_path, runtime, catalog)
    key = gone[0]
    at_link, release = hold_at_link(b_llm)
    out: list = [None, None]
    thread = spawn(b_llm, b_dir, out, 1)
    assert at_link.wait(timeout=30)
    catalog.invalidate(key)  # the entry vanishes for no recovery's reason
    other = fresh_coordinator.claim("another-runtime", key, observed=0)  # a recovery of the same key on a DIFFERENT runtime explains nothing here
    assert other.role == OWNER
    release.set()
    join_all(thread)
    error = out[1]
    assert isinstance(error, SGLangRequestError) and error.reason == "missing"
    assert str(error) == f"strict link rejected missing cache key {key}"  # exactly the historical strict failure
    assert fresh_coordinator.stats()["recovery_waiter_count"] == 0 and fresh_coordinator.stats()["recovery_settled_count"] == 0
    assert runtime.plain_generation_payloads == []
    assert runner.classify_request_error(error)["error_class"] == "host_catalog_missing_cache_key"
    assert runner.classify_request_error(error)["missing_cache_keys"] == [key]


def test_the_owner_thread_never_joins_its_own_flight_and_failures_are_remembered_only_for_requests_that_began_before_them():
    coordinator = ResidencyRecoveryCoordinator(wait_timeout_s=5)
    observed = coordinator.completions
    owner = coordinator.claim("rt", "K", observed=observed)
    assert owner.role == OWNER and coordinator.join("rt", "K", observed=observed) is None  # this thread owns it: its own Compile must not wait for itself
    other = {}
    thread = threading.Thread(target=lambda: other.setdefault("ticket", coordinator.join("rt", "K", observed=observed)), daemon=True)
    thread.start()
    thread.join(5)
    assert other["ticket"].role == WAITER  # any other thread joins
    coordinator.complete(owner, ok=False, error="compile refused")
    failed = coordinator.join("rt", "K", observed=observed)
    assert failed.role == FAILED and failed.error == "compile refused"
    assert coordinator.join("rt", "K", observed=coordinator.completions) is None  # a request that began after the failure has no provenance
    again = coordinator.claim("rt", "K", observed=coordinator.completions)
    coordinator.complete(again, ok=True)
    assert coordinator.join("rt", "K", observed=observed).role == SETTLED  # a success after `observed` supersedes the earlier failure


class _CompileStub(NOCSGLangAdapter):
    """Only what ``compile`` touches before it would send anything."""

    def __init__(self, coordinator, transport):
        self.residency_coordinator = coordinator
        self.patched_transport = transport
        self.cache_catalog = CacheCatalog()
        self.duplicate_compile_avoided = 0

    def _standalone_compile_segment(self, value):
        return None


class _Value:
    cache_key, sub_context_id, semantic_identity = "K", "sc-K", "si-K"


def test_a_compile_of_a_key_a_peer_is_recovering_sends_no_second_compile():
    coordinator = ResidencyRecoveryCoordinator(wait_timeout_s=10)
    sent: list = []
    stub = _CompileStub(coordinator, lambda payload: sent.append(payload))
    owner = coordinator.claim(stub.residency_runtime_id, "K", observed=0)  # claimed here, but only a DIFFERENT thread calls compile()
    out: list = []
    thread = threading.Thread(target=lambda: out.append(stub.compile(_Value())), daemon=True)
    thread.start()
    sf.wait_until(lambda: coordinator.stats()["recovery_waiter_count"] == 1)
    assert thread.is_alive() and sent == []
    coordinator.complete(owner, ok=True)
    thread.join(10)
    assert out == [({}, 0.0, False)] and sent == [] and stub.duplicate_compile_avoided == 1
    assert stub.__dict__["_residency_prejoin"]["waiters"] == 1  # recorded for the request's link() to report


# --- F: different keys stay independent ---------------------------------------------------------------------------------------------------------------


def test_f_requests_at_the_guard_wait_only_for_their_own_key(tmp_path, fresh_coordinator):
    class KeyGated(sf.ConcurrentRuntime):
        def __init__(self):
            super().__init__()
            self.key_gates: dict[str, threading.Event] = {}

        def __call__(self, payload):
            key = payload.get("extra_key")
            if payload.get("pic_link") is None and key in self.key_gates:
                self.compile_gate.set()  # the base class waits on the shared gate; this subclass gates per key instead
                with self.lock:
                    self.compiles[key] += 1
                    self.payloads.append(payload)
                assert self.key_gates[key].wait(timeout=30)
                with self.lock:
                    self.resident.add(key)
                return self._answer("compiled")
            return super().__call__(payload)

    runtime = KeyGated()
    catalog = WatchedCatalog()
    llms, dirs, gone = two_agents(tmp_path, runtime, catalog, tags=("alpha", "alpha", "beta", "beta"))
    assert len(gone) == 2
    (a_x, b_x, a_y, b_y), (da_x, db_x, da_y, db_y) = llms, dirs
    key_gates = {k: threading.Event() for k in gone}
    runtime.key_gates = key_gates
    at = [hold_at_link(b) for b in (b_x, b_y)]
    out: list = [None] * 4
    threads_b = [spawn(b_x, db_x, out, 1), spawn(b_y, db_y, out, 3)]
    for at_link, _ in at:
        assert at_link.wait(timeout=30)
    threads_a = [spawn(a_x, da_x, out, 0), spawn(a_y, da_y, out, 2)]
    sf.wait_until(lambda: all(runtime.compiles[k] >= 2 for k in gone))  # turn 1's Compile plus each owner's recovery Compile
    for _, release in at:
        release.set()
    sf.wait_until(lambda: fresh_coordinator.stats()["recovery_waiter_count"] == 2)
    assert fresh_coordinator.stats()["in_flight"] == 2
    # open ONE key: its owner and its waiter finish while the other key's group is still blocked
    opened = sorted(gone)[0]
    key_gates[opened].set()
    first_group_done = lambda: sum(isinstance(item, dict) for item in out) == 2  # noqa: E731
    sf.wait_until(first_group_done)
    assert sum(isinstance(item, dict) for item in out) == 2 and sum(item is None for item in out) == 2  # the other pair has not been released
    for event in key_gates.values():
        event.set()
    join_all(*threads_a, *threads_b)
    assert all(isinstance(item, dict) for item in out), out
    stats = fresh_coordinator.stats()
    assert stats["recovery_owner_count"] == 2 and stats["recovery_waiter_count"] == 2 and stats["duplicate_recovery_compile_count"] == 0
    assert sorted(Counter(catalog.invalidations).values()) == [1, 1]  # one invalidation per key, by its owner


# --- G: terminal missing-cache-key errors are part of the end-to-end correctness summary -----------------------------------------------------------


JOB768_MESSAGE = "strict link rejected missing cache key 91dd511a291d4a2a374592c6737fa1649d67087cfb7aa6d52a0fbe71fb5ea9ca"


def load_summarizer():
    spec = importlib.util.spec_from_file_location("summarize_h200_coherence", ROOT / "scripts" / "summarize_h200_multi_agent_reuse.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def error_rows(count, phase="measured"):
    return [{"cell_id": "cell", "phase": phase, **runner.classify_request_error(SGLangRequestError(JOB768_MESSAGE, reason="missing"))} for _ in range(count)]


