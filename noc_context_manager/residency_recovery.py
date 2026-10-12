"""Single-flight coordination of host residency recovery, scoped per physical cache key.

The host catalog records *intent* and the runtime RadixCache registry is the authority on physical residency. When the runtime says a
planned source is gone, the host re-issues a Compile of content it already owns (``NOCSGLangAdapter._recover_residency``). With several
agents that share one runtime, many requests can name the same gone key at once. Without coordination each of them invalidates the
shared catalog entry (one request's invalidation erases the entry another request just re-registered) and each sends its own Compile.

This module is only the *who recompiles* decision. It holds no slot, position, layout or placement information and never talks to the
runtime; the owner of a flight still uses the ordinary Compile request, and the runtime still decides where the KV lives.

Per key ``(runtime, cache_key)`` a request is exactly one of:

* **owner** - nothing is being recovered and nothing was recovered since this request's Link began: it recompiles the block once
  (invalidate, Compile, and its own Link retry is the revalidation: the runtime's strict preflight is the only residency check there is);
* **waiter** - a recovery of this key is in flight: it waits for it and then retries its Link, with no Compile of its own;
* **settled** - a recovery of this key *completed* after this request's Link was sent, so its abort was about a state that no longer
  exists: it retries without waiting and without a Compile.

A flight that fails fails every waiter with the same recorded reason. A failure is per flight, never remembered: the next request that
finds the key missing starts a new one. Flights are independent per key, so different missing keys recover concurrently, and no lock is
held across a Compile (the registry lock guards dictionary updates only).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

OWNER, WAITER, SETTLED, FAILED = "owner", "waiter", "settled", "failed"


class ResidencyRecoveryFailed(RuntimeError):
    """The owner of a flight could not recover the key; every waiter of that flight sees this, with the same reason."""

    def __init__(self, cache_key: str, reason: str) -> None:
        super().__init__(f"residency recovery of {cache_key} failed: {reason}")
        self.cache_key = cache_key
        self.reason = reason


@dataclass
class RecoveryFlight:
    runtime_id: Hashable
    cache_key: str
    owner: str
    started_s: float
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool | None = None
    error: str | None = None
    compile_requests: int = 0
    compile_latency_s: float = 0.0
    waiters: int = 0


@dataclass(frozen=True)
class RecoveryTicket:
    role: str
    runtime_id: Hashable
    cache_key: str
    flight: RecoveryFlight | None = None
    error: str | None = None  # FAILED: the reason the recovery this request depends on was recorded with


class ResidencyRecoveryCoordinator:
    """Process-wide registry of in-flight recoveries. Cross-process coordination is out of scope (see the docs)."""

    def __init__(self, *, wait_timeout_s: float = 300.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.wait_timeout_s = float(wait_timeout_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._flights: dict[tuple[Hashable, str], RecoveryFlight] = {}
        self._last_success: dict[tuple[Hashable, str], int] = {}
        self._last_failure: dict[tuple[Hashable, str], tuple[int, str]] = {}
        self._compiled_flights: set[int] = set()
        self._local = threading.local()  # the flights THIS thread owns: an owner's own Compile must never wait on its own flight
        #: monotone count of finished flights (success or failure); read without the lock (an int read) before every Link attempt
        self.completions = 0
        self._counters = {"recovery_owner_count": 0, "recovery_waiter_count": 0, "recovery_settled_count": 0, "recovery_compile_count": 0,
                          "duplicate_recovery_compile_count": 0, "recovery_failure_count": 0}
        self._wait_time_s = 0.0

    # --- the decision -----------------------------------------------------------------------------------------------------------

    def claim(self, runtime_id: Hashable, cache_key: str, *, observed: int, owner: str = "") -> RecoveryTicket:
        """Decide this request's role for a key the runtime reported gone. ``observed`` is ``completions`` read before the Link was sent."""

        slot = (runtime_id, cache_key)
        with self._lock:
            flight = self._flights.get(slot)
            if flight is not None and not flight.done.is_set():
                flight.waiters += 1
                self._counters["recovery_waiter_count"] += 1
                return RecoveryTicket(WAITER, runtime_id, cache_key, flight)
            if self._last_success.get(slot, 0) > int(observed):
                self._counters["recovery_settled_count"] += 1
                return RecoveryTicket(SETTLED, runtime_id, cache_key)
            flight = RecoveryFlight(runtime_id, cache_key, owner, self._clock())
            self._flights[slot] = flight
            self._counters["recovery_owner_count"] += 1
            self._owned_here().add(slot)
            return RecoveryTicket(OWNER, runtime_id, cache_key, flight)

    def _owned_here(self) -> set:
        owned = getattr(self._local, "owned", None)
        if owned is None:
            owned = self._local.owned = set()
        return owned

    def join(self, runtime_id: Hashable, cache_key: str, *, observed: int) -> RecoveryTicket | None:
        """A request found the HOST catalog without a key it believes it needs, outside any runtime abort. If a recovery of that key explains
        it, say how: ``WAITER`` (one is in flight: wait for it), ``SETTLED`` (one succeeded after ``observed``: nothing to wait for) or
        ``FAILED`` (one failed after ``observed``: fail the same way its waiters do). ``None`` means no recovery accounts for the miss, so it is
        a genuine catalog miss and the caller keeps its strict contract. The thread that OWNS the flight always gets ``None``: it is the
        one emptying the entry on purpose."""

        slot = (runtime_id, cache_key)
        with self._lock:
            if slot in self._owned_here():
                return None
            flight = self._flights.get(slot)
            if flight is not None and not flight.done.is_set():
                flight.waiters += 1
                self._counters["recovery_waiter_count"] += 1
                return RecoveryTicket(WAITER, runtime_id, cache_key, flight)
            if self._last_success.get(slot, 0) > int(observed):
                self._counters["recovery_settled_count"] += 1
                return RecoveryTicket(SETTLED, runtime_id, cache_key)
            failure = self._last_failure.get(slot)
            if failure is not None and failure[0] > int(observed):
                return RecoveryTicket(FAILED, runtime_id, cache_key, error=failure[1])
            return None

    def note_compile(self, ticket: RecoveryTicket, requests: int = 1) -> int:
        """The owner is about to issue ``requests`` recovery Compile(s). A second Compile in one flight is a duplicate by definition;
        returns how many duplicates this call counted (0 unless something bypassed the single-flight)."""

        if ticket.role != OWNER or ticket.flight is None or requests <= 0:
            return 0
        with self._lock:
            self._counters["recovery_compile_count"] += requests
            duplicates = requests if id(ticket.flight) in self._compiled_flights else 0
            self._counters["duplicate_recovery_compile_count"] += duplicates
            self._compiled_flights.add(id(ticket.flight))
            return duplicates

    def complete(self, ticket: RecoveryTicket, *, ok: bool, error: str | None = None, compile_requests: int = 0, compile_latency_s: float = 0.0) -> None:
        """Owner only; always called (the adapter does it in a ``finally``), so a waiter can never be left waiting on a dead owner."""

        if ticket.role != OWNER or ticket.flight is None:
            return
        flight, slot = ticket.flight, (ticket.runtime_id, ticket.cache_key)
        with self._lock:
            if flight.done.is_set():
                return
            flight.ok, flight.error = bool(ok), (None if ok else (error or "recovery failed"))
            flight.compile_requests, flight.compile_latency_s = int(compile_requests), float(compile_latency_s)
            self.completions += 1
            if ok:
                self._last_success[slot] = self.completions
                self._last_failure.pop(slot, None)
            else:
                self._last_failure[slot] = (self.completions, flight.error or "recovery failed")
                self._counters["recovery_failure_count"] += 1
            self._owned_here().discard(slot)
            if self._flights.get(slot) is flight:
                del self._flights[slot]
            self._compiled_flights.discard(id(flight))
        flight.done.set()

    def wait(self, ticket: RecoveryTicket) -> tuple[bool, str | None, float]:
        """Block until the flight finishes. Returns ``(ok, reason, seconds waited)``; a timeout is a failure with a stated reason."""

        if ticket.role != WAITER or ticket.flight is None:
            return True, None, 0.0
        started = self._clock()
        finished = ticket.flight.done.wait(self.wait_timeout_s)
        waited = self._clock() - started
        with self._lock:
            self._wait_time_s += waited
        if not finished:
            return False, f"timed out after {self.wait_timeout_s:g}s waiting for the recovery owner", waited
        return bool(ticket.flight.ok), ticket.flight.error, waited

    # --- observability -----------------------------------------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self._counters, "recovery_wait_time_s": self._wait_time_s, "in_flight": len(self._flights), "completions": self.completions}

    def reset(self) -> None:
        with self._lock:
            self._flights.clear()
            self._last_success.clear()
            self._last_failure.clear()
            self._compiled_flights.clear()
            self._local = threading.local()
            self.completions = 0
            self._counters = {name: 0 for name in self._counters}
            self._wait_time_s = 0.0


_DEFAULT = ResidencyRecoveryCoordinator()


def default_coordinator() -> ResidencyRecoveryCoordinator:
    """The process-wide coordinator every adapter uses unless it is given one: all agents of a process share a runtime's residency."""

    return _DEFAULT


__all__ = ["FAILED", "OWNER", "SETTLED", "WAITER", "RecoveryFlight", "RecoveryTicket", "ResidencyRecoveryCoordinator", "ResidencyRecoveryFailed", "default_coordinator"]
