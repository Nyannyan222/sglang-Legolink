"""Session identity for the NOC OpenAI proxy.

The proxy keeps one `NOCSGLangLLM` per conversation: its collector, sub-context
store, manifests, turn counter and generation settings. Which conversation a
request belongs to is therefore a *lookup identity*, and it decides whose
selected context a request reads and writes.

That identity used to fall back to a hash of the first chat message. A message
is content, not a capability: anyone who knows or can guess the opening prompt
(or a later attempt of the same task, which sends the same one) computed the
same key and joined the running session -- reading its selected context out of
its own upstream prompt and poisoning it. Two independent trials with identical
first messages were one session by construction.

The rule here is that a session is only ever *joined* by something a caller
cannot get from the task text alone:

* an **explicit identity** the harness chose (`X-NOC-Session-ID`, task-id
  headers, the OpenAI `user` field), exactly as before; or
* an **opaque id the proxy minted** for an earlier conversation, presented back
  as an explicit identity; or
* **transcript continuation**: the request's earlier turns must equal, byte for
  byte after whitespace trimming, the conversation the proxy last served for
  that session -- including the assistant reply the proxy itself produced. An
  opening message alone proves nothing about any earlier conversation, so a
  request that carries one never joins a session; it starts a new one under a
  fresh random id.

Transcript continuation is **unique-match only**. If more than one live session
last served the same history (two concurrent trials whose transcripts are still
byte-identical), nothing in the request says which one this is, and Terminus-2 /
LiteLLM send no `user`, header or other token that could. The request then starts
a new, isolated session (`split_ambiguous`, a SAFE_SESSION_SPLIT) rather than
taking an arbitrary candidate: order-dependent choices made two such trials swap
sessions. Once their transcripts diverge, each session is again the unique match.

When none of these holds the request starts a new session. That is the safe
direction: an agent that rewrites its history loses cross-turn reuse, but it
cannot be handed someone else's context.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

DERIVED_PREFIX = "derived-"
EXPLICIT_HEADERS = ("X-NOC-Session-ID", "X-Terminal-Bench-Task-ID", "X-Task-ID")

Turn = tuple[str, str]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")[:120] or "session"


def explicit_session_identity(
    *, headers: Any, payload: Mapping[str, Any], model: str
) -> Optional[tuple[str, str]]:
    """The harness-chosen identity of a request, as ``(session_key, raw_value)``.

    The key keeps the readable, sanitized prefix the artifact directories have
    always carried and appends a digest of the *raw* identity, so two distinct
    identities that sanitize to the same text (``run/7`` and ``run:7``) are no
    longer one session. ``raw_value`` is returned so a caller can recognise an
    opaque id the proxy minted.
    """

    for header in EXPLICIT_HEADERS:
        value = headers.get(header) if hasattr(headers, "get") else None
        if value:
            raw = str(value)
            return _explicit_key(header.lower(), raw, model), raw
    user = payload.get("user")
    if isinstance(user, str) and user:
        return _explicit_key("user", user, model), user
    return None


def _explicit_key(source: str, raw: str, model: str) -> str:
    digest = hashlib.sha256(f"{source}\0{raw}\0{model}".encode()).hexdigest()[:10]
    return f"{_safe_name(f'{source}:{raw}:{model}')}-{digest}"


def conversation_digest(model: str, turns: Sequence[Turn]) -> str:
    """Digest of a conversation prefix; whitespace at the edges of a turn is ignored."""

    canonical = json.dumps(
        [model, [[role, text.strip()] for role, text in turns]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionResolution:
    session_key: str
    #: ``explicit`` | ``minted_id`` | ``continuation`` | ``new`` | ``split_ambiguous``
    source: str
    #: Whether the registry tracks this session and must be told the outcome.
    tracked: bool


@dataclass
class _TrackedSession:
    model: str
    inflight: int = 0
    expected_digest: Optional[str] = None


class ProxySessionRegistry:
    """Maps requests to sessions without ever joining one by its opening prompt."""

    def __init__(self, *, new_id: Optional[Callable[[], str]] = None) -> None:
        self._new_id = new_id or (lambda: secrets.token_hex(12))
        self._lock = threading.Lock()
        self._sessions: dict[str, _TrackedSession] = {}
        self._expected: dict[str, list[str]] = {}
        self._ambiguous_splits = 0

    # -- resolution ---------------------------------------------------------

    def resolve(
        self,
        *,
        explicit: Optional[tuple[str, str]],
        model: str,
        turns: Sequence[Turn],
    ) -> SessionResolution:
        with self._lock:
            if explicit is not None:
                explicit_key, raw = explicit
                tracked = self._sessions.get(raw)
                if tracked is not None and tracked.model == model:
                    tracked.inflight += 1
                    return SessionResolution(raw, "minted_id", True)
                return SessionResolution(explicit_key, "explicit", False)
            source = "new"
            if len(turns) >= 2:
                digest = conversation_digest(model, turns[:-1])
                # Every live session that last served exactly this history, idle
                # or busy. Byte-identical histories carry no information about
                # which conversation a request belongs to, so a session is only
                # continued when it is the *unique* claimant and is idle.
                claimants = [
                    key
                    for key in self._expected.get(digest, ())
                    if self._sessions[key].model == model
                ]
                if len(claimants) == 1 and self._sessions[claimants[0]].inflight == 0:
                    self._sessions[claimants[0]].inflight += 1
                    return SessionResolution(claimants[0], "continuation", True)
                if claimants:
                    # Several sessions (or one that is already serving this very
                    # history) match: guessing would hand this request to
                    # whichever trial happened to be first in some order, and
                    # two such trials would swap sessions. Start a new,
                    # isolated session instead (SAFE_SESSION_SPLIT).
                    source = "split_ambiguous"
                    self._ambiguous_splits += 1
            key = f"{DERIVED_PREFIX}{self._new_id()}"
            while key in self._sessions:
                key = f"{DERIVED_PREFIX}{self._new_id()}"
            self._sessions[key] = _TrackedSession(model=model, inflight=1)
            return SessionResolution(key, source, True)

    # -- outcome ------------------------------------------------------------

    def complete(
        self, resolution: SessionResolution, *, turns: Sequence[Turn], reply: str
    ) -> None:
        """Record what the session now ends with, so its next request can continue it."""

        if not resolution.tracked:
            return
        with self._lock:
            tracked = self._sessions.get(resolution.session_key)
            if tracked is None:
                return
            tracked.inflight = max(0, tracked.inflight - 1)
            self._unindex(resolution.session_key, tracked)
            digest = conversation_digest(
                tracked.model, [*turns, ("assistant", reply)]
            )
            tracked.expected_digest = digest
            self._expected.setdefault(digest, []).append(resolution.session_key)

    def abandon(self, resolution: SessionResolution) -> None:
        """The request failed; the session keeps its previous continuation point."""

        if not resolution.tracked:
            return
        with self._lock:
            tracked = self._sessions.get(resolution.session_key)
            if tracked is not None:
                tracked.inflight = max(0, tracked.inflight - 1)

    def release(self, session_key: str) -> None:
        """Forget a session and every index entry that points at it."""

        with self._lock:
            tracked = self._sessions.pop(session_key, None)
            if tracked is not None:
                self._unindex(session_key, tracked)

    def ambiguous_split_count(self) -> int:
        """How many requests were split into a new session because several sessions matched."""

        with self._lock:
            return self._ambiguous_splits

    def tracked_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def indexed_continuation_count(self) -> int:
        with self._lock:
            return sum(len(keys) for keys in self._expected.values())

    def _unindex(self, session_key: str, tracked: _TrackedSession) -> None:
        digest = tracked.expected_digest
        if digest is None:
            return
        keys = self._expected.get(digest)
        if keys is not None:
            if session_key in keys:
                keys.remove(session_key)
            if not keys:
                self._expected.pop(digest, None)
        tracked.expected_digest = None
