"""Reuse authoritative token ids the host already owns instead of re-encoding.

EXPERIMENTAL, opt-in (``NOC_HOST_TOKEN_REUSE=1``), off by default.

Who owns what, unchanged:

* a semantic member's authoritative ids are owned by its `SemanticSubContext`
  (`token_ids`, `tokenizer_fingerprint`, `token_count`), written once by
  segmentation validation;
* a request's prefix / block / mutable / final-prompt ids and hashes are owned
  by the `NormalizedChatTokenLayout` that physical framing builds;
* KV slots, offsets, RoPE, kvlink and ``imp_indices`` are owned by the runtime
  and never appear here.

What was missing is a way for framing to *consume* the member ids instead of
encoding the same text again. :class:`RequestTokenSource` is that: an
authoritative tokenizer, one per request, that answers ``tokenize(text)`` from
ids a `SemanticSubContext` already holds for that exact text under the same
tokenizer fingerprint, and otherwise encodes once and remembers for the rest of
the request. Tokenization is a pure function of (exact text, tokenizer), so the
answer is the one the tokenizer would give, and every framing check still runs
on it -- including the check that the block ids plus the mutable ids equal a
*fresh* encoding of the whole body, which is what catches a store that lies.

:class:`SharedTokenCache` holds ids for exact immutable texts no object owns
(an execution group's concatenated text) for as long as the tokenizer
fingerprint is the same: the fingerprint is part of the key, so a tokenizer
change is a miss, not a stale hit, and nothing has to be cleared by hand.
Mutable text (the question, the assembled prompt) never enters it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import Counter, OrderedDict
from typing import Any, Mapping, Sequence

from .subcontext_store import SemanticSubContext
from .token_accounting import ModelTokenCount

FEATURE_FLAG_ENV = "NOC_HOST_TOKEN_REUSE"


def host_token_reuse_enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(FEATURE_FLAG_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class HostTokenReuseDisabledError(RuntimeError):
    pass


def refusal_reason(value: SemanticSubContext, tokenizer: Any) -> str | None:
    """Why ``value.token_ids`` may not answer for ``tokenizer``, or ``None``.

    Ids are valid only in the namespace they were produced in: the same
    tokenizer fingerprint, declared model tokens, one id per counted token.
    Same model family is not the same tokenizer; a fingerprint mismatch always
    refuses.
    """

    if value.invalidated:
        return "invalidated"
    if not value.token_ids:
        return "no_token_ids"
    if len(value.token_ids) != value.token_count:
        return "token_count_mismatch"
    fingerprint = str(getattr(tokenizer, "fingerprint", "") or "")
    if not fingerprint or value.tokenizer_fingerprint != fingerprint:
        return "tokenizer_fingerprint_mismatch"
    accounting = value.token_accounting()
    if not isinstance(accounting, ModelTokenCount):
        return "not_model_tokens"
    if accounting.tokenizer_fingerprint != fingerprint:
        return "tokenizer_fingerprint_mismatch"
    if any(type(v) is not int for v in value.token_ids):
        return "non_integer_token_ids"
    return None


class SharedTokenCache:
    """Bounded, process-wide: (tokenizer fingerprint, exact immutable text) -> ids.

    Read-only sharing: values are tuples of ints. Keys compare the full text,
    not a digest and not object identity.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], tuple[int, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, fingerprint: str, text: str) -> tuple[int, ...] | None:
        with self._lock:
            ids = self._entries.get((fingerprint, text))
            if ids is None:
                self.misses += 1
                return None
            self._entries.move_to_end((fingerprint, text))
            self.hits += 1
            return ids

    def put(self, fingerprint: str, text: str, token_ids: Sequence[int]) -> tuple[int, ...]:
        ids = tuple(int(v) for v in token_ids)
        with self._lock:
            self._entries[(fingerprint, text)] = ids
            self._entries.move_to_end((fingerprint, text))
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1
        return ids

    def __len__(self) -> int:
        return len(self._entries)

    def memory_bytes(self) -> dict[str, int]:
        """Raw id payload (4 bytes/id, what an int32 array would need) and the
        Python objects actually held (tuples, ints, key strings)."""

        with self._lock:
            items = list(self._entries.items())
        raw = sum(4 * len(ids) for _, ids in items)
        python = 0
        small = set()
        for (fingerprint, text), ids in items:
            python += sys.getsizeof(ids) + sys.getsizeof(text) + sys.getsizeof((fingerprint, text))
            for v in ids:
                if -5 <= v <= 256:
                    small.add(v)
                else:
                    python += sys.getsizeof(v)
        return {"entries": len(items), "raw_token_id_bytes": raw, "python_object_bytes": python}


def _message_key(messages: Sequence[Mapping[str, Any]], kwargs: Mapping[str, Any]) -> tuple | None:
    # Framing passes tools/tool_choice/reasoning_effort=None and
    # continue_final_message=False; a caller that passes nothing asks the same
    # question. Anything else is part of the key, and anything unusual is not
    # memoized at all.
    extra = tuple(sorted((k, repr(v)) for k, v in kwargs.items() if v is not None and v is not False))
    key = []
    for message in messages:
        if set(message) - {"role", "content"} or not isinstance(message.get("content"), str):
            return None
        key.append((message.get("role"), message["content"]))
    return tuple(key), extra


class RequestTokenSource:
    """An authoritative tokenizer for one request that never encodes a text twice.

    ``adopt`` hands it a `SemanticSubContext` whose ids answer for its exact
    ``raw_text`` (refused, and counted, if the namespace differs). Everything
    else is encoded by the wrapped tokenizer at most once per request. Texts
    marked with ``mark_immutable`` are also looked up in, and written to, the
    shared cache. Identity attributes (``token_unit``, ``model_id``,
    ``fingerprint``) are the wrapped tokenizer's, so the authoritative guard
    sees the real tokenizer.

    Not shared across requests: build one per request and drop it.
    """

    def __init__(self, inner: Any, *, shared: SharedTokenCache | None = None) -> None:
        self._inner = inner
        self._shared = shared
        self._texts: dict[str, tuple[int, ...]] = {}
        self._text_origin: dict[str, str] = {}
        self._messages: dict[tuple, tuple[int, ...]] = {}
        self._immutable: set[str] = set()
        self.hits: Counter = Counter()
        self.refusals: Counter = Counter()
        self.inner_calls: Counter = Counter()
        self.inner_chars = 0
        self.inner_tokens = 0
        self.inner_ms = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    def adopt(self, value: SemanticSubContext) -> bool:
        reason = refusal_reason(value, self._inner)
        if reason is not None:
            self.refusals[reason] += 1
            return False
        self._texts[value.raw_text] = tuple(value.token_ids)
        self._text_origin[value.raw_text] = "semantic_subcontext"
        return True

    def mark_immutable(self, text: str) -> None:
        self._immutable.add(text)

    def _encode(self, method: str, call, chars: int) -> tuple[int, ...]:
        started = time.perf_counter_ns()
        ids = tuple(call())
        self.inner_ms += (time.perf_counter_ns() - started) / 1e6
        self.inner_calls[method] += 1
        self.inner_chars += chars
        self.inner_tokens += len(ids)
        return ids

    def tokenize(self, text: str) -> tuple[int, ...]:
        ids = self._texts.get(text)
        if ids is not None:
            self.hits[self._text_origin[text]] += 1
            return ids
        if self._shared is not None and text in self._immutable:
            ids = self._shared.get(self._inner.fingerprint, text)
            if ids is not None:
                self.hits["shared_immutable"] += 1
                self._texts[text], self._text_origin[text] = ids, "request"
                return ids
        ids = self._encode("tokenize", lambda: self._inner.tokenize(text), len(text))
        if self._shared is not None and text in self._immutable:
            ids = self._shared.put(self._inner.fingerprint, text, ids)
        self._texts[text], self._text_origin[text] = ids, "request"
        return ids

    def tokenize_messages(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> tuple[int, ...]:
        key = _message_key(messages, kwargs)
        if key is not None and key in self._messages:
            self.hits["request_messages"] += 1
            return self._messages[key]
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        ids = self._encode("tokenize_messages", lambda: self._inner.tokenize_messages(messages, **kwargs), chars)
        if key is not None:
            self._messages[key] = ids
        return ids

    def counts(self) -> dict[str, Any]:
        return {"tokenizer_calls": sum(self.inner_calls.values()), **{f"calls_{k}": v for k, v in self.inner_calls.items()},
                "tokenizer_chars": self.inner_chars, "tokenizer_tokens": self.inner_tokens,
                "tokenizer_ms": round(self.inner_ms, 4), "reused": dict(self.hits), "refused": dict(self.refusals)}


class CountingTokenizer:
    """Delegating wrapper that counts what reaches the real tokenizer; used to
    measure the current path the same way as :class:`RequestTokenSource`."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.inner_calls: Counter = Counter()
        self.inner_chars = 0
        self.inner_tokens = 0
        self.inner_ms = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    def _record(self, method: str, started: int, chars: int, ids) -> None:
        self.inner_ms += (time.perf_counter_ns() - started) / 1e6
        self.inner_calls[method] += 1
        self.inner_chars += chars
        self.inner_tokens += len(ids)

    def tokenize(self, text: str):
        started = time.perf_counter_ns()
        ids = self._inner.tokenize(text)
        self._record("tokenize", started, len(text), ids)
        return ids

    def tokenize_messages(self, messages, **kwargs):
        started = time.perf_counter_ns()
        ids = self._inner.tokenize_messages(messages, **kwargs)
        self._record("tokenize_messages", started, sum(len(str(m.get("content", ""))) for m in messages), ids)
        return ids

    def counts(self) -> dict[str, Any]:
        return {"tokenizer_calls": sum(self.inner_calls.values()), **{f"calls_{k}": v for k, v in self.inner_calls.items()},
                "tokenizer_chars": self.inner_chars, "tokenizer_tokens": self.inner_tokens,
                "tokenizer_ms": round(self.inner_ms, 4)}
