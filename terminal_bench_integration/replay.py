"""Replay driver for the NOC agent LLM: feed recorded Step-1 events to the real ``NOCSGLangLLM`` and talk to a (patched) SGLang server.

``make_replay_class()`` returns ``NOCSGLangLLM`` with only its Step 1 (turn events) replaced by a recorded script, so the rest of the pipeline
(extraction, segmentation, store, selection, policy, layout, plan, adapter, residency recovery) is the real code. It is what
``examples/compile_link_decode.py`` and the residency tests drive.

``ensure_terminal_bench_importable()`` installs a minimal, labelled shim for the ``terminal_bench`` modules the agent imports, when the real
``terminal-bench`` package is not installed (replay only uses ``NOCSGLangLLM.call``).

Extracted verbatim from the research runner this package was cut from (run_h200_multi_agent_reuse.py); no behaviour was changed.
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path
from typing import Any


def ensure_terminal_bench_importable() -> bool:
    """True if the real ``terminal_bench`` is imported, False if a minimal shim was installed.

    Replay uses only ``NOCSGLangLLM.call``; a host without Terminal-Bench (the H200 login/compute node) still needs the
    module names ``noc_sglang_agent`` imports. The shim is recorded in environment.json, never silent.
    """

    try:
        import terminal_bench.llms.base_llm  # noqa: F401
        import terminal_bench.agents.base_agent  # noqa: F401
        import terminal_bench.agents.terminus_1  # noqa: F401
        import terminal_bench.terminal.tmux_session  # noqa: F401

        return True
    except ImportError:
        pass

    class _BaseLLM:
        def __init__(self, **kwargs: Any) -> None:
            if kwargs:
                raise TypeError(f"unexpected BaseLLM kwargs: {sorted(kwargs)}")

    class _OutputLengthExceededError(Exception):
        def __init__(self, message: str, truncated_response: Any = None) -> None:
            super().__init__(message)
            self.truncated_response = truncated_response

    class _Terminus:
        def __init__(self, **kwargs: Any) -> None:
            self._llm = None

    definitions = {
        "terminal_bench": {},
        "terminal_bench.agents": {},
        "terminal_bench.agents.base_agent": {"AgentResult": type("AgentResult", (), {})},
        "terminal_bench.agents.terminus_1": {"Terminus": _Terminus},
        "terminal_bench.llms": {},
        "terminal_bench.llms.base_llm": {"BaseLLM": _BaseLLM, "OutputLengthExceededError": _OutputLengthExceededError},
        "terminal_bench.terminal": {},
        "terminal_bench.terminal.tmux_session": {"TmuxSession": type("TmuxSession", (), {})},
    }
    for name, attrs in definitions.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
    return False


def make_replay_class():
    ensure_terminal_bench_importable()
    from noc_context_manager.session import EventType
    from terminal_bench_integration.noc_sglang_agent import NOCSGLangLLM

    class ReplayLLM(NOCSGLangLLM):
        """The real agent LLM object, with Step 1 replaced by the recorded Step-1 output of the real run."""

        def __init__(self, *, script: list[dict[str, Any]], cold_native: bool = False, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._script = script
            self._cursor = 0
            self._cold_native = cold_native

        def _turn_events(self, prompt, history, **_ignored):  # noqa: ANN001
            step = self._script[self._cursor]
            self._cursor += 1
            return [(EventType(e["event_type"]), e["source_id"], e["raw_text"], dict(e.get("metadata") or {})) for e in step["events"]]

        def replay_step(self, logging_path: Path) -> tuple[str, dict[str, Any]]:
            step = self._script[self._cursor]
            if self._cold_native and self._adapter is not None:
                self._adapter.generation["extra_key"] = f"cold-{uuid.uuid4().hex}"
            length_error = sys.modules["terminal_bench.llms.base_llm"].OutputLengthExceededError
            try:
                output = self.call(step["prompt"], message_history=[], logging_path=logging_path)
            except length_error as error:
                # A reply cut at max_tokens is a completed request (its artifact and metrics row are written before the agent
                # raises); replay only needs the prefill and the first max_tokens of decode, and it equalizes decode across arms.
                output = getattr(error, "truncated_response", "") or ""
            return output, step

    return ReplayLLM


def classify_request_error(error: BaseException, body: str = "") -> dict[str, Any]:
    """Name what kind of failure a request raised, so an unrecovered eviction is never filed as a generic runtime error.

    ``unrecovered_source_missing``: the runtime's source preflight named keys that are not resident (missing / stale / incompatible)
    and the host did not (or could not) recover. ``host_catalog_missing_cache_key``: the host's strict pre-Link guard found a planned
    block absent from its catalog (Job768: a peer's recovery had emptied the entry). ``residency_recovery_failed``: a recovery this
    request depended on failed. ``runtime_error``: anything else.
    """

    from noc_context_manager.sglang_adapter import NOCSGLangAdapter

    text = f"{error} {body}"
    detail = NOCSGLangAdapter.residency_recovery_detail(body or str(error))
    if not any(detail.values()):
        detail = NOCSGLangAdapter.residency_recovery_detail(str(error))
    named = [key for group in ("missing", "stale", "incompatible") for key in detail[group]]
    if named or "source preflight failed" in text.lower():
        return {"error_class": "unrecovered_source_missing", "missing_cache_keys": list(detail["missing"]), "stale_cache_keys": list(detail["stale"]),
                "incompatible_cache_keys": list(detail["incompatible"])}
    import re

    lowered = text.lower()
    if "residency recovery failed" in lowered:  # a recovery this request waited on (before its Link) failed
        found = re.search(r"residency recovery failed \(([0-9a-zA-Z_.:-]+):", text)
        return {"error_class": "residency_recovery_failed", "missing_cache_keys": [found.group(1)] if found else []}
    if "strict link rejected missing cache key" in lowered:  # the host's pre-Link catalog guard, before any runtime preflight
        found = re.search(r"strict link rejected missing cache key ([0-9a-zA-Z_.:-]+)", text)
        return {"error_class": "host_catalog_missing_cache_key", "missing_cache_keys": [found.group(1)] if found else []}
    return {"error_class": "runtime_error"}
