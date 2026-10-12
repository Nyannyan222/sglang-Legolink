"""OpenAI-compatible proxy that keeps arbitrary agents on the NOC/LegoLink path.

The proxy accepts ordinary Chat Completions requests and internally routes them
through :class:`NOCSGLangLLM`.  This lets Terminal-Bench built-in agents, or any
other OpenAI-compatible agent, keep RadixCache LegoLink without knowing about
``pic_link`` metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from terminal_bench.llms.base_llm import OutputLengthExceededError

from noc_context_manager.compile_policy import CacheCatalog

from .noc_sglang_agent import NOCSGLangLLM
from .proxy_sessions import (
    DERIVED_PREFIX,
    ProxySessionRegistry,
    SessionResolution,
    explicit_session_identity,
)
from .terminal_bench_guardrail import (
    NOC_VERIFY_MARKER,
    TERMINAL_BENCH_GUARDRAIL_REPLAN_PREFIX,
    TERMINAL_BENCH_JSON_GUARDRAIL,
    TERMINAL_BENCH_SAFE_DIAGNOSTIC_COMMAND,
    terminal_bench_guardrail_replan_prompt,
    terminal_bench_guardrail_safe_diagnostic_response,
    terminal_bench_guardrail_violation,
    terminal_bench_inject_verification,
    terminal_bench_repair_json_response,
    terminal_bench_verification_probe,
)

# Re-exported so callers and tests that already import these from the proxy
# keep working; the definitions now live in terminal_bench_guardrail so
# sglang_pic_agent can apply the same rules.
__all__ = [
    "LEGOLINK_ROUTE",
    "NOC_VERIFY_MARKER",
    "PASSTHROUGH_ROUTE",
    "TERMINAL_BENCH_GUARDRAIL_REPLAN_PREFIX",
    "TERMINAL_BENCH_JSON_GUARDRAIL",
    "TERMINAL_BENCH_SAFE_DIAGNOSTIC_COMMAND",
    "NOCOpenAIProxy",
    "ProxyConfig",
    "terminal_bench_guardrail_replan_prompt",
    "terminal_bench_guardrail_safe_diagnostic_response",
    "terminal_bench_guardrail_violation",
    "terminal_bench_inject_verification",
    "terminal_bench_repair_json_response",
    "terminal_bench_verification_probe",
]

LEGOLINK_ROUTE = "selected_context_radixcache_legolink"
# Control arm for Agent Quality. The guardrail, session keying, turn counting,
# replan loop and metrics rows are the proxy's, so the only thing that differs
# from the LegoLink route is the inference path itself. An agent pointed
# straight at Original SGLang cannot receive the guardrail at all, which made
# every earlier pass-rate comparison confound LegoLink with prompt content.
PASSTHROUGH_ROUTE = "passthrough_original_sglang"


def normalize_model_name(model: str) -> str:
    """Strip LiteLLM/OpenAI provider prefixes without changing the model id."""

    for prefix in ("openai/", "hosted_vllm/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def message_content_to_text(content: Any) -> str:
    """Convert OpenAI text or multi-part content into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def messages_to_prompt_and_history(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Map OpenAI chat messages to the NOC LLM's prompt/history interface."""

    if not messages:
        raise ValueError("messages must not be empty")
    normalized = [
        {
            "role": str(message.get("role") or "user"),
            "content": message_content_to_text(message.get("content")),
        }
        for message in messages
    ]
    return normalized[-1]["content"], normalized[:-1]


def safe_session_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")[:120] or "session"


def request_session_key(
    *,
    headers: Any,
    payload: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
) -> str:
    """The session key a request names, or a fresh opaque one if it names none.

    Explicit sources are headers or the OpenAI ``user`` field and give a stable
    key. A request that names nothing gets a new random ``derived-`` id: the
    first chat message is task text, not a capability, so it is no longer used
    to find a session (see :mod:`.proxy_sessions`). Continuing a conversation
    without an explicit identity is :class:`ProxySessionRegistry`'s job, which
    needs the whole transcript and so lives on the proxy rather than here.
    """

    explicit = explicit_session_identity(headers=headers, payload=payload, model=model)
    if explicit is not None:
        return explicit[0]
    return f"{DERIVED_PREFIX}{secrets.token_hex(12)}"


def _conversation_turns(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (str(message.get("role") or "user"), message_content_to_text(message.get("content")))
        for message in messages
    ]


@dataclass(frozen=True)
class ProxyConfig:
    host: str
    port: int
    model: str
    patched_api_base: str
    original_api_base: str
    variant: str
    operation_mode: str
    segmentation_mode: str
    selection_mode: str
    compile_policy_mode: str
    context_token_budget: int
    legolink_k: int
    max_tokens: int
    temperature: float
    tokenizer_mode: str
    artifacts_root: Path
    timeout: float
    api_key: str
    allow_streaming_fallback: bool
    terminal_bench_json_guardrail: bool
    # Defaulted so existing constructions keep working. Both arms read the same
    # field, so it can never differ between them within a paired run.
    verification_probe: bool = True
    compile_cost_profile: Path | None = None
    deployment_hardware_id: str | None = None
    compile_reuse_probability: float = 0.5
    # Multi-agent topology, default off. When set, every session (agent) this proxy serves consults one
    # host catalog of compiled immutable exact-safe blocks, so a block one agent compiled is linked by the
    # next agent instead of compiled again. Sessions still keep their own stores, manifests, mutable
    # suffixes and histories; Gate 1 and Gate 2 run per request exactly as before.
    shared_block_catalog: bool = False
    # Stage-1 segmentation instruction variant; `None` is P0, the production prompt.
    segmentation_prompt_variant: str | None = None
    # Host-side cache namespace lookup arm: global | hard_namespace | logical_namespace.
    semantic_cache_namespace_mode: str = "global"
    # Sampling stated on every request (None sends nothing), and per-token logprobs kept in the turn artifact.
    repetition_penalty: float | None = None
    record_logprobs: bool = False


class PassthroughOriginalSession:
    """Forward one conversation to Original SGLang without NOC context work.

    Deliberately exposes the same surface as :class:`NOCSGLangLLM` -- ``call``,
    ``count_tokens``, ``turn``, ``metrics_path`` -- so the proxy's guardrail and
    replan loop run byte-identically on both arms and the two cannot drift.

    Token counts come from the server's own ``usage`` rather than a local
    tokenizer, so the control arm reports what Original SGLang actually billed.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_base: str,
        api_key: str,
        timeout: float,
        temperature: float,
        metrics_path: Path,
        task_id: str,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.metrics_path = metrics_path
        self.task_id = task_id
        self.turn = 0
        self.last_usage: dict[str, int] = {}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else {}
                ),
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def call(
        self,
        prompt: str,
        message_history: list[Any] | None = None,
        max_tokens: int | None = None,
        logging_path: Path | None = None,
        **_: Any,
    ) -> str:
        self.turn += 1
        # The last Terminal-Bench message is always the user turn, so
        # reconstructing it as "user" reproduces what the agent sent.
        messages = [
            *(message_history or []),
            {"role": "user", "content": prompt},
        ]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        started = time.perf_counter()
        response = self._post(payload)
        elapsed = time.perf_counter() - started
        choice = (response.get("choices") or [{}])[0]
        content = str((choice.get("message") or {}).get("content") or "")
        usage = response.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }
        if logging_path is not None:
            logging_path.parent.mkdir(parents=True, exist_ok=True)
            logging_path.write_text(
                json.dumps(
                    {
                        "route": PASSTHROUGH_ROUTE,
                        "turn": self.turn,
                        "request": payload,
                        "response": response,
                        "request_latency_s": elapsed,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self._write_metrics(elapsed)
        return content

    def _write_metrics(self, elapsed: float) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "task_id": self.task_id,
                        "turn": self.turn,
                        "variant": PASSTHROUGH_ROUTE,
                        "route": PASSTHROUGH_ROUTE,
                        "request_latency_s": elapsed,
                        # No LegoLink work happens on this arm, so the reuse
                        # columns are explicitly zero rather than absent.
                        "newly_compiled_blocks": 0,
                        "linked_tokens": self.last_usage.get("prompt_tokens", 0),
                        "reused_tokens": 0,
                        "recompute_tokens": self.last_usage.get(
                            "prompt_tokens", 0
                        ),
                        "context_cache_ratio": 0.0,
                        "fallback_reason": None,
                        **self.last_usage,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Server-reported usage when available, else a stable approximation."""

        if self.last_usage:
            roles = {str(message.get("role")) for message in messages}
            if roles == {"assistant"}:
                return self.last_usage.get("completion_tokens", 0)
            return self.last_usage.get("prompt_tokens", 0)
        return sum(
            len(str(message.get("content") or "")) // 4 for message in messages
        )


class NOCOpenAIProxy:
    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._sessions: dict[str, Any] = {}
        self._session_registry = ProxySessionRegistry()
        self._shared_cache_catalog = (
            CacheCatalog() if config.shared_block_catalog else None
        )

    def close_session(self, session_key: str) -> None:
        """Drop one session and every continuation entry that points at it."""

        with self._lock:
            self._sessions.pop(session_key, None)
        self._session_registry.release(session_key)

    def get_llm(self, session_key: str, model: str) -> Any:
        with self._lock:
            llm = self._sessions.get(session_key)
            if llm is not None and llm.model_name == model:
                return llm

            session_root = self.config.artifacts_root / safe_session_name(session_key)
            metrics_path = session_root / "proxy_metrics.jsonl"
            if self.config.variant == PASSTHROUGH_ROUTE:
                llm = PassthroughOriginalSession(
                    model_name=model,
                    api_base=self.config.original_api_base,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    temperature=self.config.temperature,
                    metrics_path=metrics_path,
                    task_id=session_key,
                )
                self._sessions[session_key] = llm
                return llm
            llm = NOCSGLangLLM(
                model_name=model,
                api_base=self.config.patched_api_base,
                original_api_base=self.config.original_api_base,
                variant=self.config.variant,
                operation_mode=self.config.operation_mode,
                segmentation_mode=self.config.segmentation_mode,
                selection_mode=self.config.selection_mode,
                compile_policy_mode=self.config.compile_policy_mode,
                compile_cost_profile=(
                    str(self.config.compile_cost_profile)
                    if self.config.compile_cost_profile is not None
                    else None
                ),
                deployment_hardware_id=self.config.deployment_hardware_id,
                compile_reuse_probability=self.config.compile_reuse_probability,
                segmentation_prompt_variant=self.config.segmentation_prompt_variant,
                semantic_cache_namespace_mode=self.config.semantic_cache_namespace_mode,
                repetition_penalty=self.config.repetition_penalty,
                record_logprobs=self.config.record_logprobs,
                shared_cache_catalog=self._shared_cache_catalog,
                context_token_budget=self.config.context_token_budget,
                artifacts_root=str(self.config.artifacts_root),
                tokenizer_mode=self.config.tokenizer_mode,
                recompute_mode="batched",
                legolink_k=self.config.legolink_k,
                first_block_k=self.config.legolink_k,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
                api_key=self.config.api_key,
            )
            llm.set_task_context(session_key, metrics_path)
            self._sessions[session_key] = llm
            return llm

    def models_response(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "noc-legolink-proxy",
                },
                {
                    "id": f"openai/{self.config.model}",
                    "object": "model",
                    "created": 0,
                    "owned_by": "noc-legolink-proxy",
                },
            ],
        }

    def record_error(
        self,
        *,
        request_path: str,
        payload: dict[str, Any] | None,
        error: Exception,
    ) -> None:
        """Persist compact proxy failures without storing full prompts."""

        record: dict[str, Any] = {
            "timestamp": time.time(),
            "path": request_path,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if payload is not None:
            messages = payload.get("messages")
            record.update(
                {
                    "model": payload.get("model"),
                    "stream": bool(payload.get("stream")),
                    "max_tokens": payload.get("max_tokens"),
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "user": payload.get("user") if isinstance(payload.get("user"), str) else None,
                }
            )
        try:
            self.config.artifacts_root.mkdir(parents=True, exist_ok=True)
            with (self.config.artifacts_root / "proxy_errors.jsonl").open(
                "a", encoding="utf-8"
            ) as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as log_error:  # noqa: BLE001
            print(f"[noc-openai-proxy] failed to write proxy_errors.jsonl: {log_error}")

    def chat_completion(self, payload: dict[str, Any], headers: Any) -> dict[str, Any]:
        if payload.get("stream") and not self.config.allow_streaming_fallback:
            raise ValueError(
                "stream=true is not supported by the NOC/LegoLink proxy; "
                "run the agent in non-streaming mode."
            )

        raw_model = str(payload.get("model") or self.config.model)
        model = normalize_model_name(raw_model)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise TypeError("payload.messages must be a list")
        prompt, history = messages_to_prompt_and_history(messages)
        if self.config.terminal_bench_json_guardrail:
            prompt = f"{prompt}{TERMINAL_BENCH_JSON_GUARDRAIL}"
        turns = _conversation_turns(messages)
        resolution = self._session_registry.resolve(
            explicit=explicit_session_identity(
                headers=headers, payload=payload, model=model
            ),
            model=model,
            turns=turns,
        )
        try:
            response = self._chat_completion_in_session(
                resolution=resolution,
                payload=payload,
                messages=messages,
                prompt=prompt,
                history=history,
                model=model,
                raw_model=raw_model,
            )
        except BaseException:
            self._session_registry.abandon(resolution)
            raise
        self._session_registry.complete(
            resolution,
            turns=turns,
            reply=str(response["choices"][0]["message"]["content"] or ""),
        )
        return response

    def _chat_completion_in_session(
        self,
        *,
        resolution: SessionResolution,
        payload: dict[str, Any],
        messages: list[dict[str, Any]],
        prompt: str,
        history: list[dict[str, str]],
        model: str,
        raw_model: str,
    ) -> dict[str, Any]:
        session_key = resolution.session_key
        llm = self.get_llm(session_key, model)
        max_tokens = int(payload.get("max_tokens") or self.config.max_tokens)
        request_id = uuid.uuid4().hex
        logging_path = (
            self.config.artifacts_root
            / safe_session_name(session_key)
            / f"request_{llm.turn + 1:04d}_{request_id[:8]}.json"
        )
        try:
            # An LLM that understands structured ingress gets the request's roles, tool calls, tool results
            # and tool schemas (Step 1 adapters); `prompt` / `history` stay the flat view they always were.
            structured = (
                {
                    "openai_messages": messages,
                    "tools": payload.get("tools"),
                    "functions": payload.get("functions"),
                }
                if getattr(llm, "structured_ingress", False)
                else {}
            )
            content = llm.call(
                prompt,
                message_history=history,
                max_tokens=max_tokens,
                logging_path=logging_path,
                **structured,
            )
            finish_reason = "stop"
            if self.config.terminal_bench_json_guardrail:
                content, _repair_reason = terminal_bench_repair_json_response(content)
                for _ in range(2):
                    violation = terminal_bench_guardrail_violation(content)
                    if violation is None:
                        break
                    replan_id = uuid.uuid4().hex
                    replan_logging_path = (
                        self.config.artifacts_root
                        / safe_session_name(session_key)
                        / f"request_{llm.turn + 1:04d}_{replan_id[:8]}_replan.json"
                    )
                    replan_history = [
                        *history,
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": content},
                    ]
                    content = llm.call(
                        terminal_bench_guardrail_replan_prompt(violation),
                        message_history=replan_history,
                        max_tokens=max_tokens,
                        logging_path=replan_logging_path,
                    )
                    content, _repair_reason = terminal_bench_repair_json_response(
                        content
                    )
                violation = terminal_bench_guardrail_violation(content)
                if violation is not None:
                    content = terminal_bench_guardrail_safe_diagnostic_response(
                        violation
                    )
                # Last, so only commands that already cleared the guardrail get
                # decorated and the probe can never turn a rejected command
                # into an accepted one.
                if self.config.verification_probe:
                    content, _probe_reason = terminal_bench_inject_verification(content)
        except OutputLengthExceededError as error:
            finish_reason = "length"
            if self.config.terminal_bench_json_guardrail:
                content = terminal_bench_guardrail_safe_diagnostic_response(
                    "model response exceeded the Terminal-Bench parser output limit"
                )
            else:
                content = error.truncated_response

        prompt_tokens = llm.count_tokens(
            [
                {
                    "role": str(message.get("role") or "user"),
                    "content": message_content_to_text(message.get("content")),
                }
                for message in messages
            ]
        )
        completion_tokens = llm.count_tokens([{"role": "assistant", "content": content}])
        return {
            "id": f"chatcmpl-noc-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": raw_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "noc_legolink": {
                "route": self.config.variant,
                "legolink_route_exercised": self.config.variant == LEGOLINK_ROUTE,
                "control_arm": self.config.variant == PASSTHROUGH_ROUTE,
                "session_key": session_key,
                "session_source": resolution.source,
                "metrics_path": str(llm.metrics_path) if llm.metrics_path else None,
            },
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "NOCOpenAIProxy/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[noc-openai-proxy] {self.address_string()} - {fmt % args}")

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> NOCOpenAIProxy:
        return self.server.proxy  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path in {"", "/v1", "/health"}:
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/v1/models":
            self._send_json(HTTPStatus.OK, self._proxy().models_response())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        payload: dict[str, Any] | None = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = self._proxy().chat_completion(payload, self.headers)
        except Exception as error:  # noqa: BLE001
            self._proxy().record_error(
                request_path=path,
                payload=payload,
                error=error,
            )
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "object": "error",
                    "message": str(error),
                    "type": type(error).__name__,
                },
            )
            return
        self._send_json(HTTPStatus.OK, response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32200)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--patched-api-base", required=True)
    parser.add_argument("--original-api-base")
    parser.add_argument(
        "--variant",
        choices=[
            "full_context_original_sglang",
            "selected_context_original_sglang",
            LEGOLINK_ROUTE,
            PASSTHROUGH_ROUTE,
        ],
        default=LEGOLINK_ROUTE,
    )
    parser.add_argument("--operation-mode", default="strict_experimental")
    parser.add_argument("--segmentation-mode", default="llm")
    parser.add_argument(
        "--semantic-cache-namespace-mode",
        choices=("global", "hard_namespace", "logical_namespace"),
        default="global",
        help="host-side semantic cache namespace lookup arm (see docs/semantic_cache_namespaces.md)",
    )
    parser.add_argument("--repetition-penalty", type=float, default=None, help="sent on every request when given")
    parser.add_argument("--record-logprobs", action="store_true", help="keep per-token logprobs in the turn artifact")
    parser.add_argument("--selection-mode", default="rule")
    parser.add_argument(
        "--compile-policy-mode",
        choices=("legacy_fixed_64", "cost_benefit", "compile_all_immutable"),
        default="legacy_fixed_64",
    )
    parser.add_argument(
        "--compile-cost-profile",
        type=Path,
        help="measured CompileCostProfile JSON; required by cost_benefit",
    )
    parser.add_argument(
        "--deployment-hardware-id",
        help="served hardware identity; required by cost_benefit",
    )
    parser.add_argument(
        "--compile-reuse-probability",
        type=float,
        default=0.5,
        help="discount applied to the caller's planned future turns",
    )
    parser.add_argument("--context-token-budget", type=int, default=24_000)
    parser.add_argument("--legolink-k", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tokenizer-mode", default="sglang")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("benchmark_results/noc_openai_proxy_sessions"),
    )
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--allow-streaming-fallback", action="store_true")
    parser.add_argument("--terminal-bench-json-guardrail", action="store_true")
    parser.add_argument(
        "--no-verification-probe",
        dest="verification_probe",
        action="store_false",
        help=(
            "Stop appending the read-only NOC_VERIFY evidence block after "
            "commands that change state silently."
        ),
    )
    parser.add_argument(
        "--shared-block-catalog",
        action="store_true",
        help=(
            "Multi-agent topology: all sessions share one host catalog of "
            "compiled immutable exact-safe blocks. Default off."
        ),
    )
    parser.add_argument(
        "--segmentation-prompt-variant",
        default=None,
        help="Stage-1 segmentation prompt: p0_current (default), p1_independent, p2_reuse.",
    )
    parser.set_defaults(verification_probe=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_api_base = args.original_api_base or args.patched_api_base
    config = ProxyConfig(
        host=args.host,
        port=args.port,
        model=normalize_model_name(args.model),
        patched_api_base=args.patched_api_base,
        original_api_base=original_api_base,
        variant=args.variant,
        operation_mode=args.operation_mode,
        segmentation_mode=args.segmentation_mode,
        selection_mode=args.selection_mode,
        compile_policy_mode=args.compile_policy_mode,
        context_token_budget=args.context_token_budget,
        legolink_k=args.legolink_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tokenizer_mode=args.tokenizer_mode,
        artifacts_root=args.artifacts_root,
        timeout=args.timeout,
        api_key=args.api_key,
        allow_streaming_fallback=args.allow_streaming_fallback,
        terminal_bench_json_guardrail=args.terminal_bench_json_guardrail,
        verification_probe=args.verification_probe,
        compile_cost_profile=args.compile_cost_profile,
        deployment_hardware_id=args.deployment_hardware_id,
        compile_reuse_probability=args.compile_reuse_probability,
        shared_block_catalog=args.shared_block_catalog,
        segmentation_prompt_variant=args.segmentation_prompt_variant,
        semantic_cache_namespace_mode=args.semantic_cache_namespace_mode,
        repetition_penalty=args.repetition_penalty,
        record_logprobs=args.record_logprobs,
    )
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    proxy = NOCOpenAIProxy(config)
    server = ThreadingHTTPServer((config.host, config.port), Handler)
    server.proxy = proxy  # type: ignore[attr-defined]
    print(
        "NOC/LegoLink OpenAI proxy listening on "
        f"http://{config.host}:{config.port}/v1"
    )
    print(f"route={config.variant} patched_api_base={config.patched_api_base}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
