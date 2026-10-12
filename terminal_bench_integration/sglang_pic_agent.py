"""Terminus agent backed by the patched SGLang PIC compile/link API."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shlex
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.terminus_1 import Terminus
from terminal_bench.llms.base_llm import BaseLLM, OutputLengthExceededError
from terminal_bench.terminal.tmux_session import TmuxSession

from noc_context_manager.chat_transport import post_chat
from noc_context_manager.inference_metrics import compute_stream_metrics
from noc_context_manager.pic_link_contract import (
    build_pic_link_payload,
    build_pic_link_request,
)
from noc_context_manager.real_ingress import (
    KIND_AGENT_DIRECTIVE,
    KIND_TOOL_OUTPUT,
    PromptPart,
    split_around,
)
from noc_context_manager.runtime_isolation import estimate_tokens

# Absolute, matching the noc_context_manager import above: several tests load
# this file by path with importlib, where a relative import has no parent
# package to resolve against.
from terminal_bench_integration.terminal_bench_guardrail import (
    TERMINAL_BENCH_JSON_GUARDRAIL,
    terminal_bench_verification_probe,
)


@dataclass
class _PICCommand:
    keystrokes: str
    is_blocking: bool
    timeout_sec: float


@dataclass
class _PICAgentResponse:
    commands: list[_PICCommand]
    is_task_complete: bool
    raw_response: str
    feedback: str = ""

    def model_dump_json(self) -> str:
        return self.raw_response


def _root_cause(error: BaseException) -> tuple[str, str]:
    """Name the failure that actually happened, not its wrapper.

    The harness retries the model call through tenacity, so an exhausted
    retry surfaces as ``RetryError`` and the real cause -- an HTTP 400, a
    refused Link, a context-length rejection -- is reachable only through
    the last attempt. Recording only the wrapper leaves a failed trial with
    no diagnosable reason, which is worst on the long remote runs where
    re-running to reproduce is expensive.
    """

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        attempt = getattr(current, "last_attempt", None)
        nested = None
        if attempt is not None and hasattr(attempt, "exception"):
            try:
                nested = attempt.exception()
            except Exception:
                nested = None
        if nested is None:
            # __cause__ only: it is set by an explicit ``raise ... from``.
            # __context__ is set implicitly for anything raised inside an
            # except block, including a failure that was already handled,
            # so following it names the wrong exception.
            nested = current.__cause__
        if nested is None:
            break
        current = nested
    resolved = current if current is not None else error
    return type(resolved).__name__, str(resolved)[:2000]


def _ms_to_s(value: Any) -> float | None:
    return None if value is None else float(value) / 1000.0


def _message_value(message: Any, name: str, default: str = "") -> str:
    if isinstance(message, dict):
        value = message.get(name, default)
    else:
        value = getattr(message, name, default)
    return str(value if value is not None else default)


def _split_text(text: str, max_chars: int) -> list[str]:
    remaining = text.strip()
    chunks: list[str] = []
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, max_chars + 1)
        if split_at < max_chars // 2:
            split_at = remaining.rfind("\n", 0, max_chars + 1)
        if split_at < max_chars // 2:
            split_at = max_chars
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks or [""]


class SGLangPICLLM(BaseLLM):
    def __init__(
        self,
        *,
        model_name: str,
        api_base: str,
        recompute_mode: str = "batched",
        legolink_k: int = 256,
        first_block_k: int | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        block_chars: int = 6000,
        timeout: float = 1200,
        api_key: str = "",
        stream_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if recompute_mode not in {"batched", "full_causal"}:
            raise ValueError("recompute_mode must be batched or full_causal")
        if (
            legolink_k < 0
            or (first_block_k is not None and first_block_k < 0)
            or max_tokens <= 0
            or block_chars <= 0
        ):
            raise ValueError("invalid LegoLink K, max_tokens, or block_chars")
        self.model_name = model_name
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.recompute_mode = recompute_mode
        self.legolink_k = int(legolink_k)
        self.first_block_k = (
            int(first_block_k) if first_block_k is not None else int(legolink_k)
        )
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.block_chars = int(block_chars)
        self.timeout = float(timeout)
        self.api_key = api_key
        self.stream_metrics = bool(stream_metrics)
        self.session_id = uuid.uuid4().hex
        self.task_id = "unknown"
        self.turn = 0
        self.metrics_path: Path | None = None
        self._compiled_keys: set[str] = set()
        self._compiled_token_counts: dict[str, int] = {}

    def set_task_context(self, task_id: str, metrics_path: Path | None) -> None:
        self.task_id = task_id
        self.metrics_path = metrics_path
        self.session_id = uuid.uuid4().hex
        self.turn = 0
        self._compiled_keys.clear()
        self._compiled_token_counts.clear()
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("", encoding="utf-8")

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        try:
            result = post_chat(
                self.endpoint, payload, timeout=self.timeout, headers=self._headers()
            )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang HTTP {error.code}: {detail}") from error
        return result.response, result.latency_s

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _post_stream(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float, dict[str, Any]]:
        """Post one generation request over SSE and time its first token.

        Whole-request latency cannot separate prefill from decode, so it
        cannot show that reuse skipped a prefill. TTFT is measured rather than
        derived, by the same shared transport the NOC pipeline uses -- the two
        used to be separate implementations and only one of them could stream.

        The validation below stays here rather than in the transport: these are
        judgements about whether a response is usable as evidence, and the
        adapter's routes answer that differently.
        """

        result = post_chat(
            self.endpoint,
            payload,
            timeout=self.timeout,
            headers={"Accept": "text/event-stream", **self._headers()},
            stream=True,
        )
        if not result.saw_chunk:
            # No SSE frame arrived at all, so this deployment did not stream
            # the request. Worth falling back and recording as unavailable.
            raise RuntimeError("no SSE frames arrived; this route did not stream")
        text = str(
            (result.response["choices"][0].get("message") or {}).get("content") or ""
        )
        if not text and result.finish_reason is None:
            # An empty completion is only believable when the server said why
            # it stopped. With no content and no finish_reason there is no way
            # to tell a silent model from a broken stream, and accepting it
            # writes a metric row built on a prompt_tokens the request never
            # really had.
            raise RuntimeError("stream ended with no content and no finish_reason")
        # An empty completion carrying a finish_reason is a real answer. Re-
        # sending it unstreamed would double the work and change nothing.
        # Reuse accounting is deliberately omitted here. The patched runtime
        # emits no reuse fields in usage, and this call site has no access to
        # the compiled block token counts; the authoritative reused/recomputed
        # figures are computed by the caller and written into the same row.
        usage = result.response.get("usage") or {}
        metrics = compute_stream_metrics(
            request_started_ns=result.request_started_ns,
            token_ready_ns=list(result.token_ready_ns),
            request_completed_ns=result.request_completed_ns,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens") or len(result.token_ready_ns),
        )
        return result.response, result.latency_s, metrics

    def _post_generation(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float, dict[str, Any]]:
        """Stream when asked, but never let instrumentation fail a trial.

        A deployment that does not stream this route must still produce the
        run. The metric row then records why TTFT is absent instead of
        leaving an unexplained null that later reads as a measured zero.
        """

        if not self.stream_metrics:
            response, latency = self._post(payload)
            return response, latency, {"stream_metrics_status": "disabled"}
        try:
            response, latency, metrics = self._post_stream(payload)
        except Exception as error:  # instrumentation must not fail a run
            response, latency = self._post(payload)
            return (
                response,
                latency,
                {
                    "stream_metrics_status": "unavailable",
                    "stream_metrics_error": f"{type(error).__name__}: {error}",
                },
            )
        metrics["stream_metrics_status"] = "ok"
        return response, latency, metrics

    def _message_blocks(
        self, role: str, content: str, *, force_two: bool = False
    ) -> list[tuple[str, str]]:
        chunks = _split_text(content, self.block_chars)
        if force_two and len(chunks) == 1 and len(chunks[0]) > 1:
            midpoint = len(chunks[0]) // 2
            chunks = [chunks[0][:midpoint], chunks[0][midpoint:]]
        blocks: list[tuple[str, str]] = []
        for index, chunk in enumerate(chunks):
            text = (
                f"<terminal_bench_message role={role} "
                f"chunk={index + 1}/{len(chunks)}>\n{chunk}\n"
                "</terminal_bench_message>"
            )
            cache_key = hashlib.sha256(
                f"{self.model_name}\0{text}".encode("utf-8")
            ).hexdigest()
            blocks.append((cache_key, text))
        return blocks

    def _prepare_turn(
        self, prompt: str, message_history: list[Any]
    ) -> tuple[list[tuple[str, str]], str]:
        blocks: list[tuple[str, str]] = []
        for message_index, message in enumerate(message_history):
            role = _message_value(message, "role", "user")
            blocks.extend(
                self._message_blocks(
                    role,
                    _message_value(message, "content"),
                    force_two=message_index == 0 and role == "user",
                )
            )
        current_blocks = self._message_blocks(
            "user", prompt, force_two=not message_history
        )
        blocks.extend(current_blocks[:-1])
        mutable_text = current_blocks[-1][1]
        return blocks, mutable_text

    def _semantic_block_id(self, cache_key: str) -> str:
        """Name a block the same way at Compile and at Link.

        The runtime registers this id when the block is compiled and looks
        the block up by it when a later request links it, so the two call
        sites must agree. Deriving it from the cache key also keeps the id
        stable when the same block appears at a different position in a
        later turn.
        """

        return f"tb::{self.task_id}::{cache_key[:16]}"

    def _compile_block(self, cache_key: str, text: str) -> tuple[int, float]:
        block_id = self._semantic_block_id(cache_key)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "seed": 0,
            "max_tokens": 1,
            "extra_key": cache_key,
            "custom_labels": {
                "prefill_unit_id": block_id,
                "cache_key": cache_key,
                "pic_storage_mode": "reference",
            },
            "sub_context": {
                "prefill_unit_id": block_id,
                "cache_key": cache_key,
                "task_id": self.task_id,
                "context_field": "terminal_bench_conversation",
                "functional_role": "immutable_agent_history",
            },
        }
        response, latency = self._post(payload)
        token_count = int((response.get("usage") or {}).get("prompt_tokens") or 0)
        return token_count, latency

    def _write_metrics(self, row: dict[str, Any], logging_path: Path | None) -> None:
        if self.metrics_path is not None:
            with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        if logging_path is not None:
            logging_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_request_capture(
        self,
        *,
        blocks: list[tuple[str, str]],
        mutable_text: str,
        payload: dict[str, Any],
        logging_path: Path | None,
    ) -> None:
        if logging_path is None:
            return
        capture = {
            "schema_version": 1,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "turn": self.turn,
            "model": self.model_name,
            "immutable_blocks": [
                {
                    "cache_key": cache_key,
                    "text": text,
                    "token_count": self._compiled_token_counts.get(cache_key, 0),
                }
                for cache_key, text in blocks
            ],
            "mutable_text": mutable_text,
            "generation": {
                key: payload[key]
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                    "seed",
                    "max_tokens",
                )
                if key in payload
            },
            "response_format": payload.get("response_format"),
        }
        (logging_path.parent / "sglang_request.json").write_text(
            json.dumps(capture, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def call(
        self,
        prompt: str,
        message_history: list[Any] | None = None,
        response_format: Any = None,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> str:
        self.turn += 1
        blocks, mutable_text = self._prepare_turn(prompt, message_history or [])
        compile_latency = 0.0
        newly_compiled = 0
        for cache_key, block_text in blocks:
            if cache_key in self._compiled_keys:
                continue
            token_count, latency = self._compile_block(cache_key, block_text)
            self._compiled_keys.add(cache_key)
            self._compiled_token_counts[cache_key] = token_count
            compile_latency += latency
            newly_compiled += 1

        cache_keys = [cache_key for cache_key, _ in blocks]
        block_ids = [self._semantic_block_id(cache_key) for cache_key in cache_keys]
        variant = (
            "full_causal"
            if self.recompute_mode == "full_causal"
            else f"adaptive_l{self.legolink_k}"
        )
        payload = build_pic_link_payload(
            build_pic_link_request(
                request_id=f"tb::{self.session_id}::turn-{self.turn}",
                task_id=self.task_id,
                variant=variant,
                cache_keys=cache_keys,
                block_ids=block_ids,
                recompute_mode=self.recompute_mode,
                recompute_initial_tokens=self.legolink_k,
                recompute_first_block=self.first_block_k > 0,
                recompute_first_block_tokens=self.first_block_k,
            ),
            model=self.model_name,
            mutable_text=mutable_text,
            generation={
                "temperature": self.temperature,
                "top_p": 1.0,
                "top_k": 1 if self.temperature <= 1e-6 else -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "seed": 0,
                "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            },
        )
        if response_format is not None:
            schema = (
                response_format
                if isinstance(response_format, dict)
                else response_format.model_json_schema()
            )
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(response_format, "__name__", "response"),
                    "schema": schema,
                    "strict": True,
                },
            }
        self._write_request_capture(
            blocks=blocks,
            mutable_text=mutable_text,
            payload=payload,
            logging_path=logging_path,
        )
        response, link_latency, stream_row = self._post_generation(payload)
        choice = (response.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = response.get("usage") or {}
        linked_tokens = int(usage.get("prompt_tokens") or 0)
        immutable_tokens = sum(
            self._compiled_token_counts.get(cache_key, 0) for cache_key in cache_keys
        )
        mutable_tokens = max(linked_tokens - immutable_tokens, 0)
        block_token_counts = [
            self._compiled_token_counts.get(cache_key, 0) for cache_key in cache_keys
        ]
        block_ranges: list[dict[str, Any]] = []
        offset = 0
        for index, (cache_key, token_count) in enumerate(
            zip(cache_keys, block_token_counts)
        ):
            block_ranges.append(
                {
                    "block_id": block_ids[index],
                    "cache_key": cache_key,
                    "start_token": offset,
                    "end_token": offset + token_count,
                    "token_count": token_count,
                }
            )
            offset += token_count
        per_block_k = [self.first_block_k] + [self.legolink_k] * max(
            len(cache_keys) - 1, 0
        )
        recompute_ranges = [
            {
                "block_index": index,
                "start_token": block_range["start_token"],
                "end_token": min(
                    block_range["start_token"] + per_block_k[index],
                    block_range["end_token"],
                ),
            }
            for index, block_range in enumerate(block_ranges)
            if per_block_k[index] > 0 and block_range["token_count"] > 0
        ]
        recompute_tokens = (
            linked_tokens
            if self.recompute_mode == "full_causal"
            else mutable_tokens
            + sum(
                min(block_k, token_count)
                for block_k, token_count in zip(per_block_k, block_token_counts)
            )
        )
        self._write_metrics(
            {
                "task_id": self.task_id,
                "session_id": self.session_id,
                "turn": self.turn,
                "recompute_mode": self.recompute_mode,
                "variant": variant,
                "legolink_k": self.legolink_k,
                "first_block_k": self.first_block_k,
                "history_blocks": len(blocks),
                "newly_compiled_blocks": newly_compiled,
                "compile_latency_s": compile_latency,
                "link_latency_s": link_latency,
                "ttft_s": _ms_to_s(stream_row.get("ttft_ms")),
                "decode_s": _ms_to_s(stream_row.get("decode_ms")),
                "itl_p50_ms": stream_row.get("itl_p50_ms"),
                "itl_p95_ms": stream_row.get("itl_p95_ms"),
                "output_tokens_per_second": stream_row.get(
                    "output_tokens_per_second"
                ),
                "stream_metrics_status": stream_row.get("stream_metrics_status"),
                "stream_metrics_error": stream_row.get("stream_metrics_error"),
                "linked_tokens": linked_tokens,
                "reused_tokens": max(linked_tokens - recompute_tokens, 0),
                "recompute_tokens": recompute_tokens,
                "context_cache_ratio": (
                    max(linked_tokens - recompute_tokens, 0) / linked_tokens
                    if linked_tokens
                    else None
                ),
                "request_id": payload["pic_link"]["request_id"],
                "selected_cache_keys": cache_keys,
                "linked_block_ranges": block_ranges,
                "global_offsets": [row["start_token"] for row in block_ranges],
                "sparse_recompute_ranges": recompute_ranges,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "sampling_temperature": self.temperature,
                "sampling_top_k": 1 if self.temperature <= 1e-6 else -1,
                "sampling_seed": 0,
            },
            logging_path,
        )
        if choice.get("finish_reason") == "length":
            raise OutputLengthExceededError(
                "SGLang response reached max_tokens", truncated_response=content
            )
        return content

    def count_tokens(self, messages: list[dict]) -> int:
        return sum(
            len(estimate_tokens(_message_value(message, "content")))
            for message in messages
        )


class SGLangPICAgent(Terminus):
    """Terminal-Bench Terminus loop using SGLang PIC for every model turn."""

    RECOVERY_GUIDANCE = """
Execution policy:
- Inspect the filesystem and service endpoints before assuming paths or commands.
- Never repeat an identical failed command batch. Read the terminal error and change strategy.
- For aggregation and structured output, use a short Python script that computes actual totals and writes the required schema. Do not treat per-file counts as a total.
- Inspect every generated artifact with `cat`, `head`, or a small Python validator before declaring completion.
- Every command must be self-contained with balanced shell quoting. Never leave placeholder values such as `{}` in generated files.
- Avoid interactive editors such as `nano`, `vim`, and `less`; create files with a quoted heredoc, `printf`, or a short Python script.
- In shell commands, never place `!` inside double quotes because interactive Bash expands it. Use single-quoted literals, for example `printf '%s\n' 'Hello, world!'`.
- If a script "will not run" or may have a permission problem, inspect it with `ls -l` or `stat` and fix the executable bit with `chmod +x`; do not treat `bash script.sh` succeeding as proof that the script itself is runnable.
- When a command times out, inspect or interrupt it before issuing more commands.
- Grader tests are unavailable during the agent phase. Validate against the task specification using independent shell or Python checks.
- Set realistic command timeouts (at least 60 seconds for compilation, tests, or data processing).
""".strip()

    # The same guardrail the proxy applies. It used to reach only the proxy
    # loop, so the H100 formal runs -- which use this agent, not the proxy --
    # never received it.
    RECOVERY_GUIDANCE = RECOVERY_GUIDANCE + TERMINAL_BENCH_JSON_GUARDRAIL

    @staticmethod
    def name() -> str:
        return "sglang-pic-terminus"

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://127.0.0.1:30000/v1",
        recompute_mode: str = "batched",
        legolink_k: int = 256,
        first_block_k: int | None = None,
        max_tokens: int = 1024,
        max_episodes: int = 12,
        temperature: float = 0.0,
        block_chars: int = 6000,
        timeout: float = 1200,
        api_key: str = "",
        verification_probe: Any = True,
        stream_metrics: Any = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_episodes=max_episodes,
            api_base=api_base,
            temperature=temperature,
            **kwargs,
        )
        self._llm = SGLangPICLLM(
            model_name=model_name,
            api_base=api_base,
            recompute_mode=recompute_mode,
            legolink_k=legolink_k,
            first_block_k=first_block_k,
            max_tokens=max_tokens,
            temperature=temperature,
            block_chars=block_chars,
            timeout=timeout,
            api_key=api_key,
            stream_metrics=self._coerce_optin_flag(stream_metrics),
        )
        self._step_metrics_path: Path | None = None
        self._task_metrics_path: Path | None = None
        self._step_rows: list[dict[str, Any]] = []
        self._base_temperature = float(temperature)
        self._verification_probe = self._coerce_probe_flag(verification_probe)
        self._last_termination_reason = "unclassified"
        self._invalid_action_count = 0
        self._tool_error_count = 0
        self._timeout_count = 0

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _declare_prompt_structure(self, prompt: str, parts: tuple[PromptPart, ...] | None) -> None:
        """Tell the LLM how the next prompt was assembled, when it can use that (the NOC LLM can).

        The loop built ``prompt`` from a terminal observation and its own directive text, so it knows which
        piece is which; the LLM boundary (`call(prompt, message_history)`) does not. The LLM re-verifies that
        the parts re-assemble ``prompt`` exactly before it uses them.
        """

        declare = getattr(self._llm, "declare_prompt_structure", None)
        if callable(declare):
            declare(parts or (PromptPart(KIND_AGENT_DIRECTIVE, prompt),))

    def _write_step_metric(self, row: dict[str, Any]) -> None:
        self._step_rows.append(row)
        if self._step_metrics_path is not None:
            with self._step_metrics_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )

    def _resolve_completion(
        self, requested: bool, terminal_output: str
    ) -> tuple[bool, str | None]:
        return requested, None

    @staticmethod
    def _compact_terminal_state(terminal_output: str, max_chars: int = 6000) -> str:
        if len(terminal_output) <= max_chars:
            return terminal_output
        return (
            "[terminal output truncated to the most recent state]\n"
            + terminal_output[-max_chars:]
        )

    @staticmethod
    def _blocked_command_recovery_prompt(
        *,
        reason: str,
        terminal_output: str,
    ) -> str:
        return (
            "RECOVERY REQUIRED.\n"
            f"Reason: {reason}\n\n"
            "Your next response must follow these rules:\n"
            "1. Do not repeat any previous command batch.\n"
            "2. If the terminal is in a continuation prompt or a command failed, "
            "recover by inspecting current state first.\n"
            "3. Prefer one short, concrete command batch that either validates the "
            "current artifact or applies a different direct fix.\n"
            "   For script permission tasks, inspect `ls -l`/`stat` and use "
            "`chmod +x` when needed; do not rerun the script through `bash`.\n"
            "4. If the task is already satisfied, return task_complete=true with "
            "no commands.\n"
            "5. Return only the required JSON response; keep analysis concise.\n\n"
            "Current terminal state:\n"
            + SGLangPICAgent._compact_terminal_state(terminal_output)
        )

    @staticmethod
    def _interrupt_terminal(session: TmuxSession) -> None:
        try:
            session.send_keys("C-c", min_timeout_sec=0.1)
        except Exception:
            # Recovery should not fail only because the interrupt could not be sent.
            pass

    @staticmethod
    def _command_batch_signature(commands: list[Any]) -> tuple[tuple[str, bool], ...]:
        return tuple(
            (
                str(getattr(command, "keystrokes", "")).strip(),
                bool(getattr(command, "is_blocking", False)),
            )
            for command in commands
        )

    @staticmethod
    def _coerce_optin_flag(value: Any) -> bool:
        """Read an opt-in flag that may arrive as a Terminal-Bench string.

        The mirror of :meth:`_coerce_probe_flag`: this one defaults to off, so
        only an explicit affirmative turns it on. An unset or unparsed value
        must never silently change the transport of a formal run.
        """

        return str(value).strip().lower() in {"true", "1", "yes", "on"}

    @staticmethod
    def _coerce_probe_flag(value: Any) -> bool:
        """Read a probe flag that may arrive as a Terminal-Bench string kwarg.

        ``--agent-kwarg verification_probe=false`` delivers the string
        ``"false"``, which is truthy and would silently leave the probe on.
        """

        return str(value).strip().lower() not in {"false", "0", "no", "off"}

    @staticmethod
    def _apply_verification_probes(commands: list[Any]) -> list[str]:
        """Append read-only evidence to commands that change state silently.

        A successful redirect, mkdir or chmod prints nothing, so the pane the
        model sees next is indistinguishable from the one before the write and
        it cannot tell a finished task from a no-op. Applied after the repeat
        and syntax checks, so the batch signature stays keyed on what the model
        actually emitted.
        """

        probed: list[str] = []
        for command in commands:
            keystrokes = str(getattr(command, "keystrokes", ""))
            probe = terminal_bench_verification_probe(keystrokes)
            if probe is None:
                continue
            if not keystrokes.endswith("\n"):
                keystrokes += "\n"
            command.keystrokes = keystrokes + probe
            # The probe has to finish inside the command's own window or the
            # pane is captured before it prints.
            try:
                command.timeout_sec = float(command.timeout_sec) + 5.0
            except (TypeError, ValueError):
                pass
            probed.append(keystrokes.strip().splitlines()[0] if keystrokes.strip() else "")
        return probed

    @staticmethod
    def _command_syntax_errors(commands: list[Any]) -> list[str]:
        errors: list[str] = []
        for index, command in enumerate(commands, start=1):
            keystrokes = str(getattr(command, "keystrokes", "")).strip()
            if not keystrokes or keystrokes in {"C-c", "C-d", "Enter"}:
                continue
            # Heredocs contain another language whose quotes should not be parsed as
            # shell tokens. The terminal's blocking safeguards already special-case
            # commands ending in EOF.
            if "<<" in keystrokes:
                continue
            if re.search(r'"(?:\\.|[^"\\])*?(?<!\\)!', keystrokes):
                errors.append(
                    f"command {index}: unescaped ! inside double quotes would "
                    "trigger interactive Bash history expansion; use a "
                    "single-quoted literal"
                )
                continue
            try:
                shlex.split(keystrokes, posix=True)
            except ValueError as error:
                errors.append(f"command {index}: {error}")
        return errors

    def _run_agent_loop(
        self,
        initial_prompt: str,
        session: TmuxSession,
        chat: Any,
        logging_dir: Path | None = None,
    ) -> None:
        """Run Terminus while recording ThunderAgent-style reasoning/acting steps."""

        prompt = f"{initial_prompt}\n\n{self.RECOVERY_GUIDANCE}"
        seen_command_batches: set[tuple[tuple[str, bool], ...]] = set()
        self._last_termination_reason = "max_rounds_exhausted"
        self._invalid_action_count = 0
        self._tool_error_count = 0
        self._timeout_count = 0
        for episode in range(self._max_episodes):
            logging_paths = self._setup_episode_logging(logging_dir, episode)
            step_started_at = self._utc_now()
            # Wall-clock ISO is for humans; order and durations use the monotonic
            # clock, which a host clock adjustment cannot move backwards.
            step_started_monotonic_ns = time.monotonic_ns()
            step_started = time.perf_counter()
            llm_started = time.perf_counter()
            try:
                parsed_response = self._handle_llm_interaction(
                    chat, prompt, logging_paths
                )
            except Exception as error:
                llm_latency = time.perf_counter() - llm_started
                self._write_step_metric(
                    {
                        "task_id": self._llm.task_id,
                        "session_id": self._llm.session_id,
                        "round_index": episode + 1,
                        "loop_stage": "planner" if episode == 0 else "replanner",
                        "session_revision": episode + 1,
                        "step": episode + 1,
                        "status": "llm_error",
                        "parse_status": "llm_request_error",
                        "termination_candidate": "llm_request_error",
                        "error_type": type(error).__name__,
                        "error_cause_type": _root_cause(error)[0],
                        "error_cause_message": _root_cause(error)[1],
                        "reasoning_latency_s": llm_latency,
                        "acting_latency_s": 0.0,
                        "request_latency": llm_latency,
                        "request_latency_ms": llm_latency * 1000,
                        "tool_latency": 0.0,
                        "tool_latency_ms": 0.0,
                        "step_latency_s": time.perf_counter() - step_started,
                        "command_count": 0,
                        "llm_calls": 1,
                        "tool_calls": 0,
                        "is_task_complete": False,
                        "terminal_timeout": False,
                        "step_started_at": step_started_at,
                        "step_ended_at": self._utc_now(),
                        "step_started_monotonic_ns": step_started_monotonic_ns,
                        "step_ended_monotonic_ns": time.monotonic_ns(),
                    }
                )
                self._last_termination_reason = "llm_request_error"
                raise

            llm_latency = time.perf_counter() - llm_started
            self._record_asciinema_marker(
                parsed_response.model_dump_json(), session
            )
            response_feedback = str(getattr(parsed_response, "feedback", ""))
            if response_feedback.startswith("ERROR:"):
                self._invalid_action_count += 1
                self._write_step_metric(
                    {
                        "task_id": self._llm.task_id,
                        "session_id": self._llm.session_id,
                        "round_index": episode + 1,
                        "loop_stage": "planner" if episode == 0 else "replanner",
                        "session_revision": episode + 1,
                        "step": episode + 1,
                        "status": "parser_recovery",
                        "parse_status": "invalid_json",
                        "termination_candidate": "parser_error",
                        "reasoning_latency_s": llm_latency,
                        "acting_latency_s": 0.0,
                        "request_latency": llm_latency,
                        "request_latency_ms": llm_latency * 1000,
                        "tool_latency": 0.0,
                        "tool_latency_ms": 0.0,
                        "step_latency_s": time.perf_counter() - step_started,
                        "command_count": 0,
                        "llm_calls": 1,
                        "tool_calls": 0,
                        "is_task_complete": False,
                        "terminal_timeout": False,
                        "step_started_at": step_started_at,
                        "step_ended_at": self._utc_now(),
                        "step_started_monotonic_ns": step_started_monotonic_ns,
                        "step_ended_monotonic_ns": time.monotonic_ns(),
                    }
                )
                prompt = (
                    "Your previous response could not be executed. "
                    + response_feedback
                    + "\nReturn one valid response in the required format."
                )
                self._declare_prompt_structure(prompt, None)
                continue
            acting_started = time.perf_counter()
            command_signature = self._command_batch_signature(
                parsed_response.commands
            )
            repeated_command_blocked = bool(command_signature) and (
                command_signature in seen_command_batches
            )
            syntax_errors = self._command_syntax_errors(parsed_response.commands)
            invalid_command_blocked = bool(syntax_errors)
            parse_status = "valid_tool_action"
            blocked_recovery_prompt: str | None = None
            if invalid_command_blocked:
                self._invalid_action_count += 1
                parse_status = "invalid_arguments"
                timeout_occurred = False
                self._interrupt_terminal(session)
                terminal_output = (
                    "RECOVERY_REQUIRED: The command batch was not executed because "
                    "its shell syntax is incomplete: "
                    + "; ".join(syntax_errors)
                    + ". Use a self-contained command with balanced quotes. Prefer "
                    "writing a short Python script when shell quoting is complex.\n\n"
                    + session.capture_pane()
                )
                blocked_recovery_prompt = self._blocked_command_recovery_prompt(
                    reason="invalid shell syntax",
                    terminal_output=terminal_output,
                )
            elif repeated_command_blocked:
                parse_status = "tool_execution_error"
                timeout_occurred = False
                self._interrupt_terminal(session)
                terminal_output = (
                    "RECOVERY_REQUIRED: This exact command batch already ran without "
                    "completing the task. Do not repeat it. Inspect the current state, "
                    "read the prior error, and choose a different command or implementation.\n\n"
                    + session.capture_pane()
                )
                blocked_recovery_prompt = self._blocked_command_recovery_prompt(
                    reason="repeated command batch blocked",
                    terminal_output=terminal_output,
                )
                self._llm.temperature = max(self._base_temperature, 0.2)
            else:
                if command_signature:
                    seen_command_batches.add(command_signature)
                # After the signature is recorded, so repeat detection keys on
                # the model's own output rather than on the probe.
                if self._verification_probe:
                    self._apply_verification_probes(parsed_response.commands)
                try:
                    timeout_occurred, terminal_output = self._execute_commands(
                        parsed_response.commands, session
                    )
                except Exception:
                    self._tool_error_count += 1
                    self._last_termination_reason = "tool_execution_error"
                    raise
                self._llm.temperature = self._base_temperature
            acting_latency = time.perf_counter() - acting_started
            self._timeout_count += int(bool(timeout_occurred))
            episode_dir = next(
                (
                    path.parent
                    for path in logging_paths
                    if path is not None
                ),
                None,
            )
            if episode_dir is not None:
                (episode_dir / "terminal_commands.json").write_text(
                    json.dumps(
                        [
                            {
                                "keystrokes": str(
                                    getattr(command, "keystrokes", "")
                                ),
                                "is_blocking": bool(
                                    getattr(command, "is_blocking", False)
                                ),
                                "timeout_sec": float(
                                    getattr(command, "timeout_sec", 0.0)
                                ),
                            }
                            for command in parsed_response.commands
                        ],
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (episode_dir / "terminal_output.txt").write_text(
                    terminal_output,
                    encoding="utf-8",
                )
            is_task_complete = bool(parsed_response.is_task_complete) and not (
                repeated_command_blocked or invalid_command_blocked
            )
            is_task_complete, completion_prompt = self._resolve_completion(
                is_task_complete, terminal_output
            )
            termination_candidate = (
                "explicit_final"
                if is_task_complete
                else "timeout"
                if timeout_occurred
                else "invalid_action"
                if invalid_command_blocked
                else "tool_execution_error"
                if repeated_command_blocked
                else None
            )
            self._write_step_metric(
                {
                    "task_id": self._llm.task_id,
                    "session_id": self._llm.session_id,
                    "round_index": episode + 1,
                    "loop_stage": "planner" if episode == 0 else "replanner",
                    "session_revision": episode + 1,
                    "step": episode + 1,
                    "status": "completed",
                    "parse_status": parse_status,
                    "termination_candidate": termination_candidate,
                    "reasoning_latency_s": llm_latency,
                    "acting_latency_s": acting_latency,
                    "request_latency": llm_latency,
                    "request_latency_ms": llm_latency * 1000,
                    "tool_latency": acting_latency,
                    "tool_latency_ms": acting_latency * 1000,
                    "step_latency_s": time.perf_counter() - step_started,
                    "command_count": len(parsed_response.commands),
                    "llm_calls": 1,
                    "tool_calls": int(bool(parsed_response.commands)),
                    "is_task_complete": is_task_complete,
                    "terminal_timeout": bool(timeout_occurred),
                    "repeated_command_blocked": repeated_command_blocked,
                    "invalid_command_blocked": invalid_command_blocked,
                    "next_turn_temperature": self._llm.temperature,
                    "terminal_output_chars": len(terminal_output),
                    "step_started_at": step_started_at,
                    "step_ended_at": self._utc_now(),
                    "step_started_monotonic_ns": step_started_monotonic_ns,
                    "step_ended_monotonic_ns": time.monotonic_ns(),
                }
            )

            if is_task_complete:
                self._last_termination_reason = "explicit_final"
                break
            # The terminal output is a tool result only when a command batch really ran. A blocked batch
            # (invalid syntax, repeated) produced no execution: its text is the agent's own recovery directive.
            output_meta = {
                "tool_name": "terminal",
                "tool_call_id": f"{self._llm.task_id}:episode-{episode + 1}",
                "commands": [str(getattr(command, "keystrokes", "")) for command in parsed_response.commands],
                "terminal_timeout": bool(timeout_occurred),
            }
            executed = not (invalid_command_blocked or repeated_command_blocked)
            if completion_prompt is not None:
                prompt = completion_prompt
                parts = split_around(prompt, terminal_output, needle_metadata=output_meta) if executed else None
            elif blocked_recovery_prompt is not None:
                prompt = blocked_recovery_prompt
                parts = None
            elif response_feedback:
                prompt = response_feedback + "\n\n" + terminal_output
                parts = (
                    (
                        PromptPart(KIND_AGENT_DIRECTIVE, response_feedback + "\n\n"),
                        PromptPart(KIND_TOOL_OUTPUT, terminal_output, output_meta),
                    )
                    if executed and terminal_output
                    else None
                )
            else:
                prompt = terminal_output
                parts = (PromptPart(KIND_TOOL_OUTPUT, terminal_output, output_meta),) if executed and terminal_output else None
            self._declare_prompt_structure(prompt, parts)

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        task_id = (
            logging_dir.parent.parent.name
            if logging_dir is not None and len(logging_dir.parents) >= 2
            else hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]
        )
        metrics_path = logging_dir / "sglang_pic_metrics.jsonl" if logging_dir else None
        self._llm.set_task_context(task_id, metrics_path)
        self._step_metrics_path = (
            logging_dir / "sglang_agent_steps.jsonl" if logging_dir else None
        )
        self._task_metrics_path = (
            logging_dir / "sglang_agent_task.json" if logging_dir else None
        )
        self._step_rows = []
        self._last_termination_reason = "unclassified"
        self._invalid_action_count = 0
        self._tool_error_count = 0
        self._timeout_count = 0
        if self._step_metrics_path is not None:
            self._step_metrics_path.write_text("", encoding="utf-8")

        task_started_at = self._utc_now()
        task_started = time.perf_counter()
        status = "completed"
        error_type: str | None = None
        try:
            return super().perform_task(instruction, session, logging_dir)
        except Exception as error:
            status = "agent_error"
            error_type = type(error).__name__
            raise
        finally:
            duration = time.perf_counter() - task_started
            fallback_count = sum(
                int(bool(row.get("fallback_reason"))) for row in self._step_rows
            )
            termination_reason = (
                "runtime_error"
                if status == "agent_error"
                else self._last_termination_reason
            )
            failure_category = (
                "success"
                if status == "completed" and termination_reason == "explicit_final"
                else termination_reason
                if termination_reason != "unclassified"
                else "unknown_failure"
            )
            summary = {
                "task_id": task_id,
                "session_id": self._llm.session_id,
                "status": status,
                "error_type": error_type,
                "max_rounds": self._max_episodes,
                "steps": len(self._step_rows),
                "rounds_used": len(self._step_rows),
                "llm_calls": sum(
                    int(row.get("llm_calls") or 0) for row in self._step_rows
                ),
                "tool_calls": sum(
                    int(row.get("tool_calls") or 0) for row in self._step_rows
                ),
                "termination_reason": termination_reason,
                "failure_category": failure_category,
                "verifier_pass": None,
                "agent_duration_s": duration,
                "reasoning_latency_s": sum(
                    float(row["reasoning_latency_s"]) for row in self._step_rows
                ),
                "acting_latency_s": sum(
                    float(row["acting_latency_s"]) for row in self._step_rows
                ),
                "total_request_latency_ms": sum(
                    float(row.get("request_latency_ms") or 0.0)
                    for row in self._step_rows
                ),
                "total_tool_latency_ms": sum(
                    float(row.get("tool_latency_ms") or 0.0)
                    for row in self._step_rows
                ),
                "total_e2e_latency_ms": duration * 1000,
                "invalid_action_count": self._invalid_action_count,
                "tool_error_count": self._tool_error_count,
                "timeout_count": self._timeout_count,
                "fallback_count": fallback_count,
                "steps_per_minute": (
                    60.0 * len(self._step_rows) / duration if duration else None
                ),
                "task_started_at": task_started_at,
                "task_ended_at": self._utc_now(),
            }
            if self._task_metrics_path is not None:
                self._task_metrics_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                (self._task_metrics_path.parent / "trial_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )


class SGLangPICTerminus2Agent(SGLangPICAgent):
    """Terminus-2 parser and completion policy backed by SGLang PIC."""

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "plan": {"type": "string"},
            "commands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keystrokes": {"type": "string"},
                        "duration": {"type": "number"},
                    },
                    "required": ["keystrokes", "duration"],
                    "additionalProperties": False,
                },
            },
            "task_complete": {"type": "boolean"},
        },
        "required": ["analysis", "plan", "commands", "task_complete"],
        "additionalProperties": False,
    }

    @staticmethod
    def name() -> str:
        return "sglang-pic-terminus-2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from terminal_bench.agents.terminus_2.terminus_json_plain_parser import (
            TerminusJSONPlainParser,
        )

        self._terminus2_parser = TerminusJSONPlainParser()
        parser_path = Path(inspect.getfile(TerminusJSONPlainParser))
        self._prompt_template = (
            parser_path.parents[1] / "prompt-templates" / "terminus-json-plain.txt"
        ).read_text(encoding="utf-8")
        self._pending_completion = False

    def _handle_llm_interaction(
        self,
        chat: Any,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
    ) -> _PICAgentResponse:
        logging_path, prompt_path, response_path = logging_paths
        if prompt_path is not None:
            prompt_path.write_text(prompt, encoding="utf-8")
        response = chat.chat(
            prompt,
            response_format=self.RESPONSE_SCHEMA,
            logging_path=logging_path,
        )
        if response_path is not None:
            response_path.write_text(response, encoding="utf-8")

        result = self._terminus2_parser.parse_response(response)
        feedback_parts: list[str] = []
        if result.error:
            feedback_parts.append("ERROR: " + result.error)
        if result.warning:
            feedback_parts.append("WARNINGS: " + result.warning)
        commands = [
            _PICCommand(
                keystrokes=command.keystrokes,
                is_blocking=True,
                timeout_sec=max(0.1, min(float(command.duration), 60.0)),
            )
            for command in result.commands
        ]
        return _PICAgentResponse(
            commands=commands,
            is_task_complete=bool(result.is_task_complete),
            raw_response=response,
            feedback="\n".join(feedback_parts),
        )

    def _resolve_completion(
        self, requested: bool, terminal_output: str
    ) -> tuple[bool, str | None]:
        if not requested:
            self._pending_completion = False
            return False, None
        if self._pending_completion:
            return True, None
        self._pending_completion = True
        return False, (
            "Current terminal state:\n"
            + terminal_output
            + "\n\nAre you sure the task is complete? Inspect the generated "
            "artifacts against every requirement. If no correction is needed, "
            'return valid JSON with "task_complete": true again and no commands.'
        )
