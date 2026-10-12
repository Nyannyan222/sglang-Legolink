"""One place where a chat-completions request is issued and timed.

There were two. `SGLangPICLLM` posted directly and could stream;
`JSONHTTPTransport`, which is what the NOC pipeline and therefore the formal
harness actually use, posted without streaming. So `stream_metrics` existed,
was wired through the runners, and produced `stream_metrics_status: None` on
all 48 requests of the 2026-09-03 paired run: the capability was on the path
nobody measured with.

That is the third time one capability lived on one of two request paths --
the Terminal-Bench guardrail and the verification probe were the first two, both
on 2026-08-28. Porting a fourth would repeat it, so this module is the path and
both callers delegate to it.

TTFT is the reason it matters. The objective is to lower time to first token,
end-to-end latency is decode-dominated and hides it -- the same non-prefix
requests measured 1.83x on TTFT and 1.08x end to end -- and nothing that runs
through the pipeline could report TTFT at all.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


class ChatStreamError(RuntimeError):
    """A streamed request the server aborted.

    An aborted stream is delivered as one SSE `data: {"error": ...}` frame
    inside an HTTP 200, so a reader that only collects `choices[].delta.content`
    assembles an empty completion and reports success. That is how a LegoLink
    source-preflight abort reached an agent as an empty assistant message and
    became a JSON parse failure (results/cache_residency_correctness/failure_trace.md).
    An error the server took the trouble to send must not look like an answer.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = dict(payload or {})


@dataclass(frozen=True)
class ChatResponse:
    """A completed chat request, with the two timings worth keeping."""

    response: dict[str, Any]
    latency_s: float
    # None when the request was not streamed, which is different from a request
    # that streamed and produced no content: that reports 0 tokens, not no
    # measurement.
    ttft_s: float | None = None
    streamed: bool = False
    # Per-token arrival, kept because inter-token latency is derived from it.
    # The agent computed this before the two paths merged; dropping it here to
    # keep the shared function tidy would have made the merge a regression.
    request_started_ns: int = 0
    token_ready_ns: tuple[int, ...] = ()
    request_completed_ns: int = 0
    saw_chunk: bool = False
    finish_reason: str | None = None
    # Wall-clock anchors (time.time_ns), diagnostics only: the perf_counter
    # values above cannot be compared with another process's clock, and
    # concurrent requests have to be placed on one axis to show they overlapped.
    request_started_unix_ns: int = 0
    first_token_unix_ns: int = 0
    request_completed_unix_ns: int = 0


def post_chat(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    stream: bool = False,
) -> ChatResponse:
    """Issue one chat-completions request.

    Transport errors are raised unchanged. Callers classify them differently --
    the adapter maps them onto `SGLangRequestError` with a reason -- and
    swallowing that distinction here would flatten it.
    """

    if not stream:
        request = _build(endpoint, payload, headers)
        started_unix_ns = time.time_ns()
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return ChatResponse(
            body,
            time.perf_counter() - started,
            request_started_unix_ns=started_unix_ns,
            request_completed_unix_ns=time.time_ns(),
        )

    body = dict(payload)
    body["stream"] = True
    # Without this the usage block never arrives and token accounting, which
    # every downstream metric row reads, silently becomes zero.
    body["stream_options"] = {"include_usage": True}
    request = _build(endpoint, body, headers)

    started_unix_ns = time.time_ns()
    first_token_unix_ns = 0
    started = time.perf_counter()
    started_ns = time.perf_counter_ns()
    ttft: float | None = None
    token_ready_ns: list[int] = []
    saw_chunk = False
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            saw_chunk = True
            error = chunk.get("error")
            if error:
                detail = error if isinstance(error, dict) else {"message": str(error)}
                raise ChatStreamError(
                    str(detail.get("message") or detail),
                    status=detail.get("code"),
                    payload=detail,
                )
            response_id = chunk.get("id") or response_id
            model = chunk.get("model") or model
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                content = (choice.get("delta") or {}).get("content")
                if content:
                    # First chunk carrying content, not first chunk: servers
                    # open with a role-only delta, and timing to that reports a
                    # TTFT no reader ever waited for.
                    if ttft is None:
                        ttft = time.perf_counter() - started
                        first_token_unix_ns = time.time_ns()
                    token_ready_ns.append(time.perf_counter_ns())
                    pieces.append(content)
    completed_ns = time.perf_counter_ns()
    latency = time.perf_counter() - started

    assembled = {
        "id": response_id,
        "model": model,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(pieces)},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {},
    }
    return ChatResponse(
        assembled,
        latency,
        ttft,
        streamed=True,
        request_started_ns=started_ns,
        token_ready_ns=tuple(token_ready_ns),
        request_completed_ns=completed_ns,
        saw_chunk=saw_chunk,
        finish_reason=finish_reason,
        request_started_unix_ns=started_unix_ns,
        first_token_unix_ns=first_token_unix_ns,
        request_completed_unix_ns=time.time_ns(),
    )


def _build(
    endpoint: str, payload: dict[str, Any], headers: dict[str, str] | None
) -> urllib.request.Request:
    merged = {"Content-Type": "application/json", "Accept": "application/json"}
    merged.update(headers or {})
    return urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=merged,
        method="POST",
    )
