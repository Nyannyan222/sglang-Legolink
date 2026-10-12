"""A strict stand-in for the patched, LegoLink-only SGLang server, at the HTTP contract level. CPU only, standard library only.

It exists so the host half of the architecture can be exercised without a GPU. It is **not** the runtime and says nothing about the runtime's
numerical behaviour or performance. It enforces three things of the real contract:

* a Compile request (``extra_key`` = the block's cache key, no ``pic_link``) registers that key as resident;
* a Link request (``pic_link``) needs every cache key it names to be resident, otherwise it is refused with the runtime's residency-recovery
  JSON (the host then recompiles the missing block and retries);
* a plain generation request (neither of the above) is refused, as the LegoLink-only deployment refuses it.

``StrictFakeRuntime`` is a payload-to-response callable (the shape the adapter's transport has); ``serve`` puts the same object behind
``POST /v1/chat/completions`` so the real HTTP transport code can be exercised too.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from noc_context_manager.sglang_adapter import SGLangRequestError


class StrictFakeRuntime:
    def __init__(self) -> None:
        self.resident: set[str] = set()
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(payload)
            link = payload.get("pic_link")
            if link is None:
                if "extra_key" not in payload:
                    raise SGLangRequestError("LegoLink-only deployment refuses plain generation", reason="incompatible", status=400)
                self.resident.add(payload["extra_key"])  # Compile: prefill the block alone, register it under its cache key
                return self._answer("compiled")
            missing = [key for key in link["cache_keys"] if key not in self.resident]
            if missing:
                detail = {"noc_residency_recovery": {"reason": "scheduler_source_preflight_failed", "missing_cache_keys": missing,
                                                     "stale_cache_keys": [], "incompatible_cache_keys": []}}
                raise SGLangRequestError("LegoLink source preflight failed before destination KV allocation: " + ", ".join(missing)
                                         + " | " + json.dumps(detail), reason="missing", status=400)
            return self._answer('{"commands": [], "is_task_complete": true}')  # Link: copy KV, remap RoPE, repair, decode

    @staticmethod
    def _answer(content: str) -> dict[str, Any]:
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 400, "completion_tokens": 8}}

    def evict_everything(self) -> list[str]:
        """Legal after release: the runtime simply stops holding the blocks."""

        with self._lock:
            gone = sorted(self.resident)
            self.resident.clear()
        return gone

    def compile_requests(self, since: int = 0) -> list[dict[str, Any]]:
        return [r for r in self.requests[since:] if not r.get("pic_link") and "extra_key" in r]

    def link_requests(self, since: int = 0) -> list[dict[str, Any]]:
        return [r for r in self.requests[since:] if r.get("pic_link")]

    def plain_generation_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if not r.get("pic_link") and "extra_key" not in r]


def serve(runtime: StrictFakeRuntime, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Serve ``runtime`` on ``POST /v1/chat/completions``. The caller owns the server (``shutdown()`` / ``server_close()``)."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                body, status = runtime(payload), 200
            except SGLangRequestError as error:
                body, status = {"object": "error", "message": str(error), "code": error.status or 400}, int(error.status or 400)
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # silence the default stderr access log
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
