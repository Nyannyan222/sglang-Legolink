#!/usr/bin/env python3
"""Check that a running SGLang server plays the role it was started for. Standard library only.

    python scripts/check_server.py --role patched --api-base http://127.0.0.1:30000/v1
    python scripts/check_server.py --role native  --api-base http://127.0.0.1:30001/v1

* ``patched``: the server answers ``GET /v1/models`` and **refuses** a plain chat request (HTTP 4xx). The LegoLink-only deployment never serves plain
  generation, and Original SGLang is never an internal fallback of it; a patched server that answers a plain request is not the deployment this
  repository describes.
* ``native``: the server answers ``GET /v1/models`` and **serves** a plain chat request (HTTP 200).

Exit 0 when the role holds, 1 when it does not, 2 when the server is unreachable. This checks the role only; it does not exercise Compile or Link
(``examples/compile_link_decode.py --api-base ...`` does).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def request(url: str, payload: dict | None = None, *, timeout: float) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", required=True, choices=("patched", "native"))
    parser.add_argument("--api-base", required=True, help="e.g. http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default=None, help="served model name (default: the first model /v1/models lists)")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base = args.api_base.rstrip("/")

    try:
        status, body = request(f"{base}/models", timeout=args.timeout)
    except (urllib.error.URLError, OSError) as error:
        print(f"unreachable: {error}", file=sys.stderr)
        return 2
    if status != 200:
        print(f"GET {base}/models -> HTTP {status}", file=sys.stderr)
        return 1
    model = args.model or (json.loads(body).get("data") or [{}])[0].get("id")
    if not model:
        print("the server lists no model", file=sys.stderr)
        return 1

    status, body = request(f"{base}/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, timeout=args.timeout)
    served = status == 200
    ok = (not served) if args.role == "patched" else served
    print(json.dumps({"role": args.role, "model": model, "plain_chat_http_status": status, "plain_generation_served": served, "role_holds": ok}, indent=2))
    if not ok:
        print("the patched role requires plain generation to be refused" if args.role == "patched" else "the native role requires plain generation to be served", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
