#!/usr/bin/env python3
"""Start the NOC OpenAI-compatible proxy (``POST /v1/chat/completions``) in front of a patched (and optionally an original) SGLang server.

The proxy's agent code imports a few names from the ``terminal_bench`` package. If the real ``terminal-bench`` distribution is installed it is
used; otherwise a minimal, labelled shim from ``terminal_bench_integration.replay`` is installed first, so the proxy runs without it. The shim
only provides the names ``NOCSGLangLLM`` imports; none of the Terminal-Bench harness is involved.

    python scripts/run_openai_proxy.py --help
    python scripts/run_openai_proxy.py --model Qwen/Qwen2.5-7B-Instruct --patched-api-base http://127.0.0.1:30000/v1 ...

Every option is the proxy's own (``terminal_bench_integration/noc_openai_proxy.py``); this wrapper forwards ``sys.argv`` unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terminal_bench_integration.replay import ensure_terminal_bench_importable  # noqa: E402

if __name__ == "__main__":
    using_real_package = ensure_terminal_bench_importable()
    if not using_real_package:
        print("note: the terminal-bench package is not installed; using the minimal compatibility shim", file=sys.stderr)
    from terminal_bench_integration.noc_openai_proxy import main  # noqa: E402

    raise SystemExit(main())
