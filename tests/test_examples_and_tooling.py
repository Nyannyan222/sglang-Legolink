"""The shipped examples and scripts run: the CPU example in both transports, the OpenAI-compatible proxy in front of the strict fake runtime,
and the server-role checker. Everything here is CPU-only and talks to a fake; nothing proves anything about a GPU runtime."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from strict_fake_runtime import StrictFakeRuntime, serve  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.cpu]


def run_python(*args: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("extra", [[], ["--fake-http"]], ids=["in_process", "fake_http"])
def test_the_compile_link_decode_example_holds_every_check(extra, tmp_path):
    done = run_python("examples/compile_link_decode.py", "--workdir", str(tmp_path), *extra)
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout)
    turns = report["turns"]
    assert turns[0]["compile_requests_sent"] >= 1 and turns[0]["reused_tokens"] == 0  # the first block is compiled
    assert turns[1]["compile_requests_sent"] == 0 and turns[1]["reused_tokens"] > 0  # and reused, not recompiled, a turn later
    assert all(turn["runtime_owned_fields_on_the_wire"] == [] for turn in turns)
    assert report["after_eviction"]["outcome"] == "recovered_after_source_missing"
    assert report["after_eviction"]["plain_generation_requests_ever_sent"] == 0


def test_the_proxy_keeps_an_openai_style_client_on_the_compile_link_path():
    runtime = StrictFakeRuntime()
    server = serve(runtime)
    port = free_port()
    proxy = subprocess.Popen(
        [sys.executable, "scripts/run_openai_proxy.py", "--port", str(port), "--model", "mock-model", "--patched-api-base",
         f"http://127.0.0.1:{server.server_address[1]}/v1", "--variant", "selected_context_radixcache_legolink", "--tokenizer-mode", "regex",
         "--segmentation-mode", "deterministic", "--artifacts-root", str(ROOT / ".tmp" / f"proxy_{port}")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 60
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except Exception:  # noqa: BLE001 - still starting
                assert proxy.poll() is None, "the proxy exited during startup"
                assert time.time() < deadline, "the proxy did not start"
                time.sleep(0.2)
        answers = []
        for turn in range(3):
            messages = [{"role": "system", "content": "You are a careful coding agent. " * 40}, {"role": "user", "content": "Fix the failing parser test. " * 40}]
            if turn:
                messages += [{"role": "assistant", "content": "Running pytest. " * 40}, {"role": "user", "content": "FAILED test_empty " * 40}]
            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", headers={"Content-Type": "application/json"},
                                             data=json.dumps({"model": "mock-model", "messages": messages, "max_tokens": 16}).encode())
            with urllib.request.urlopen(request, timeout=60) as response:
                answers.append(json.loads(response.read())["choices"][0]["message"]["content"])
        assert all(answer for answer in answers)
        assert len(runtime.link_requests()) == 3 and len(runtime.compile_requests()) >= 1
        assert runtime.plain_generation_requests() == []  # the patched server is never asked for plain generation
    finally:
        proxy.terminate()
        try:
            proxy.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proxy.kill()
        server.shutdown()
        server.server_close()


def test_check_server_distinguishes_the_patched_role_from_the_native_role():
    class Plain(StrictFakeRuntime):
        """A server that serves plain generation, i.e. a native/unpatched SGLang."""

        def __call__(self, payload):
            self.requests.append(payload)
            return self._answer("hello")

    strict, plain = StrictFakeRuntime(), Plain()

    # the checker also needs GET /v1/models, which the example's fake does not serve, so wrap the two fakes in a small handler
    import http.server
    import threading

    class Models(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data = json.dumps({"data": [{"id": "mock-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                body, status = self.server.runtime(payload), 200
            except Exception as error:  # noqa: BLE001
                body, status = {"message": str(error)}, 400
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    servers = []
    try:
        for runtime in (strict, plain):
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Models)
            httpd.runtime = runtime
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            servers.append(httpd)
        patched_base, native_base = (f"http://127.0.0.1:{s.server_address[1]}/v1" for s in servers)
        assert run_python("scripts/check_server.py", "--role", "patched", "--api-base", patched_base).returncode == 0
        assert run_python("scripts/check_server.py", "--role", "patched", "--api-base", native_base).returncode == 1  # a server that serves plain generation is not the patched role
        assert run_python("scripts/check_server.py", "--role", "native", "--api-base", native_base).returncode == 0
        assert run_python("scripts/check_server.py", "--role", "native", "--api-base", patched_base).returncode == 1
        assert run_python("scripts/check_server.py", "--role", "patched", "--api-base", f"http://127.0.0.1:{free_port()}/v1", "--timeout", "2").returncode == 2
    finally:
        for httpd in servers:
            httpd.shutdown()
            httpd.server_close()
