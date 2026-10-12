"""A LegoLink-only server has to warm itself, because SGLang's warmup is refused.

Measured on 2026-09-02, patched arm against a warmed Original: the first Compile
cost 2.564 s against ~0.28 s once warm, the first Link 7.101 s against ~2.0 s.
Those 9.665 s were 48% of everything the optimized arm spent across six calls,
and dropping the first call from both arms moved the ratio from 1.79x to 1.19x.
The cost is the Triton JIT and the FlashInfer plan build, charged to the first
real request because `--skip-server-warmup` is forced on the strict deployment.
"""

from __future__ import annotations

import json

import pytest

from tests.pic.support import load_srt_module, SRT_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "{}") -> None:
        self.status_code = status_code
        self.text = text


class FakeRequests:
    """Records what a warmup would have sent, without a server."""

    def __init__(self, *, post_status: int = 200, probe_status: int = 200) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self.post_status = post_status
        self.probe_status = probe_status

    def get(self, url, **_kwargs):
        self.gets.append(url)
        return FakeResponse(self.probe_status)

    def post(self, url, json=None, **_kwargs):
        self.posts.append((url, json))
        return FakeResponse(self.post_status, text="boom")


@pytest.fixture
def fake_requests(pic, monkeypatch):
    import sys
    import types

    def install(**kwargs):
        module = types.ModuleType("requests")
        stub = FakeRequests(**kwargs)
        module.get = stub.get
        module.post = stub.post
        monkeypatch.setitem(sys.modules, "requests", module)
        monkeypatch.setattr(pic.time, "sleep", lambda _s: None)
        return stub

    return install


def test_the_two_payloads_are_a_compile_and_the_link_that_consumes_it(pic) -> None:
    compile_request, link_request = pic.legolink_warmup_payloads("some/model")
    # A Compile is identified by sub_context metadata carrying a cache key.
    assert compile_request["sub_context"]["cache_key"] == pic._WARMUP_CACHE_KEY
    assert "pic_link" not in compile_request
    # And the Link must ask for exactly the block the Compile registered,
    # otherwise it warms nothing and reports success.
    assert link_request["pic_link"]["cache_keys"] == [pic._WARMUP_CACHE_KEY]
    assert "sub_context" not in link_request
    assert compile_request["model"] == link_request["model"] == "some/model"


def test_a_request_cannot_compile_and_link_at_once(pic) -> None:
    # The runtime rejects that combination outright; a warmup that tripped it
    # would fail on every start.
    compile_request, link_request = pic.legolink_warmup_payloads("m")
    assert not ({"sub_context", "pic_link"} <= set(compile_request))
    assert not ({"sub_context", "pic_link"} <= set(link_request))


def test_it_sends_compile_before_link(pic, fake_requests) -> None:
    stub = fake_requests()
    assert pic.execute_legolink_warmup("http://server:30000", model="m") is True
    assert [payload for _url, payload in stub.posts][0].get("sub_context") is not None
    assert [payload for _url, payload in stub.posts][1].get("pic_link") is not None
    assert len(stub.posts) == 2


def test_it_waits_for_the_server_before_posting(pic, fake_requests) -> None:
    # The skipped-warmup path does not otherwise wait for the app to listen.
    stub = fake_requests()
    pic.execute_legolink_warmup("http://server:30000", model="m")
    assert stub.gets and stub.gets[0].endswith("/model_info")


def test_a_refused_warmup_does_not_fail_startup(pic, fake_requests) -> None:
    stub = fake_requests(post_status=400)
    assert pic.execute_legolink_warmup("http://server:30000", model="m") is False
    # Stopped at the Compile rather than sending a Link that cannot work.
    assert len(stub.posts) == 1


def test_an_unreachable_server_does_not_fail_startup(pic, fake_requests) -> None:
    stub = fake_requests(probe_status=503)
    assert pic.execute_legolink_warmup("http://server:30000", model="m") is False
    assert stub.posts == []


def test_the_warmup_is_on_by_default_and_can_be_turned_off(pic, monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_PIC_WARMUP", raising=False)
    assert pic.legolink_warmup_enabled() is True
    monkeypatch.setenv("SGLANG_PIC_WARMUP", "0")
    assert pic.legolink_warmup_enabled() is False


def test_the_server_only_warms_itself_in_legolink_only_mode() -> None:
    from pathlib import Path

    source = (
        SRT_ROOT
        / "entrypoints"
        / "http_server.py"
    ).read_text(encoding="utf-8", errors="replace")
    assert "if legolink_only_mode_enabled() and legolink_warmup_enabled():" in source
    # Only on the branch where the native warmup was skipped -- a server that
    # ran the native warmup has already paid for itself.
    skipped = source.index("if not server_args.skip_server_warmup:")
    assert source.index("execute_legolink_warmup(") > skipped


def test_the_payloads_are_json_serializable(pic) -> None:
    for payload in pic.legolink_warmup_payloads("m"):
        json.dumps(payload)


def test_the_warmup_shape_matches_the_traffic_not_a_toy(pic, monkeypatch) -> None:
    """The first version warmed a 48-token block with k=8 and saved nothing on Link.

    Compile went 2.564 s to 0.710 s; Link stayed at 8.887 s against a warm
    1.8 s. Triton specializes on constexpr values and FlashInfer plans per
    shape, so a warmup an order of magnitude below the traffic compiles kernels
    the traffic never calls.
    """

    monkeypatch.delenv("SGLANG_PIC_WARMUP_BLOCK_TOKENS", raising=False)
    monkeypatch.delenv("SGLANG_PIC_WARMUP_LEGOLINK_K", raising=False)
    tokens, k = pic._warmup_shape()
    # The 2026-09-02 runs sent 1,556-token selected contexts at k=128.
    assert tokens >= 1024
    assert k == 128
    _compile_request, link_request = pic.legolink_warmup_payloads("m")
    assert link_request["pic_link"]["recompute_initial_tokens"] == 128


def test_a_deployment_can_declare_its_own_shape(pic, monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_PIC_WARMUP_BLOCK_TOKENS", "600")
    monkeypatch.setenv("SGLANG_PIC_WARMUP_LEGOLINK_K", "64")
    tokens, k = pic._warmup_shape()
    assert (tokens, k) == (600, 64)


def test_k_can_never_exceed_the_block_it_recomputes_from(pic, monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_PIC_WARMUP_BLOCK_TOKENS", "32")
    monkeypatch.setenv("SGLANG_PIC_WARMUP_LEGOLINK_K", "128")
    tokens, k = pic._warmup_shape()
    assert k <= tokens


def test_the_warmup_block_is_actually_that_long(pic, monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_PIC_WARMUP_BLOCK_TOKENS", "1536")
    compile_request, _link = pic.legolink_warmup_payloads("m")
    words = compile_request["messages"][0]["content"].split()
    # Filler is one word per token, so the block must be within a small factor
    # of what was asked for -- a warmup that quietly sends 48 tokens is the bug.
    assert 1000 <= len(words) <= 2000


def test_the_warmup_generates_enough_tokens_to_reach_decode(pic) -> None:
    """At max_tokens=1 nothing decodes, so the decode path stays cold.

    The single token comes out of the prefill or Link forward. SGLang's own
    warmup asks for eight for exactly this reason; the 2026-09-02 warmup asked
    for one and left the first real Link at 8.9 s, unchanged from no warmup at
    all.
    """

    for payload in pic.legolink_warmup_payloads("m"):
        assert payload["max_tokens"] >= 8
