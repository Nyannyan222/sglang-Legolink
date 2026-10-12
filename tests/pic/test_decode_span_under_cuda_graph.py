"""Decode has to stay measurable when CUDA graphs are on.

`forward_decode` emits the decode span, and CUDA graph replay does not go
through it. So the 2026-09-02 local graph run produced a trace with 868 sampling
events and **zero** `native_decode_start`, and the 22.81 s of `link_decode` it
measured could not be attributed to anything. Decode instrumentation and the
configuration the project should actually be running were mutually exclusive.

The replay site now emits the same span. These are AST contract checks: the
call site needs a GPU and a captured graph to execute, so what is testable on
CPU is that the emit exists, wraps the replay, and labels which path produced
it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.pic.support import SRT_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

MODEL_RUNNER = (
    SRT_ROOT
    / "model_executor"
    / "model_runner.py"
)


def emitted_events(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "_emit_pic_trace":
            if node.args and isinstance(node.args[0], ast.Constant):
                names.append(node.args[0].value)
    return names


@pytest.fixture(scope="module")
def tree() -> ast.AST:
    return ast.parse(MODEL_RUNNER.read_text(encoding="utf-8", errors="replace"))


def test_the_decode_span_is_emitted_twice_in_the_file(tree: ast.AST) -> None:
    events = emitted_events(tree)
    # Once in forward_decode, once around graph replay.
    assert events.count("native_decode_start") == 2
    assert events.count("native_decode_complete") == 2


def test_the_replay_site_emits_it(tree: ast.AST) -> None:
    source = MODEL_RUNNER.read_text(encoding="utf-8", errors="replace")
    replay = source.index("ret = self.graph_runner.replay(")
    before = source[:replay]
    after = source[replay:]
    # Opened before the replay, closed after it, so the span covers the replay
    # and nothing else.
    assert before.rindex('_emit_pic_trace("native_decode_start"') > before.rindex(
        "if can_run_graph:"
    )
    assert '"native_decode_complete"' in after[: after.index("return ModelRunnerOutput")]


def test_each_path_says_which_one_it_is(tree: ast.AST) -> None:
    source = MODEL_RUNNER.read_text(encoding="utf-8", errors="replace")
    # Without this a reader compares an eager 41 ms step against a replayed one
    # and believes the difference is the model.
    assert '"execution_path": "eager"' in source
    assert '"execution_path": "cuda_graph_replay"' in source


def test_prefill_replay_does_not_borrow_the_decode_span() -> None:
    source = MODEL_RUNNER.read_text(encoding="utf-8", errors="replace")
    # The replay site serves both modes; only the decode one is instrumented,
    # and native prefill has never had a span on either path.
    assert 'if phase == "native_decode":' in source
    assert "decode_metadata = None" in source
    assert "if decode_metadata is not None:" in source
