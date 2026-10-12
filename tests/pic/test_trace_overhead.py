"""Diagnostic tracing must cost nothing while it is disabled.

``_emit_pic_trace`` returns immediately when ``SGLANG_PIC_TRACE_PATH`` is unset,
but Python evaluates its keyword arguments first. Several call sites inside the
KV-link path passed ``_trace_values(<cuda tensor>)``, and ``.cpu()`` on a CUDA
tensor is a device-to-host copy that synchronises the stream. One site also
concatenated a tensor to build its argument.

Profiler-free ABBA runs are the project's formal latency evidence and they run
with tracing disabled, so that synchronisation and allocation sat inside the
measured path for no observable benefit. These tests pin the two properties that
matter: nothing is computed while tracing is off, and a written record is
byte-identical to what the eager implementation produced.
"""

from __future__ import annotations

import json

import pytest
import torch

from .support import load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def pic_module():
    return load_srt_module("pic_runtime", "pic_runtime.py")


class ExplodingSource:
    """Fails loudly if anything materializes it."""

    def __iter__(self):
        raise AssertionError("trace values materialized while tracing disabled")


class CountingSource:
    def __init__(self, items):
        self.items = items
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.items


@pytest.fixture
def tracing_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_PIC_TRACE_PATH", raising=False)


@pytest.fixture
def trace_path(tmp_path, monkeypatch):
    path = tmp_path / "runtime_trace.jsonl"
    monkeypatch.setenv("SGLANG_PIC_TRACE_PATH", str(path))
    return path


def written_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# Nothing is computed while tracing is disabled
# --------------------------------------------------------------------------


def test_values_are_not_materialized_when_tracing_is_disabled(tracing_disabled):
    module = pic_module()

    module._emit_pic_trace("event", slots=module._trace_values(ExplodingSource()))


def test_callable_source_is_not_invoked_when_tracing_is_disabled(tracing_disabled):
    module = pic_module()
    source = CountingSource([1, 2, 3])

    module._emit_pic_trace("event", slots=module._trace_values(source))

    assert source.calls == 0


def test_deferred_field_is_not_built_when_tracing_is_disabled(tracing_disabled):
    module = pic_module()
    source = CountingSource([{"start_token": 0, "end_token": 4}])

    module._emit_pic_trace("event", ranges=module._lazy_trace(source))

    assert source.calls == 0


def test_cuda_tensor_is_not_copied_to_host_when_tracing_is_disabled(tracing_disabled):
    module = pic_module()

    class TensorThatRefusesToSync(torch.Tensor):
        pass

    # A real .cpu() on a CUDA tensor synchronises the stream. Standing in for
    # that here keeps the test hardware-independent while pinning the property.
    calls = {"count": 0}

    def exploding_cpu():
        calls["count"] += 1
        raise AssertionError("device-to-host copy while tracing disabled")

    tensor = torch.arange(4)
    tensor.cpu = exploding_cpu  # type: ignore[method-assign]

    module._emit_pic_trace("event", slots=module._trace_values(tensor))

    assert calls["count"] == 0


# --------------------------------------------------------------------------
# A written record is unchanged
# --------------------------------------------------------------------------


def test_written_record_holds_the_bounded_representation(trace_path):
    module = pic_module()

    module._emit_pic_trace("event", slots=module._trace_values([1, 2, 3]))

    record = written_records(trace_path)[0]
    assert record["event"] == "event"
    assert record["slots"] == {"count": 3, "values": [1, 2, 3], "truncated": False}


def test_written_record_truncates_long_index_lists(trace_path, monkeypatch):
    module = pic_module()
    monkeypatch.setenv("SGLANG_PIC_TRACE_MAX_ITEMS", "4")

    module._emit_pic_trace("event", slots=module._trace_values(list(range(100))))

    slots = written_records(trace_path)[0]["slots"]
    assert slots["count"] == 100
    assert slots["truncated"] is True
    assert slots["values"] == [0, 1, 98, 99]


def test_tensor_source_is_materialized_when_tracing_is_enabled(trace_path):
    module = pic_module()

    module._emit_pic_trace("event", slots=module._trace_values(torch.arange(3)))

    assert written_records(trace_path)[0]["slots"]["values"] == [0, 1, 2]


def test_callable_source_is_materialized_when_tracing_is_enabled(trace_path):
    module = pic_module()
    source = CountingSource(torch.arange(3))

    module._emit_pic_trace("event", slots=module._trace_values(source))

    assert source.calls == 1
    assert written_records(trace_path)[0]["slots"]["values"] == [0, 1, 2]


def test_deferred_field_is_materialized_when_tracing_is_enabled(trace_path):
    module = pic_module()

    module._emit_pic_trace(
        "event",
        ranges=module._lazy_trace(lambda: [{"start_token": 0, "end_token": 4}]),
    )

    assert written_records(trace_path)[0]["ranges"] == [
        {"start_token": 0, "end_token": 4}
    ]


def test_deferred_values_nested_in_a_field_are_resolved(trace_path):
    module = pic_module()

    module._emit_pic_trace(
        "event",
        breakdown={"slots": module._trace_values([7, 8])},
        rows=[module._trace_values([9])],
    )

    record = written_records(trace_path)[0]
    # A record must never carry an unserialized placeholder.
    assert record["breakdown"]["slots"]["values"] == [7, 8]
    assert record["rows"][0]["values"] == [9]


def test_records_stay_json_serializable(trace_path):
    module = pic_module()

    module._emit_pic_trace(
        "event",
        a=module._trace_values(torch.arange(2)),
        b=module._lazy_trace(lambda: {"inner": module._trace_values([5])}),
    )

    record = written_records(trace_path)[0]
    assert record["a"]["values"] == [0, 1]
    assert record["b"]["inner"]["values"] == [5]
