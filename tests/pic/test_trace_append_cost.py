"""Trace appends must not cost more than the thing they observe.

Measured on 2026-09-02: with CUDA graphs on, the optimized route spent 42.57 s
with tracing and 18.00 s without, same 3/3 and same six LLM calls. 24.6 s --
58% -- was `_append_jsonl` doing `os.makedirs`, `os.open`, `os.write` and
`os.close` for every event, four events per generated token, across a bind
mount. And only the patched deployment pays it: stock SGLang has no trace call,
so it cannot appear on the other arm of any ratio.

The descriptor is now held open. What must not change is the write itself: one
`O_APPEND` write per record and no user-space buffering, because a trace whose
scheduler died on SIGQUIT is exactly the trace this project has had to read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.pic.support import load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


@pytest.fixture(autouse=True)
def clean_descriptors(pic):
    yield
    for descriptor in pic._TRACE_DESCRIPTORS.values():
        try:
            os.close(descriptor)
        except OSError:
            pass
    pic._TRACE_DESCRIPTORS.clear()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_records_are_appended_in_order(pic, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "runtime_trace.jsonl"
    for index in range(5):
        pic._append_jsonl(str(target), {"event": "e", "index": index})
    assert [record["index"] for record in read(target)] == [0, 1, 2, 3, 4]


def test_the_parent_directory_is_created_once_not_per_event(pic, tmp_path: Path) -> None:
    target = tmp_path / "made" / "runtime_trace.jsonl"
    pic._append_jsonl(str(target), {"event": "first"})
    assert target.parent.is_dir()
    calls = {"n": 0}
    real = os.makedirs

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    pic.os.makedirs = counting
    try:
        for _ in range(10):
            pic._append_jsonl(str(target), {"event": "later"})
    finally:
        pic.os.makedirs = real
    assert calls["n"] == 0
    assert len(read(target)) == 11


def test_the_descriptor_is_reused_across_events(pic, tmp_path: Path) -> None:
    target = tmp_path / "runtime_trace.jsonl"
    pic._append_jsonl(str(target), {"event": "first"})
    first = dict(pic._TRACE_DESCRIPTORS)
    pic._append_jsonl(str(target), {"event": "second"})
    assert dict(pic._TRACE_DESCRIPTORS) == first
    assert len(first) == 1


def test_each_record_reaches_the_file_before_the_next_call(pic, tmp_path: Path) -> None:
    # No buffering: a scheduler that dies mid-request must leave its last
    # records on disk, which is how the workspace-capacity failures were read.
    target = tmp_path / "runtime_trace.jsonl"
    pic._append_jsonl(str(target), {"event": "crash_context"})
    assert read(target) == [{"event": "crash_context"}]


def test_a_stale_descriptor_is_reopened_not_raised(pic, tmp_path: Path) -> None:
    target = tmp_path / "runtime_trace.jsonl"
    pic._append_jsonl(str(target), {"event": "before"})
    for path, descriptor in list(pic._TRACE_DESCRIPTORS.items()):
        os.close(descriptor)
        pic._TRACE_DESCRIPTORS[path] = descriptor  # now stale
    pic._append_jsonl(str(target), {"event": "after"})
    assert [record["event"] for record in read(target)] == ["before", "after"]


def test_tracing_never_takes_the_request_down(pic, tmp_path: Path) -> None:
    # A directory where the file should be: every open fails, forever.
    target = tmp_path / "runtime_trace.jsonl"
    target.mkdir()
    pic._append_jsonl(str(target), {"event": "unwritable"})


def test_separate_paths_get_separate_descriptors(pic, tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    pic._append_jsonl(str(first), {"event": "a"})
    pic._append_jsonl(str(second), {"event": "b"})
    assert len(pic._TRACE_DESCRIPTORS) == 2
    assert read(first) == [{"event": "a"}]
    assert read(second) == [{"event": "b"}]
