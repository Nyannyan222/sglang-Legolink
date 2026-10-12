"""Experimental flag: admit a Link while other requests decode.

`results/link_admission_concurrency/scheduler_guard_audit.md` found that the
empty-engine clause of the Link admission guard protects no identified
invariant, while the batch-of-one clauses protect `run_link`'s batch_size == 1
contract. The flag removes only the empty-engine clause, is off by default, and
must leave every Link alone in its own prefill forward.

The scheduler needs a GPU deployment to import, so its guard is checked on the
source; the flag predicate runs against the real `pic_runtime` module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.pic.support import PROJECT_ROOT, SRT_ROOT, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

FLAG = "SGLANG_PIC_ADMIT_LINK_WHILE_DECODING"



@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


def test_the_flag_is_off_by_default(pic, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert pic.link_admission_while_decoding_enabled() is False


@pytest.mark.parametrize("value", ["1", "true"])
def test_the_flag_turns_on_explicitly(pic, monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    assert pic.link_admission_while_decoding_enabled() is True


def _prefill_function():
    source = (SRT_ROOT / "managers" / "scheduler.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_new_batch_prefill_raw":
            return node
    raise AssertionError("_get_new_batch_prefill_raw not found")


def _link_skip_condition(function):
    # The Link guard is the `is_pic_link and (...)` skip. The same-key Compile
    # guard (`not is_pic_link and ...`, test_compile_race_guard.py) also skips
    # with `continue` but is a different rule.
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and isinstance(node.test.values[0], ast.Name)
        and node.test.values[0].id == "is_pic_link"
        and any(isinstance(stmt, ast.Continue) for stmt in node.body)
    ]
    assert len(matches) == 1, "expected exactly one Link skip condition"
    return matches[0].test


def test_the_flag_is_read_once_per_prefill_scheduling_pass():
    body = ast.unparse(_prefill_function())
    assert body.count("link_admission_while_decoding_enabled()") == 1
    assert "admit_link_while_decoding = link_admission_while_decoding_enabled()" in body


def test_only_the_empty_engine_clause_depends_on_the_flag():
    condition = _link_skip_condition(_prefill_function())
    text = ast.unparse(condition)

    # The batch-of-one and chunked-prefill clauses are unconditional.
    assert "adder.can_run_list" in text
    assert "self.chunked_req is not None" in text

    # The running-batch clause exists only together with the flag.
    guarded = [
        node
        for node in ast.walk(condition)
        if isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.And)
        and "self.running_batch.is_empty()" in ast.unparse(node)
        and "is_pic_link" not in ast.unparse(node)
    ]
    assert len(guarded) == 1
    assert "not admit_link_while_decoding" in ast.unparse(guarded[0])
    assert text.count("self.running_batch.is_empty()") == 1


def test_a_link_still_runs_alone_in_its_prefill_forward():
    body = ast.unparse(_prefill_function())
    # Nothing joins a batch whose head is a Link.
    assert (
        "if adder.can_run_list and get_pic_link_plan(adder.can_run_list[0].sampling_params) is not None:\n"
        in body
    )
    link_blocks = [
        node
        for node in ast.walk(_prefill_function())
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "is_pic_link"
        and "release_link_reservation" in ast.unparse(node)
    ]
    # A Link ends the scan right after it is added.
    assert len(link_blocks) == 1
    assert any(isinstance(stmt, ast.Break) for stmt in link_blocks[0].body)

