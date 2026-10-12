"""Same-key Compiles must not finish in one prefill batch.

Terminal-Bench C=8 with Link admission crashed the scheduler: three byte-identical
task copies sent a Compile of the same block at the same moment, both landed in
one prefill batch, the first finished Compile released the runtime's local
snapshot, and the second raised ``PIC cannot rebind missing block``
(`results/link_admission_concurrency/c8_crash_diagnosis.md`). The fix is in the
scheduler only: a Compile whose cache key is already in the batch being built
waits one batch and then takes RadixCache's existing duplicate-registration path.

The scheduler needs a GPU deployment to import, so its guard is checked on the
source; the helper runs against the real `pic_runtime` module.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from tests.pic.support import SRT_ROOT, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


def _compile(cache_key):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            custom_params={"sub_context_metadata": {"cache_key": cache_key}}
        )
    )


def _link():
    return SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={"pic_link": {"cache_keys": ["k"]}})
    )


def _native():
    return SimpleNamespace(sampling_params=SimpleNamespace(custom_params=None))


def test_a_compile_of_a_key_already_in_the_batch_is_detected(pic):
    batch = [_native(), _compile("block-a"), _link()]
    assert pic.pic_compile_key_in_batch("block-a", batch) is True


def test_other_keys_links_and_native_requests_do_not_collide(pic):
    batch = [_native(), _compile("block-b"), _link()]
    assert pic.pic_compile_key_in_batch("block-a", batch) is False
    assert pic.pic_compile_key_in_batch("block-a", []) is False


def test_a_request_that_is_not_a_compile_never_collides(pic):
    batch = [_compile("block-a")]
    assert pic.pic_compile_key_in_batch(None, batch) is False


def _prefill_function():
    source = (SRT_ROOT / "managers" / "scheduler.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_new_batch_prefill_raw":
            return node
    raise AssertionError("_get_new_batch_prefill_raw not found")


def _waiting_queue_loop(function):
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "self.waiting_queue"
    ]
    assert len(loops) == 1, "expected one scan of the waiting queue"
    return loops[0]


def test_the_scheduler_defers_a_same_key_compile_before_adding_it():
    loop = _waiting_queue_loop(_prefill_function())
    guards = [
        (index, stmt)
        for index, stmt in enumerate(loop.body)
        if isinstance(stmt, ast.If)
        and "pic_compile_key_in_batch" in ast.unparse(stmt.test)
    ]
    assert len(guards) == 1, "expected exactly one same-key Compile guard"
    index, guard = guards[0]
    text = ast.unparse(guard.test)
    assert "get_pic_compile_cache_key(req.sampling_params)" in text
    assert "adder.can_run_list" in text
    assert "not is_pic_link" in text
    # The request stays in the waiting queue for the next batch.
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Continue)
    # The guard runs before the request is added to the batch.
    add_positions = [
        i for i, stmt in enumerate(loop.body) if "adder.add_one_req(" in ast.unparse(stmt)
    ]
    assert add_positions and index < min(add_positions)


def test_the_link_admission_rules_are_unchanged_by_the_compile_guard():
    loop_text = ast.unparse(_waiting_queue_loop(_prefill_function()))
    # Links remain alone in their prefill forward and the flag still gates only
    # the empty-engine clause (tests/pic/test_link_admission_flag.py checks the
    # details); the Compile guard does not apply to Links.
    assert "not self.running_batch.is_empty() and (not admit_link_while_decoding)" in loop_text
    assert "if adder.can_run_list and get_pic_link_plan(adder.can_run_list[0].sampling_params) is not None:" in loop_text
