"""Overlap suppression belongs to the PIC batch, not to the deployment.

`PICRuntime.has_link` only fires on an extend batch holding exactly one
request, so a server-wide `--disable-overlap-schedule` makes every decode step
pay for a constraint that belongs to the link batches. The 2026-09-02 local
terminus2 trace put numbers on it: 10 links, 1,325 decode steps, decode and
sampling strictly sequential.

These tests cover the predicate that replaces the server-wide answer. They run
on CPU against the real `pic_runtime` module; the scheduler wiring that calls
it needs a GPU deployment and is not exercised here.
"""

from __future__ import annotations

import pytest

from tests.pic.support import load_srt_module, SRT_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


class FakeSamplingParams:
    def __init__(self, **custom_params) -> None:
        self.custom_params = custom_params


class FakeRequest:
    def __init__(self, sampling_params) -> None:
        self.sampling_params = sampling_params


class FakeBatch:
    def __init__(self, *reqs) -> None:
        self.reqs = list(reqs)


@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


def link_request(pic) -> FakeRequest:
    return FakeRequest(FakeSamplingParams(pic_link={"request_id": "r1"}))


def compile_request(pic) -> FakeRequest:
    return FakeRequest(
        FakeSamplingParams(sub_context_metadata={"cache_key": "block-1"})
    )


def plain_request() -> FakeRequest:
    return FakeRequest(FakeSamplingParams())


def test_a_link_batch_needs_the_boundary(pic) -> None:
    assert pic.batch_needs_synchronous_boundary(FakeBatch(link_request(pic)))


def test_a_compile_batch_needs_it_too(pic) -> None:
    # Compile registers blocks the RadixCache then hands to a later link. It has
    # never run under overlap either, because the server-wide flag covered it.
    assert pic.batch_needs_synchronous_boundary(FakeBatch(compile_request(pic)))


def test_a_decode_batch_does_not(pic) -> None:
    # This is the whole point: these are the 1,325 steps that were paying.
    batch = FakeBatch(plain_request(), plain_request(), plain_request())
    assert not pic.batch_needs_synchronous_boundary(batch)


def test_one_pic_request_in_a_mixed_batch_is_enough(pic) -> None:
    batch = FakeBatch(plain_request(), link_request(pic), plain_request())
    assert pic.batch_needs_synchronous_boundary(batch)


def test_absent_batch_and_absent_params_are_not_boundaries(pic) -> None:
    # The scheduler applies the predicate to self.last_batch, which is None on
    # the first iteration and after an idle stretch.
    assert not pic.batch_needs_synchronous_boundary(None)
    assert not pic.batch_needs_synchronous_boundary(FakeBatch())
    assert not pic.batch_needs_synchronous_boundary(FakeBatch(FakeRequest(None)))


def test_the_flag_defaults_to_todays_behaviour(pic, monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_PIC_PER_BATCH_OVERLAP", raising=False)
    assert pic.per_batch_overlap_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
def test_the_flag_opts_in(pic, monkeypatch, value: str) -> None:
    monkeypatch.setenv("SGLANG_PIC_PER_BATCH_OVERLAP", value)
    assert pic.per_batch_overlap_enabled() is True


def test_a_link_request_still_gets_a_batch_to_itself() -> None:
    """The predicate is safe only while this holds.

    `has_link` requires ``len(metadata) == 1``, so a link request that gets
    batched with anything else silently does not link -- it answers through the
    native path and the trace just shows one link fewer. Keeping overlap on
    raises concurrency, which is exactly the condition that would start merging
    batches, so the admission guard becomes load-bearing here.
    """

    scheduler = _scheduler_source()
    assert (
        "if is_pic_link and (\n"
        "                adder.can_run_list\n"
        "                or (\n"
        "                    not self.running_batch.is_empty()\n"
        "                    and not admit_link_while_decoding\n"
        "                )\n"
        "                or self.chunked_req is not None\n"
        "            ):\n"
        "                continue"
    ) in scheduler
    # And nothing joins a link request once it is in the list.
    assert (
        "if adder.can_run_list and get_pic_link_plan(\n"
        "                adder.can_run_list[0].sampling_params\n"
        "            ) is not None:\n"
        "                break"
    ) in scheduler


def test_the_arrival_guard_accepts_overlap_only_behind_the_flag() -> None:
    scheduler = _scheduler_source()
    assert "and not per_batch_overlap_enabled()" in scheduler
    assert "SGLANG_PIC_PER_BATCH_OVERLAP=1" in scheduler
    # Both sides of a PIC batch are fenced, not just the batch itself.
    assert (
        "need_pic_sync = batch_needs_synchronous_boundary(\n"
        "                batch\n"
        "            ) or batch_needs_synchronous_boundary(self.last_batch)"
    ) in scheduler
    assert "or need_pic_sync" in scheduler


def test_the_fence_never_fires_with_nothing_in_flight() -> None:
    """The first Compile batch of the 2026-09-02 local A/B killed the scheduler.

    `event_loop_overlap` calls `pop_and_process()` unconditionally once
    `is_disable_overlap_for_batch` returns True, so returning True with an empty
    `result_queue` raises `IndexError: pop from an empty deque` and the run dies
    on its first PIC request. Both upstream conditions imply a predecessor
    already; this one has to check.
    """

    scheduler = _scheduler_source()
    assert (
        "if per_batch_overlap_enabled() and len(self.result_queue) > 0:"
        in scheduler
    )


def _scheduler_source() -> str:
    from pathlib import Path

    return (
        SRT_ROOT
        / "managers"
        / "scheduler.py"
    ).read_text(encoding="utf-8", errors="replace")
