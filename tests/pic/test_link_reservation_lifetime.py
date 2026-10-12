"""Reservation ownership and retraction for LegoLink requests.

Two latent defects surfaced while auditing concurrent Link admission
(`results/link_admission_concurrency/invariant_audit.md` §2-§3):

* `_scheduler_reserved_cache_keys` outlived the pins it names. After a Link
  completed the marker stayed on the plan, so a re-admission skipped pinning
  and the next release decremented pins this request never took -- possibly
  another request's.
* A retracted Link request was re-queued with its Link plan. Its re-prefill
  (prompt + generated tokens) cannot match the linked layout; in LegoLink-only
  mode that raises inside ModelRunner and the scheduler process exits.

Ownership rule under test: the marker is present iff the request holds exactly
one scheduler pin on each listed key. These run on CPU against the real
`pic_runtime` module; the scheduler wiring is checked on its source.
"""

from __future__ import annotations

import ast
import collections
from types import SimpleNamespace

import pytest

from tests.pic.support import SRT_ROOT, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

MARKER = "_scheduler_reserved_cache_keys"


@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


@pytest.fixture(autouse=True)
def legolink_only(monkeypatch):
    monkeypatch.setenv("SGLANG_PIC_LEGOLINK_ONLY", "1")
    monkeypatch.setenv("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1")


class _Lookup:
    def __init__(self, keys):
        self.matched_handles = [SimpleNamespace(cache_key=key) for key in keys]
        self.missing_blocks, self.stale_blocks, self.incompatible_blocks = [], [], []


class CountingTree:
    """RadixCache pin accounting, including its clamp-at-zero unpin."""

    def __init__(self):
        self.pins = collections.Counter()

    def lookup_legolink_blocks(self, cache_keys, **_):
        return _Lookup(cache_keys)

    def pin_legolink_blocks(self, handles):
        for key in dict.fromkeys(handle.cache_key for handle in handles):
            self.pins[key] += 1

    def unpin_legolink_blocks(self, cache_keys):
        for key in cache_keys:
            self.pins[key] = max(self.pins[key] - 1, 0)


def link_request(keys, request_id):
    plan = {"request_id": request_id, "cache_keys": list(keys)}
    params = SimpleNamespace(custom_params={"pic_link": plan})
    return SimpleNamespace(rid=request_id, sampling_params=params), plan


def native_request():
    return SimpleNamespace(
        rid="native", sampling_params=SimpleNamespace(custom_params={})
    )


def registry_on(pic, tree):
    registry = pic.PICRequestTokenRegistry()
    registry._tree_cache = tree
    return registry


def run_link_for(pic, tree, plan, *, fail=False):
    class Runtime:
        _legolink_cache = tree

        @pic._guard_legolink_release
        def run_link(self, forward_batch):
            if fail:
                raise pic.PICRuntimeError("injected link failure")
            return "linked"

    batch = SimpleNamespace(pic_request_metadata=[{"link_plan": plan}])
    if fail:
        with pytest.raises(pic.PICRuntimeError):
            Runtime().run_link(batch)
    else:
        assert Runtime().run_link(batch) == "linked"


def test_the_marker_is_cleared_when_the_link_releases_its_pins(pic):
    tree = CountingTree()
    registry = registry_on(pic, tree)
    req, plan = link_request(["a", "b"], "B")

    assert registry.reserve_link_sources(req.sampling_params) == ["a", "b"]
    assert tree.pins == {"a": 1, "b": 1}
    assert plan[MARKER] == ["a", "b"]

    run_link_for(pic, tree, plan)

    assert tree.pins == {"a": 0, "b": 0}
    assert MARKER not in plan


def test_the_marker_is_cleared_when_the_link_raises(pic):
    tree = CountingTree()
    registry = registry_on(pic, tree)
    req, plan = link_request(["a"], "B")
    registry.reserve_link_sources(req.sampling_params)

    run_link_for(pic, tree, plan, fail=True)

    assert tree.pins["a"] == 0
    assert MARKER not in plan


def test_readmission_after_a_completed_link_pins_again(pic):
    tree = CountingTree()
    registry = registry_on(pic, tree)
    req, plan = link_request(["a"], "B")
    registry.reserve_link_sources(req.sampling_params)
    run_link_for(pic, tree, plan)

    registry.reserve_link_sources(req.sampling_params)

    assert tree.pins["a"] == 1, "a re-admitted Link must not run on unpinned sources"
    assert plan[MARKER] == ["a"]


def test_a_completed_link_cannot_release_another_requests_reservation(pic):
    tree = CountingTree()
    registry = registry_on(pic, tree)
    b, b_plan = link_request(["a"], "B")
    c, c_plan = link_request(["a"], "C")
    registry.reserve_link_sources(b.sampling_params)
    registry.reserve_link_sources(c.sampling_params)
    assert tree.pins["a"] == 2

    run_link_for(pic, tree, b_plan)
    assert tree.pins["a"] == 1

    # B is retracted afterwards, and even re-enters the guarded Link path by
    # mistake: none of it may consume the pin C still holds.
    assert registry.retracted_link_abort_reason(b) is not None
    run_link_for(pic, tree, b_plan)
    assert tree.pins["a"] == 1
    assert c_plan[MARKER] == ["a"]


def test_retracting_a_link_request_aborts_instead_of_replaying(pic):
    tree = CountingTree()
    registry = registry_on(pic, tree)
    req, plan = link_request(["a"], "B")
    registry.reserve_link_sources(req.sampling_params)
    run_link_for(pic, tree, plan)

    reason = registry.retracted_link_abort_reason(req)

    assert isinstance(reason, str) and "retract" in reason.lower()
    assert tree.pins["a"] == 0


def test_retraction_releases_a_still_held_reservation_exactly_once(pic):
    tree = CountingTree()
    registry = registry_on(pic, tree)
    held, held_plan = link_request(["a"], "B")
    other, _ = link_request(["a"], "C")
    registry.reserve_link_sources(held.sampling_params)
    registry.reserve_link_sources(other.sampling_params)

    assert registry.retracted_link_abort_reason(held) is not None
    assert registry.retracted_link_abort_reason(held) is not None

    assert tree.pins["a"] == 1
    assert MARKER not in held_plan


def test_a_retracted_native_request_is_left_to_the_native_path(pic):
    registry = registry_on(pic, CountingTree())

    assert registry.retracted_link_abort_reason(native_request()) is None


# --------------------------------------------------------------------------
# Scheduler wiring. The scheduler cannot be imported without a GPU deployment,
# so its use of the rule above is checked on the source.
# --------------------------------------------------------------------------


def _scheduler_ast():
    source = (SRT_ROOT / "managers" / "scheduler.py").read_text(encoding="utf-8")
    return ast.parse(source)


def test_no_retracted_request_is_requeued_without_the_link_filter():
    for node in ast.walk(_scheduler_ast()):
        if not isinstance(node, ast.For):
            continue
        requeues = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_add_request_to_queue"
            for call in ast.walk(node)
        )
        if requeues:
            assert ast.unparse(node.iter) != "retracted_reqs", (
                f"scheduler.py:{node.lineno} re-queues retracted requests "
                "without _pic_filter_retracted_reqs"
            )


def test_both_retraction_sites_use_the_link_filter():
    source = ast.unparse(_scheduler_ast())
    assert source.count("self._pic_filter_retracted_reqs(retracted_reqs)") == 2


def test_the_link_filter_aborts_through_the_tokenizer():
    helpers = [
        node
        for node in ast.walk(_scheduler_ast())
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pic_filter_retracted_reqs"
    ]
    assert len(helpers) == 1
    body = ast.unparse(helpers[0])
    assert "retracted_link_abort_reason" in body
    assert "AbortReq" in body
    assert "FINISH_ABORT" in body
