"""L3_: a same-key Compile in flight must not corrupt another Compile's completion.

Audit finding L-3 (`pic-token-registry-inflight-overwrite-fatal-pin`, CWE-362 /
CWE-248). `PICRequestTokenRegistry.prepare_request` overwrote
`_blocks[cache_key]` at admission without regard to a Compile of the same key
still running. When the first Compile finished, `pin_finished_compile` re-read
the key's tokens from that shared map, got the *other* request's tokens, found
them absent from the radix tree and raised from the batch-result loop -- which
ends the scheduler process. Two more request-reachable causes reached the same
raise: a Compile aborted after admission, and one whose KV was kept out of the
radix tree by the upstream sentinel `bootstrap_host`.

The invariant these tests pin:

* a cache key is ``absent`` / ``compiling`` / ``resident``;
* a Compile that disagrees with one in flight is refused at admission
  (request-scoped), and one that agrees is attached as its own record;
* completion registers the tokens *its own* request was admitted with;
* a Compile that cannot register degrades to "no LegoLink name" -- traced,
  never raised -- and returns the key to its previous state.

`RadixCache`, `ModelRunner` and the scheduler need a GPU deployment to import,
so the radix tree and runtime are the smallest stand-ins that honour the same
calls; the scheduler wiring is pinned on the source.
"""

from __future__ import annotations

import ast
import gc
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.pic.support import SRT_ROOT, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


# --- stand-ins ---------------------------------------------------------------


class DuplicateError(ValueError):
    pass


@dataclass
class MatchPrefixParams:
    key: object


@dataclass
class RadixKey:
    token_ids: list
    extra_key: object = None


class FakeTree:
    """A radix tree reduced to: which token sequences are resident, and the registry."""

    def __init__(self):
        self.sequences: list[tuple] = []
        self.registered: dict[str, dict] = {}

    def insert(self, tokens):
        self.sequences.append(tuple(tokens))

    def match_prefix(self, params):
        wanted = tuple(params.key.token_ids)
        best = 0
        for sequence in self.sequences:
            shared = 0
            for left, right in zip(sequence, wanted):
                if left != right:
                    break
                shared += 1
            best = max(best, shared)
        return SimpleNamespace(
            device_indices=list(range(best)), last_device_node=object()
        )

    def register_legolink_block(self, **kwargs):
        key = kwargs["cache_key"]
        resident = self.registered.get(key)
        if resident is not None:
            if resident["token_ids"] != kwargs["token_ids"]:
                raise DuplicateError(key)
            return resident
        self.registered[key] = kwargs
        return kwargs


class FakeRuntime:
    def __init__(self):
        self.local_blocks: set[str] = set()
        self.released: list[str] = []

    def rebind_reference_slots(self, cache_key, slots):
        return SimpleNamespace(cache_key=cache_key, slots=list(slots))

    def set_legolink_cache(self, tree_cache):
        pass

    def get_legolink_compatibility(self, adapter_id=None):
        return {"model_fingerprint": "m"}

    def release_local_block(self, cache_key):
        self.released.append(cache_key)
        self.local_blocks.discard(cache_key)


class AbortReason:
    def to_json(self):
        return {"type": "abort"}


class LengthReason:
    def to_json(self):
        return {"type": "length"}


def modules():
    return {
        "sglang.srt.mem_cache.base_prefix_cache": SimpleNamespace(
            MatchPrefixParams=MatchPrefixParams
        ),
        "sglang.srt.mem_cache.radix_cache": SimpleNamespace(
            RadixKey=RadixKey, LegoLinkDuplicateRegistrationError=DuplicateError
        ),
    }


def compile_sp(cache_key, mode=None):
    metadata = {"cache_key": cache_key}
    if mode is not None:
        metadata["pic_storage_mode"] = mode
    return SimpleNamespace(custom_params={"sub_context_metadata": metadata})


class FakeReq:
    """A plain class, like the scheduler's `Req`: instances can be weakly referenced."""


def make_req(cache_key, *, mode=None, finished_reason=None, **overrides):
    request = FakeReq()
    request.sampling_params = compile_sp(cache_key, mode)
    request.extra_key = None
    request.rid = f"rid-{id(request)}"
    request.finished_reason = finished_reason or LengthReason()
    request.to_finish = None
    request.skip_radix_cache_insert = False
    request.lora_id = None
    for name, value in overrides.items():
        setattr(request, name, value)
    return request


class Harness:
    def __init__(self, pic):
        self.pic = pic
        self.registry = pic.PICRequestTokenRegistry()
        self.tree = FakeTree()
        self.runtime = FakeRuntime()
        self.worker = SimpleNamespace(
            model_runner=SimpleNamespace(_pic_runtime=self.runtime)
        )

    def admit(self, cache_key, tokens, **req_overrides):
        request = make_req(cache_key, **req_overrides)
        self.registry.prepare_request(list(tokens), request.sampling_params, owner=request)
        self.runtime.local_blocks.add(cache_key)
        return request

    def finish(self, request, *, insert=None):
        """The result loop: KV enters the tree, then the block is pinned."""

        if insert is not None:
            self.tree.insert(insert)
        with mock.patch.dict(sys.modules, modules()):
            return self.registry.pin_finished_compile(
                request, self.tree, self.worker
            )


@pytest.fixture
def h(pic):
    return Harness(pic)


A = (1, 2, 3)
B = (7, 8, 9, 10)


# --- 1. one Compile ---------------------------------------------------------------


def test_L3_one_compile_moves_absent_compiling_resident(h):
    assert h.registry.compile_state("K") == "absent"
    request = h.admit("K", A)
    assert h.registry.compile_state("K") == "compiling"
    assert h.finish(request, insert=A) is True
    assert h.registry.compile_state("K") == "resident"
    assert h.tree.registered["K"]["token_ids"] == A
    assert not h.registry._compile_records


# --- 2. sequential same-key Compile -------------------------------------------------


def test_L3_a_sequential_same_key_compile_is_a_normal_duplicate(h):
    first = h.admit("K", A)
    assert h.finish(first, insert=A) is True
    second = h.admit("K", A)
    assert h.registry.compile_state("K") == "compiling"
    assert h.finish(second) is True
    assert h.registry.compile_state("K") == "resident"


# --- 3. simultaneous / in-flight same-key Compile -----------------------------------


def test_L3_identical_in_flight_compiles_attach_and_each_finishes_on_its_own(h):
    first = h.admit("K", A)
    second = h.admit("K", A)
    assert len(h.registry._compile_records) == 2
    assert h.finish(first, insert=A) is True
    # The first finishing must not end the second's in-flight life.
    assert h.registry.compile_state("K") == "compiling"
    assert h.finish(second) is True
    assert h.registry.compile_state("K") == "resident"
    assert not h.registry._compile_records


def test_L3_the_audit_trigger_a_conflicting_compile_no_longer_poisons_the_first(h, pic):
    """Trigger 1 of the audit: B overwrote A's tokens, then A's pin raised."""

    first = h.admit("K", A)
    with pytest.raises(pic.PICRuntimeError, match="already in flight"):
        h.admit("K", B)
    # Nothing of the refused request leaked into the shared state.
    assert h.registry._blocks["K"] == A
    assert len(h.registry._compile_records) == 1
    # The first Compile completes with *its* tokens and does not raise.
    assert h.finish(first, insert=A) is True
    assert h.tree.registered["K"]["token_ids"] == A


# --- 4. same-key different-content Compile -------------------------------------------


def test_L3_a_different_content_compile_is_refused_while_one_is_in_flight(h, pic):
    in_flight = h.admit("K", A)  # kept alive: the record dies with its request
    with pytest.raises(pic.PICRuntimeError):
        h.admit("K", B)
    with pytest.raises(pic.PICRuntimeError):
        h.admit("K", A + (4,))
    assert in_flight is not None


def test_L3_a_different_storage_mode_is_refused_while_one_is_in_flight(h, pic):
    in_flight = h.admit("K", A)
    with pytest.raises(pic.PICRuntimeError):
        h.admit("K", A, mode="clone")
    assert in_flight is not None


def test_L3_a_different_content_compile_after_completion_keeps_the_existing_rule(h):
    """Resident + different content: unchanged. The Compile runs and RadixCache
    refuses the duplicate name (first registration wins); nothing is fatal."""

    first = h.admit("K", A)
    assert h.finish(first, insert=A) is True
    second = h.admit("K", B)
    assert h.finish(second, insert=B) is False  # registration refused, not raised
    assert h.tree.registered["K"]["token_ids"] == A


# --- 5. a batch containing two same-key Compiles ---------------------------------------


def test_L3_a_batch_of_two_same_key_compiles_with_different_tokens(h, pic):
    """`/generate` admits every item of a batch before any runs."""

    first = make_req("K")
    second = make_req("K")
    h.registry.prepare_request(list(A), first.sampling_params, owner=first)
    with pytest.raises(pic.PICRuntimeError):
        h.registry.prepare_request(list(B), second.sampling_params, owner=second)
    assert h.finish(first, insert=A) is True


def test_L3_a_batch_of_two_same_key_compiles_with_the_same_tokens(h):
    first = make_req("K")
    second = make_req("K")
    h.registry.prepare_request(list(A), first.sampling_params, owner=first)
    h.registry.prepare_request(list(A), second.sampling_params, owner=second)
    assert h.finish(first, insert=A) is True
    assert h.finish(second) is True


def test_L3_completion_uses_the_requests_own_tokens_not_the_shared_map(h):
    """Even if the shared map is overwritten behind its back, the record decides."""

    request = h.admit("K", A)
    h.registry._blocks["K"] = B  # the old attack: someone else's tokens under the key
    assert h.finish(request, insert=A) is True
    assert h.tree.registered["K"]["token_ids"] == A


# --- 6. failed Compile cleanup -----------------------------------------------------------


def test_L3_an_aborted_compile_does_not_raise_and_leaves_no_phantom_block(h):
    request = h.admit("K", A, finished_reason=AbortReason())
    assert h.finish(request) is False  # nothing in the tree, aborted
    assert h.registry.compile_state("K") == "absent"
    assert "K" not in h.registry._blocks
    assert "K" not in h.registry._storage_modes
    assert "K" not in h.tree.registered
    assert h.runtime.released == ["K"]
    assert not h.registry._compile_records
    assert not h.registry._precompile_state


def test_L3_the_sentinel_bootstrap_host_compile_does_not_raise(h):
    request = h.admit("K", A, skip_radix_cache_insert=True)
    assert h.finish(request) is False
    assert h.registry.compile_state("K") == "absent"


def test_L3_a_compile_whose_block_is_not_in_the_tree_does_not_raise(h):
    request = h.admit("K", A)
    assert h.finish(request) is False  # tree is empty: matched=0, expected=3
    assert h.registry.compile_state("K") == "absent"


def test_L3_a_partially_inserted_block_is_not_registered(h):
    request = h.admit("K", A)
    assert h.finish(request, insert=A[:2]) is False
    assert "K" not in h.tree.registered


def test_L3_a_failed_compile_restores_the_previous_resident_block(h):
    first = h.admit("K", A)
    assert h.finish(first, insert=A) is True
    second = h.admit("K", B, finished_reason=AbortReason())
    assert h.registry._blocks["K"] == B  # admission still overwrites, as before
    assert h.finish(second) is False
    assert h.registry._blocks["K"] == A
    assert h.registry.compile_state("K") == "resident"


def test_L3_a_failure_with_another_compile_still_in_flight_keeps_the_key_compiling(h):
    first = h.admit("K", A, finished_reason=AbortReason())
    second = h.admit("K", A)
    assert h.finish(first) is False
    assert h.registry.compile_state("K") == "compiling"
    # The survivor still owns the shared entry and the runtime snapshot.
    assert h.registry._blocks["K"] == A
    assert h.runtime.released == []
    assert h.finish(second, insert=A) is True


def test_L3_a_queued_compile_dropped_before_it_ran_is_discarded(h):
    request = h.admit("K", A)
    h.registry.discard_compile(request)
    assert h.registry.compile_state("K") == "absent"
    assert not h.registry._compile_records


def test_L3_a_request_that_is_garbage_collected_does_not_wedge_its_key(h, pic):
    request = h.admit("K", A)
    assert h.registry.compile_state("K") == "compiling"
    del request
    gc.collect()
    assert h.registry.compile_state("K") == "absent"
    # A Compile with different content is admissible again.
    later = h.admit("K", B)
    assert h.finish(later, insert=B) is True


def test_L3_finishing_a_request_the_registry_never_admitted_is_a_no_op(h):
    stranger = make_req("K")
    assert h.finish(stranger, insert=A) is False
    assert not h.tree.registered


def test_L3_clone_mode_completion_closes_its_record(h):
    request = h.admit("K", A, mode="clone")
    assert h.finish(request) is False
    assert not h.registry._compile_records
    assert h.registry.compile_state("K") == "resident"


# --- 7. later valid Compile after a failure ---------------------------------------------


def test_L3_a_valid_compile_after_a_failure_registers_normally(h):
    failed = h.admit("K", A, finished_reason=AbortReason())
    assert h.finish(failed) is False
    again = h.admit("K", B)  # different content is fine: the failed one left nothing
    assert h.finish(again, insert=B) is True
    assert h.tree.registered["K"]["token_ids"] == B
    assert h.registry.compile_state("K") == "resident"


def test_L3_clear_forgets_in_flight_compiles(h):
    in_flight = h.admit("K", A)
    h.registry.clear()
    assert in_flight is not None
    assert h.registry.compile_state("K") == "absent"
    assert not h.registry._compile_records and not h.registry._inflight_owners


# --- internal invariants are still loud ------------------------------------------------


def test_L3_a_missing_runtime_is_still_a_runtime_invariant_failure(h, pic):
    request = h.admit("K", A)
    h.worker = SimpleNamespace(model_runner=SimpleNamespace())
    h.tree.insert(A)
    with mock.patch.dict(sys.modules, modules()):
        with pytest.raises(pic.PICRuntimeError, match="runtime is unavailable"):
            h.registry.pin_finished_compile(request, h.tree, h.worker)


# --- scheduler wiring (source) ------------------------------------------------------------


def scheduler_function(name):
    source = (SRT_ROOT / "managers" / "scheduler.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


def test_L3_the_scheduler_identifies_the_compile_by_its_request():
    text = ast.unparse(scheduler_function("handle_generate_request"))
    assert "prepare_request(list(req.origin_input_ids), req.sampling_params, owner=req)" in text


def test_L3_a_queue_abort_discards_the_in_flight_record():
    text = ast.unparse(scheduler_function("abort_request"))
    assert "self.pic_token_registry.discard_compile(req)" in text
    assert text.index("waiting_queue.pop(i)") < text.index("discard_compile(req)")


def test_L3_the_result_processor_still_pins_through_the_registry():
    source = (
        SRT_ROOT / "managers" / "scheduler_components" / "batch_result_processor.py"
    ).read_text(encoding="utf-8")
    assert "self.pic_token_registry.pin_finished_compile(" in source


# --- review additions: commit semantics and "no stale record" -------------------------


def test_L3_a_failed_sibling_does_not_roll_back_a_sibling_that_completed(h):
    """A and B attach to one key; A registers; B's failure must not undo A's state."""

    first = h.admit("K", A)
    second = h.admit("K", A, finished_reason=AbortReason())
    assert h.finish(first, insert=A) is True
    assert h.finish(second) is False
    assert h.registry._blocks["K"] == A
    assert h.registry.compile_state("K") == "resident"
    assert h.tree.registered["K"]["token_ids"] == A


OUTCOMES = [
    ("registered", {}, A, True),
    ("aborted", {"finished_reason": AbortReason()}, None, False),
    ("insert_suppressed", {"skip_radix_cache_insert": True}, None, False),
    ("block_not_in_tree", {}, None, False),
    ("clone_mode", {"mode": "clone"}, None, False),
]


@pytest.mark.parametrize(
    "overrides,insert,expected",
    [(o, i, e) for _, o, i, e in OUTCOMES],
    ids=[name for name, *_ in OUTCOMES],
)
def test_L3_no_record_or_saved_state_survives_any_completion_path(h, overrides, insert, expected):
    request = h.admit("K", A, **overrides)
    assert h.finish(request, insert=insert) is expected
    assert h.registry._compile_records == {}
    assert h.registry._inflight_owners == {}
    assert h.registry._precompile_state == {}
    assert h.registry.compile_state("K") != "compiling"


def test_L3_the_registry_adds_no_ownership_beyond_the_in_flight_records(h, pic):
    """RadixCache sees only the registrations it always saw: one per completed block."""

    first = h.admit("K", A)
    second = h.admit("K", A)
    h.finish(first, insert=A)
    h.finish(second)
    assert list(h.tree.registered) == ["K"]
    assert set(vars(h.registry)) >= {"_blocks", "_storage_modes", "_pinned_blocks"}
