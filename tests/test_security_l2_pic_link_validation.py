"""L2_: untyped `pic_link` values must be a request-scoped rejection.

Audit finding L-2 (`pic-link-untyped-value-nonpic-exception-scheduler-fatal`,
CWE-20 / CWE-248). `pic_link` reaches the scheduler as the JSON body's dict,
unchanged. `_prepare_link` then applied `int()`, `len()` and indexing to what it
found, so a wrong-shaped value surfaced as TypeError, ValueError, OverflowError
or KeyError. The scheduler turns only `PICRuntimeError` into a per-request abort
and re-raises anything else out of the event loop, where the process signals its
parent: one malformed request ended the server.

These tests keep the audit's triggering shape -- the real
`PICRequestTokenRegistry.prepare_request` fed a hostile `pic_link` -- and assert
the exception *type*, because the type is the contract: `PICRuntimeError` is
what admission catches. A valid Link is retained as the control, so a validator
that rejected everything would fail here too.
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


KEY = "block-a"
BLOCK_TOKENS = [11, 12, 13, 14]
QUERY_TOKENS = [91, 92, 93]


def compile_params(cache_key):
    return SimpleNamespace(
        custom_params={"sub_context_metadata": {"cache_key": cache_key}}
    )


def link_params(link_plan):
    return SimpleNamespace(custom_params={"pic_link": link_plan})


def registry_with_block(pic):
    registry = pic.PICRequestTokenRegistry()
    registry.prepare_request(list(BLOCK_TOKENS), compile_params(KEY))
    return registry


def valid_plan(**overrides):
    plan = {
        "request_id": "r-1",
        "task_id": "t-1",
        "cache_keys": [KEY],
        "block_ids": [KEY],
        "recompute_mode": "batched",
        "recompute_initial_tokens": 2,
    }
    plan.update(overrides)
    return plan


def admit(pic, plan):
    registry = registry_with_block(pic)
    return registry.prepare_request(list(QUERY_TOKENS), link_params(plan))


# --- control -----------------------------------------------------------------


def test_L2_a_valid_link_is_still_admitted(pic):
    linked = admit(pic, valid_plan())
    assert linked == BLOCK_TOKENS + QUERY_TOKENS


def test_L2_a_valid_link_with_every_optional_field_is_still_admitted(pic):
    plan = valid_plan(
        variant="v",
        profiling_warmup=False,
        concurrency=4,
        adapter_id=None,
        segmentation_policy_version="field-first-v1",
        legolink_policy_version="p1",
        recompute_first_block=True,
        recompute_first_block_tokens=1,
        recompute_tokens_per_block=[3],
    )
    assert admit(pic, plan) == BLOCK_TOKENS + QUERY_TOKENS


def test_L2_absent_optional_fields_keep_their_defaults(pic):
    plan = {"cache_keys": [KEY]}
    assert admit(pic, plan) == BLOCK_TOKENS + QUERY_TOKENS


# --- the audit's trigger shapes, each of which escaped as a non-PIC exception --


AUDIT_PAYLOADS = [
    pytest.param({"recompute_initial_tokens": None}, id="audit-null-recompute"),
    pytest.param({"recompute_initial_tokens": "abc"}, id="audit-string-recompute"),
    pytest.param({"recompute_initial_tokens": [1]}, id="audit-list-recompute"),
    pytest.param(
        {"recompute_initial_tokens": float("inf")}, id="audit-infinite-recompute"
    ),
    pytest.param({"recompute_tokens_per_block": ["x"]}, id="audit-string-window"),
    pytest.param({"recompute_first_block_tokens": "z"}, id="audit-string-first-block"),
    pytest.param({"block_ids": 5}, id="audit-scalar-block-ids"),
    pytest.param({"block_ids": {"k": 1}}, id="audit-dict-block-ids"),
]


@pytest.mark.parametrize("override", AUDIT_PAYLOADS)
def test_L2_the_audited_payloads_are_pic_runtime_errors(pic, override):
    with pytest.raises(pic.PICRuntimeError):
        admit(pic, valid_plan(**override))


@pytest.mark.parametrize("cache_keys", [None, 5, 1.5, True])
def test_L2_cache_keys_of_the_wrong_type_with_a_governance_wire(pic, cache_keys):
    """`validate_execution_intent` called len() on cache_keys before anything checked it."""

    plan = valid_plan(
        cache_keys=cache_keys,
        framing_spec={
            "framing_layout_version": "noc-single-chat-template-split-v1",
            "model_id": "m",
            "tokenizer_fingerprint": "t",
            "chat_template_sha256": "0" * 64,
            "profile_id": "p",
        },
        repair_decision={
            "selected_k": 2,
            "windows": [2],
            "policy": "fixed_unvalidated_debug",
            "policy_version": "repair-decision-v1",
            "framing_layout_version": "noc-single-chat-template-split-v1",
            "quality_profile_id": None,
            "evidence_id": "e",
            "reason": "r",
            "workload_class": "w",
        },
    )
    with pytest.raises(pic.PICRuntimeError):
        admit(pic, plan)


@pytest.mark.parametrize("profile_id", [["x"], {"a": 1}, 7])
def test_L2_an_unhashable_quality_profile_id_is_a_pic_runtime_error(pic, profile_id):
    plan = valid_plan(
        framing_spec={
            "framing_layout_version": "noc-single-chat-template-split-v1",
            "model_id": "m",
            "tokenizer_fingerprint": "t",
            "chat_template_sha256": "0" * 64,
            "profile_id": "p",
        },
        repair_decision={
            "selected_k": 2,
            "windows": [2],
            "policy": "quality_guarded",
            "policy_version": "repair-decision-v1",
            "framing_layout_version": "noc-single-chat-template-split-v1",
            "quality_profile_id": profile_id,
            "evidence_id": "e",
            "reason": "r",
            "workload_class": "w",
        },
        physical_token_layout_version="noc-single-chat-template-split-v1",
        execution_mode="formal",
    )
    with pytest.raises(pic.PICRuntimeError):
        pic.validate_pic_link_request(plan)


# --- mutation table over every field ----------------------------------------

HUGE = 2**200

MUTATIONS = [
    # wrong scalar type
    ("recompute_initial_tokens", "16"),
    ("recompute_initial_tokens", 16.0),
    ("recompute_initial_tokens", True),
    ("recompute_initial_tokens", {"n": 1}),
    ("legolink_k", "16"),
    ("recompute_first_block_tokens", 1.5),
    ("recompute_first_block", "yes"),
    ("recompute_first_block", 1),
    ("concurrency", "4"),
    ("concurrency", True),
    ("profiling_warmup", "false"),
    ("variant", 7),
    ("variant", ["v"]),
    ("adapter_id", 3),
    ("segmentation_policy_version", {"v": 1}),
    ("legolink_policy_version", ["p"]),
    ("promote_reconstructed_group", "true"),
    ("execution_group_key", ["g"]),
    ("request_id", {"id": 1}),
    ("request_id", [1]),
    ("task_id", 1.5),
    ("task_id", True),
    # wrong list type
    ("cache_keys", KEY),
    ("cache_keys", {KEY: 1}),
    ("cache_keys", 5),
    ("block_ids", KEY),
    ("recompute_tokens_per_block", 2),
    ("recompute_tokens_per_block", "2"),
    ("recompute_tokens_per_block", {"0": 2}),
    # wrong list element type
    ("cache_keys", [1]),
    ("cache_keys", [None]),
    ("cache_keys", [KEY, ["nested"]]),
    ("block_ids", [1]),
    ("block_ids", [None]),
    ("recompute_tokens_per_block", [None]),
    ("recompute_tokens_per_block", [True]),
    ("recompute_tokens_per_block", [1.5]),
    ("recompute_tokens_per_block", [[2]]),
    # list shape
    ("cache_keys", []),
    ("block_ids", [KEY, KEY]),
    ("recompute_tokens_per_block", []),
    ("recompute_tokens_per_block", [1, 2]),
    # invalid enum
    ("recompute_mode", "turbo"),
    ("recompute_mode", ""),
    ("recompute_mode", 1),
    ("recompute_mode", None),
    ("recompute_mode", ["batched"]),
    # negative where illegal
    ("recompute_initial_tokens", -1),
    ("legolink_k", -1),
    ("recompute_first_block_tokens", -1),
    ("recompute_tokens_per_block", [-1]),
    ("concurrency", -1),
    # huge integers / overflow-like values
    ("recompute_initial_tokens", HUGE),
    ("recompute_initial_tokens", 2**31),
    ("recompute_first_block_tokens", HUGE),
    ("recompute_tokens_per_block", [HUGE]),
    ("concurrency", HUGE),
    ("recompute_initial_tokens", float("inf")),
    ("recompute_initial_tokens", float("-inf")),
    ("recompute_initial_tokens", float("nan")),
    # unexpected null
    ("cache_keys", None),
    ("recompute_initial_tokens", None),
    ("legolink_k", None),
    # malformed nested
    ("block_ids", [[KEY]]),
    ("block_ids", [{"id": KEY}]),
    ("recompute_tokens_per_block", [{"n": 2}]),
]


@pytest.mark.parametrize(
    "field,value",
    MUTATIONS,
    ids=[f"{field}={value!r}"[:60] for field, value in MUTATIONS],
)
def test_L2_every_mutated_field_is_a_pic_runtime_error(pic, field, value):
    plan = valid_plan()
    plan[field] = value
    if field == "recompute_tokens_per_block" and isinstance(value, list):
        # keep a matching-length control out of the way: the length rule is its own case
        pass
    with pytest.raises(pic.PICRuntimeError):
        admit(pic, plan)


def test_L2_a_missing_required_field_is_a_pic_runtime_error(pic):
    plan = valid_plan()
    del plan["cache_keys"]
    with pytest.raises(pic.PICRuntimeError):
        admit(pic, plan)


@pytest.mark.parametrize("plan", [None, 5, "x", [], [KEY], 1.5, True])
def test_L2_pic_link_that_is_not_an_object_is_a_pic_runtime_error(pic, plan):
    with pytest.raises(pic.PICRuntimeError):
        pic.validate_pic_link_request(plan)


@pytest.mark.parametrize("mode", ["batched", "Tokenwise", "FULL_CAUSAL"])
def test_L2_the_runtime_recompute_modes_are_accepted_in_any_case(pic, mode):
    pic.validate_pic_link_request(valid_plan(recompute_mode=mode))


def test_L2_the_enum_matches_what_run_link_implements(pic):
    """The accepted set must be exactly the modes `run_link` branches on."""

    source = (SRT_ROOT / "pic_runtime.py").read_text(encoding="utf-8")
    for mode in pic.PIC_LINK_RECOMPUTE_MODES:
        assert f'"{mode}"' in source
    assert pic.PIC_LINK_RECOMPUTE_MODES == {"batched", "tokenwise", "full_causal"}


# --- no hostile pic_link may produce anything but PICRuntimeError ---------------


@pytest.mark.parametrize("field,value", MUTATIONS)
def test_L2_no_mutation_escapes_as_another_exception_type(pic, field, value):
    plan = valid_plan()
    plan[field] = value
    try:
        admit(pic, plan)
    except pic.PICRuntimeError:
        return
    except BaseException as error:  # noqa: BLE001 - the point of the test
        pytest.fail(
            f"{field}={value!r} escaped as {type(error).__name__}: {error}"
        )
    pytest.fail(f"{field}={value!r} was admitted")


def test_L2_the_scheduler_still_aborts_only_on_pic_runtime_error():
    """Containment policy is unchanged: validation errors must be PICRuntimeError."""

    source = (SRT_ROOT / "managers" / "scheduler.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    handler_texts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "handle_generate_request":
            for inner in ast.walk(node):
                if isinstance(inner, ast.ExceptHandler):
                    handler_texts.append(ast.unparse(inner))
    joined = "\n".join(handler_texts)
    assert "PICRuntimeError" in joined
    assert "set_finish_with_abort" in joined
