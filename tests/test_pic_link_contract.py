"""One logical request contract, and a runtime that owns every physical field.

Three builders used to construct `pic_link` by hand and their field sets barely
overlapped: the bridge sent `block_ranges`/`imp_indices`/`mutable_start_token`,
the adapter sent framing policy and equivalence evidence, the Terminal-Bench
agent sent neither. The performance result came out of the first path and the
correctness result out of the second, so "it is one system" was an assumption
rather than something anything checked.

The runtime was already the authority -- `_prepare_link` overwrote every
physical field a host sent -- but nothing said so, nothing rejected a host that
sent them anyway, and a stale `imp_indices` in a saved plan file still read like
it was steering the run. These tests pin the contract at both ends: what a host
may send, what happens when it sends more, and that the three producers resolve
to the same physical plan.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from noc_context_manager.manager import TokenAccountingMode

from noc_context_manager.context_selection import AgentContextManifest
from noc_context_manager.pic_link_contract import (
    PIC_LINK_DEPRECATED_ALIASES,
    PIC_LINK_REQUEST_FIELDS,
    PIC_LINK_RUNTIME_OWNED_FIELDS,
    PICLinkContractError,
    build_pic_link_request,
)
from noc_context_manager.session import SourceSpan
from noc_context_manager.sglang_adapter import NOCSGLangAdapter
from noc_context_manager.sglang_bridge import build_sglang_block_runtime_plan
from noc_context_manager.subcontext_store import SemanticSubContext
from noc_context_manager.tokenizer import RegexTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from tests.pic.support import SRT_ROOT  # noqa: E402

PIC_RUNTIME_PATH = SRT_ROOT / "pic_runtime.py"


@pytest.fixture(scope="module")
def runtime():
    spec = importlib.util.spec_from_file_location(
        "pic_runtime_contract_test_module", PIC_RUNTIME_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def link_params(link_plan):
    return SimpleNamespace(custom_params={"pic_link": link_plan})


def compile_params(cache_key):
    return SimpleNamespace(
        custom_params={"sub_context_metadata": {"cache_key": cache_key}}
    )


def registry_with_blocks(runtime, blocks):
    """A registry holding `blocks` as {cache_key: token_ids}, already compiled."""

    registry = runtime.PICRequestTokenRegistry()
    for cache_key, token_ids in blocks.items():
        registry.prepare_request(list(token_ids), compile_params(cache_key))
    return registry


def resolve(runtime, blocks, link_plan, mutable=(90, 91)):
    registry = registry_with_blocks(runtime, blocks)
    registry.prepare_request(list(mutable), link_params(link_plan))
    return registry.last_resolved_link_plan


# --------------------------------------------------------------------------
# The two copies of the field list
# --------------------------------------------------------------------------


def test_the_host_and_runtime_field_lists_agree(runtime) -> None:
    """The runtime copy is the one that rejects; this one only warns earlier.

    They cannot be a single import -- `external/sglang` is gitignored and its
    package will not import on Windows -- so the drift has to be a test.
    """

    assert runtime.PIC_LINK_REQUEST_FIELDS == PIC_LINK_REQUEST_FIELDS
    assert runtime.PIC_LINK_RUNTIME_OWNED_FIELDS == PIC_LINK_RUNTIME_OWNED_FIELDS
    assert runtime.PIC_LINK_DEPRECATED_ALIASES == PIC_LINK_DEPRECATED_ALIASES
    # And the two halves stay disjoint: a field cannot be both the host's to
    # send and the runtime's to derive.
    assert not PIC_LINK_REQUEST_FIELDS & PIC_LINK_RUNTIME_OWNED_FIELDS


# --------------------------------------------------------------------------
# Task 8: a typo must not be silence
# --------------------------------------------------------------------------


def test_a_misspelt_field_is_refused_by_the_runtime(runtime) -> None:
    with pytest.raises(runtime.PICRuntimeError, match="unknown fields: selected_ordre"):
        resolve(
            runtime,
            {"a": (10, 11, 12)},
            {"cache_keys": ["a"], "selected_ordre": ["a"]},
        )


def test_a_misspelt_field_is_refused_by_the_host_builder() -> None:
    request = build_pic_link_request(
        request_id="r", task_id="t", cache_keys=["a"]
    )
    wire = request.to_wire()
    wire["selected_ordre"] = ["a"]
    from noc_context_manager.pic_link_contract import validate_pic_link_request

    with pytest.raises(PICLinkContractError, match="selected_ordre"):
        validate_pic_link_request(wire)


def test_host_correctness_evidence_is_not_a_request_field(runtime) -> None:
    """These five reached the server and were read by nothing.

    Keeping them is right -- they are how a run defends its own correctness --
    but a request contract is the wrong carrier, because a field nobody reads
    looks exactly like a field that works.
    """

    for name in (
        "selected_order",
        "expected_cache_keys",
        "framing_policy",
        "separator_token_ids",
        "equivalence_validation",
    ):
        with pytest.raises(runtime.PICRuntimeError, match="unknown fields"):
            resolve(runtime, {"a": (10, 11, 12)}, {"cache_keys": ["a"], name: []})


# --------------------------------------------------------------------------
# Task 7: the runtime does not trust a host physical plan
# --------------------------------------------------------------------------


POISON = {
    "block_ranges": [{"cache_key": "a", "start_token": 999, "end_token": 1000}],
    "imp_indices": [7, 3, 1],
    "kvlink": [42],
    "mutable_start_token": 12345,
    "total_token_count": -1,
}


@pytest.mark.parametrize("field_name", sorted(POISON))
def test_a_host_supplied_physical_field_is_refused(runtime, field_name) -> None:
    with pytest.raises(runtime.PICRuntimeError, match="the runtime derives"):
        resolve(
            runtime,
            {"a": (10, 11, 12), "b": (20, 21)},
            {"cache_keys": ["a", "b"], field_name: POISON[field_name]},
        )


def test_a_poisoned_legacy_plan_resolves_exactly_like_a_clean_one(
    runtime, monkeypatch
) -> None:
    """The shim discards the values; it does not honour them.

    A pre-contract plan file carries these fields, and regenerating every saved
    plan is not always possible. Replaying one must give the same physical plan
    as sending the logical request alone -- otherwise the shim would be a second
    contract rather than a way onto the first.
    """

    blocks = {"a": (10, 11, 12), "b": (20, 21)}
    logical = {
        "cache_keys": ["a", "b"],
        "block_ids": ["block-a", "block-b"],
        "recompute_initial_tokens": 1,
    }

    clean = resolve(runtime, blocks, dict(logical))

    monkeypatch.setenv("SGLANG_PIC_ALLOW_LEGACY_LINK_FIELDS", "1")
    poisoned = resolve(runtime, blocks, {**logical, **POISON})

    assert poisoned == clean
    # Not vacuously: the poison values are nowhere in the resolved plan.
    assert clean.mutable_start_token == 5
    assert clean.total_token_count == 7
    assert clean.recompute_positions == (3, 5, 6)


def test_the_legacy_shim_is_off_by_default(runtime) -> None:
    assert runtime.legacy_link_fields_allowed() is False


# --------------------------------------------------------------------------
# Task 6: every producer resolves to the same physical plan
# --------------------------------------------------------------------------


def sub_context(source_id: str, text: str) -> SemanticSubContext:
    return SemanticSubContext.create(
        task_id="task",
        session_revision=1,
        context_field="workspace_evidence",
        title=source_id,
        question_or_objective="request path equivalence",
        raw_text=text,
        source_spans=[SourceSpan(source_id, 0, len(text))],
        token_count=RegexTokenizer().count(text),
        is_mutable=False,
        reuse_candidate=True,
        semantic_independence_reason="test fixture",
        segmentation_confidence=1.0,
        model_fingerprint="model",
        tokenizer_fingerprint=RegexTokenizer.fingerprint,
    )


def adapter_link_plan(contexts) -> dict:
    """The pic_link the live NOC pipeline sends, captured off the transport."""

    calls: list[dict] = []

    def transport(payload):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }

    selected_tokens = sum(value.token_count for value in contexts)
    request_manifest = AgentContextManifest(
        task_id="task",
        agent_id="planner",
        request_id="request",
        session_revision=1,
        objective="test",
        selected_sub_context_ids=tuple(value.sub_context_id for value in contexts),
        selected_cache_keys=tuple(value.cache_key for value in contexts),
        mutable_text="current objective",
        full_session_tokens=selected_tokens + 10,
        candidate_context_tokens=selected_tokens,
        selected_context_tokens=selected_tokens,
        context_reduction_tokens=10,
        context_reduction_ratio=10 / (selected_tokens + 10),
    )
    adapter = NOCSGLangAdapter(
        token_accounting_mode=TokenAccountingMode.ESTIMATE_ONLY,
        model="model",
        patched_transport=transport,
        recompute_initial_tokens=1,
    )
    decisions, _, _ = adapter.compile_selected(request_manifest, list(contexts))
    adapter.link(request_manifest, list(contexts), decisions=decisions)
    return calls[-1]["pic_link"]


def bridge_link_plan(contexts, block_ids) -> dict:
    """The pic_link the plan-file microbenchmarks send."""

    from noc_context_manager.runtime_isolation import PrefillBlock

    blocks = [
        PrefillBlock(
            block_id=block_ids[index],
            cache_key=value.cache_key,
            source_sub_context_id=value.sub_context_id,
            context_field=value.context_field,
            functional_role="evidence",
            text=value.raw_text,
            token_count=value.token_count,
            metadata={"task_id": "task"},
        )
        for index, value in enumerate(contexts)
    ]
    plan = build_sglang_block_runtime_plan(
        blocks,
        request_id="request",
        task_id="task",
        model="model",
        recompute_initial_tokens=1,
        # Same repair policy as the adapter. Until the bridge could say this,
        # the two paths repaired different tokens and the equivalence below
        # could not have held for a reason that had nothing to do with wiring.
        recompute_first_block=True,
        mutable_text="current objective",
    )
    return plan.pic_link_payload(max_tokens=1)["pic_link"]


def agent_link_plan(contexts, block_ids) -> dict:
    """The pic_link the Terminal-Bench agent sends, built through the same door."""

    return build_pic_link_request(
        request_id="request",
        task_id="task",
        variant="adaptive_l1",
        cache_keys=[value.cache_key for value in contexts],
        block_ids=block_ids,
        recompute_mode="batched",
        recompute_initial_tokens=1,
        recompute_first_block=True,
        recompute_first_block_tokens=1,
    ).to_wire()


def test_every_producer_resolves_to_the_same_physical_plan(runtime) -> None:
    """The point of the whole exercise.

    Not "the request JSON looks similar" -- the three still differ in labels,
    repair policy and the token-layout override, which is legitimate. What has
    to match is what the runtime derives from them: the block order, the global
    offsets, the mutable boundary and the sparse-repair positions.
    """

    contexts = [
        sub_context("alpha", "workspace evidence alpha " * 80),
        sub_context("beta", "workspace evidence beta " * 80),
    ]
    blocks = {
        contexts[0].cache_key: (10, 11, 12),
        contexts[1].cache_key: (20, 21),
    }

    # Same blocks, same order, same semantic labels: the three requests differ
    # only in what each producer additionally knows how to say.
    adapter_plan = adapter_link_plan(contexts)
    block_ids = adapter_plan["block_ids"]
    plans = {
        "bridge": bridge_link_plan(contexts, block_ids),
        "adapter": adapter_plan,
        "agent": agent_link_plan(contexts, block_ids),
    }

    # Each producer really is a distinct request; equivalence is a property of
    # the resolution, not of the payloads being identical.
    assert plans["bridge"] != plans["adapter"] != plans["agent"]

    resolved = {}
    for name, plan in plans.items():
        # Whatever else a producer sends, none of it is a physical plan.
        assert not set(plan) & PIC_LINK_RUNTIME_OWNED_FIELDS, name
        resolved[name] = resolve(runtime, blocks, dict(plan))

    reference = resolved["bridge"]
    for name, value in resolved.items():
        assert value.block_ranges == reference.block_ranges, name
        assert value.cache_keys == reference.cache_keys, name
        assert value.mutable_start_token == reference.mutable_start_token, name
        assert value.total_token_count == reference.total_token_count, name
        assert value.recompute_positions == reference.recompute_positions, name
        assert value.linked_token_ids == reference.linked_token_ids, name
        assert (
            value.recompute_tokens_per_block == reference.recompute_tokens_per_block
        ), name

    # And the resolution is the real one, not an empty tuple compared to itself.
    assert reference.linked_token_ids == (10, 11, 12, 20, 21, 90, 91)
    assert reference.mutable_start_token == 5


def test_block_order_is_the_one_thing_the_host_does_decide(runtime) -> None:
    """Reordering the request must reorder the resolved plan.

    Guards the equivalence test above from passing because the runtime ignores
    `cache_keys` order too -- which would make block-level reuse meaningless,
    since ordering is precisely what RadixCache prefix matching cannot express.
    """

    blocks = {"a": (10, 11, 12), "b": (20, 21)}
    forward = resolve(runtime, blocks, {"cache_keys": ["a", "b"]})
    reversed_ = resolve(runtime, blocks, {"cache_keys": ["b", "a"]})

    assert forward.linked_token_ids == (10, 11, 12, 20, 21, 90, 91)
    assert reversed_.linked_token_ids == (20, 21, 10, 11, 12, 90, 91)
    assert forward.block_ranges != reversed_.block_ranges


# --------------------------------------------------------------------------
# Task 3: no producer builds its own field set any more
# --------------------------------------------------------------------------


def pic_link_bindings(tree: ast.AST) -> set[str]:
    """Names in one module that hold a pic_link object.

    Bind-tracking rather than a name heuristic: `link` in the hotset stress
    runner is the whole chat payload, and flagging its `max_tokens` would have
    made this test noise.
    """

    def is_pic_link_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.BoolOp):
            return any(is_pic_link_expr(value) for value in node.values)
        if isinstance(node, ast.Subscript):
            key = node.slice
            return isinstance(key, ast.Constant) and key.value == "pic_link"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"get", "setdefault"} and node.args:
                first = node.args[0]
                return isinstance(first, ast.Constant) and first.value == "pic_link"
        return False

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and is_pic_link_expr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_no_benchmark_runner_writes_an_off_contract_field() -> None:
    """The runners mutate a loaded plan rather than build one.

    They set K, the recompute mode, the diagnostic switches -- legitimately,
    that is what a sweep varies -- but each assignment is a hand-written key,
    and the server used to accept every one of them. A misspelt key here
    produced a run that looked like it swept something and swept nothing.
    """

    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "pic_link" not in text:
            continue
        tree = ast.parse(text)
        bound = pic_link_bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                key = target.slice
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                holder = target.value
                on_pic_link = (
                    isinstance(holder, ast.Name) and holder.id in bound
                ) or (
                    isinstance(holder, ast.Subscript)
                    and isinstance(holder.slice, ast.Constant)
                    and holder.slice.value == "pic_link"
                )
                if on_pic_link and key.value not in PIC_LINK_REQUEST_FIELDS:
                    offenders.append(f"{path.name}:{node.lineno}: {key.value}")
    assert offenders == []


def test_the_audit_sidecar_is_not_mistaken_for_a_generation_parameter() -> None:
    """Moving evidence beside the request adds a top-level key to the payload.

    Two scripts reconstruct the generation parameters by subtracting the known
    request keys from the payload. A new sibling they do not know about becomes
    a phantom sampling parameter, and the generation-parameter hash is what a
    paired A/B uses to claim both arms asked for the same thing.
    """

    for name in (
        "scripts/audit_noc_physical_token_equivalence.py",
        "scripts/run_noc_paired_replay.py",
    ):
        if not (PROJECT_ROOT / name).is_file():
            continue  # a research script the standalone package does not ship
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert '"pic_link_audit",' in text, name


def test_no_producer_hand_builds_a_pic_link_object() -> None:
    """A fourth hand-built producer is how the third one got here."""

    producers = (
        "noc_context_manager/sglang_adapter.py",
        "noc_context_manager/sglang_bridge.py",
        "terminal_bench_integration/sglang_pic_agent.py",
        "scripts/run_terminal_bench_paired_replay.py",
    )
    # Stage 8 (`LogicalPICRequestFramer`, in pic_link_contract.py) is the one
    # place the live adapter's request is built. The adapter reaches the
    # canonical builder through it, so the adapter must name the framer instead
    # of the builder -- and the framer itself must use the builder.
    via_stage8_framer = {"noc_context_manager/sglang_adapter.py"}
    for name in producers:
        if not (PROJECT_ROOT / name).is_file():
            continue  # a research script the standalone package does not ship
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        if name in via_stage8_framer:
            assert "LogicalPICRequestFramer" in text, name
        else:
            assert "build_pic_link_request" in text, name
        # The literal key assignment is what each of them used to do.
        assert '"pic_link": {' not in text, name
        assert 'payload["pic_link"] = {' not in text, name
    contract = (PROJECT_ROOT / "noc_context_manager/pic_link_contract.py").read_text(
        encoding="utf-8"
    )
    framer_body = contract.split("class LogicalPICRequestFramer", 1)[1]
    assert "build_pic_link_request(" in framer_body
    assert "build_pic_link_payload(" in framer_body
    assert '"pic_link": {' not in contract
    assert 'payload["pic_link"] = {' not in contract
