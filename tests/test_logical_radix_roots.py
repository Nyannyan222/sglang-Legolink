"""One category, one LOGICAL RadixTree root -- over one shared physical SGLang RadixCache.

These tests pin the host-side logical-root layer (`semantic_cache_index` + `cache_namespace`):

* every persistent category has a root; the two special categories have none;
* lookup is namespace-first, and the mode decides what a miss may fall back to;
* a name in one root can reference the physical block another root compiled, so exact-compatible blocks are
  compiled once, however many logical names they have;
* an alias is derived state that dies with the physical block, and the runtime's own acquire/release stays
  balanced when several names point at one block;
* nothing here creates a second cache, and nothing the runtime owns reaches the wire.

The GPU numerics (Native vs full repair) are covered by the local runtime validation, not by CPU tests.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from noc_context_manager.cache_namespace import (  # noqa: E402
    CacheNamespace,
    CacheNamespacePolicy,
    NamespaceMode,
    NamespaceResolver,
    summarize_namespace_request,
)
from noc_context_manager.compile_policy import CacheCatalog, CompileAction  # noqa: E402
from noc_context_manager.persistent_catalog import PersistentCacheCatalog  # noqa: E402
from noc_context_manager.pic_link_contract import (  # noqa: E402
    PIC_LINK_RUNTIME_OWNED_FIELDS,
    validate_pic_link_request,
)
from noc_context_manager.semantic_cache_index import (  # noqa: E402
    PERSISTENT_ROOTS,
    SemanticCacheIndex,
)
from noc_context_manager.sglang_adapter import NOCSGLangAdapter  # noqa: E402
from tests.test_noc_module_boundaries import manifest as boundary_manifest  # noqa: E402
from tests.test_radix_cache_legolink import MockAllocator, RadixCache, radix_cache  # noqa: E402
from tests.test_semantic_cache_namespaces import (  # noqa: E402
    TEXT,
    TOK,
    WORKSPACE_FROM_TOOL_OUTPUT,
    Server,
    make_adapter,
    make_block,
    plan_for,
    two_namespace_scenario,
)

INSTRUCTION_TO_WORKSPACE = ((CacheNamespace.WORKSPACE, CacheNamespace.INSTRUCTION),)
FIELD_ROOT = {
    "task_instruction": "instruction",
    "tool_definitions": "tool_schema",
    "tool_history_output": "tool_output",
    "workspace_evidence": "workspace",
}


def link_once(adapter, server, block):
    manifest = boundary_manifest([block])
    plan, _requests, _latency = adapter.compile_plan(adapter.planner.plan(manifest, [block]))
    adapter.link(manifest, [block], execution_plan=plan)
    return plan, dict(server.links[-1]["pic_link"])


def instruction_then_workspace(mode, pairs=INSTRUCTION_TO_WORKSPACE):
    """The same exact block, first as task instruction (producer), then as workspace evidence (consumer)."""

    adapter, server = make_adapter(mode, pairs=pairs)
    producer = make_block("task_instruction", task="t1")
    consumer = make_block("workspace_evidence", task="t2", source="same-bytes-elsewhere")
    adapter.compile(producer)
    return adapter, server, producer, consumer


# --- 1. logical roots exist, one per persistent category -------------------------------------


def test_there_are_exactly_five_logical_roots_and_they_are_the_persistent_namespaces():
    assert PERSISTENT_ROOTS == ("instruction", "tool_schema", "conversation", "tool_output", "workspace")
    assert set(PERSISTENT_ROOTS) == {
        item.value
        for item in CacheNamespace
        if CacheNamespacePolicy().is_persistent(item)
    }
    snapshot = make_adapter(NamespaceMode.LOGICAL_NAMESPACE)[0].cache_catalog.semantic_index.snapshot()
    assert snapshot["logical_root_count"] == 5 and snapshot["physical_cache_instances"] == 1
    assert list(snapshot["by_root"]) == list(PERSISTENT_ROOTS)


@pytest.mark.parametrize("mode", list(NamespaceMode))
@pytest.mark.parametrize(("field", "root"), sorted(FIELD_ROOT.items()))
def test_a_block_resolves_to_the_root_of_its_namespace(field, root, mode):
    adapter, _ = make_adapter(mode)
    block = make_block(field)
    resolution = plan_for(adapter, block).namespace_resolutions[0]
    assert resolution.namespace == root and resolution.logical_root == root
    assert resolution.logical_entry_key == block.cache_key
    assert resolution.physical_kv_identity, "the namespace-independent KV identity is reported with it"
    adapter.compile(block)
    entries = adapter.cache_catalog.semantic_index.roots()[root].primary_entries
    assert [entry.logical_entry_key for entry in entries] == [block.cache_key]
    assert entries[0].runtime_cache_key == block.cache_key and not entries[0].is_alias


def test_a_finalized_conversation_block_has_the_conversation_root():
    adapter, _ = make_adapter(NamespaceMode.HARD_NAMESPACE)
    block = make_block("conversation_history", mutable=False)
    assert plan_for(adapter, block).namespace_resolutions[0].logical_root == "conversation"


# --- 2. the two special categories have no root ------------------------------------------------


@pytest.mark.parametrize("mode", list(NamespaceMode))
@pytest.mark.parametrize(
    "block",
    [
        make_block("user_prompt", mutable=False),
        make_block("conversation_history", TEXT + " open", mutable=True),
        make_block("task_metadata"),
    ],
    ids=["mutable_current_turn", "open_conversation_turn", "no_cache"],
)
def test_mutable_current_turn_and_no_cache_never_get_a_persistent_root_entry(block, mode):
    adapter, _ = make_adapter(mode)
    resolution = plan_for(adapter, block).namespace_resolutions[0]
    assert resolution.logical_root is None and resolution.logical_entry_key is None
    adapter.compile(block)  # GLOBAL still compiles it (legacy); a root must not list it either way
    roots = adapter.cache_catalog.semantic_index.roots()
    assert all(not root.entries for root in roots.values())
    with pytest.raises(ValueError):
        adapter.cache_catalog.semantic_index.record_alias(
            namespace=resolution.namespace,
            logical_entry_key=block.cache_key,
            target=SemanticCacheIndexTarget(),
        )


def SemanticCacheIndexTarget():  # noqa: N802 - a tiny stand-in target for the rejection check
    from noc_context_manager.semantic_cache_index import LogicalEntry

    return LogicalEntry("instruction", "k", "k", "identity")


# --- 3. same-root reuse ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(NamespaceMode))
def test_the_second_request_for_a_root_entry_reuses_its_physical_kv(mode):
    adapter, server = make_adapter(mode)
    first = make_block("task_instruction", task="t1")
    second = make_block("task_instruction", task="t2", source="elsewhere")
    assert first.cache_key == second.cache_key
    adapter.compile(first)
    plan = plan_for(adapter, second)
    resolution = plan.namespace_resolutions[0]
    assert resolution.same_namespace_hit and resolution.physical_kv_reused
    assert resolution.alias_kind is None and not resolution.cross_namespace_hit
    assert plan.reuse_cached_contexts and not plan.compile_on_miss_contexts
    assert len(server.compiles) == 1


@pytest.mark.parametrize("mode", list(NamespaceMode))
def test_two_fields_of_one_namespace_share_one_physical_block_in_every_mode(mode):
    """`task_metadata` opts into the instruction root: another name in the SAME root -- not a cross-root alias."""

    adapter, server = make_adapter(mode)
    opted_in = {"model_visible": True, "stable": True, "reusable": True}
    producer = make_block("task_instruction", task="t1")
    sibling = make_block("task_metadata", task="t2", metadata=opted_in, source="elsewhere")
    adapter.compile(producer)
    resolution = plan_for(adapter, sibling).namespace_resolutions[0]
    assert resolution.namespace == "instruction" == resolution.logical_root
    assert resolution.alias_kind == "same_root" and not resolution.cross_namespace_hit
    assert resolution.physical_cache_key == producer.cache_key
    plan, wire = link_once(adapter, server, sibling)
    assert len(server.compiles) == 1 and wire["cache_keys"] == [producer.cache_key]


# --- 4. hard isolation -------------------------------------------------------------------------


def test_hard_namespace_never_reuses_across_roots_even_for_an_exact_match():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.HARD_NAMESPACE)
    resolution = plan_for(adapter, consumer).namespace_resolutions[0]
    assert not resolution.resident and not resolution.fallback_attempted and resolution.alias_kind is None
    assert resolution.missed_reuse_tokens == consumer.token_count
    plan, _ = link_once(adapter, server, consumer)
    assert len(server.compiles) == 2, "isolation costs a second physical copy -- that is what it measures"
    assert {entry["sub_context"]["cache_key"] for entry in server.compiles} == {
        producer.cache_key,
        consumer.cache_key,
    }


# --- 5. logical cross-root alias --------------------------------------------------------------


def test_logical_namespace_resolves_the_workspace_name_to_the_instruction_physical_kv():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    index = adapter.cache_catalog.semantic_index
    resolution = plan_for(adapter, consumer).namespace_resolutions[0]
    assert resolution.logical_root == "workspace" and resolution.cross_namespace_hit
    assert resolution.alias_kind == "cross_root" and resolution.fallback_attempted
    assert resolution.physical_cache_key == producer.cache_key
    assert resolution.physical_alias_count == 2, "the instruction name plus this workspace name"
    assert index.roots()["workspace"].entries == (), "an alias is only recorded once a Link used it"

    _plan, wire = link_once(adapter, server, consumer)
    assert wire["cache_keys"] == [producer.cache_key]
    assert len(server.compiles) == 1, "no second Compile"

    alias = index.roots()["workspace"].alias_entries
    assert [(a.logical_entry_key, a.runtime_cache_key) for a in alias] == [
        (consumer.cache_key, producer.cache_key)
    ]
    # And the next lookup says the name is already recorded.
    again = plan_for(adapter, consumer).namespace_resolutions[0]
    assert again.alias_recorded and again.physical_alias_count == 2


def test_the_cross_root_policy_still_needs_an_explicit_pair():
    adapter, _, _, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE, pairs=())
    resolution = plan_for(adapter, consumer).namespace_resolutions[0]
    assert not resolution.resident and not resolution.fallback_attempted
    assert resolution.missed_reuse_tokens == consumer.token_count


def test_a_pair_is_directional():
    adapter, _, _, consumer = instruction_then_workspace(
        NamespaceMode.LOGICAL_NAMESPACE, pairs=((CacheNamespace.INSTRUCTION, CacheNamespace.WORKSPACE),)
    )
    assert not plan_for(adapter, consumer).namespace_resolutions[0].resident


def test_similarity_is_never_evidence_for_sharing_a_physical_block():
    adapter, server = make_adapter(NamespaceMode.LOGICAL_NAMESPACE, pairs=INSTRUCTION_TO_WORKSPACE)
    adapter.compile(make_block("task_instruction", TEXT + "alpha"))
    near = make_block("workspace_evidence", TEXT + "alpho")
    assert not plan_for(adapter, near).namespace_resolutions[0].resident


# --- 6. global ---------------------------------------------------------------------------------


def test_global_mode_does_not_let_a_namespace_block_an_exact_compatible_reuse():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.GLOBAL, pairs=())
    resolution = plan_for(adapter, consumer).namespace_resolutions[0]
    assert resolution.alias_kind == "cross_root" and resolution.physical_cache_key == producer.cache_key
    link_once(adapter, server, consumer)
    assert len(server.compiles) == 1


def test_global_mode_reuses_for_every_pair_of_roots():
    names = {
        "task_instruction": CacheNamespace.INSTRUCTION,
        "tool_definitions": CacheNamespace.TOOL_SCHEMA,
        "tool_history_output": CacheNamespace.TOOL_OUTPUT,
        "workspace_evidence": CacheNamespace.WORKSPACE,
    }
    for producer_field in names:
        for consumer_field in names:
            if producer_field == consumer_field:
                continue
            adapter, _ = make_adapter(NamespaceMode.GLOBAL)
            adapter.compile(make_block(producer_field, task="a"))
            resolution = plan_for(adapter, make_block(consumer_field, task="b")).namespace_resolutions[0]
            assert resolution.cross_namespace_hit, (producer_field, consumer_field)


# --- 7. physical deduplication -------------------------------------------------------------------


def test_two_logical_names_are_one_compiled_physical_block():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    link_once(adapter, server, consumer)
    snapshot = adapter.cache_catalog.semantic_index.snapshot()
    assert len(adapter.cache_catalog.live_entries()) == 1 == snapshot["physical_blocks"]
    assert snapshot["logical_entries"] == 2 and snapshot["physical_kv_dedup_count"] == 1
    assert snapshot["by_root"]["instruction"]["primary_entries"] == 1
    assert snapshot["by_root"]["workspace"]["alias_entries"] == 1
    assert len(server.compiles) == 1


def test_a_third_name_adds_a_name_not_a_block():
    adapter, server, producer, consumer = instruction_then_workspace(
        NamespaceMode.LOGICAL_NAMESPACE,
        pairs=INSTRUCTION_TO_WORKSPACE + ((CacheNamespace.TOOL_OUTPUT, CacheNamespace.INSTRUCTION),),
    )
    third = make_block("tool_history_output", task="t3", source="third-name")
    link_once(adapter, server, consumer)
    link_once(adapter, server, third)
    snapshot = adapter.cache_catalog.semantic_index.snapshot()
    assert snapshot["physical_blocks"] == 1 and snapshot["logical_entries"] == 3
    assert snapshot["physical_kv_dedup_count"] == 2 and len(server.compiles) == 1
    assert adapter.cache_catalog.semantic_index.logical_refs(producer.cache_key) == 3


def test_the_namespace_is_not_part_of_the_physical_identity():
    a = make_block("task_instruction", task="t1")
    b = make_block("workspace_evidence", task="t2", source="elsewhere")
    resolver = NamespaceResolver(NamespaceMode.LOGICAL_NAMESPACE)
    assert resolver.policy.namespace_of(a) != resolver.policy.namespace_of(b)
    catalog = CacheCatalog()
    ra, rb = resolver.resolve(a, catalog), resolver.resolve(b, catalog)
    assert ra.physical_kv_identity == rb.physical_kv_identity
    # Nor the session or the task: the identity moves only with tokens, content, model and policy versions.
    assert resolver.resolve(make_block("task_instruction", task="other"), catalog).physical_kv_identity == (
        ra.physical_kv_identity
    )
    assert resolver.resolve(make_block("task_instruction", TEXT + "x"), catalog).physical_kv_identity != (
        ra.physical_kv_identity
    )


# --- 8. acquire / release stay balanced with several names -------------------------------------------


def make_runtime_block():
    allocator = MockAllocator()
    tree = RadixCache.create_simulated(mock_allocator=allocator)
    tree.insert(
        radix_cache.InsertParams(key=radix_cache.RadixKey([1, 2, 3]), value=torch.tensor([11, 12, 13]))
    )
    match = tree.match_prefix(radix_cache.MatchPrefixParams(key=radix_cache.RadixKey([1, 2, 3])))
    lock = tree.inc_lock_ref(match.last_device_node)
    tree.register_legolink_block(
        cache_key="producer-key",
        token_ids=(1, 2, 3),
        kv_slots=match.device_indices,
        radix_node=match.last_device_node,
        ownership_lock_params=lock.to_dec_params(),
    )
    return tree, allocator


def test_two_names_for_one_block_in_one_request_acquire_once_and_release_once():
    """The runtime lookup returns one handle per requested position; it pins and releases the distinct ones."""

    tree, _ = make_runtime_block()
    # Both logical names (instruction/X and workspace/X) are linked under the producer's key.
    result = tree.lookup_legolink_blocks(["producer-key", "producer-key"])
    assert len(result.matched_handles) == 2
    distinct = list({handle.cache_key: handle for handle in result.matched_handles}.values())
    assert len(distinct) == 1
    handles = tree.acquire_legolink_blocks(distinct, pin=True)
    assert handles[0].active_link_refcount == 1 and handles[0].pin_count == 1
    tree.release_legolink_blocks(handles, unpin=True)
    assert handles[0].active_link_refcount == 0 and handles[0].pin_count == 0
    with pytest.raises(RuntimeError):
        tree.release_legolink_blocks(handles)  # a second release is the bug this guards against


def test_the_host_names_one_runtime_key_for_every_alias_in_a_request():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    # One request carrying both names: the producer's own block and the alias of it.
    blocks = [producer, consumer]
    manifest = boundary_manifest(blocks)
    plan, requests, _ = adapter.compile_plan(adapter.planner.plan(manifest, blocks))
    adapter.link(manifest, blocks, execution_plan=plan)
    wire = server.links[-1]["pic_link"]
    assert set(wire["cache_keys"]) == {producer.cache_key}, "one physical block, named twice"
    assert requests == 0 and len(server.compiles) == 1
    validate_pic_link_request(dict(wire))


# --- 9. eviction: every alias goes stale with the physical block -------------------------------------


def test_evicting_the_physical_block_makes_every_alias_stale_and_never_serves_stale_reuse():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    index = adapter.cache_catalog.semantic_index
    link_once(adapter, server, consumer)
    assert index.aliases_of(producer.cache_key)

    # The runtime evicts the block; the host learns it from the runtime's own residency answer.
    adapter.cache_catalog.mark_stale(producer.cache_key)

    assert index.aliases_of(producer.cache_key) == [], "all aliases are invalid in the same call"
    assert index.lookup_alias("workspace", consumer.cache_key) is None
    assert index.roots()["instruction"].entries == () and index.roots()["workspace"].entries == ()
    resolution = plan_for(adapter, consumer).namespace_resolutions[0]
    assert not resolution.resident, "no stale reuse"
    log = index.eviction_log[-1]
    assert log["namespace"] == "instruction" and log["runtime_cache_key"] == producer.cache_key
    assert log["aliases_invalidated"] == [
        {"namespace": "workspace", "logical_entry_key": consumer.cache_key}
    ]
    assert log["residency_age_s"] is not None
    assert adapter.cache_catalog.removals_by_namespace["instruction"] == 1


def test_after_eviction_the_block_is_compiled_once_more_and_the_names_resolve_again():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    link_once(adapter, server, consumer)
    adapter.cache_catalog.invalidate(producer.cache_key)
    adapter.compile(producer)  # residency recovery recompiles the producer's key
    assert len(server.compiles) == 2
    link_once(adapter, server, consumer)
    assert len(server.compiles) == 2, "the alias is re-derived from the recompiled block, not recompiled"
    assert [a.logical_entry_key for a in adapter.cache_catalog.semantic_index.aliases_of(producer.cache_key)] == [
        consumer.cache_key
    ]


def test_recompiling_an_alias_view_registers_the_owners_root_and_field_not_the_aliass():
    """Residency recovery compiles the view the Link named; that view stands for the instruction's block."""

    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    view = adapter.planner.plan(boundary_manifest([consumer]), [consumer]).ordered_contexts[0]
    assert view.cache_key == producer.cache_key and view.context_field == "workspace_evidence"
    adapter.cache_catalog.invalidate(producer.cache_key)
    adapter.compile(view)
    row = adapter.cache_catalog.metadata(producer.cache_key)
    assert row["namespace"] == "instruction" and row["context_field"] == "task_instruction"
    assert server.compiles[-1]["sub_context"]["context_field"] == "task_instruction"
    roots = adapter.cache_catalog.semantic_index.roots()
    assert [e.logical_entry_key for e in roots["instruction"].primary_entries] == [producer.cache_key]
    assert roots["workspace"].entries == ()


def test_an_alias_does_not_survive_its_key_being_recompiled_with_different_content():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    link_once(adapter, server, consumer)
    index = adapter.cache_catalog.semantic_index
    adapter.cache_catalog.mark_compiled(
        producer.cache_key, namespace="instruction", physical_kv_identity="a-different-block"
    )
    assert index.lookup_alias("workspace", consumer.cache_key) is None


def test_a_removal_another_process_made_is_noticed_by_the_persistent_catalog(tmp_path):
    path = tmp_path / "catalog.jsonl"
    mine, theirs = PersistentCacheCatalog(path), PersistentCacheCatalog(path)
    adapter = NOCSGLangAdapter(
        model="model",
        patched_transport=Server(),
        tokenizer=TOK,
        cache_catalog=mine,
        semantic_cache_namespace_mode=NamespaceMode.LOGICAL_NAMESPACE,
        cache_namespace_policy=CacheNamespacePolicy(cross_namespace_pairs=INSTRUCTION_TO_WORKSPACE),
    )
    producer = make_block("task_instruction", task="t1")
    consumer = make_block("workspace_evidence", task="t2", source="elsewhere")
    adapter.compile(producer)
    plan, _requests, _ = adapter.compile_plan(adapter.planner.plan(boundary_manifest([consumer]), [consumer]))
    adapter.link(boundary_manifest([consumer]), [consumer], execution_plan=plan)
    assert mine.semantic_index.aliases_of(producer.cache_key)
    theirs.invalidate(producer.cache_key)  # no hook fires in `mine`
    assert mine.semantic_index.aliases_of(producer.cache_key) == [], "validity is derived from the catalog"
    assert mine.semantic_index.sweep() >= 0


# --- 10. persistence invariant -------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [NamespaceMode.HARD_NAMESPACE, NamespaceMode.LOGICAL_NAMESPACE])
def test_no_cache_and_mutable_current_turn_never_become_a_compile_through_prefix_closure(mode):
    adapter, server = make_adapter(mode)
    blocks = [
        make_block("task_metadata", TEXT + "meta"),
        make_block("user_prompt", TEXT + "ask", mutable=False),
        make_block("task_instruction", TEXT + "instr"),
    ]
    manifest = boundary_manifest(blocks)
    plan, requests, _ = adapter.compile_plan(adapter.planner.plan(manifest, blocks))
    assert requests == 0 and server.compiles == []
    assert all(decision.action is CompileAction.KEEP_MUTABLE for decision in plan.decisions)
    assert all(not root.entries for root in adapter.cache_catalog.semantic_index.roots().values())


# --- 11. runtime ownership -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(NamespaceMode))
def test_the_wire_carries_no_runtime_owned_or_namespace_field_in_any_mode(mode):
    adapter, server, producer, consumer = instruction_then_workspace(mode)
    _plan, wire = link_once(adapter, server, consumer)
    assert not set(wire) & PIC_LINK_RUNTIME_OWNED_FIELDS
    validate_pic_link_request(dict(wire))
    assert not any("namespace" in key or "logical" in key or "physical_kv" in key for key in wire)
    for compile_payload in server.compiles:
        assert not any("namespace" in key or "logical_root" in key for key in compile_payload["sub_context"])


# --- 12. what reaches the runtime does not depend on the logical label -----------------------------------


def test_a_block_linked_through_an_alias_sends_the_same_request_as_through_its_own_name():
    def wire_for(consumer_field):
        adapter, server = make_adapter(NamespaceMode.LOGICAL_NAMESPACE, pairs=INSTRUCTION_TO_WORKSPACE)
        producer = make_block("task_instruction", task="t1")
        adapter.compile(producer)
        consumer = (
            producer
            if consumer_field == "task_instruction"
            else make_block(consumer_field, task="t2", source="elsewhere")
        )
        _plan, wire = link_once(adapter, server, consumer)
        wire.pop("request_id", None)
        # Provenance (the occurrence's own id) is the block's, not the physical block's; everything else is.
        for segment in wire.get("physical_block_segments", []):
            segment.pop("sub_context_id", None)
        return wire

    assert wire_for("workspace_evidence") == wire_for("task_instruction")


# --- 14. metrics ------------------------------------------------------------------------------------------


def test_the_request_summary_reports_dedup_and_the_second_compile_it_avoided():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    plan = adapter.planner.plan(boundary_manifest([consumer]), [consumer])
    summary = plan.namespace_summary()
    totals = summary["totals"]
    assert totals["cross_namespace_attempt"] == 1 and totals["cross_namespace_hit"] == 1
    assert totals["logical_lookup_miss"] == 0 and totals["logical_lookup_hit"] == 1
    assert totals["physical_kv_dedup_count"] == 1
    assert totals["cross_namespace_second_compile_avoided"] == 1
    assert totals["physical_kv_reused"] == 1 and totals["reused_token_count"] == consumer.token_count
    assert totals["compiled_token_count"] == 0
    assert summary["by_namespace"]["workspace"]["alias_count"] == 1
    row = plan.namespace_resolutions[0].to_dict()
    for key in (
        "context_field",
        "cache_namespace",
        "logical_root",
        "logical_entry_key",
        "logical_lookup_hit",
        "logical_lookup_miss",
        "cross_namespace_attempt",
        "cross_namespace_hit",
        "physical_kv_identity",
        "physical_kv_reused",
        "second_compile_avoided",
        "cross_namespace_second_compile_avoided",
        "physical_alias_count",
        "physical_block_residency_age_s",
        "lookup_latency_s",
    ):
        assert key in row, key


def test_hard_namespace_reports_a_miss_and_a_compile_instead_of_a_dedup():
    adapter, _, _, consumer = instruction_then_workspace(NamespaceMode.HARD_NAMESPACE)
    totals = adapter.planner.plan(boundary_manifest([consumer]), [consumer]).namespace_summary()["totals"]
    assert totals["physical_kv_dedup_count"] == 0 and totals["logical_lookup_miss"] == 1
    assert totals["cross_namespace_missed_reuse_tokens"] == consumer.token_count


def test_the_catalog_snapshot_exposes_what_a_later_residency_study_needs():
    adapter, server, producer, consumer = instruction_then_workspace(NamespaceMode.LOGICAL_NAMESPACE)
    link_once(adapter, server, consumer)
    adapter.cache_catalog.mark_stale(producer.cache_key)
    snapshot = adapter.cache_catalog.semantic_index.snapshot()
    assert snapshot["physical_removals_by_namespace"] == {"instruction": 1}
    assert snapshot["removed_tokens_by_namespace"]["instruction"] > 0
    assert snapshot["aliases_invalidated"] == 1


# --- one physical cache -----------------------------------------------------------------------------------


def test_the_host_never_creates_a_second_physical_cache():
    forbidden = {"RadixCache", "TokenToKVPoolAllocator", "ReqToTokenPool", "PICRuntime", "MHATokenToKVPool"}
    for path in (ROOT / "noc_context_manager").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                callee = node.func
                name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
                assert name not in forbidden, (path.name, name)
    assert NamespaceResolver(NamespaceMode.LOGICAL_NAMESPACE).describe()["physical_cache_instances"] == 1


def test_the_index_is_a_view_over_the_catalog_not_a_copy_of_it():
    catalog = CacheCatalog()
    index = catalog.semantic_index
    assert isinstance(index, SemanticCacheIndex)
    catalog.mark_compiled("k", namespace="workspace", physical_kv_identity="i", semantic_identity="s")
    assert [e.logical_entry_key for e in index.roots()["workspace"].primary_entries] == ["k"]
    catalog.invalidate("k")
    assert index.roots()["workspace"].entries == ()


# --- the documentation states the architecture it implements ----------------------------------------------


def test_the_architecture_doc_states_the_invariant_and_carries_the_diagram():
    text = (ROOT / "docs" / "logical_radix_roots.md").read_text(encoding="utf-8")
    assert "One category has one logical RadixTree namespace/root," in text
    assert "but all logical roots share the same physical SGLang RadixCache and KV pool." in text
    assert "It must not become a hard physical reuse barrier unless" in text
    order = ["Functional Classification", "Cache Namespace", "Logical RadixTree Roots",
             "Shared Physical RadixCache", "Shared Semantic KV Blocks", "LegoLink", "Repair", "Native Decode"]
    diagram = text[text.index("Host / CPU"):]
    positions = [diagram.index(item) for item in order]
    assert positions == sorted(positions), "the diagram keeps the host -> runtime order"
    for root in PERSISTENT_ROOTS:
        assert f"+-- {root}" in text
