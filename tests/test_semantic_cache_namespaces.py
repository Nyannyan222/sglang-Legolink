"""Functional fields, cache namespaces and physical KV identity are three different things.

These tests pin the host-side classification and namespace layer:

* Step 2 classifies deterministically first and asks an (injected, mocked) Jev
  only about an event the table cannot decide;
* a cache namespace is a pure function of the field, never part of the compiled
  KV's identity, and never a second physical cache;
* GLOBAL / HARD_NAMESPACE / LOGICAL_NAMESPACE differ only in which already
  compiled entries the *host* is willing to name in a Link -- the Link request
  itself stays the contract the runtime validates;
* finalized segmentation boundaries do not move when a source grows.

Nothing here claims namespaces improve reuse. They make it measurable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noc_context_manager.atomic_units import AtomicUnitBuilder  # noqa: E402
from noc_context_manager.cache_namespace import (  # noqa: E402
    NAMESPACE_NOT_PERSISTENT,
    CacheNamespace,
    CacheNamespacePolicy,
    NamespaceMode,
    NamespaceResolver,
    physical_kv_identity,
)
from noc_context_manager.compile_policy import CompileAction  # noqa: E402
from noc_context_manager.context_selection import RuleBasedSemanticContextSelector  # noqa: E402
from noc_context_manager.field_classification import (  # noqa: E402
    SOURCE_DETERMINISTIC,
    SOURCE_EXPLICIT_FALLBACK,
    SOURCE_JEV,
    DeterministicFieldClassifier,
    FieldClassificationError,
    HybridFieldClassifier,
    JevFieldClassifier,
)
from noc_context_manager.manager import IncrementalContextManager, TokenAccountingMode  # noqa: E402
from noc_context_manager.pic_link_contract import (  # noqa: E402
    PIC_LINK_RUNTIME_OWNED_FIELDS,
    validate_pic_link_request,
)
from noc_context_manager.segmentation_validation import SemanticSegmentationPipeline  # noqa: E402
from noc_context_manager.semantic_segmentation import OpenAICompatibleSemanticSegmenter  # noqa: E402
from noc_context_manager.session import (  # noqa: E402
    CONTEXT_FIELDS,
    EventType,
    FieldFirstExtractor,
    SessionEventCollector,
    SourceSpan,
)
from noc_context_manager.sglang_adapter import NOCSGLangAdapter  # noqa: E402
from noc_context_manager.subcontext_store import (  # noqa: E402
    SemanticSubContext,
    SQLiteSubContextStore,
)
from noc_context_manager.token_accounting import ModelTokenCount  # noqa: E402
from tests.test_noc_module_boundaries import manifest as boundary_manifest  # noqa: E402
from tests.test_token_accounting_contract import chat_model_tokenizer  # noqa: E402

TOK = chat_model_tokenizer()
TEXT = "shared reusable evidence block " * 30


def make_block(
    field: str,
    text: str = TEXT,
    *,
    mutable: bool = False,
    reuse: bool = True,
    task: str = "task",
    metadata: dict | None = None,
    source: str | None = None,
) -> SemanticSubContext:
    ids = TOK.tokenize(text)
    return SemanticSubContext.create(
        task_id=task,
        session_revision=1,
        context_field=field,
        title=source or field,
        question_or_objective="namespace fixture",
        raw_text=text,
        source_spans=[SourceSpan(source or f"{task}-{field}", 0, len(text))],
        token_count=len(ids),
        token_accounting=ModelTokenCount(
            value=len(ids), tokenizer_fingerprint=TOK.fingerprint, model_id=TOK.model_id
        ),
        is_mutable=mutable,
        reuse_candidate=reuse,
        semantic_independence_reason="fixture",
        segmentation_confidence=1.0,
        model_fingerprint="model",
        tokenizer_fingerprint=TOK.fingerprint,
        token_ids=ids,
        metadata=metadata,
    )


class Server:
    """A scripted SGLang endpoint that records every request it receives."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 1},
        }

    @property
    def compiles(self) -> list[dict]:
        return [item for item in self.payloads if "sub_context" in item]

    @property
    def links(self) -> list[dict]:
        return [item for item in self.payloads if "pic_link" in item]


def make_adapter(mode, *, pairs=(), server=None):
    server = server or Server()
    adapter = NOCSGLangAdapter(
        model="model",
        patched_transport=server,
        tokenizer=TOK,
        semantic_cache_namespace_mode=mode,
        cache_namespace_policy=CacheNamespacePolicy(cross_namespace_pairs=pairs),
    )
    return adapter, server


def collect(events):
    collector = SessionEventCollector("task")
    collector.add_events(events)
    return collector


# --- 1. deterministic field classification ------------------------------------


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        (EventType.USER_REQUEST, "user_prompt"),
        (EventType.TOOL_DEFINITION, "tool_definitions"),
        (EventType.TOOL_OUTPUT, "tool_history_output"),
        (EventType.TOOL_CALL, "tool_history_output"),
        (EventType.WORKSPACE_FILE, "workspace_evidence"),
        (EventType.TASK_INSTRUCTION, "task_instruction"),
        (EventType.CONVERSATION_MESSAGE, "conversation_history"),
        (EventType.TASK_METADATA, "task_metadata"),
    ],
)
def test_structurally_known_events_get_their_field_deterministically(event_type, expected):
    collector = collect([(event_type, "s", "text", {})])
    classification = DeterministicFieldClassifier().classify(collector.events()[0])
    assert classification.field == expected
    assert classification.confidence == 1.0
    assert classification.source == SOURCE_DETERMINISTIC
    extraction = FieldFirstExtractor().extract(collector.events())
    assert [name for name, sources in extraction.fields.items() if sources] == [expected]


def test_the_table_classifier_declines_what_the_table_does_not_cover():
    collector = collect([(EventType.UNCLASSIFIED, "s", "who knows", {})])
    assert DeterministicFieldClassifier().classify(collector.events()[0]) is None


def test_an_unclassified_event_fails_extraction_when_no_classifier_is_configured():
    collector = collect([(EventType.UNCLASSIFIED, "s", "who knows", {})])
    with pytest.raises(FieldClassificationError):
        FieldFirstExtractor().extract(collector.events())


# --- 2. Jev fallback --------------------------------------------------------


class MockJev:
    def __init__(self, response):
        self.response = response
        self.requests: list[dict] = []

    def __call__(self, request):
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def hybrid(response, **kwargs):
    mock = MockJev(response)
    return HybridFieldClassifier(jev=JevFieldClassifier(mock), **kwargs), mock


def test_jev_is_not_invoked_for_structurally_known_events():
    classifier, mock = hybrid({"field": "workspace_evidence", "confidence": 0.99})
    collector = collect(
        [
            (EventType.USER_REQUEST, "a", "do it", {}),
            (EventType.TOOL_DEFINITION, "b", "{}", {}),
            (EventType.TOOL_OUTPUT, "c", "out", {}),
            (EventType.WORKSPACE_FILE, "d", "code", {}),
        ]
    )
    FieldFirstExtractor(classifier=classifier).extract(collector.events())
    assert mock.requests == []
    assert classifier.jev_invocations == 0


def test_jev_is_invoked_for_an_ambiguous_event_once_and_only_picks_from_the_closed_set():
    classifier, mock = hybrid({"field": "tool_history_output", "confidence": 0.93, "reason": "log"})
    collector = collect([(EventType.UNCLASSIFIED, "s", "make: *** error 2", {"k": 1})])
    extractor = FieldFirstExtractor(classifier=classifier)
    first = extractor.extract(collector.events())
    second = extractor.extract(collector.events())  # extraction runs several times a turn
    assert len(mock.requests) == 1, "the answer is remembered, so a block cannot change field"
    assert first.fields["tool_history_output"][0].metadata["field_classification"]["source"] == SOURCE_JEV
    assert second.fields["tool_history_output"][0].raw_text == "make: *** error 2"
    request = mock.requests[0]
    assert request["schema"]["properties"]["field"]["enum"] == list(CONTEXT_FIELDS)
    assert request["event"]["metadata_keys"] == ["k"]


@pytest.mark.parametrize(
    "response",
    [
        {"field": "tool_history_output", "confidence": 0.2},
        {"field": "not_a_field", "confidence": 0.99},
        {"field": "tool_history_output", "confidence": 0.99, "cache_key": "x"},
        {"confidence": 0.99},
        RuntimeError("jev down"),
    ],
)
def test_a_low_confidence_or_malformed_or_failed_jev_answer_is_not_guessed_at(response):
    classifier, _ = hybrid(response)
    collector = collect([(EventType.UNCLASSIFIED, "s", "text", {})])
    with pytest.raises(FieldClassificationError):
        FieldFirstExtractor(classifier=classifier).extract(collector.events())


def test_an_explicit_fallback_is_recorded_as_a_fallback_not_as_jev():
    classifier, _ = hybrid({"field": "tool_history_output", "confidence": 0.1}, fallback_field="conversation_history")
    collector = collect([(EventType.UNCLASSIFIED, "s", "text", {})])
    extraction = FieldFirstExtractor(classifier=classifier).extract(collector.events())
    recorded = extraction.fields["conversation_history"][0].metadata["field_classification"]
    assert recorded["source"] == SOURCE_EXPLICIT_FALLBACK
    assert recorded["confidence"] == 0.0


def test_a_hidden_source_is_never_sent_to_jev():
    classifier, mock = hybrid({"field": "workspace_evidence", "confidence": 0.99})
    collector = collect([(EventType.UNCLASSIFIED, "s", "secret answer", {"hidden": True})])
    extraction = FieldFirstExtractor(classifier=classifier).extract(collector.events())
    assert mock.requests == []
    assert extraction.filtered_event_ids


# --- 3-5. namespace mapping ---------------------------------------------------


def test_field_to_namespace_mapping_is_one_policy():
    policy = CacheNamespacePolicy()
    assert policy.namespace_for("task_instruction") is CacheNamespace.INSTRUCTION
    assert policy.namespace_for("tool_definitions") is CacheNamespace.TOOL_SCHEMA
    assert policy.namespace_for("tool_history_output") is CacheNamespace.TOOL_OUTPUT
    assert policy.namespace_for("conversation_history") is CacheNamespace.CONVERSATION
    assert policy.namespace_for("workspace_evidence") is CacheNamespace.WORKSPACE
    assert policy.namespace_for("user_prompt") is CacheNamespace.MUTABLE_CURRENT_TURN
    assert policy.namespace_for("task_metadata") is CacheNamespace.NO_CACHE
    assert {policy.namespace_for(name) for name in CONTEXT_FIELDS}  # every field is mapped


def test_task_metadata_is_cached_only_when_explicitly_model_visible_stable_and_reusable():
    policy = CacheNamespacePolicy()
    assert policy.namespace_for("task_metadata", explicitly_cacheable=True) is CacheNamespace.INSTRUCTION
    partial = make_block("task_metadata", metadata={"model_visible": True, "stable": True})
    assert policy.namespace_of(partial) is CacheNamespace.NO_CACHE
    full = make_block("task_metadata", metadata={"model_visible": True, "stable": True, "reusable": True})
    assert policy.namespace_of(full) is CacheNamespace.INSTRUCTION


def test_conversation_history_is_reusable_only_once_finalized():
    policy = CacheNamespacePolicy()
    open_turn = make_block("conversation_history", mutable=True)
    settled = make_block("conversation_history", mutable=False)
    assert policy.namespace_of(open_turn) is CacheNamespace.MUTABLE_CURRENT_TURN
    assert policy.namespace_of(settled) is CacheNamespace.CONVERSATION
    assert policy.is_persistent(policy.namespace_of(settled))
    assert not policy.is_persistent(policy.namespace_of(open_turn))


def plan_for(adapter, block):
    return adapter.planner.plan(boundary_manifest([block]), [block])


def test_the_current_user_prompt_is_not_persisted_by_default():
    block = make_block("user_prompt", mutable=False, reuse=True)
    for mode in (NamespaceMode.HARD_NAMESPACE, NamespaceMode.LOGICAL_NAMESPACE):
        adapter, _ = make_adapter(mode)
        plan = plan_for(adapter, block)
        assert plan.decisions[0].action is CompileAction.KEEP_MUTABLE
        assert plan.decisions[0].reason == NAMESPACE_NOT_PERSISTENT
        assert plan.namespace_resolutions[0].namespace == "mutable_current_turn"
        assert plan.planning_invariant_violations() == ()
    # GLOBAL is the pre-namespace behaviour and is left exactly as it was.
    adapter, _ = make_adapter(NamespaceMode.GLOBAL)
    assert plan_for(adapter, block).decisions[0].action is CompileAction.COMPILE


def test_finalized_conversation_history_becomes_reusable_and_an_open_one_does_not():
    adapter, _ = make_adapter(NamespaceMode.HARD_NAMESPACE)
    settled = plan_for(adapter, make_block("conversation_history", mutable=False))
    assert settled.decisions[0].action is CompileAction.COMPILE
    assert settled.namespace_resolutions[0].namespace == "conversation"
    open_turn = make_block("conversation_history", TEXT + " open", mutable=True)
    assert plan_for(adapter, open_turn).decisions[0].action is CompileAction.KEEP_MUTABLE


# --- 6. same-namespace reuse / physical identity ------------------------------


def test_physical_identity_ignores_field_namespace_task_and_session_but_not_tokens():
    a = make_block("workspace_evidence", task="t1")
    b = make_block("tool_history_output", task="t2", source="elsewhere")
    assert a.cache_key != b.cache_key, "the cache key carries the field today"
    assert physical_kv_identity(a) == physical_kv_identity(b)
    assert physical_kv_identity(a) != physical_kv_identity(make_block("workspace_evidence", TEXT + " more"))


def test_a_block_with_no_token_ids_has_no_physical_identity():
    ids = TOK.tokenize(TEXT)
    text_only = SemanticSubContext.create(
        task_id="t", session_revision=1, context_field="workspace_evidence", title="x",
        question_or_objective="x", raw_text=TEXT, source_spans=[SourceSpan("s", 0, len(TEXT))],
        token_count=len(ids), is_mutable=False, reuse_candidate=True,
        semantic_independence_reason="x", segmentation_confidence=1.0,
    )
    assert physical_kv_identity(text_only) is None


@pytest.mark.parametrize("mode", list(NamespaceMode))
def test_same_namespace_reuse_resolves_to_the_same_physical_entry_in_every_mode(mode):
    adapter, server = make_adapter(mode)
    first = make_block("workspace_evidence", task="t1")
    second = make_block("workspace_evidence", task="t2", source="other-source")
    assert first.cache_key == second.cache_key
    adapter.compile(first)
    plan = plan_for(adapter, second)
    resolution = plan.namespace_resolutions[0]
    assert resolution.resident and resolution.same_namespace_hit
    assert resolution.physical_cache_key == first.cache_key
    assert plan.reuse_cached_contexts, "linked from the existing entry"
    assert len(server.compiles) == 1


# --- 7-9. isolation, fallback, no duplicate KV --------------------------------


def two_namespace_scenario(mode, pairs):
    """A block compiled under tool_output, then the same tokens arrive as workspace evidence."""

    adapter, server = make_adapter(mode, pairs=pairs)
    produced = make_block("tool_history_output", task="t1")
    wanted = make_block("workspace_evidence", task="t2", source="same-bytes-elsewhere")
    adapter.compile(produced)
    return adapter, server, produced, wanted


WORKSPACE_FROM_TOOL_OUTPUT = ((CacheNamespace.WORKSPACE, CacheNamespace.TOOL_OUTPUT),)


def test_hard_namespace_mode_never_looks_across_namespaces():
    adapter, server, produced, wanted = two_namespace_scenario(
        NamespaceMode.HARD_NAMESPACE, WORKSPACE_FROM_TOOL_OUTPUT
    )
    plan = plan_for(adapter, wanted)
    resolution = plan.namespace_resolutions[0]
    assert not resolution.resident
    assert not resolution.fallback_attempted and not resolution.cross_namespace_hit
    assert resolution.missed_reuse_tokens == wanted.token_count
    assert plan.compile_on_miss_contexts and not plan.reuse_cached_contexts
    adapter.compile_plan(plan)
    assert len(server.compiles) == 2, "isolation costs a second physical copy -- that is what it measures"


def test_global_mode_lets_an_exact_physical_match_serve_a_block_of_another_field():
    """GLOBAL: the namespace never stops exact-compatible reuse (it used to be field-partitioned)."""

    adapter, server, produced, wanted = two_namespace_scenario(
        NamespaceMode.GLOBAL, WORKSPACE_FROM_TOOL_OUTPUT
    )
    resolution = plan_for(adapter, wanted).namespace_resolutions[0]
    assert resolution.resident and resolution.cross_namespace_hit
    assert resolution.physical_cache_key == produced.cache_key
    assert resolution.missed_reuse_tokens == 0


def test_the_field_partitioned_global_registry_is_still_reproducible_on_request():
    adapter = NOCSGLangAdapter(
        model="model",
        patched_transport=Server(),
        tokenizer=TOK,
        semantic_cache_namespace_mode=NamespaceMode.GLOBAL,
        global_physical_dedup=False,
    )
    adapter.compile(make_block("tool_history_output", task="t1"))
    wanted = make_block("workspace_evidence", task="t2", source="same-bytes-elsewhere")
    resolution = plan_for(adapter, wanted).namespace_resolutions[0]
    assert not resolution.resident and not resolution.cross_namespace_hit
    assert resolution.missed_reuse_tokens == wanted.token_count


def test_logical_namespace_fallback_is_off_until_a_pair_is_named():
    adapter, _, _, wanted = two_namespace_scenario(NamespaceMode.LOGICAL_NAMESPACE, ())
    resolution = plan_for(adapter, wanted).namespace_resolutions[0]
    assert not resolution.fallback_attempted and not resolution.cross_namespace_hit
    assert resolution.missed_reuse_tokens == wanted.token_count


def test_logical_namespace_fallback_resolves_through_a_compatible_entry_and_shares_one_kv():
    adapter, server, produced, wanted = two_namespace_scenario(
        NamespaceMode.LOGICAL_NAMESPACE, WORKSPACE_FROM_TOOL_OUTPUT
    )
    manifest = boundary_manifest([wanted])
    plan = adapter.planner.plan(manifest, [wanted])
    resolution = plan.namespace_resolutions[0]
    assert resolution.cross_namespace_hit and resolution.fallback_attempted
    assert resolution.physical_cache_key == produced.cache_key
    assert resolution.missed_reuse_tokens == 0
    assert plan.ordered_contexts[0].cache_key == produced.cache_key
    assert plan.ordered_contexts[0].sub_context_id == wanted.sub_context_id, "provenance stays the block's own"

    plan, requests, _ = adapter.compile_plan(plan)
    assert requests == 0
    result = adapter.link(manifest, [wanted], execution_plan=plan)

    wire = server.links[-1]["pic_link"]
    assert wire["cache_keys"] == [produced.cache_key]
    assert len(server.compiles) == 1, "no second Compile: one physical KV, two logical names"
    assert len(adapter.cache_catalog._entries) == 1
    assert result.route.endswith("legolink")


def test_fallback_needs_the_exact_physical_identity_not_similarity():
    adapter, server = make_adapter(NamespaceMode.LOGICAL_NAMESPACE, pairs=WORKSPACE_FROM_TOOL_OUTPUT)
    adapter.compile(make_block("tool_history_output", TEXT + "alpha"))
    near = make_block("workspace_evidence", TEXT + "alpho")
    resolution = plan_for(adapter, near).namespace_resolutions[0]
    assert not resolution.cross_namespace_hit and resolution.missed_reuse_tokens == 0


def test_a_stale_entry_is_never_an_alias_target():
    adapter, _, produced, wanted = two_namespace_scenario(
        NamespaceMode.LOGICAL_NAMESPACE, WORKSPACE_FROM_TOOL_OUTPUT
    )
    adapter.cache_catalog.mark_stale(produced.cache_key)
    resolution = plan_for(adapter, wanted).namespace_resolutions[0]
    assert not resolution.cross_namespace_hit


# --- numerical behaviour does not depend on the mode -------------------------


def link_wire(mode):
    adapter, server = make_adapter(mode)
    blocks = [make_block("task_instruction"), make_block("tool_definitions", TEXT + "schema")]
    manifest = boundary_manifest(blocks)
    plan = adapter.planner.plan(manifest, blocks)
    plan, _, _ = adapter.compile_plan(plan)
    adapter.link(manifest, blocks, execution_plan=plan)
    wire = dict(server.links[-1]["pic_link"])
    wire.pop("request_id", None)
    return wire, [item["sub_context"]["cache_key"] for item in server.compiles]


def test_the_link_request_is_identical_in_all_three_modes_when_nothing_crosses():
    reference = link_wire(NamespaceMode.GLOBAL)
    assert link_wire(NamespaceMode.HARD_NAMESPACE) == reference
    assert link_wire(NamespaceMode.LOGICAL_NAMESPACE) == reference


def test_global_mode_plans_exactly_what_a_planner_without_a_resolver_plans():
    blocks = [make_block("task_instruction"), make_block("workspace_evidence", TEXT + "w")]
    adapter, _ = make_adapter(NamespaceMode.GLOBAL)
    manifest = boundary_manifest(blocks)
    with_resolver = adapter.planner.plan(manifest, blocks)
    adapter.planner.namespace_resolver = None
    without = adapter.planner.plan(manifest, blocks)
    assert with_resolver.decisions == without.decisions
    assert with_resolver.cached_cache_keys == without.cached_cache_keys
    assert without.namespace_resolutions == ()
    assert "namespace_resolution" not in without.to_dict()
    assert "namespace_resolution" in with_resolver.to_dict()


# --- 13/14. ownership and the native path ---------------------------------


def test_the_host_payload_never_carries_a_runtime_owned_field_even_through_an_alias():
    adapter, server, produced, wanted = two_namespace_scenario(
        NamespaceMode.LOGICAL_NAMESPACE, WORKSPACE_FROM_TOOL_OUTPUT
    )
    manifest = boundary_manifest([wanted])
    plan, _, _ = adapter.compile_plan(adapter.planner.plan(manifest, [wanted]))
    adapter.link(manifest, [wanted], execution_plan=plan)
    wire = server.links[-1]["pic_link"]
    assert not set(wire) & PIC_LINK_RUNTIME_OWNED_FIELDS
    validate_pic_link_request(wire)
    assert not any(
        "namespace" in key or "physical_kv" in key for key in wire
    ), "namespaces are host bookkeeping and never reach the runtime"


def test_a_runtime_owned_field_is_still_rejected():
    adapter, server, _, wanted = two_namespace_scenario(
        NamespaceMode.LOGICAL_NAMESPACE, WORKSPACE_FROM_TOOL_OUTPUT
    )
    manifest = boundary_manifest([wanted])
    plan, _, _ = adapter.compile_plan(adapter.planner.plan(manifest, [wanted]))
    adapter.link(manifest, [wanted], execution_plan=plan)
    wire = dict(server.links[-1]["pic_link"])
    wire["imp_indices"] = [0]
    with pytest.raises(Exception):
        validate_pic_link_request(wire)


@pytest.mark.parametrize("mode", list(NamespaceMode))
def test_the_native_prefix_cache_path_is_untouched_by_the_namespace_mode(mode):
    adapter, server = make_adapter(mode)
    blocks = [make_block("workspace_evidence")]
    manifest = boundary_manifest(blocks)
    result = adapter.native_selected(manifest, blocks)
    payload = server.payloads[-1]
    assert "pic_link" not in payload and "sub_context" not in payload and "extra_key" not in payload
    assert payload["messages"][0]["content"] == result.selected_content
    assert adapter.cache_catalog._entries == {}


# --- metrics ----------------------------------------------------------------


def test_the_request_summary_counts_hits_misses_and_the_missed_reuse_tokens():
    adapter, server, produced, wanted = two_namespace_scenario(
        NamespaceMode.HARD_NAMESPACE, WORKSPACE_FROM_TOOL_OUTPUT
    )
    summary = plan_for(adapter, wanted).namespace_summary()
    row = summary["by_namespace"]["workspace"]
    assert row["namespace_lookup_miss"] == 1 and row["namespace_lookup_hit"] == 0
    assert row["cross_namespace_missed_reuse_tokens"] == wanted.token_count
    assert row["compiled_token_count"] == wanted.token_count
    assert row["reused_token_count"] == 0
    assert row["functional_fields"] == ["workspace_evidence"]
    assert summary["totals"]["cross_namespace_missed_reuse_tokens"] == wanted.token_count
    assert summary["totals"]["namespace_lookup_latency_s"] >= 0


def test_the_summary_counts_reused_and_repair_tokens_for_a_resident_block():
    adapter, _ = make_adapter(NamespaceMode.LOGICAL_NAMESPACE)
    block = make_block("workspace_evidence")
    adapter.compile(block)
    plan = plan_for(adapter, block)
    summary = plan.namespace_summary(repair_tokens_by_cache_key={block.cache_key: 16})
    row = summary["by_namespace"]["workspace"]
    assert row["reused_token_count"] == block.token_count
    assert row["repair_token_count"] == 16
    assert row["compiled_token_count"] == 0
    assert plan.namespace_resolutions[0].entry_age_s is not None


def test_the_catalog_reports_active_blocks_and_host_observed_removals_by_namespace():
    adapter, _ = make_adapter(NamespaceMode.GLOBAL)
    adapter.compile(make_block("workspace_evidence"))
    adapter.compile(make_block("tool_definitions", TEXT + "s"))
    assert adapter.cache_catalog.active_by_namespace() == {"workspace": 1, "tool_schema": 1}
    adapter.cache_catalog.invalidate(make_block("workspace_evidence").cache_key)
    assert adapter.cache_catalog.removals_by_namespace == {"workspace": 1}
    assert adapter.cache_catalog.active_by_namespace() == {"tool_schema": 1}


def test_the_mode_is_validated_and_reported():
    assert NamespaceMode.parse(None) is NamespaceMode.GLOBAL
    assert NamespaceMode.parse("Hard_Namespace") is NamespaceMode.HARD_NAMESPACE
    with pytest.raises(ValueError, match="semantic_cache_namespace_mode"):
        NamespaceMode.parse("sharded")
    assert NamespaceResolver("logical_namespace").describe()["semantic_cache_namespace_mode"] == "logical_namespace"


# --- 10/11. segmentation determinism and boundary locking -------------------


PARAGRAPHS = [f"paragraph{index} " * 70 for index in range(1, 7)]


def paragraphs(indices):
    return "".join(PARAGRAPHS[i] + "\n\n" for i in indices)


class MiddleSplitLLM:
    """Splits after the middle unit of whatever it is shown -- so the cut depends on what it sees."""

    def __init__(self):
        self.seen_units: list[list[str]] = []

    def __call__(self, payload):
        body = payload["messages"][-1]["content"].split("\n\n", 1)[1]
        units, _ = json.JSONDecoder().raw_decode(body)
        units = units["units"]
        self.seen_units.append([unit["text"] for unit in units])
        middle = len(units) // 2
        if len(units) < 2 or middle < 1:
            return {"split_points": []}
        return {"split_points": [{"after_unit_id": units[middle - 1]["unit_id"], "reason": "middle"}]}


def llm_manager(llm, tmp_path):
    segmenter = OpenAICompatibleSemanticSegmenter(
        api_base="http://native/v1", model="seg", transport=llm, retries=0,
        semantic_classification_enabled=False,
    )
    return IncrementalContextManager(
        task_id="task",
        segment_pipeline=SemanticSegmentationPipeline(segmenter),
        store=SQLiteSubContextStore(),
        selector=RuleBasedSemanticContextSelector(),
        authoritative_tokenizer=TOK,
        token_accounting_mode=TokenAccountingMode.AUTHORITATIVE,
    )


def test_segmentation_is_deterministic_for_the_same_source_field_and_policy(tmp_path):
    source_text = paragraphs(range(4))

    def run():
        manager = llm_manager(MiddleSplitLLM(), tmp_path)
        manager.ingest([(EventType.WORKSPACE_FILE, "file", source_text, {"path": "a.py"})])
        return [
            (value.sub_context_id, value.semantic_identity, value.cache_key, value.raw_text)
            for value in sorted(manager.store.list_by_task("task"), key=lambda v: v.sub_context_id)
        ]

    first, second = run(), run()
    assert first == second and len(first) == 2

    class Source:
        raw_text = source_text
        source_id = "file"
        context_field = "workspace_evidence"

    units_a = AtomicUnitBuilder(TOK).build(Source())
    units_b = AtomicUnitBuilder(TOK).build(Source())
    assert [(u.unit_id, u.start_char, u.end_char) for u in units_a] == [
        (u.unit_id, u.start_char, u.end_char) for u in units_b
    ]


def test_finalized_boundaries_do_not_move_when_a_source_grows(tmp_path):
    llm = MiddleSplitLLM()
    manager = llm_manager(llm, tmp_path)
    old_text = paragraphs(range(4))
    manager.ingest([(EventType.WORKSPACE_FILE, "file", old_text, {"path": "a.py"})])
    before = sorted(manager.store.list_by_task("task"), key=lambda v: v.source_spans[0].start)
    assert [value.raw_text for value in before] == [paragraphs([0, 1]), paragraphs([2, 3])]

    manager.ingest([(EventType.WORKSPACE_FILE, "file", old_text + paragraphs([4, 5]), {"path": "a.py"})])
    after = sorted(
        (value for value in manager.store.list_by_task("task") if not value.invalidated),
        key=lambda v: v.source_spans[0].start,
    )
    # Left alone, a middle-splitting LLM shown all six units would cut [0 1 2][3 4 5].
    assert [value.raw_text for value in after[:2]] == [value.raw_text for value in before]
    assert [value.cache_key for value in after[:2]] == [value.cache_key for value in before]
    assert "".join(value.raw_text for value in after) == old_text + paragraphs([4, 5])
    # The second call saw only the appended units: the old ones were never re-offered.
    shown = [text for units in llm.seen_units[1:] for text in units]
    assert shown and all(PARAGRAPHS[0] not in text and PARAGRAPHS[3] not in text for text in shown)
    assert len(after) == 4


def test_an_edit_rather_than_an_append_is_segmented_as_a_whole(tmp_path):
    llm = MiddleSplitLLM()
    manager = llm_manager(llm, tmp_path)
    manager.ingest([(EventType.WORKSPACE_FILE, "file", paragraphs(range(4)), {"path": "a.py"})])
    edited = paragraphs([5]) + paragraphs([1, 2, 3])
    manager.ingest([(EventType.WORKSPACE_FILE, "file", edited, {"path": "a.py"})])
    active = [value for value in manager.store.list_by_task("task") if not value.invalidated]
    assert "".join(value.raw_text for value in sorted(active, key=lambda v: v.source_spans[0].start)) == edited
    assert any(PARAGRAPHS[1] in text for text in llm.seen_units[-1]), "the whole edited source was re-offered"


def test_mutable_blocks_are_not_frozen(tmp_path):
    llm = MiddleSplitLLM()
    manager = llm_manager(llm, tmp_path)
    first = paragraphs(range(2))
    manager.ingest([(EventType.TOOL_OUTPUT, "log", first, {})])  # tool_history_output: mutable
    manager.ingest([(EventType.TOOL_OUTPUT, "log", first + paragraphs([2, 3]), {})])
    assert any(PARAGRAPHS[0] in text for text in llm.seen_units[-1]), "a mutable block may still be re-cut"


# --- 12. retry correctness -------------------------------------------------


def test_a_failure_after_events_are_accepted_does_not_lose_them_on_retry():
    flaky = {"down": True}

    def jev(request):
        if flaky["down"]:
            raise RuntimeError("jev unavailable")
        return {"field": "workspace_evidence", "confidence": 0.95}

    classifier = HybridFieldClassifier(jev=JevFieldClassifier(jev))
    segmenter = OpenAICompatibleSemanticSegmenter(
        api_base="http://native/v1", model="seg", transport=MiddleSplitLLM(), retries=0,
        semantic_classification_enabled=False,
    )
    manager = IncrementalContextManager(
        task_id="task",
        segment_pipeline=SemanticSegmentationPipeline(segmenter),
        store=SQLiteSubContextStore(),
        selector=RuleBasedSemanticContextSelector(),
        extractor=FieldFirstExtractor(classifier=classifier),
        authoritative_tokenizer=TOK,
        token_accounting_mode=TokenAccountingMode.AUTHORITATIVE,
    )
    events = [
        (EventType.USER_REQUEST, "request", "inspect the parser", {}),
        (EventType.UNCLASSIFIED, "blob", paragraphs([0, 1]), {}),
    ]
    with pytest.raises(FieldClassificationError):
        manager.ingest(events)
    assert manager.collector.events() == [], "the accepted events were rolled back"
    assert manager.store.list_by_task("task") == []

    flaky["down"] = False
    result = manager.ingest(events)
    assert len(result.events) == 2, "the same events are offered, accepted and ingested again"
    fields = {value.context_field for value in manager.store.list_by_task("task")}
    assert fields == {"user_prompt", "workspace_evidence"}


# --- the agent logs the arm and the counters ------------------------------


def agent_llm(tmp_path, mode):
    from tests import test_terminal_bench_noc_sglang_agent as harness
    from terminal_bench_integration.noc_sglang_agent import NOCSGLangLLM

    llm = NOCSGLangLLM(
        model_name="model",
        api_base="http://patched/v1",
        original_api_base="http://original/v1",
        segmentation_mode="deterministic",
        legolink_k=16,
        max_tokens=64,
        timeout=1,
        artifacts_root=str(tmp_path),
        tokenizer_mode="regex",
        semantic_cache_namespace_mode=mode,
    )
    llm._patched_transport = harness.transport
    llm._original_transport = harness.transport
    llm.set_task_context("task", tmp_path / "metrics.jsonl")
    return llm


@pytest.mark.parametrize("mode", [m.value for m in NamespaceMode])
def test_each_row_records_the_namespace_arm_and_per_namespace_counters(tmp_path, mode):
    llm = agent_llm(tmp_path, mode)
    history = [{"role": "system", "content": "preserve behavior " * 100}]
    llm.call("fix parser", message_history=history)
    llm.call("fix parser again", message_history=history)
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    last = rows[-1]
    assert last["semantic_cache_namespace_mode"] == mode
    assert llm._adapter.namespace_resolver.mode.value == mode
    metrics = last["namespace_metrics"]
    assert set(metrics["totals"]) >= {
        "namespace_lookup_hit",
        "namespace_lookup_miss",
        "cross_namespace_fallback_attempted",
        "cross_namespace_hit",
        "cross_namespace_missed_reuse_tokens",
        "reused_token_count",
        "compiled_token_count",
        "repair_token_count",
        "namespace_lookup_latency_s",
    }
    assert sum(row["blocks"] for row in metrics["by_namespace"].values()) == metrics["blocks"]
    assert isinstance(last["namespace_active_blocks"], dict)
    assert last["namespace_active_blocks"], "the first turn compiled a reusable block"
    assert metrics["totals"]["reused_token_count"] > 0, "the second turn linked what the first compiled"


def test_a_misspelled_namespace_mode_fails_at_construction(tmp_path):
    with pytest.raises(ValueError, match="semantic_cache_namespace_mode"):
        agent_llm(tmp_path, "hard_namspace")


def test_the_agent_and_proxy_forward_the_namespace_mode():
    agent = (ROOT / "terminal_bench_integration/noc_sglang_agent.py").read_text(encoding="utf-8")
    proxy = (ROOT / "terminal_bench_integration/noc_openai_proxy.py").read_text(encoding="utf-8")
    assert agent.count('"semantic_cache_namespace_mode",') == 2, "both agent kwarg filters"
    assert "semantic_cache_namespace_mode=self._semantic_cache_namespace_mode" in agent
    assert "--semantic-cache-namespace-mode" in proxy
    assert "semantic_cache_namespace_mode=self.config.semantic_cache_namespace_mode" in proxy
