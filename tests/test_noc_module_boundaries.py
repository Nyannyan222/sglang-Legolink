from __future__ import annotations

from dataclasses import replace

import pytest

from noc_context_manager.compile_policy import CacheCatalog, CompileAction
from noc_context_manager.context_selection import (
    AgentContextManifest,
    RuleBasedSemanticContextSelector,
)
from noc_context_manager.manager import IncrementalContextManager
from noc_context_manager.pipeline import NOCRequestPipeline, PipelineRoute
from noc_context_manager.scheduler import NOCRequestPlanner, NOCRequestScheduler
from noc_context_manager.segmentation_validation import (
    SemanticSegmentationPipeline,
    SemanticSegmentationValidator,
)
from noc_context_manager.semantic_segmentation import DeterministicFieldSegmenter
from noc_context_manager.session import EventType, SourceSpan
from noc_context_manager.sglang_adapter import (
    NOCSGLangAdapter,
    SGLangRequestError,
)
from noc_context_manager.stitch import ContextStitcher, PromptFramingPolicy
from noc_context_manager.subcontext_store import (
    SemanticSubContext,
    SQLiteSubContextStore,
)
from noc_context_manager.tokenizer import (
    CallableTokenizer,
    HuggingFaceTokenizer,
    RegexTokenizer,
)


from noc_context_manager.manager import TokenAccountingMode  # noqa: E402
from noc_context_manager.token_accounting import ModelTokenCount  # noqa: E402

#: These fixtures stand in for the agent path, which injects the serving
#: model's tokenizer. They declare the unit they are in rather than leaving the
#: count unlabelled, so the compile floor's model-token guard applies to them
#: the way it applies in production.
FIXTURE_TOKENIZER = CallableTokenizer(
    lambda text: RegexTokenizer().tokenize(text),
    fingerprint="fixture-model-tokenizer-v1",
    model_id="test/fixture-model",
    authoritative=True,
)


def fixture_model_tokens(text: str) -> ModelTokenCount:
    return ModelTokenCount(
        value=FIXTURE_TOKENIZER.count(text),
        tokenizer_fingerprint=FIXTURE_TOKENIZER.fingerprint,
        model_id=FIXTURE_TOKENIZER.model_id,
    )


def context(
    source_id: str,
    text: str,
    *,
    mutable: bool = False,
    reuse: bool = True,
) -> SemanticSubContext:
    return SemanticSubContext.create(
        task_id="task",
        session_revision=1,
        context_field="conversation_history" if mutable else "workspace_evidence",
        title=source_id,
        question_or_objective="test modular request planning",
        raw_text=text,
        source_spans=[SourceSpan(source_id, 0, len(text))],
        token_count=fixture_model_tokens(text).value,
        token_accounting=fixture_model_tokens(text),
        is_mutable=mutable,
        reuse_candidate=reuse,
        semantic_independence_reason="test fixture",
        segmentation_confidence=1.0,
        model_fingerprint="model",
        tokenizer_fingerprint=FIXTURE_TOKENIZER.fingerprint,
    )


def manifest(
    values: list[SemanticSubContext],
    *,
    mutable_text: str = "current objective",
) -> AgentContextManifest:
    selected_tokens = sum(value.token_count for value in values)
    return AgentContextManifest(
        task_id="task",
        agent_id="planner",
        request_id="request",
        session_revision=1,
        objective="test",
        selected_sub_context_ids=tuple(value.sub_context_id for value in values),
        selected_cache_keys=tuple(
            value.cache_key for value in values if not value.is_mutable
        ),
        mutable_text=mutable_text,
        full_session_tokens=selected_tokens + 10,
        candidate_context_tokens=selected_tokens,
        selected_context_tokens=selected_tokens,
        context_reduction_tokens=10,
        context_reduction_ratio=10 / (selected_tokens + 10),
    )


def test_tokenizer_module_supports_default_and_model_adapters():
    regex = RegexTokenizer()
    assert regex.tokenize("alpha/beta 中文!") == ("alpha/beta", "中", "文", "!")
    assert regex.count("alpha/beta 中文!") == 4

    model = CallableTokenizer(
        lambda text: [index for index, _ in enumerate(text.split())],
        fingerprint="fake-model-tokenizer-v1",
    )
    assert model.count("one two three") == 3
    assert model.fingerprint == "fake-model-tokenizer-v1"


def test_preliminary_count_precedes_authoritative_reconstructed_tokenization(
    tmp_path,
):
    calls = []

    def preliminary_encode(text):
        calls.append(("preliminary", text))
        return tuple(text.split())

    def authoritative_encode(text):
        calls.append(("authoritative", text))
        return tuple(range(len(text.split())))

    preliminary = CallableTokenizer(
        preliminary_encode,
        fingerprint="preliminary-v1",
    )
    authoritative = CallableTokenizer(
        authoritative_encode,
        fingerprint="authoritative-v1",
    )
    manager = IncrementalContextManager(
        task_id="task",
        segment_pipeline=SemanticSegmentationPipeline(
            DeterministicFieldSegmenter(),
            validator=SemanticSegmentationValidator(
                tokenizer=authoritative
            ),
            preliminary_tokenizer=preliminary,
            report_dir=tmp_path,
        ),
        store=SQLiteSubContextStore(),
        selector=RuleBasedSemanticContextSelector(),
        preliminary_tokenizer=preliminary,
        authoritative_tokenizer=authoritative,
        model_fingerprint="model-v1",
    )
    result = manager.ingest(
        [
            (
                EventType.TASK_INSTRUCTION,
                "instruction",
                "preserve exact source bytes",
                {},
            )
        ]
    )

    stages = [stage for stage, _text in calls]
    assert stages.index("authoritative") > stages.index("preliminary")
    value = result.stored[0]
    assert value.raw_text == "preserve exact source bytes"
    assert value.token_ids == (0, 1, 2, 3)
    assert value.token_count == 4
    assert value.tokenizer_fingerprint == "authoritative-v1"
    assert value.metadata["tokenization_stage"] == (
        "after_validation_and_exact_reconstruction"
    )
    assert (
        result.segmentation.preliminary_token_count.tokenizer_fingerprint
        == "preliminary-v1"
    )


def test_oversized_preliminary_source_uses_exact_contiguous_windows(tmp_path):
    text = "one two three four five six seven"
    pipeline = SemanticSegmentationPipeline(
        DeterministicFieldSegmenter(),
        preliminary_tokenizer=RegexTokenizer(),
        segmentation_estimate_token_budget=3,
        report_dir=tmp_path,
    )
    manager = IncrementalContextManager(
        task_id="task",
        segment_pipeline=pipeline,
        store=SQLiteSubContextStore(),
        selector=RuleBasedSemanticContextSelector(),
    )
    result = manager.ingest(
        [(EventType.TASK_INSTRUCTION, "long", text, {})]
    )

    assert len(result.stored) >= 2
    ordered = sorted(
        result.stored,
        key=lambda value: value.source_spans[0].start,
    )
    assert "".join(value.raw_text for value in ordered) == text
    preliminary = result.segmentation.preliminary_token_count
    assert preliminary is not None
    assert preliminary.oversized_source_ids == ("long",)


def test_huggingface_tokenizer_uses_raw_model_token_ids():
    class Backend:
        name_or_path = "fake/model"
        vocab_size = 100
        special_tokens_map = {"eos_token": "<eos>"}
        init_kwargs = {"_commit_hash": "revision"}

        @staticmethod
        def encode(text, *, add_special_tokens):
            assert add_special_tokens is False
            return [ord(value) for value in text]

    first = HuggingFaceTokenizer(Backend(), model_id="fake/model")
    second = HuggingFaceTokenizer(Backend(), model_id="fake/model")
    assert first.tokenize("ab") == (97, 98)
    assert first.count("abc") == 3
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.startswith("hf-")


def test_scheduler_preserves_manifest_order_and_separates_actions():
    stable = context("stable", "stable " * 80)
    mutable = context("mutable", "latest tool result", mutable=True, reuse=False)
    schedule = NOCRequestScheduler().schedule(
        manifest([mutable, stable]),
        [stable, mutable],
    )

    assert schedule.ordered_contexts == (mutable, stable)
    assert [value.action for value in schedule.decisions] == [
        CompileAction.KEEP_MUTABLE,
        CompileAction.KEEP_MUTABLE,
    ]
    assert schedule.linked_contexts == ()
    assert schedule.mutable_contexts == (mutable, stable)


def test_request_planner_is_logical_and_contains_no_physical_link_state():
    stable = context("stable", "stable " * 80)
    plan = NOCRequestPlanner().plan(manifest([stable]), [stable])
    serialized = str(plan.to_dict()).lower()

    assert plan.selected_order == (stable.sub_context_id,)
    assert "kv_slots" not in serialized
    assert "imp_indices" not in serialized
    assert "rope_position" not in serialized
    assert "destination_slots" not in serialized


def test_scheduler_rejects_incomplete_external_decisions():
    stable = context("stable", "stable " * 80)
    mutable = context("mutable", "latest", mutable=True, reuse=False)
    scheduler = NOCRequestScheduler()
    schedule = scheduler.schedule(manifest([stable, mutable]), [stable, mutable])

    with pytest.raises(ValueError, match="do not match"):
        scheduler.bind_decisions(
            schedule.manifest,
            schedule.ordered_contexts,
            schedule.decisions[:1],
        )


def test_stitcher_materializes_selected_and_legolink_views():
    stable = context("stable", "stable block " * 40)
    mutable = context("mutable", "latest tool result", mutable=True, reuse=False)
    schedule = NOCRequestScheduler().schedule(
        manifest([stable, mutable], mutable_text="current objective"),
        [mutable, stable],
    )
    stitched = ContextStitcher().stitch(schedule)

    assert stitched.selected_text == (
        f"{stable.raw_text}\n\n{mutable.raw_text}\n\ncurrent objective"
    )
    assert stitched.mutable_text == f"{mutable.raw_text}\n\ncurrent objective"
    # The wire carries the content identity so another task can match it; the
    # occurrence id stays available beside it for provenance.
    assert stitched.block_ids == (stable.semantic_identity,)
    assert stitched.occurrence_ids == (stable.sub_context_id,)
    assert stitched.cache_keys == (stable.cache_key,)
    assert stable.raw_text not in stitched.mutable_text


def test_stitcher_allows_equal_text_at_distinct_semantic_occurrences():
    repeated_text = "the same user-visible turn"
    stable = context("stable-occurrence", repeated_text)
    mutable = context(
        "mutable-occurrence",
        repeated_text,
        mutable=True,
        reuse=False,
    )
    schedule = NOCRequestScheduler().schedule(
        manifest([stable, mutable], mutable_text="current objective"),
        [stable, mutable],
    )

    stitched = ContextStitcher().stitch(schedule)

    assert stable.sub_context_id != mutable.sub_context_id
    assert stitched.equivalence.immutable_content_not_duplicated is True
    assert stitched.equivalence.complete_selected_text_matches is True


def test_scheduler_uses_shared_cache_catalog():
    stable = context("stable", "stable " * 80)
    catalog = CacheCatalog()
    catalog.mark_compiled(stable.cache_key, token_count=stable.token_count)
    scheduler = NOCRequestScheduler(cache_catalog=catalog)

    schedule = scheduler.schedule(manifest([stable]), [stable])
    assert schedule.decisions[0].action == CompileAction.COMPILE
    assert "already compiled" in schedule.decisions[0].reason

    stale = replace(stable, invalidated=True)
    stale_schedule = scheduler.schedule(manifest([stale]), [stale])
    assert stale_schedule.decisions[0].action == CompileAction.DO_NOT_INCLUDE


def test_pipeline_facade_executes_ingest_schedule_stitch_and_link(tmp_path):
    # The facade plans, so it needs the unit the compile floor is expressed in.
    # AUTHORITATIVE is asserted rather than inferred: the manager proves it has
    # the serving model's tokenizer at construction instead of discovering the
    # problem when a regex count reaches the floor.
    manager = IncrementalContextManager(
        task_id="task",
        segment_pipeline=SemanticSegmentationPipeline(
            DeterministicFieldSegmenter(),
            report_dir=tmp_path / "validation",
        ),
        store=SQLiteSubContextStore(),
        selector=RuleBasedSemanticContextSelector(),
        authoritative_tokenizer=FIXTURE_TOKENIZER,
        token_accounting_mode=TokenAccountingMode.AUTHORITATIVE,
    )
    calls = []

    def transport(payload):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 1},
        }

    adapter = NOCSGLangAdapter(token_accounting_mode=TokenAccountingMode.ESTIMATE_ONLY, model="model", patched_transport=transport)
    pipeline = NOCRequestPipeline(context_manager=manager, adapter=adapter)
    execution = pipeline.execute(
        [
            (EventType.USER_REQUEST, "request", "inspect parser", {}),
            (
                EventType.TASK_INSTRUCTION,
                "instruction",
                "preserve behavior " * 80,
                {},
            ),
        ],
        route=PipelineRoute.SELECTED_CONTEXT_LEGOLINK,
        agent_id="planner",
        loop_stage="planner",
        objective="inspect parser",
        plan_step="inspect parser",
        available_tools=("terminal",),
        token_budget=10_000,
    )

    assert execution.adapter_result.route.endswith("legolink")
    assert execution.schedule is not None
    assert execution.stitched is not None
    assert execution.compile_requests >= 1
    assert any("sub_context" in payload for payload in calls)
    assert any("pic_link" in payload for payload in calls)


def test_adapter_reuses_pipeline_schedule_and_stitched_context():
    stable = context("stable", "stable block " * 80)
    request_manifest = manifest([stable])
    calls = []

    def transport(payload):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 1},
        }

    adapter = NOCSGLangAdapter(token_accounting_mode=TokenAccountingMode.ESTIMATE_ONLY, model="model", patched_transport=transport)
    decisions, _, _ = adapter.compile_selected(request_manifest, [stable])
    schedule = adapter.scheduler.bind_decisions(
        request_manifest,
        [stable],
        decisions,
    )
    stitched = adapter.stitcher.stitch(
        schedule,
        generation_parameters=adapter.generation,
    )
    result = adapter.link(
        request_manifest,
        [stable],
        decisions=decisions,
        schedule=schedule,
        stitched=stitched,
    )

    assert result.selected_content == stitched.selected_text
    assert calls[-1]["pic_link"]["cache_keys"] == list(stitched.cache_keys)
    assert (
        calls[-1]["pic_link"]["segmentation_policy_version"]
        == stable.policy_version
    )


def test_adapter_rejects_empty_mutable_suffix_before_link_transport():
    stable = context("stable", "stable block " * 80)
    request_manifest = manifest([stable], mutable_text="")
    calls = []

    def transport(payload):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 1},
        }

    adapter = NOCSGLangAdapter(token_accounting_mode=TokenAccountingMode.ESTIMATE_ONLY, model="model", patched_transport=transport)
    adapter.compile(stable)
    calls.clear()

    with pytest.raises(SGLangRequestError) as caught:
        adapter.link(request_manifest, [stable])

    assert caught.value.reason == "empty_mutable_suffix"
    assert calls == []


def test_adapter_rejects_mixed_segmentation_policy_versions_before_transport():
    first = context("first", "first stable block " * 80)
    second = replace(
        context("second", "second stable block " * 80),
        policy_version="semantic-v2",
    )
    calls = []
    adapter = NOCSGLangAdapter(
        token_accounting_mode=TokenAccountingMode.ESTIMATE_ONLY,
        model="model",
        patched_transport=lambda payload: calls.append(payload),
    )

    with pytest.raises(SGLangRequestError) as caught:
        adapter.link(manifest([first, second]), [first, second])

    assert caught.value.reason == "incompatible"
    assert calls == []


def test_stitcher_rejects_generation_contract_mismatch():
    stable = context("stable", "stable block " * 80)
    plan = NOCRequestPlanner().plan(manifest([stable]), [stable])

    with pytest.raises(ValueError, match="not equivalent"):
        ContextStitcher().stitch(
            plan,
            baseline_generation_parameters={"temperature": 0},
            legolink_generation_parameters={"temperature": 1},
        )


def test_stitcher_rejects_baseline_and_legolink_framing_mismatch():
    stable = context("stable", "stable block " * 80)
    plan = NOCRequestPlanner().plan(manifest([stable]), [stable])

    with pytest.raises(ValueError, match="not equivalent"):
        ContextStitcher(
            baseline_framing=PromptFramingPolicy(
                role_template="single-user-message"
            ),
            legolink_framing=PromptFramingPolicy(
                role_template="system-plus-user-message"
            ),
        ).stitch(plan)
