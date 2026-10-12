"""Approximate token counts and model token counts are different things.

`count_tokens(text)` meant "whatever tokenizer happened to be available", and
what happened to be available was `RegexTokenizer`. The resulting integer
reached the compile floor, the context budget, the LegoLink recompute ranges
and the benchmark tables, and nothing about it said which unit it was in.

The loud version of this failure is on record: a workload cut to "30,013
tokens" on regex counts was refused by the server at 36,573 model tokens.
Measured regex-to-BPE ratios across candidate tasks run from 1.19 to 7.63, so
the quiet version is a 64-token compile floor that is somewhere between 8 and 54
model tokens depending on which path built the block.

These tests are about the refusal, not about any particular count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noc_context_manager.compile_policy import (  # noqa: E402
    CompileAction,
    CompilePolicy,
    CompilePolicyContext,
)
from noc_context_manager.session import SourceSpan  # noqa: E402
from noc_context_manager.sglang_adapter import NOCSGLangAdapter  # noqa: E402
from noc_context_manager.subcontext_store import SemanticSubContext  # noqa: E402
from noc_context_manager.token_accounting import (  # noqa: E402
    EstimatedTokenCount,
    ModelTokenCount,
    TokenAccountingError,
    TokenUnit,
    authoritative_identity,
    count_from_provenance,
    is_authoritative_tokenizer,
    require_authoritative_tokenizer,
    require_model_tokens,
    token_unit_of,
)
from noc_context_manager.tokenizer import (  # noqa: E402
    DEFAULT_TOKENIZER,
    ESTIMATE_TOKENIZER,
    CallableTokenizer,
    RegexTokenizer,
    count_model_tokens,
    count_tokens,
    estimate_tokens,
)


def model_tokenizer(model_id: str = "test/model") -> CallableTokenizer:
    """A tokenizer that declares itself the model's and can back the claim."""

    return CallableTokenizer(
        lambda text: RegexTokenizer().tokenize(text),
        fingerprint=f"fixture-{model_id}",
        model_id=model_id,
        authoritative=True,
    )


class ChatModelTokenizer:
    """Authoritative *and* able to render a chat template.

    Both properties are needed and they are not the same one:
    `CallableTokenizer` can be authoritative for counting yet cannot reproduce
    the server's chat template, and chat rendering changes token IDs
    independently of the vocabulary. Physical equivalence needs this stronger
    kind; token accounting needs only the weaker one.
    """

    token_unit = TokenUnit.MODEL
    backend_name = "fixture-chat"

    def __init__(self, model_id: str = "test/model") -> None:
        self.model_id = model_id
        self.fingerprint = f"fixture-chat-{model_id}"
        self._inner = RegexTokenizer()

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(hash(token) % 50_000 for token in self._inner.tokenize(text))

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize_messages(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        reasoning_effort=None,
        continue_final_message=False,
        chat_template_kwargs=None,
    ) -> tuple[int, ...]:
        rendered = "".join(
            f"<|{message.get('role')}|>{message.get('content')}"
            for message in messages
        )
        return self.tokenize(rendered)


def chat_model_tokenizer(model_id: str = "test/model") -> ChatModelTokenizer:
    return ChatModelTokenizer(model_id)


def block(
    text: str = "parser implementation " * 40,
    *,
    accounting=None,
    tokenizer_fingerprint: str = "regex-ascii-cjk-punct-v1",
) -> SemanticSubContext:
    count = len(RegexTokenizer().tokenize(text))
    return SemanticSubContext.create(
        task_id="task",
        session_revision=1,
        context_field="workspace_evidence",
        title="fixture",
        question_or_objective="unit contract",
        raw_text=text,
        source_spans=[SourceSpan("src", 0, len(text))],
        token_count=count,
        token_accounting=accounting,
        is_mutable=False,
        reuse_candidate=True,
        semantic_independence_reason="fixture",
        segmentation_confidence=1.0,
        model_fingerprint="model",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


# --- A and B: the two counting APIs return the two units --------------------


def test_estimate_tokens_returns_an_estimate() -> None:
    count = estimate_tokens("hello there world")

    assert isinstance(count, EstimatedTokenCount)
    assert count.unit is TokenUnit.ESTIMATED
    assert count.value == 3
    assert count.to_dict()["unit"] == "estimated_tokens"


def test_count_model_tokens_returns_model_tokens() -> None:
    count = count_model_tokens("hello there world", model_tokenizer())

    assert isinstance(count, ModelTokenCount)
    assert count.unit is TokenUnit.MODEL
    assert count.value == 3
    assert count.model_id == "test/model"
    assert count.to_dict()["unit"] == "model_tokens"


# --- C and D: authoritative counting cannot be faked ------------------------


def test_authoritative_counting_without_a_tokenizer_fails() -> None:
    with pytest.raises(TokenAccountingError) as excinfo:
        count_model_tokens("text", None)

    assert "requires the serving model's tokenizer" in str(excinfo.value)


def test_regex_cannot_satisfy_the_authoritative_requirement() -> None:
    with pytest.raises(TokenAccountingError):
        count_model_tokens("text", RegexTokenizer())
    with pytest.raises(TokenAccountingError):
        require_authoritative_tokenizer(RegexTokenizer(), purpose="test")

    assert token_unit_of(RegexTokenizer()) is TokenUnit.ESTIMATED
    assert is_authoritative_tokenizer(RegexTokenizer()) is False
    assert authoritative_identity(RegexTokenizer()) is None


def test_the_compatibility_default_tokenizer_is_still_only_an_estimator() -> None:
    """`DEFAULT_TOKENIZER` kept its name; it did not keep any authority."""

    assert DEFAULT_TOKENIZER is ESTIMATE_TOKENIZER
    assert is_authoritative_tokenizer(DEFAULT_TOKENIZER) is False


def test_count_tokens_no_longer_picks_a_tokenizer_for_you() -> None:
    """The int return is why: it cannot carry the unit the caller assumed."""

    with pytest.raises(TokenAccountingError) as excinfo:
        count_tokens("text")

    assert "no longer defaults to the regex estimator" in str(excinfo.value)
    assert count_tokens("a b c", RegexTokenizer()) == 3


# --- E: agreement between two estimators is not evidence about the model ----


def test_two_matching_regex_fingerprints_do_not_pass_authoritative_identity() -> None:
    """The specific hole this closes.

    The previous guard compared one tokenizer's fingerprint against another's
    and passed when they matched. Two `RegexTokenizer` instances match
    perfectly, always, and neither has ever seen the model.
    """

    left, right = RegexTokenizer(), RegexTokenizer()

    assert left.fingerprint == right.fingerprint
    assert authoritative_identity(left) is None
    assert authoritative_identity(right) is None
    with pytest.raises(TokenAccountingError):
        require_authoritative_tokenizer(left, purpose="formal run")


def test_an_authoritative_wrapper_must_name_its_model() -> None:
    with pytest.raises(TokenAccountingError):
        CallableTokenizer(lambda t: t.split(), fingerprint="fp", authoritative=True)


# --- F: an injected model tokenizer passes ---------------------------------


def test_an_injected_model_tokenizer_passes_the_guard() -> None:
    identity = require_authoritative_tokenizer(
        model_tokenizer("Qwen/Qwen2.5-3B-Instruct"), purpose="formal run"
    )

    assert identity.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert identity.tokenizer_class.endswith("CallableTokenizer")
    assert identity.fingerprint


# --- G and H: the compile floor's unit, with its numbers unchanged ----------


def test_compile_policy_refuses_an_estimated_count() -> None:
    estimated = block()
    context = CompilePolicyContext(require_model_tokens=True)

    assert isinstance(estimated.token_accounting(), EstimatedTokenCount)
    with pytest.raises(TokenAccountingError) as excinfo:
        CompilePolicy().decide(estimated, context)

    assert "model-token floor" in str(excinfo.value)
    assert "Estimates do not convert" in str(excinfo.value)


def test_compile_policy_is_numerically_unchanged_for_model_tokens() -> None:
    """Same threshold, same branches. Only the unit is now stated."""

    text = "parser implementation " * 40
    count = len(RegexTokenizer().tokenize(text))
    accounting = ModelTokenCount(
        value=count, tokenizer_fingerprint="fixture-test/model", model_id="test/model"
    )
    value = block(text, accounting=accounting)

    gated = CompilePolicy().decide(value, CompilePolicyContext(require_model_tokens=True))
    ungated = CompilePolicy().decide(value, CompilePolicyContext())

    assert gated.action is CompileAction.COMPILE
    assert gated.action == ungated.action
    assert gated.reason == ungated.reason
    assert CompilePolicyContext().min_compile_model_tokens == 64
    # The pre-unit spelling still reads, and reads the same number.
    assert CompilePolicyContext().min_compile_tokens == 64


def test_the_floor_still_declines_a_sub_floor_block() -> None:
    text = "tiny"
    accounting = ModelTokenCount(
        value=1, tokenizer_fingerprint="fixture-test/model", model_id="test/model"
    )
    small = block(text, accounting=accounting)

    decision = CompilePolicy().decide(
        small, CompilePolicyContext(require_model_tokens=True)
    )

    assert decision.action is CompileAction.KEEP_MUTABLE
    assert "compile floor" in decision.reason


def test_require_model_tokens_refuses_a_bare_int() -> None:
    """A plain integer is the shape the whole defect had. It has no unit."""

    with pytest.raises(TokenAccountingError) as excinfo:
        require_model_tokens(64, purpose="test")

    assert "has no unit" in str(excinfo.value)


# --- I: the planning path does not silently fall back to regex --------------


def test_an_adapter_defaults_to_authoritative_and_proves_it() -> None:
    from noc_context_manager.manager import TokenAccountingMode

    adapter = NOCSGLangAdapter(
        model="m",
        patched_transport=lambda payload: {},
        tokenizer=chat_model_tokenizer(),
    )

    assert adapter.token_accounting_mode is TokenAccountingMode.AUTHORITATIVE
    assert adapter.require_model_tokens is True
    assert adapter.planner.require_model_tokens is True


@pytest.mark.parametrize("tokenizer", [RegexTokenizer(), None])
def test_a_missing_tokenizer_fails_closed_instead_of_downgrading(tokenizer) -> None:
    """The escape hatch this audit closed.

    An earlier version read `require_model_tokens = is_authoritative_tokenizer(
    tokenizer)`, so an adapter built without the serving model's tokenizer
    quietly dropped the requirement and planned on estimates. Absence is now a
    failure, not a downgrade: the mode is a statement about intended use.
    """

    with pytest.raises(TokenAccountingError) as excinfo:
        NOCSGLangAdapter(
            model="m", patched_transport=lambda payload: {}, tokenizer=tokenizer
        )

    assert "AUTHORITATIVE token accounting mode" in str(excinfo.value)


@pytest.mark.parametrize("tokenizer", [RegexTokenizer(), None])
def test_estimate_only_remains_possible_but_must_be_asked_for(tokenizer) -> None:
    from noc_context_manager.manager import TokenAccountingMode

    adapter = NOCSGLangAdapter(
        model="m",
        patched_transport=lambda payload: {},
        tokenizer=tokenizer,
        token_accounting_mode=TokenAccountingMode.ESTIMATE_ONLY,
    )

    assert adapter.require_model_tokens is False
    assert adapter.planner.require_model_tokens is False


def test_a_planner_defaults_to_requiring_model_tokens() -> None:
    from noc_context_manager.scheduler import NOCRequestPlanner

    assert NOCRequestPlanner().require_model_tokens is True


# --- J: estimate-only paths still work -------------------------------------


def test_estimate_only_paths_are_not_broken_by_the_contract() -> None:
    from noc_context_manager.manager import TokenAccountingMode

    assert estimate_tokens("some rough text").value > 0
    assert TokenAccountingMode.infer(RegexTokenizer()) is (
        TokenAccountingMode.ESTIMATE_ONLY
    )
    assert TokenAccountingMode.infer(model_tokenizer()) is (
        TokenAccountingMode.AUTHORITATIVE
    )


def test_a_manager_asked_to_prove_authority_without_it_fails_closed() -> None:
    from noc_context_manager.manager import (
        IncrementalContextManager,
        TokenAccountingMode,
    )
    from noc_context_manager.segmentation_validation import (
        DeterministicFieldSegmenter,
        SemanticSegmentationPipeline,
    )
    from noc_context_manager.context_selection import (
        RuleBasedSemanticContextSelector,
    )
    from noc_context_manager.subcontext_store import SQLiteSubContextStore

    with pytest.raises(TokenAccountingError) as excinfo:
        IncrementalContextManager(
            task_id="task",
            segment_pipeline=SemanticSegmentationPipeline(DeterministicFieldSegmenter()),
            store=SQLiteSubContextStore(),
            selector=RuleBasedSemanticContextSelector(),
            authoritative_tokenizer=RegexTokenizer(),
            token_accounting_mode=TokenAccountingMode.AUTHORITATIVE,
        )

    assert "AUTHORITATIVE token accounting mode" in str(excinfo.value)


# --- K: serialized artifacts distinguish the two units ----------------------


def test_a_stored_block_serializes_its_unit() -> None:
    estimated = block().to_dict()
    modelled = block(
        accounting=ModelTokenCount(
            value=len(RegexTokenizer().tokenize("parser implementation " * 40)),
            tokenizer_fingerprint="fixture-test/model",
            model_id="test/model",
        )
    ).to_dict()

    assert estimated["token_accounting"]["unit"] == "estimated_tokens"
    assert modelled["token_accounting"]["unit"] == "model_tokens"
    assert modelled["token_accounting"]["model_id"] == "test/model"
    assert json.loads(json.dumps(modelled))["token_accounting"]["unit"] == (
        "model_tokens"
    )


def test_a_plan_prep_block_is_serialized_as_an_estimate() -> None:
    from noc_context_manager.runtime_isolation import PrefillBlock

    row = PrefillBlock.from_subcontext_row(
        {"sub_context": "some text here", "id": "b1", "content_hash": "h"}
    ).to_dict()

    assert row["token_unit"] == "estimated_tokens"


def test_the_manifest_records_the_unit_its_budget_is_in() -> None:
    from noc_context_manager.context_selection import AgentContextManifest

    assert AgentContextManifest(
        task_id="t",
        agent_id="a",
        request_id="r",
        session_revision=1,
        objective="o",
        selected_sub_context_ids=(),
        selected_cache_keys=(),
        mutable_text="",
        full_session_tokens=0,
        candidate_context_tokens=0,
        selected_context_tokens=0,
        context_reduction_tokens=0,
        context_reduction_ratio=0.0,
    ).token_unit == "unknown_tokens"


# --- L: legacy integers are not promoted -----------------------------------


def test_a_legacy_row_without_provenance_reads_as_an_estimate() -> None:
    """Historical artifacts carry a bare integer. Age is not authority."""

    legacy = block()
    legacy.metadata.pop("token_unit", None)

    accounting = legacy.token_accounting()

    assert isinstance(accounting, EstimatedTokenCount)
    assert accounting.unit is TokenUnit.ESTIMATED


def test_count_from_provenance_is_conservative() -> None:
    assert isinstance(
        count_from_provenance(100, tokenizer_fingerprint=None), EstimatedTokenCount
    )
    assert isinstance(
        count_from_provenance(100, tokenizer_fingerprint="regex-ascii-cjk-punct-v1"),
        EstimatedTokenCount,
    )
    # Only a fingerprint the caller positively vouches for, with a model.
    promoted = count_from_provenance(
        100,
        tokenizer_fingerprint="hf-abc",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        authoritative_fingerprints={"hf-abc"},
    )
    assert isinstance(promoted, ModelTokenCount)
    assert promoted.model_id == "Qwen/Qwen2.5-3B-Instruct"


def test_a_serialized_unit_claim_is_not_trusted_on_read() -> None:
    """An artifact saying `model_tokens` does not make its integer authoritative."""

    row = block().to_dict()
    row["token_accounting"] = {"value": 999, "unit": "model_tokens"}

    restored = SemanticSubContext.from_dict(row)

    assert isinstance(restored.token_accounting(), EstimatedTokenCount)
    assert restored.token_accounting().value == restored.token_count


def test_the_two_counts_never_convert_into_one_another() -> None:
    estimated = EstimatedTokenCount(value=100, tokenizer_fingerprint="regex")
    modelled = ModelTokenCount(
        value=100, tokenizer_fingerprint="hf-abc", model_id="m"
    )

    assert estimated != modelled
    assert not isinstance(estimated, ModelTokenCount)
    with pytest.raises(TokenAccountingError):
        require_model_tokens(estimated, purpose="test")
    assert require_model_tokens(modelled, purpose="test") is modelled
