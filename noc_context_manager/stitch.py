"""Deterministic baseline/LegoLink representation construction and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any

from .context_selection import AgentContextManifest
from .scheduler import NOCRequestExecutionPlan
from .source_text import compose_source_text, source_boundary
from .subcontext_store import SemanticSubContext
from .tokenizer import TextTokenizer


@dataclass(frozen=True)
class PromptFramingPolicy:
    """Logical prompt framing shared by baseline and LegoLink representations."""

    version: str = "noc-openai-chat-body-v1"
    role_template: str = "single-user-message"
    separator: str = "\n\n"
    bos_policy: str = "server-chat-template"
    eos_policy: str = "server-chat-template"
    newline_policy: str = "verbatim-source-plus-explicit-separator"


@dataclass(frozen=True)
class StitchEquivalenceReport:
    selected_sub_contexts_match: bool
    block_order_matches: bool
    role_template_matches: bool
    separator_tokens_match: bool
    bos_policy_matches: bool
    eos_policy_matches: bool
    newline_policy_matches: bool
    mutable_suffix_matches: bool
    generation_parameters_match: bool
    immutable_content_not_duplicated: bool
    complete_selected_text_matches: bool
    complete_selected_token_ids_match: bool
    runtime_token_validation_scope: str

    @property
    def valid(self) -> bool:
        return all(
            (
                self.selected_sub_contexts_match,
                self.block_order_matches,
                self.role_template_matches,
                self.separator_tokens_match,
                self.bos_policy_matches,
                self.eos_policy_matches,
                self.newline_policy_matches,
                self.mutable_suffix_matches,
                self.generation_parameters_match,
                self.immutable_content_not_duplicated,
                self.complete_selected_text_matches,
                self.complete_selected_token_ids_match,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


@dataclass(frozen=True)
class StitchedContext:
    """Equivalent logical views derived from one NOC execution plan."""

    selected_text: str
    mutable_text: str
    linked_contexts: tuple[SemanticSubContext, ...]
    mutable_contexts: tuple[SemanticSubContext, ...]
    omitted_contexts: tuple[SemanticSubContext, ...]
    selected_order: tuple[str, ...]
    baseline_context_ids: tuple[str, ...]
    legolink_context_ids: tuple[str, ...]
    expected_cache_keys: tuple[str, ...]
    baseline_framing: PromptFramingPolicy
    legolink_framing: PromptFramingPolicy
    separator_token_ids: tuple[Any, ...]
    expected_complete_selected_token_ids: tuple[Any, ...]
    generation_parameters: dict[str, Any]
    generation_parameters_sha256: str
    selected_content_sha256: str
    equivalence: StitchEquivalenceReport

    @property
    def block_ids(self) -> tuple[str, ...]:
        """The semantic identity the runtime matches a cached block against.

        This is the content identity, not `sub_context_id`, because the runtime
        rejects a block whose recorded `semantic_block_id` is neither the id the
        link asked for nor the block's own cache key. Sending the occurrence id
        would make a block compiled under one task unusable by another that has
        byte-identical content -- the reuse this project exists to measure.
        Rows written before the identity split fall back to their occurrence id,
        which is what they were compiled under.
        """

        return tuple(
            value.semantic_identity or value.sub_context_id
            for value in self.linked_contexts
        )

    @property
    def occurrence_ids(self) -> tuple[str, ...]:
        """Provenance: which occurrence of each block this request selected."""

        return tuple(value.sub_context_id for value in self.linked_contexts)

    @property
    def cache_keys(self) -> tuple[str, ...]:
        return tuple(value.cache_key for value in self.linked_contexts)

    @property
    def mutable_suffix_nonempty(self) -> bool:
        return bool(self.mutable_text)


class ContextStitcher:
    """Create B/C views from one plan without deciding physical KV placement."""

    def __init__(
        self,
        *,
        separator: str = "\n\n",
        tokenizer: TextTokenizer | None = None,
        framing: PromptFramingPolicy | None = None,
        baseline_framing: PromptFramingPolicy | None = None,
        legolink_framing: PromptFramingPolicy | None = None,
    ) -> None:
        if framing is not None and (
            baseline_framing is not None or legolink_framing is not None
        ):
            raise ValueError(
                "framing cannot be combined with baseline_framing or legolink_framing"
            )
        self.tokenizer = tokenizer
        shared = framing or PromptFramingPolicy(separator=separator)
        self.baseline_framing = baseline_framing or shared
        self.legolink_framing = legolink_framing or shared
        # Compatibility for callers that used one shared framing policy.
        self.framing = self.baseline_framing
        self.separator = separator

    def _join(self, values: Iterable[str]) -> str:
        return self.separator.join(value for value in values if value)

    def _tokenize(self, text: str) -> tuple[Any, ...]:
        if self.tokenizer is None:
            return ()
        return tuple(self.tokenizer.tokenize(text))

    @staticmethod
    def _generation_hash(generation: dict[str, Any]) -> str:
        encoded = json.dumps(
            generation,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def render_selected(
        self,
        manifest: AgentContextManifest,
        contexts: Iterable[SemanticSubContext],
    ) -> str:
        context_list = list(contexts)
        by_id = {value.sub_context_id: value for value in context_list}
        if len(by_id) != len(context_list):
            raise ValueError("contexts contain duplicate sub_context_id values")
        missing = [
            value for value in manifest.selected_sub_context_ids if value not in by_id
        ]
        if missing:
            raise ValueError(f"manifest references unknown sub-context ids: {missing}")
        values = [by_id[value] for value in manifest.selected_sub_context_ids]
        return compose_source_text(
            values,
            separator=self.separator,
            trailing_text=manifest.mutable_text,
        )

    def stitch(
        self,
        plan: NOCRequestExecutionPlan,
        *,
        generation_parameters: dict[str, Any] | None = None,
        baseline_generation_parameters: dict[str, Any] | None = None,
        legolink_generation_parameters: dict[str, Any] | None = None,
    ) -> StitchedContext:
        if plan.text_separator != self.separator:
            raise ValueError("execution plan and stitcher separators differ")
        shared_generation = dict(generation_parameters or {})
        baseline_generation = dict(
            baseline_generation_parameters
            if baseline_generation_parameters is not None
            else shared_generation
        )
        legolink_generation = dict(
            legolink_generation_parameters
            if legolink_generation_parameters is not None
            else shared_generation
        )
        selected_text = plan.baseline_selected_text
        linked = plan.linked_contexts
        mutable = plan.mutable_contexts
        linked_views: list[SemanticSubContext] = []
        for index, value in enumerate(linked):
            next_value = (
                linked[index + 1]
                if index + 1 < len(linked)
                else mutable[0]
                if mutable
                else None
            )
            suffix = (
                self.separator
                if next_value is None and plan.manifest.mutable_text
                else ""
                if next_value is None
                else source_boundary(value, next_value, self.separator)
            )
            linked_views.append(replace(value, raw_text=value.raw_text + suffix))
        mutable_text = compose_source_text(
            mutable,
            separator=self.separator,
            trailing_text=plan.manifest.mutable_text,
        )
        linked_ids = tuple(value.sub_context_id for value in linked)
        mutable_ids = tuple(value.sub_context_id for value in plan.mutable_contexts)
        runtime_order = (*linked_ids, *mutable_ids)
        baseline_order = tuple(value.sub_context_id for value in plan.included_contexts)
        declared_order = (
            plan.manifest.selection_order or plan.manifest.selected_sub_context_ids
        )
        logical_link_text = "".join(value.raw_text for value in linked_views) + mutable_text
        baseline_separator_tokens = self._tokenize(self.baseline_framing.separator)
        legolink_separator_tokens = self._tokenize(self.legolink_framing.separator)
        complete_tokens = self._tokenize(selected_text)
        logical_link_tokens = self._tokenize(logical_link_text)
        # Equal text may legitimately occur at different conversation turns.
        # Ownership is defined by sub-context identity; complete text/token equality
        # below separately proves that no occurrence was inserted or removed.
        immutable_not_duplicated = set(linked_ids).isdisjoint(mutable_ids)
        baseline_framing = replace(
            self.baseline_framing,
            newline_policy="verbatim-source-with-semantic-sibling-continuation-v1",
        )
        # Physical framing already concatenates every linked block with this
        # suffix. Boundary-aware linked views carry the configured separator
        # only at true source boundaries, so its uniform suffix must be empty.
        legolink_framing = replace(
            self.legolink_framing,
            separator="",
            newline_policy="verbatim-source-with-semantic-sibling-continuation-v1",
        )
        equivalence = StitchEquivalenceReport(
            selected_sub_contexts_match=(
                baseline_order == tuple(declared_order)
                and baseline_order == runtime_order
            ),
            block_order_matches=baseline_order == runtime_order,
            role_template_matches=(
                baseline_framing.role_template == legolink_framing.role_template
            ),
            separator_tokens_match=(
                self.baseline_framing.separator == self.legolink_framing.separator
                and self.baseline_framing.separator == self.separator
                and baseline_separator_tokens == legolink_separator_tokens
            ),
            bos_policy_matches=(
                baseline_framing.bos_policy == legolink_framing.bos_policy
            ),
            eos_policy_matches=(
                baseline_framing.eos_policy == legolink_framing.eos_policy
            ),
            newline_policy_matches=(
                baseline_framing.newline_policy == legolink_framing.newline_policy
            ),
            mutable_suffix_matches=(
                mutable_text
                == compose_source_text(
                    mutable,
                    separator=self.separator,
                    trailing_text=plan.manifest.mutable_text,
                )
            ),
            generation_parameters_match=(baseline_generation == legolink_generation),
            immutable_content_not_duplicated=immutable_not_duplicated,
            complete_selected_text_matches=selected_text == logical_link_text,
            complete_selected_token_ids_match=(
                selected_text == logical_link_text
                if self.tokenizer is None
                else complete_tokens == logical_link_tokens
            ),
            runtime_token_validation_scope=(
                "authoritative reconstructed-content token IDs; "
                "physical_framing builds one normalized chat-template layout, "
                "and patched SGLang validates its token counts/hashes before "
                "Runtime LinkPlan execution"
            ),
        )
        if not equivalence.valid:
            raise ValueError(
                "baseline and LegoLink logical representations are not equivalent: "
                f"{equivalence.to_dict()}"
            )
        return StitchedContext(
            selected_text=selected_text,
            mutable_text=mutable_text,
            linked_contexts=tuple(linked_views),
            mutable_contexts=plan.mutable_contexts,
            omitted_contexts=plan.omitted_contexts,
            selected_order=plan.selected_order,
            baseline_context_ids=baseline_order,
            legolink_context_ids=runtime_order,
            expected_cache_keys=plan.expected_cache_keys,
            baseline_framing=baseline_framing,
            legolink_framing=legolink_framing,
            separator_token_ids=self._tokenize(legolink_framing.separator),
            expected_complete_selected_token_ids=complete_tokens,
            generation_parameters=legolink_generation,
            generation_parameters_sha256=self._generation_hash(legolink_generation),
            selected_content_sha256=hashlib.sha256(
                selected_text.encode("utf-8")
            ).hexdigest(),
            equivalence=equivalence,
        )

    @staticmethod
    def validate_link_suffix(stitched: StitchedContext) -> None:
        if not stitched.mutable_suffix_nonempty:
            raise ValueError(
                "LegoLink requires a non-empty mutable suffix; the runtime "
                "samples from the final mutable token"
            )


def render_selected_context(
    manifest: AgentContextManifest,
    contexts: Iterable[SemanticSubContext],
) -> str:
    """Compatibility helper for callers that only need selected source text."""

    return ContextStitcher().render_selected(manifest, contexts)
