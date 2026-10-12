"""Relevant semantic context selection and request manifests.

ARCHITECTURE_STAGE: 4 -- Context Selection + SubContextStore
CANONICAL_ENTRYPOINT: ContextSelectionStage (ingest / select / assess_reuse),
    defined at the bottom of this module. The selectors and
    ``AgentContextManifest`` above it are this stage's own types.
DELEGATED_HELPERS: manager.py (IncrementalContextManager: ingest transaction and
    select), subcontext_store.py (SubContextStore, SemanticSubContext),
    reuse_eligibility.py (ReuseEligibilityPolicy). The stage imports them lazily
    because each of them imports this module.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .subcontext_store import SemanticSubContext
from .token_accounting import TokenUnit


SELECTION_SYSTEM_PROMPT = """Select only sub-context metadata relevant to the current agent request.
Always retain the original task, required system/tool policy, current objective, and
necessary current-plan state. Include recent tool results only when relevant. Respect
the token budget. Return selected and excluded IDs with concise reasons as strict JSON.
You receive metadata and short previews, never the complete source text.
"""

SELECTION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["selected", "excluded"],
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sub_context_id", "reason", "priority"],
                "properties": {
                    "sub_context_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "priority": {"type": "integer"},
                },
            },
        },
        "excluded": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sub_context_id", "reason"],
                "properties": {
                    "sub_context_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class SelectionRequest:
    task_id: str
    agent_id: str
    loop_stage: str
    current_objective: str
    current_plan_step: str
    available_tools: tuple[str, ...]
    token_budget: int
    mutable_suffix_token_count: int = 0


@dataclass(frozen=True)
class SelectionEntry:
    sub_context_id: str
    reason: str
    priority: int = 100


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[SelectionEntry, ...]
    excluded: tuple[SelectionEntry, ...]
    fallback_used: bool = False


@dataclass(frozen=True)
class AgentContextManifest:
    task_id: str
    agent_id: str
    request_id: str
    session_revision: int
    objective: str
    selected_sub_context_ids: tuple[str, ...]
    selected_cache_keys: tuple[str, ...]
    mutable_text: str
    full_session_tokens: int
    candidate_context_tokens: int
    selected_context_tokens: int
    context_reduction_tokens: int
    context_reduction_ratio: float
    selection_reason: dict[str, str] = field(default_factory=dict)
    excluded_reason: dict[str, str] = field(default_factory=dict)
    selection_order: tuple[str, ...] = ()
    excluded_sub_context_ids: tuple[str, ...] = ()
    context_token_budget: int = 0
    #: The unit every token field on this manifest is counted in. The budget,
    #: the selected total and the mutable suffix all come from one tokenizer, so
    #: the comparison inside a manifest is always self-consistent -- but whether
    #: `context_token_budget` means 2000 model tokens or 2000 regex tokens
    #: decides whether a real workload fits, and that was never recorded.
    token_unit: str = TokenUnit.UNKNOWN.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticContextSelector(ABC):
    @abstractmethod
    def select_for_request(
        self, request: SelectionRequest, candidates: list[SemanticSubContext]
    ) -> SelectionResult: ...

    def explain_selection(self, result: SelectionResult) -> dict[str, dict[str, str]]:
        return {
            "selected": {entry.sub_context_id: entry.reason for entry in result.selected},
            "excluded": {entry.sub_context_id: entry.reason for entry in result.excluded},
        }

    def validate_selection(
        self,
        request: SelectionRequest,
        candidates: list[SemanticSubContext],
        result: SelectionResult,
    ) -> None:
        by_id = {value.sub_context_id: value for value in candidates}
        ids = [entry.sub_context_id for entry in result.selected]
        if len(ids) != len(set(ids)) or any(value not in by_id for value in ids):
            raise ValueError("selection contains duplicate or unknown ids")
        if sum(by_id[value].token_count for value in ids) + request.mutable_suffix_token_count > request.token_budget:
            raise ValueError("selection exceeds token budget")
        mandatory_fields = {"user_prompt", "task_instruction"}
        available = {value.context_field for value in candidates}
        selected_fields = {by_id[value].context_field for value in ids}
        missing = (mandatory_fields & available) - selected_fields
        if missing:
            raise ValueError(f"selection omitted mandatory fields: {sorted(missing)}")


def _terms(text: str) -> set[str]:
    # Split paths and qualified names so an objective such as "parser" matches
    # metadata like "src/parser.py" without exposing full source to an LLM.
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))


class RuleBasedSemanticContextSelector(SemanticContextSelector):
    _MANDATORY_FIELDS = {"user_prompt", "task_instruction", "tool_definitions"}

    def select_for_request(
        self, request: SelectionRequest, candidates: list[SemanticSubContext]
    ) -> SelectionResult:
        objective_terms = _terms(
            " ".join((request.current_objective, request.current_plan_step, *request.available_tools))
        )
        ranked: list[tuple[int, int, SemanticSubContext, str]] = []
        for value in candidates:
            mandatory = value.context_field in self._MANDATORY_FIELDS
            overlap = len(
                objective_terms
                & _terms(" ".join((value.title, value.question_or_objective, value.raw_text[:240])))
            )
            recent_tool = value.context_field == "tool_history_output" and value.session_revision == max(
                (item.session_revision for item in candidates), default=0
            )
            relevant = mandatory or overlap > 0 or recent_tool
            priority = 0 if mandatory else 10 if overlap else 20
            reason = "mandatory task/system/tool context" if mandatory else "objective term match" if overlap else "latest tool result"
            if relevant:
                ranked.append((priority, -value.session_revision, value, reason))
        ranked.sort(key=lambda row: (row[0], row[1], row[2].sub_context_id))
        budget = max(0, request.token_budget - request.mutable_suffix_token_count)
        selected: list[SelectionEntry] = []
        selected_ids: set[str] = set()
        used = 0
        for priority, _revision, value, reason in ranked:
            mandatory = value.context_field in self._MANDATORY_FIELDS
            if used + value.token_count > budget:
                if mandatory:
                    raise ValueError("token budget cannot fit mandatory context")
                continue
            selected.append(SelectionEntry(value.sub_context_id, reason, priority))
            selected_ids.add(value.sub_context_id)
            used += value.token_count
        excluded = tuple(
            SelectionEntry(
                value.sub_context_id,
                "token budget" if any(row[2].sub_context_id == value.sub_context_id for row in ranked) else "not relevant",
            )
            for value in candidates
            if value.sub_context_id not in selected_ids
        )
        result = SelectionResult(tuple(selected), excluded)
        self.validate_selection(request, candidates, result)
        return result


class LLMMetadataSemanticContextSelector(SemanticContextSelector):
    def __init__(self, transport: Callable[[dict[str, Any]], Any], model: str) -> None:
        self.transport = transport
        self.model = model

    def select_for_request(
        self, request: SelectionRequest, candidates: list[SemanticSubContext]
    ) -> SelectionResult:
        metadata = [
            {
                "sub_context_id": value.sub_context_id,
                "context_field": value.context_field,
                "title": value.title,
                "question_or_objective": value.question_or_objective,
                "token_count": value.token_count,
                "recency": value.session_revision,
                "source_type": value.context_field,
                "preview": value.raw_text[:160],
                "is_mutable": value.is_mutable,
                "reuse_candidate": value.reuse_candidate,
            }
            for value in candidates
        ]
        response = self.transport(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "context_selection",
                        "strict": True,
                        "schema": SELECTION_RESPONSE_SCHEMA,
                    },
                },
                "messages": [
                    {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"request": asdict(request), "candidates": metadata})},
                ],
            }
        )
        if isinstance(response, str):
            response = json.loads(response)
        if "choices" in response:
            response = json.loads(response["choices"][0]["message"]["content"])
        result = SelectionResult(
            selected=tuple(
                SelectionEntry(str(row["sub_context_id"]), str(row["reason"]), int(row.get("priority", 100)))
                for row in response["selected"]
            ),
            excluded=tuple(
                SelectionEntry(str(row["sub_context_id"]), str(row["reason"]))
                for row in response.get("excluded", [])
            ),
        )
        self.validate_selection(request, candidates, result)
        return result


class HybridSemanticContextSelector(SemanticContextSelector):
    def __init__(
        self,
        llm_selector: SemanticContextSelector,
        fallback: SemanticContextSelector | None = None,
    ) -> None:
        self.llm_selector = llm_selector
        self.fallback = fallback or RuleBasedSemanticContextSelector()

    def select_for_request(
        self, request: SelectionRequest, candidates: list[SemanticSubContext]
    ) -> SelectionResult:
        try:
            return self.llm_selector.select_for_request(request, candidates)
        except Exception:
            result = self.fallback.select_for_request(request, candidates)
            return SelectionResult(result.selected, result.excluded, fallback_used=True)


def build_manifest(
    *,
    request: SelectionRequest,
    request_id: str,
    candidates: list[SemanticSubContext],
    selection: SelectionResult,
    mutable_text: str,
    full_session_tokens: int,
    token_unit: TokenUnit = TokenUnit.UNKNOWN,
) -> AgentContextManifest:
    by_id = {value.sub_context_id: value for value in candidates}
    selected_values = [by_id[entry.sub_context_id] for entry in selection.selected]

    def ingest_order(value: SemanticSubContext) -> tuple[Any, ...]:
        """Order context the way the session produced it.

        A selector ranks candidates by relevance, and the rule-based selector
        sorts newest first so a recent tool result outranks an older one. That
        is a selection priority, not a reading order: rendering an agent's
        conversation newest-first misstates what happened. Presentation is
        therefore ordered by ingestion instead, which is chronological and
        stable across turns.
        """

        spans = sorted(
            value.source_spans,
            key=lambda span: (span.source_id, span.start, span.end),
        )
        first = spans[0] if spans else None
        return (
            value.session_revision,
            value.source_order,
            first.start if first else 0,
            first.end if first else 0,
            value.sub_context_id,
        )

    # Runtime LinkPlan supports an ordered immutable prefix followed by one
    # mutable suffix, so the two groups stay separate. Within each group the
    # order is chronological.
    selected_values = [
        *sorted(
            (
                value
                for value in selected_values
                if value.reuse_candidate and not value.is_mutable
            ),
            key=ingest_order,
        ),
        *sorted(
            (
                value
                for value in selected_values
                if not value.reuse_candidate or value.is_mutable
            ),
            key=ingest_order,
        ),
    ]
    selected_tokens = sum(value.token_count for value in selected_values)
    candidate_tokens = sum(value.token_count for value in candidates)
    reduction = max(0, full_session_tokens - selected_tokens)
    return AgentContextManifest(
        task_id=request.task_id,
        agent_id=request.agent_id,
        request_id=request_id,
        session_revision=max((value.session_revision for value in candidates), default=0),
        objective=request.current_objective,
        selected_sub_context_ids=tuple(value.sub_context_id for value in selected_values),
        selected_cache_keys=tuple(
            value.cache_key
            for value in selected_values
            if value.reuse_candidate and not value.is_mutable
        ),
        mutable_text=mutable_text,
        full_session_tokens=full_session_tokens,
        token_unit=TokenUnit(token_unit).value,
        candidate_context_tokens=candidate_tokens,
        selected_context_tokens=selected_tokens,
        context_reduction_tokens=reduction,
        context_reduction_ratio=(reduction / full_session_tokens if full_session_tokens else 0.0),
        selection_reason={entry.sub_context_id: entry.reason for entry in selection.selected},
        excluded_reason={entry.sub_context_id: entry.reason for entry in selection.excluded},
        selection_order=tuple(value.sub_context_id for value in selected_values),
        excluded_sub_context_ids=tuple(
            entry.sub_context_id for entry in selection.excluded
        ),
        context_token_budget=request.token_budget,
    )


class ContextSelectionStage:
    """Stage 4 facade: ingest into the SubContextStore, select, and assess reuse.

    ARCHITECTURE_STAGE: 4
    CANONICAL_ENTRYPOINT: ContextSelectionStage.ingest / .select / .assess_reuse

    It owns no state and no semantics. ``ingest`` and ``select`` are
    ``IncrementalContextManager.ingest`` and ``.select`` (which carry the
    transactional ingest, the source-coverage fail-closed check and the
    sibling-atomicity rule); ``assess_reuse`` is ``ReuseEligibilityPolicy.assess``
    against a ``CacheCatalog``. The Gate 1 *decision* that gates a compile
    decision is still taken inside ``NOCRequestPlanner`` (Stage 6) -- this method
    only exposes the per-block verdict to a Stage 4 caller.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    @property
    def store(self) -> Any:
        """The SubContextStore the manager reads and writes."""

        return self.manager.store

    def ingest(self, events: Any) -> Any:
        return self.manager.ingest(events)

    def select(
        self,
        *,
        agent_id: str,
        loop_stage: str,
        objective: str,
        plan_step: str,
        available_tools: tuple[str, ...],
        token_budget: int,
        mutable_text: str,
    ) -> tuple[AgentContextManifest, list[SemanticSubContext]]:
        return self.manager.select(
            agent_id=agent_id,
            loop_stage=loop_stage,
            objective=objective,
            plan_step=plan_step,
            available_tools=available_tools,
            token_budget=token_budget,
            mutable_text=mutable_text,
        )

    @staticmethod
    def assess_reuse(
        contexts: list[SemanticSubContext],
        cache_catalog: Any,
        policy: Any = None,
    ) -> tuple[Any, ...]:
        from .reuse_eligibility import ReuseEligibilityPolicy

        policy = policy or ReuseEligibilityPolicy()
        return tuple(
            policy.assess(value, cache_catalog.metadata(value.cache_key))
            for value in contexts
        )
