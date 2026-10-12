"""Logical request planning for selected NOC semantic sub-contexts.

ARCHITECTURE_STAGE: 6 -- Logical Context Assembly + Reuse Plan
CANONICAL_ENTRYPOINT: NOCRequestPlanner.plan() (produces NOCRequestExecutionPlan)
DELEGATED_HELPERS: compile_policy.CompileAndLinkFeasibilityPlanner (Stage 5:
    provisional decisions and layout feasibility), reuse_eligibility.py (Gate 1),
    reuse_cost_model.py (Gate 3, observe-only), stitch.ContextStitcher (text
    assembly of the plan, invoked after planning by the pipeline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping

from .cache_namespace import (
    NAMESPACE_NOT_PERSISTENT,
    NamespaceLookupResult,
    NamespaceResolver,
    summarize_namespace_request,
)
from .compile_cost import REPAIR_WINDOW_CALLER_SUPPLIED
from .compile_policy import (
    BELOW_COMPILE_FLOOR,
    LEGACY_COMPILE_COST_REASON,
    NET_BENEFIT_NEGATIVE,
    NO_EXPECTED_REUSE,
    CacheCatalog,
    CompileAction,
    CompileAndLinkFeasibilityPlanner,
    CompileDecision,
    CompilePolicy,
)
from .context_selection import AgentContextManifest
from .link_layout import (
    POSITIONAL_LAYOUT_DEMOTION,
    POSITIONAL_LAYOUT_PROMOTION,  # noqa: F401 -- re-exported; imported from here by tests
    POSITIONAL_LAYOUT_REASONS,
    LayoutAdjustment,
    LayoutDecision,
    LayoutPlanningSummary,
    LinkLayoutFeasibility,
)
from .reuse_eligibility import (
    ELIGIBILITY_DECLINE_REASONS,
    REUSE_ELIGIBILITY_POLICY_VERSION,
    EligibilityDecision,
    ReuseEligibility,
    ReuseEligibilityPolicy,
    identity_basis,
    is_semantic_candidate,
)
from .reuse_cost_model import (
    ObserveOnlyProfitabilityPolicy,
    ProfitabilityVerdict,
    RequestReuseAccounting,
    ReuseProfitabilityPolicy,
    require_observe_only,
)
from .subcontext_store import SemanticSubContext


class PlannerCategory(str, Enum):
    """Where a selected block ended up. Every selected block gets exactly one."""

    REUSE_CACHED = "REUSE_CACHED"
    COMPILE_ON_MISS = "COMPILE_ON_MISS"
    KEEP_MUTABLE = "KEEP_MUTABLE"
    DROPPED = "DROPPED"


# A reusable, immutable block is normally expected to be linked. These are the
# reasons the planner is allowed to decline one anyway. Anything else keeping
# such a block mutable is a planning bug, not a policy decision -- see
# `NOCRequestExecutionPlan.planning_invariant_violations`.
LEGITIMATE_KEEP_MUTABLE_REASONS = frozenset(
    {
        # CompilePolicy: the block is smaller than min_compile_tokens, so a
        # compile round trip buys less than it costs. Measured cause of every
        # unlinked non-prefix block in the budget-2000 runs (blocks of 12, 41,
        # 44, 45, 48 tokens against a floor of 64).
        BELOW_COMPILE_FLOOR,
        NO_EXPECTED_REUSE,
        # CostBenefitCompilePolicy: the block's expected reuse savings did not
        # cover compiling, linking and holding it. The milliseconds behind this
        # are on the decision's provenance, so unlike the floor it can be
        # re-derived rather than taken on trust.
        NET_BENEFIT_NEGATIVE,
        # Both of the above were one branch, with one reason, until
        # 2026-09-03. Artifacts recorded before then still audit.
        LEGACY_COMPILE_COST_REASON,
        "high KV memory pressure",
        # _preserve_link_layout: a compiled block after mutable content would
        # silently reorder the manifest, because the Runtime LinkPlan is
        # immutable-prefix + one mutable suffix.
        POSITIONAL_LAYOUT_DEMOTION,
        # Gate 1 (reuse_eligibility): the cached state behind the key is not
        # established as reusable, or is a foreign contextual state with the
        # experiment flag off. A semantic match is not permission to link.
        *ELIGIBILITY_DECLINE_REASONS,
        # Namespace policy (cache_namespace): the block's namespace is not
        # persistent (current-turn content, or `no_cache`), so it is never
        # compiled into the semantic cache. Only applied outside GLOBAL mode.
        NAMESPACE_NOT_PERSISTENT,
    }
)


class FeasibilityReason(str, Enum):
    """Machine-readable form of `LayoutAdjustment`, for the planning record."""

    NONE = "none"
    LAYOUT_FORCED_COMPILE = "layout_forced_compile"
    LAYOUT_FORCED_NATIVE = "layout_forced_native"


_FEASIBILITY_REASON = {
    LayoutAdjustment.NONE.value: FeasibilityReason.NONE.value,
    LayoutAdjustment.FORCE_COMPILE_FOR_PREFIX_CLOSURE.value: (
        FeasibilityReason.LAYOUT_FORCED_COMPILE.value
    ),
    LayoutAdjustment.FORCE_MUTABLE_FOR_SUFFIX_CLOSURE.value: (
        FeasibilityReason.LAYOUT_FORCED_NATIVE.value
    ),
}


@dataclass(frozen=True)
class BlockPlanningRecord:
    """One selected block, and the category the planner actually gave it.

    This exists because the metrics rows recorded what was *selected* and what
    the runtime *linked*, with nothing in between. A block could be selected,
    be reusable and immutable, and still never reach the runtime -- and the
    artifacts said only that a link had completed.
    """

    block_id: str
    cache_key: str
    selected: bool
    immutable: bool
    cache_candidate: bool
    reuse_decision: str
    planner_category: str
    reason: str
    token_count: int = 0
    #: Which policy produced the underlying decision, and what it computed.
    #: Empty for a decision made before provenance existed; `None` net benefit
    #: for the legacy floor, which computes no milliseconds at all.
    compile_policy: str = ""
    net_benefit_ms: float | None = None
    #: The two-stage split. `reuse_decision` and `reason` above describe the
    #: *effective* action, which is what the runtime executed and what every
    #: existing consumer of this record already reads. These say what the
    #: policy asked for and what the LinkPlan shape then required, so a forced
    #: compile is never reported as a policy choice.
    policy_action: str = ""
    policy_reason: str = ""
    policy_expected_saved_ms: float | None = None
    policy_total_cost_ms: float | None = None
    layout_adjustment: str = LayoutAdjustment.NONE.value
    layout_adjustment_reason: str = ""
    profile_id: str | None = None
    #: Gate 1, kept apart from the two semantic flags above. `cache_candidate`
    #: (kept for artifact compatibility) means *semantic* reuse candidacy only;
    #: `cache_available` means the host catalog holds a non-stale entry, which
    #: is bookkeeping and not runtime residency; `reuse_eligibility` is the
    #: verdict about the *state* behind the key. Defaults describe a record
    #: built without Gate 1 (an artifact written before it existed).
    semantic_candidate: bool | None = None
    cache_available: bool | None = None
    cache_availability: str = ""
    reuse_eligibility: str = ""
    eligibility_reason: str = ""
    state_origin: str = ""
    identity_basis: str = ""
    #: The compile policy's action before Gate 1. `policy_action` is what
    #: entered layout feasibility, i.e. after Gate 1.
    provisional_action: str = ""
    eligibility_adjusted: bool = False
    experimental: bool = False
    preceding_context_tokens: int | None = None

    @property
    def feasibility_reason(self) -> str:
        return _FEASIBILITY_REASON.get(self.layout_adjustment, self.layout_adjustment)

    @property
    def effective_decision(self) -> str:
        """Where the block ended up; the planner category, named for the gates."""

        return self.planner_category

    @property
    def effective_action(self) -> str:
        """Alias for `reuse_decision`, named for what it now contrasts with."""

        return self.reuse_decision

    @property
    def layout_forced(self) -> bool:
        return self.layout_adjustment != LayoutAdjustment.NONE.value

    @property
    def positional_override(self) -> bool:
        """The layout constraint, not a policy, decided this block's fate.

        Reads the recorded adjustment rather than matching the reason string;
        the string comparison is kept as a fallback so records rebuilt from
        artifacts written before the split still classify correctly.
        """

        if self.layout_adjustment != LayoutAdjustment.NONE.value:
            return True
        return self.reason in POSITIONAL_LAYOUT_REASONS

    @property
    def planned_for_link(self) -> bool:
        """Host *intent*: the planner routed this block through the link path.

        This is not evidence that the runtime linked it. The runtime can still
        report the block missing, stale or incompatible, and only the runtime
        trace (`kv_block_linked`) says what physically happened. Read this when
        you mean "what did the planner ask for"; read the trace when you mean
        "what did the runtime do".
        """

        return self.planner_category in {
            PlannerCategory.REUSE_CACHED.value,
            PlannerCategory.COMPILE_ON_MISS.value,
        }

    @property
    def linked(self) -> bool:
        """Deprecated alias for :attr:`planned_for_link`.

        Kept because artifacts and summarizers already read it. The name says
        more than the value knows, which is why the honest name exists above.
        """

        return self.planned_for_link

    @property
    def reusable_selection(self) -> bool:
        """Whether selection offered this block as a reuse opportunity.

        Mirrors the filter `context_selection` applies when it builds
        `manifest.selected_cache_keys`, so host accounting and the manifest
        cannot disagree about what was on the table.
        """

        return self.selected and self.cache_candidate and self.immutable

    @property
    def declined_legitimately(self) -> bool:
        return (
            self.planner_category == PlannerCategory.KEEP_MUTABLE.value
            and self.reason in LEGITIMATE_KEEP_MUTABLE_REASONS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "cache_key": self.cache_key,
            "selected": self.selected,
            "immutable": self.immutable,
            "cache_candidate": self.cache_candidate,
            "reuse_decision": self.reuse_decision,
            "planner_category": self.planner_category,
            "reason": self.reason,
            "token_count": self.token_count,
            "compile_policy": self.compile_policy,
            "net_benefit_ms": self.net_benefit_ms,
            "positional_override": self.positional_override,
            "policy_action": self.policy_action,
            "policy_reason": self.policy_reason,
            "policy_net_benefit_ms": self.net_benefit_ms,
            "policy_expected_saved_ms": self.policy_expected_saved_ms,
            "policy_total_cost_ms": self.policy_total_cost_ms,
            "layout_adjustment": self.layout_adjustment,
            "layout_adjustment_reason": self.layout_adjustment_reason,
            "layout_forced": self.layout_forced,
            "effective_action": self.effective_action,
            "policy_name": self.compile_policy,
            "profile_id": self.profile_id,
            "semantic_candidate": self.semantic_candidate,
            "cache_available": self.cache_available,
            "cache_availability": self.cache_availability,
            "reuse_eligibility": self.reuse_eligibility,
            "eligibility_reason": self.eligibility_reason,
            "state_origin": self.state_origin,
            "identity_basis": self.identity_basis,
            "provisional_decision": self.provisional_action,
            "eligibility_adjusted": self.eligibility_adjusted,
            "feasibility_adjustment": self.layout_adjustment,
            "feasibility_reason": self.feasibility_reason,
            "effective_decision": self.effective_decision,
            "experimental": self.experimental,
            "preceding_context_tokens": self.preceding_context_tokens,
        }


@dataclass(frozen=True)
class NOCRequestExecutionPlan:
    """Side-effect-free NOC plan; it never contains physical GPU/KV state."""

    manifest: AgentContextManifest
    ordered_contexts: tuple[SemanticSubContext, ...]
    decisions: tuple[CompileDecision, ...]
    cached_cache_keys: tuple[str, ...] = ()
    text_separator: str = "\n\n"
    #: One per ordered block: the policy's decision, the layout adjustment (if
    #: any), and the effective decision. `decisions` above holds the effective
    #: ones, which is what the runtime executes and what every existing
    #: consumer already reads; this is what lets a report separate the two.
    #: Empty on a plan built before the split, and on `bind_decisions` callers
    #: that supply decisions directly.
    layout_decisions: tuple[LayoutDecision, ...] = ()
    #: Gate 1, one per ordered block: the compile policy's provisional
    #: decision, the eligibility verdict, and the decision that entered layout.
    #: Empty only on a plan constructed by hand.
    eligibility_decisions: tuple[EligibilityDecision, ...] = ()
    #: Gate 3, observe-only: what the profitability policy could say about
    #: this plan. Never applied; `None` on a plan constructed by hand.
    profitability: ProfitabilityVerdict | None = None
    #: One per ordered block when a `NamespaceResolver` was consulted: the
    #: namespace, whether the block was resident through it, and any alias.
    #: Observation of the lookup; the decisions above already reflect it.
    namespace_resolutions: tuple[NamespaceLookupResult, ...] = ()

    def namespace_summary(
        self,
        *,
        repair_tokens_by_cache_key: Mapping[str, int] | None = None,
        recovered_cache_keys: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        """Per-namespace counters for this request, or `None` if no resolver ran."""

        if not self.namespace_resolutions:
            return None
        decision_by_id = {value.sub_context_id: value for value in self.decisions}
        linked = [
            value.sub_context_id
            for value in self.ordered_contexts
            if decision_by_id[value.sub_context_id].action == CompileAction.COMPILE
        ]
        return summarize_namespace_request(
            self.namespace_resolutions,
            linked_block_ids=linked,
            link_cache_key_by_block={
                value.sub_context_id: value.cache_key for value in self.ordered_contexts
            },
            repair_tokens_by_cache_key=repair_tokens_by_cache_key,
            recovered_cache_keys=recovered_cache_keys,
        )

    def reuse_accounting(self, **measurements) -> RequestReuseAccounting:
        """Per-request cost accounting; see `reuse_cost_model.RequestReuseAccounting.from_plan`."""

        return RequestReuseAccounting.from_plan(self, **measurements)

    @property
    def experimental_links(self) -> tuple[str, ...]:
        """Blocks linked through experiment-only foreign state. Empty on every canonical plan."""

        eligibility = {
            value.sub_context_id: value.assessment for value in self.eligibility_decisions
        }
        return tuple(
            value.sub_context_id
            for value in self.linked_contexts
            if value.sub_context_id in eligibility
            and eligibility[value.sub_context_id].experimental
        )

    @property
    def canonical(self) -> bool:
        return not self.experimental_links

    def eligibility_summary(self) -> dict[str, Any]:
        """Per-request Gate-1 counts. Planning diagnostics, not a performance claim."""

        records = self.planning_records
        counts = {value.value: 0 for value in ReuseEligibility}
        for record in records:
            if record.reuse_eligibility:
                counts[record.reuse_eligibility] += 1
        return {
            "reuse_eligibility_policy_version": REUSE_ELIGIBILITY_POLICY_VERSION,
            "selected_block_count": len(records),
            "semantic_candidate_count": sum(bool(r.semantic_candidate) for r in records),
            "cache_available_count": sum(bool(r.cache_available) for r in records),
            "eligibility_counts": counts,
            "eligible_reuse_block_count": sum(
                r.reuse_eligibility == ReuseEligibility.EXACT_SAFE.value for r in records
            ),
            "eligibility_declined_count": sum(r.eligibility_adjusted for r in records),
            "experimental_link_count": len(self.experimental_links),
            "canonical": self.canonical,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize only logical state; physical KV state is deliberately absent."""

        return {
            "manifest": self.manifest.to_dict(),
            "selected_order": list(self.selected_order),
            "reuse_cached": [
                value.sub_context_id for value in self.reuse_cached_contexts
            ],
            "compile_on_miss": [
                value.sub_context_id for value in self.compile_on_miss_contexts
            ],
            "keep_mutable": [
                value.sub_context_id for value in self.keep_mutable_contexts
            ],
            "excluded": [
                value.sub_context_id for value in self.excluded_contexts
            ],
            "expected_cache_keys": list(self.expected_cache_keys),
            "block_planning": [
                record.to_dict() for record in self.planning_records
            ],
            "layout_decisions": [
                value.to_dict() for value in self.layout_decisions
            ],
            "layout_summary": self.layout_summary.to_dict(),
            "reuse_eligibility_summary": self.eligibility_summary(),
            "profitability": (
                None if self.profitability is None else self.profitability.to_dict()
            ),
            "baseline_selected_text": self.baseline_selected_text,
            "text_separator": self.text_separator,
            **(
                {"namespace_resolution": [row.to_dict() for row in self.namespace_resolutions]}
                if self.namespace_resolutions
                else {}
            ),
            "decisions": [
                {
                    **asdict(decision),
                    "action": decision.action.value,
                    # asdict() would flatten the provenance into raw field
                    # names and nest the cost breakdown; the curated form is
                    # what the audit scripts and reports read.
                    "provenance": (
                        None
                        if decision.provenance is None
                        else decision.provenance.to_dict()
                    ),
                }
                for decision in self.decisions
            ],
        }

    @property
    def planning_records(self) -> tuple[BlockPlanningRecord, ...]:
        """Every selected block with an explicit category and reason.

        There is no default branch: a decision is COMPILE (and then either
        already cached or not), DO_NOT_INCLUDE, or it keeps the block in the
        mutable suffix. The reason is the planner's own, carried through
        unmodified so it can be audited against the branch that produced it.
        """

        decision_by_id = {
            decision.sub_context_id: decision for decision in self.decisions
        }
        layout_by_id = {
            value.sub_context_id: value for value in self.layout_decisions
        }
        eligibility_by_id = {
            value.sub_context_id: value for value in self.eligibility_decisions
        }
        cached = set(self.cached_cache_keys)
        records: list[BlockPlanningRecord] = []
        preceding = 0
        for value in self.ordered_contexts:
            gate = eligibility_by_id.get(value.sub_context_id)
            assessment = None if gate is None else gate.assessment
            # Selected-body tokens before this block (chat prefix excluded; a
            # dropped block occupies no range). A host estimate of the position
            # shift a standalone-compiled block is placed at.
            preceding_here = preceding
            if decision_by_id[value.sub_context_id].action != CompileAction.DO_NOT_INCLUDE:
                preceding += int(value.token_count)
            decision = decision_by_id[value.sub_context_id]
            if decision.action == CompileAction.COMPILE:
                category = (
                    PlannerCategory.REUSE_CACHED
                    if value.cache_key in cached
                    else PlannerCategory.COMPILE_ON_MISS
                )
            elif decision.action == CompileAction.DO_NOT_INCLUDE:
                category = PlannerCategory.DROPPED
            else:
                category = PlannerCategory.KEEP_MUTABLE
            provenance = decision.provenance
            cost = None if provenance is None else provenance.cost
            layout = layout_by_id.get(value.sub_context_id)
            policy_decision = decision if layout is None else layout.policy_decision
            records.append(
                BlockPlanningRecord(
                    block_id=value.sub_context_id,
                    cache_key=value.cache_key,
                    selected=True,
                    immutable=not value.is_mutable,
                    cache_candidate=value.reuse_candidate,
                    reuse_decision=decision.action.value,
                    planner_category=category.value,
                    reason=decision.reason,
                    token_count=value.token_count,
                    compile_policy=(
                        "" if provenance is None else provenance.compile_policy
                    ),
                    net_benefit_ms=None if cost is None else cost.net_benefit_ms,
                    policy_action=policy_decision.action.value,
                    policy_reason=policy_decision.reason,
                    policy_expected_saved_ms=(
                        None if cost is None else cost.expected_saved_ms
                    ),
                    policy_total_cost_ms=None if cost is None else cost.total_cost_ms,
                    layout_adjustment=(
                        LayoutAdjustment.NONE.value
                        if layout is None
                        else layout.adjustment.value
                    ),
                    layout_adjustment_reason=(
                        "" if layout is None else layout.adjustment_reason
                    ),
                    profile_id=None if provenance is None else provenance.profile_id,
                    semantic_candidate=(
                        is_semantic_candidate(value)
                        if assessment is None
                        else assessment.semantic_candidate
                    ),
                    cache_available=(
                        None if assessment is None else assessment.cache_available
                    ),
                    cache_availability=(
                        "" if assessment is None else assessment.cache_availability.value
                    ),
                    reuse_eligibility=(
                        "" if assessment is None else assessment.eligibility.value
                    ),
                    eligibility_reason=(
                        "" if assessment is None else assessment.reason.value
                    ),
                    state_origin="" if assessment is None else assessment.state_origin,
                    identity_basis=(
                        identity_basis(value).value
                        if assessment is None
                        else assessment.identity_basis.value
                    ),
                    provisional_action=(
                        policy_decision.action.value
                        if gate is None
                        else gate.provisional_decision.action.value
                    ),
                    eligibility_adjusted=(
                        False if gate is None else gate.eligibility_adjusted
                    ),
                    experimental=(
                        False
                        if assessment is None
                        else assessment.experimental
                        and decision.action == CompileAction.COMPILE
                    ),
                    preceding_context_tokens=preceding_here,
                )
            )
        return tuple(records)

    @property
    def selected_reusable_blocks(self) -> tuple[BlockPlanningRecord, ...]:
        """Blocks selection offered for reuse: selected, immutable, cacheable."""

        return tuple(
            record for record in self.planning_records if record.reusable_selection
        )

    def planning_invariant_violations(self) -> tuple[BlockPlanningRecord, ...]:
        """Reusable selected blocks that silently stopped being reusable.

        A block that selection offered for reuse must end up REUSE_CACHED or
        COMPILE_ON_MISS, unless the planner declined it for a reason on the
        allow-list. A KEEP_MUTABLE with any other reason means a block changed
        role without anyone saying why, which is the failure this record exists
        to make visible.
        """

        return tuple(
            record
            for record in self.selected_reusable_blocks
            if not record.linked and not record.declined_legitimately
        )

    @property
    def layout_summary(self) -> LayoutPlanningSummary:
        """Counts separating policy choices from layout-forced changes.

        Planning diagnostics, not a performance claim: nothing here is a
        latency ratio, so nothing here is governed by the cache-regime harness.
        """

        return LayoutPlanningSummary.from_decisions(self.layout_decisions, self.cached_cache_keys)

    @property
    def policy_decisions(self) -> tuple[CompileDecision, ...]:
        """What the compile policy asked for, before layout feasibility.

        Falls back to the effective decisions on a plan built without layout
        records, which is honest for those plans: nothing adjusted them.
        """

        if not self.layout_decisions:
            return self.decisions
        return tuple(value.policy_decision for value in self.layout_decisions)

    def layout_adjusted_decisions(self) -> tuple[LayoutDecision, ...]:
        """Blocks whose effective action differs from the policy's."""

        return tuple(
            value for value in self.layout_decisions if value.adjustment.is_adjustment
        )

    def positional_overrides(self) -> tuple[BlockPlanningRecord, ...]:
        """Blocks whose outcome the link-layout constraint decided.

        The compile policies are position-free by construction -- neither reads
        a block index, an insertion order, or a prompt offset. What remains is
        `_preserve_link_layout`, which is a physical feasibility constraint on
        the Runtime LinkPlan rather than a compile heuristic. Counting its
        effect here is the difference between a documented constraint and an
        undocumented one.
        """

        return tuple(
            record for record in self.planning_records if record.positional_override
        )

    @property
    def selected_order(self) -> tuple[str, ...]:
        return tuple(value.sub_context_id for value in self.included_contexts)

    @property
    def included_contexts(self) -> tuple[SemanticSubContext, ...]:
        decision_by_id = {
            decision.sub_context_id: decision for decision in self.decisions
        }
        return tuple(
            value
            for value in self.ordered_contexts
            if decision_by_id[value.sub_context_id].action
            != CompileAction.DO_NOT_INCLUDE
        )

    @property
    def expected_cache_keys(self) -> tuple[str, ...]:
        decision_by_id = {
            decision.sub_context_id: decision for decision in self.decisions
        }
        return tuple(
            value.cache_key
            for value in self.ordered_contexts
            if decision_by_id[value.sub_context_id].action == CompileAction.COMPILE
        )

    @property
    def reuse_cached_contexts(self) -> tuple[SemanticSubContext, ...]:
        cached = set(self.cached_cache_keys)
        return tuple(
            value for value in self.linked_contexts if value.cache_key in cached
        )

    @property
    def compile_on_miss_contexts(self) -> tuple[SemanticSubContext, ...]:
        cached = set(self.cached_cache_keys)
        return tuple(
            value for value in self.linked_contexts if value.cache_key not in cached
        )

    @property
    def keep_mutable_contexts(self) -> tuple[SemanticSubContext, ...]:
        return self.mutable_contexts

    @property
    def excluded_contexts(self) -> tuple[SemanticSubContext, ...]:
        return self.omitted_contexts

    @property
    def baseline_selected_text(self) -> str:
        from .source_text import compose_source_text

        return compose_source_text(
            self.included_contexts,
            separator=self.text_separator,
            trailing_text=self.manifest.mutable_text,
        )

    @property
    def linked_contexts(self) -> tuple[SemanticSubContext, ...]:
        decision_by_id = {
            decision.sub_context_id: decision for decision in self.decisions
        }
        return tuple(
            value
            for value in self.ordered_contexts
            if decision_by_id[value.sub_context_id].action == CompileAction.COMPILE
        )

    @property
    def mutable_contexts(self) -> tuple[SemanticSubContext, ...]:
        decision_by_id = {
            decision.sub_context_id: decision for decision in self.decisions
        }
        return tuple(
            value
            for value in self.ordered_contexts
            if decision_by_id[value.sub_context_id].action
            not in {CompileAction.COMPILE, CompileAction.DO_NOT_INCLUDE}
        )

    @property
    def omitted_contexts(self) -> tuple[SemanticSubContext, ...]:
        decision_by_id = {
            decision.sub_context_id: decision for decision in self.decisions
        }
        return tuple(
            value
            for value in self.ordered_contexts
            if decision_by_id[value.sub_context_id].action
            == CompileAction.DO_NOT_INCLUDE
        )


# Compatibility name for serialized artifacts and older callers.
ScheduledRequest = NOCRequestExecutionPlan


class NOCRequestPlanner:
    """Create a logical execution plan without batching or allocating KV slots."""

    def __init__(
        self,
        *,
        compile_policy: CompilePolicy | None = None,
        cache_catalog: CacheCatalog | None = None,
        require_model_tokens: bool = True,
        reuse_eligibility_policy: ReuseEligibilityPolicy | None = None,
        profitability_policy: ReuseProfitabilityPolicy | None = None,
        namespace_resolver: NamespaceResolver | None = None,
    ) -> None:
        """Plan reuse and compilation for one request.

        `require_model_tokens` defaults to on: a plan is what reaches the
        runtime, so the counts behind it must be in the unit the compile floor
        is expressed in. A caller running the pipeline in an estimate-only
        configuration -- a regex-tokenizer agent, an offline replay -- turns it
        off deliberately and gets decisions that are explicitly approximate.
        """

        self.require_model_tokens = require_model_tokens
        #: Namespace-first lookup (cache_namespace). `None` is the pre-namespace
        #: behaviour exactly: no resolution, no records, no extra decisions.
        self.namespace_resolver = namespace_resolver
        self.compile_policy = compile_policy or CompilePolicy()
        self.cache_catalog = cache_catalog or CacheCatalog()
        # Gate 1. Canonical by default: foreign contextual state is refused
        # unless a caller constructs the policy with the experiment flag.
        self.reuse_eligibility_policy = (
            reuse_eligibility_policy or ReuseEligibilityPolicy()
        )
        # Gate 3. Observe-only is the only mode that may run: no cost model
        # has been validated on formal measurements, and no threshold may
        # stand in for one. An enforcing policy is refused here.
        self.profitability_policy = require_observe_only(
            profitability_policy or ObserveOnlyProfitabilityPolicy()
        )

    def _stage5(self) -> CompileAndLinkFeasibilityPlanner:
        """The Stage 5 entrypoint over this planner's *current* collaborators.

        Built per call rather than cached, so a policy or catalog a caller
        replaced on the planner after construction is the one that decides.
        """

        return CompileAndLinkFeasibilityPlanner(
            compile_policy=self.compile_policy,
            cache_catalog=self.cache_catalog,
            layout_feasibility=self.layout_feasibility,
        )

    def _gate(
        self,
        ordered: tuple[SemanticSubContext, ...],
        decisions: tuple[CompileDecision, ...],
    ) -> tuple[EligibilityDecision, ...]:
        """Gate 1 over provisional decisions: decline Links the state does not permit."""

        return tuple(
            self.reuse_eligibility_policy.gate(
                decision,
                self.reuse_eligibility_policy.assess(
                    value, self.cache_catalog.metadata(value.cache_key)
                ),
            )
            for value, decision in zip(ordered, decisions, strict=True)
        )

    def _assemble(
        self,
        manifest: AgentContextManifest,
        ordered: tuple[SemanticSubContext, ...],
        provisional: tuple[CompileDecision, ...],
        *,
        persistence_permitted: tuple[bool, ...] | None = None,
    ) -> NOCRequestExecutionPlan:
        """Gate 1 -> Gate 2 -> effective decisions. The single path both entry points share.

        ``persistence_permitted`` is the namespace layer's answer, one per block, to "may the layout stage
        compile this block?". It is combined with Gate 1's answer, so a block in a non-persistent namespace
        is never promoted by prefix closure; the later compiled blocks are demoted instead.
        """

        eligibility = self._gate(ordered, provisional)
        stage5 = self._stage5()
        link_permitted = tuple(value.assessment.permits_link for value in eligibility)
        if persistence_permitted is not None:
            link_permitted = tuple(
                gate and allowed for gate, allowed in zip(link_permitted, persistence_permitted, strict=True)
            )
        layout_decisions = stage5.apply_feasibility(
            ordered,
            tuple(value.gated_decision for value in eligibility),
            link_permitted=link_permitted,
        )
        decisions = stage5.effective_decisions(layout_decisions)
        cached_cache_keys = tuple(
            value.cache_key
            for value, decision in zip(ordered, decisions, strict=True)
            if decision.action == CompileAction.COMPILE
            and self.cache_catalog.contains(value.cache_key)
        )
        plan = NOCRequestExecutionPlan(
            manifest=manifest,
            ordered_contexts=ordered,
            decisions=decisions,
            cached_cache_keys=cached_cache_keys,
            layout_decisions=layout_decisions,
            eligibility_decisions=eligibility,
        )
        # Gate 3 observes the plan Gate 2 produced and changes nothing: the
        # effective decisions above are final whatever the verdict says.
        verdict = self.profitability_policy.evaluate(plan.reuse_accounting())
        if verdict.enforced:  # pragma: no cover - refused at construction
            raise NotImplementedError("enforced profitability is not supported")
        return replace(plan, profitability=verdict)

    @staticmethod
    def _ordered_contexts(
        manifest: AgentContextManifest,
        contexts: Iterable[SemanticSubContext],
    ) -> tuple[SemanticSubContext, ...]:
        context_list = list(contexts)
        by_id = {value.sub_context_id: value for value in context_list}
        if len(by_id) != len(context_list):
            raise ValueError("contexts contain duplicate sub_context_id values")
        selected_ids = manifest.selected_sub_context_ids
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("manifest contains duplicate selected sub-context ids")
        missing = [value for value in selected_ids if value not in by_id]
        if missing:
            raise ValueError(f"manifest references unknown sub-context ids: {missing}")
        return tuple(by_id[value] for value in selected_ids)

    def schedule(
        self,
        manifest: AgentContextManifest,
        contexts: Iterable[SemanticSubContext],
        *,
        expected_future_reuse: int = 1,
        memory_pressure: float = 0.0,
    ) -> NOCRequestExecutionPlan:
        """Compatibility wrapper; new code should call :meth:`plan`."""

        return self.plan(
            manifest,
            contexts,
            expected_future_reuse=expected_future_reuse,
            memory_pressure=memory_pressure,
        )

    #: Layout feasibility is a planner concern, not a policy one, and it is a
    #: single shared object precisely so an A/B over compile policies cannot
    #: accidentally vary it. See `link_layout.LinkLayoutFeasibility`.
    layout_feasibility = LinkLayoutFeasibility()

    @classmethod
    def _preserve_link_layout(
        cls,
        ordered: tuple[SemanticSubContext, ...],
        decisions: tuple[CompileDecision, ...],
    ) -> tuple[CompileDecision, ...]:
        """Deprecated shim: the effective decisions only.

        Kept because callers and tests outside the planner referred to it.
        New code should use `layout_feasibility.apply`, which returns the
        policy decision and the adjustment alongside the effective one instead
        of discarding the first two.
        """

        return tuple(
            value.effective_decision
            for value in cls.layout_feasibility.apply(ordered, decisions)
        )

    def plan(
        self,
        manifest: AgentContextManifest,
        contexts: Iterable[SemanticSubContext],
        *,
        expected_future_reuse: int = 1,
        memory_pressure: float = 0.0,
        repair_model_tokens_by_block: Mapping[str, int] | None = None,
        repair_window_source: str = REPAIR_WINDOW_CALLER_SUPPLIED,
    ) -> NOCRequestExecutionPlan:
        """Plan reuse and compilation.

        `repair_model_tokens_by_block` is only consulted by a cost-benefit
        policy whose profile prices sparse repair separately from link; the
        legacy floor ignores it. Supplying it is how a position-derived repair
        window reaches the cost model -- see `repair_window.py`, which owns the
        sizing.
        """

        ordered = self._ordered_contexts(manifest, contexts)
        resolutions: tuple[NamespaceLookupResult, ...] = ()
        if self.namespace_resolver is not None:
            # Namespace-first lookup. A block served by another namespace's entry
            # is presented under that entry's cache key and semantic identity;
            # everything after this point plans over the presented blocks.
            ordered, resolutions = self.namespace_resolver.apply(
                ordered, self.cache_catalog
            )
        raw_decisions = self._stage5().provisional_decisions(
            ordered,
            expected_future_reuse=expected_future_reuse,
            memory_pressure=memory_pressure,
            require_model_tokens=self.require_model_tokens,
            repair_model_tokens_by_block=repair_model_tokens_by_block,
            repair_window_source=repair_window_source,
        )
        if self.namespace_resolver is not None:
            raw_decisions = self.namespace_resolver.enforce_persistence(
                resolutions, raw_decisions
            )
        persistence_permitted = (
            self.namespace_resolver.layout_permissions(resolutions)
            if self.namespace_resolver is not None
            else None
        )
        plan = self._assemble(
            manifest, ordered, raw_decisions, persistence_permitted=persistence_permitted
        )
        return replace(plan, namespace_resolutions=resolutions) if resolutions else plan

    def bind_decisions(
        self,
        manifest: AgentContextManifest,
        contexts: Iterable[SemanticSubContext],
        decisions: Iterable[CompileDecision],
    ) -> NOCRequestExecutionPlan:
        ordered = self._ordered_contexts(manifest, contexts)
        decision_list = tuple(decisions)
        decision_by_id = {
            decision.sub_context_id: decision for decision in decision_list
        }
        if len(decision_by_id) != len(decision_list):
            raise ValueError("compile decisions contain duplicate sub_context_id values")
        expected_ids = {value.sub_context_id for value in ordered}
        actual_ids = set(decision_by_id)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            raise ValueError(
                "compile decisions do not match selected contexts: "
                f"missing={missing}, unexpected={unexpected}"
            )
        # Supplied decisions are provisional too: Gate 1 runs on them exactly as
        # on a policy's, so a caller cannot bind a Link to an ineligible state.
        return self._assemble(
            manifest,
            ordered,
            tuple(decision_by_id[value.sub_context_id] for value in ordered),
        )


class NOCRequestScheduler(NOCRequestPlanner):
    """Deprecated compatibility name; use :class:`NOCRequestPlanner`."""
