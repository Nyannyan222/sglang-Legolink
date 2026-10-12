"""Layout feasibility: what the runtime can represent, kept apart from policy.

Two different questions were being answered by one field.

**Is compiling this block worthwhile?** is an economic question about the block
itself, and `compile_policy.py` answers it -- from the block's token count, its
reuse expectation, and measured latency. It reads no index and no neighbour.

**Can this set of decisions be expressed as a LinkPlan?** is a question about
the *shape of the request*, and it has nothing to do with whether any
individual block is worth compiling. `PICRuntime` assembles::

    full_input_ids = prefix_input_ids
                   + block_1 + block_2 + ... + block_N     (cache_keys order)
                   + mutable_suffix_input_ids

    mutable_start_token = len(full_input_ids) after the last block

`mutable_start_token` is one scalar. There is exactly one mutable region and it
is the suffix, and `ContextStitcher` correspondingly joins every kept-mutable
block plus the manifest's mutable text into a single `mutable_text` string. So

    COMPILE A / KEEP_MUTABLE B / COMPILE C

has no representation: B's text can only be placed after C, which silently
reorders what the selector chose. The planner therefore adjusts such a set until
it is representable, and those adjustments are *not* policy.

Until now the adjustment overwrote the decision. A sub-floor block promoted for
prefix closure was recorded as a compile, indistinguishable from a block the
policy wanted; that is how 23 forced compiles came to look like policy choices,
and how the same 41-token block was `KEEP_MUTABLE` at pilot turn 4 and
`REUSE_CACHED` at turn 7 with nothing in the artifact saying why. The upcoming
A/B has to be able to ask "did cost-benefit pick better blocks?" separately from
"did the LinkPlan shape let it?", and a single overwritten field cannot answer
either.

So three things are recorded per block, and the economics are never recomputed:

    policy_action      what the policy decided, with its full cost breakdown
    layout_adjustment  NONE, or the closure that forced a change, with a reason
    effective_action   what actually reaches the runtime

`effective_action` may differ from `policy_action` only through an explicit
`LayoutAdjustment`. A forced compile keeps its negative `net_benefit_ms`,
because the point of recording it is that it was not worth doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from .compile_policy import CompileAction, CompileDecision
from .subcontext_store import SemanticSubContext


LINK_LAYOUT_POLICY_VERSION = "link-layout-v1"


class LayoutAdjustment(str, Enum):
    """Why the effective action differs from what the policy asked for.

    Both non-`NONE` values are closures of the single immutable-prefix /
    single-mutable-suffix shape, not preferences. Neither exists because a
    block was judged differently; they exist because the alternative request
    could not be built.
    """

    NONE = "none"
    #: A reusable immutable block the policy declined, with a compiled block
    #: after it. Left mutable it would be joined into the mutable suffix and
    #: emerge physically *after* that later block, reordering the manifest.
    FORCE_COMPILE_FOR_PREFIX_CLOSURE = "force_compile_for_prefix_closure"
    #: A block the policy wanted compiled, after mutable content has already
    #: begun. There is one `mutable_start_token`, so nothing can be placed
    #: before it once the suffix has opened.
    FORCE_MUTABLE_FOR_SUFFIX_CLOSURE = "force_mutable_for_suffix_closure"

    @property
    def is_adjustment(self) -> bool:
        return self is not LayoutAdjustment.NONE


#: Human-readable reasons. These strings are the ones already recorded in
#: artifacts and on `LEGITIMATE_KEEP_MUTABLE_REASONS`, kept byte-for-byte so
#: runs from before this split still audit against the same vocabulary.
POSITIONAL_LAYOUT_PROMOTION = (
    "promoted to compile to preserve selected order before later reusable blocks"
)
POSITIONAL_LAYOUT_DEMOTION = (
    "kept mutable to preserve manifest order in the single LegoLink mutable suffix"
)

ADJUSTMENT_REASONS = {
    LayoutAdjustment.FORCE_COMPILE_FOR_PREFIX_CLOSURE: POSITIONAL_LAYOUT_PROMOTION,
    LayoutAdjustment.FORCE_MUTABLE_FOR_SUFFIX_CLOSURE: POSITIONAL_LAYOUT_DEMOTION,
}

POSITIONAL_LAYOUT_REASONS = frozenset(ADJUSTMENT_REASONS.values())


@dataclass(frozen=True)
class LayoutDecision:
    """One block's policy decision, the adjustment, and what will execute.

    Frozen, and it holds the original `CompileDecision` object rather than a
    copy: the policy's action, reason, and cost breakdown are carried through
    untouched, so nothing downstream can quietly restate a forced compile as a
    chosen one.
    """

    policy_decision: CompileDecision
    effective_decision: CompileDecision
    adjustment: LayoutAdjustment = LayoutAdjustment.NONE
    model_tokens: int = 0

    @property
    def sub_context_id(self) -> str:
        return self.policy_decision.sub_context_id

    @property
    def cache_key(self) -> str:
        return self.policy_decision.cache_key

    @property
    def policy_action(self) -> CompileAction:
        return self.policy_decision.action

    @property
    def effective_action(self) -> CompileAction:
        return self.effective_decision.action

    @property
    def adjustment_reason(self) -> str:
        return ADJUSTMENT_REASONS.get(self.adjustment, "")

    @property
    def net_benefit_ms(self) -> float | None:
        """The policy's arithmetic, never the layout's.

        `None` for the legacy floor, which computes no milliseconds at all.
        """

        provenance = self.policy_decision.provenance
        cost = None if provenance is None else provenance.cost
        return None if cost is None else cost.net_benefit_ms

    @property
    def forced_negative_benefit_compile(self) -> bool:
        """Compiled by layout although the policy priced it as a loss.

        The category the upcoming experiment most needs counted: these are
        compiles the cost-benefit policy is not responsible for and must not be
        credited or blamed for.
        """

        benefit = self.net_benefit_ms
        return (
            self.adjustment is LayoutAdjustment.FORCE_COMPILE_FOR_PREFIX_CLOSURE
            and benefit is not None
            and benefit <= 0.0
        )

    @property
    def suppressed_positive_benefit_compile(self) -> bool:
        """A profitable compile the LinkPlan shape would not let us have."""

        benefit = self.net_benefit_ms
        return (
            self.adjustment is LayoutAdjustment.FORCE_MUTABLE_FOR_SUFFIX_CLOSURE
            and benefit is not None
            and benefit > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        provenance = self.policy_decision.provenance
        cost = None if provenance is None else provenance.cost
        return {
            "sub_context_id": self.sub_context_id,
            "cache_key": self.cache_key,
            "policy_action": self.policy_action.value,
            "policy_reason": self.policy_decision.reason,
            "policy_net_benefit_ms": None if cost is None else cost.net_benefit_ms,
            "policy_expected_saved_ms": (
                None if cost is None else cost.expected_saved_ms
            ),
            "policy_total_cost_ms": None if cost is None else cost.total_cost_ms,
            "layout_adjustment": self.adjustment.value,
            "layout_adjustment_reason": self.adjustment_reason,
            "effective_action": self.effective_action.value,
            "effective_reason": self.effective_decision.reason,
            "policy_name": (
                None if provenance is None else provenance.compile_policy
            ),
            "profile_id": None if provenance is None else provenance.profile_id,
            "model_token_count": int(self.model_tokens),
        }


@dataclass(frozen=True)
class LayoutPlanningSummary:
    """Per-request counts, separating what policy chose from what shape forced.

    Planning diagnostics. None of these is a performance claim, and none of them
    is gated by the cache-regime harness, because none of them is a latency
    ratio -- they describe the plan, not the run.
    """

    policy_compile_count: int = 0
    policy_decline_count: int = 0
    layout_promotion_count: int = 0
    layout_demotion_count: int = 0
    effective_compile_count: int = 0
    total_block_decisions: int = 0
    requested_reuse_count: int = 0
    non_prefix_reuse_opportunities: int = 0
    forced_negative_benefit_compile_count: int = 0
    suppressed_positive_benefit_compile_count: int = 0

    policy_compile_model_tokens: int = 0
    policy_decline_model_tokens: int = 0
    layout_promotion_model_tokens: int = 0
    layout_demotion_model_tokens: int = 0
    effective_compile_model_tokens: int = 0

    @classmethod
    def from_decisions(
        cls, decisions: Sequence[LayoutDecision], cached_cache_keys: Sequence[str] = ()
    ) -> "LayoutPlanningSummary":
        def total(predicate) -> tuple[int, int]:
            chosen = [value for value in decisions if predicate(value)]
            return len(chosen), sum(int(value.model_tokens) for value in chosen)

        policy_compiles, policy_compile_tokens = total(
            lambda value: value.policy_action is CompileAction.COMPILE
        )
        # A decline is a reusable block the policy chose not to compile. A
        # dropped or genuinely mutable block was never on the table.
        declines, decline_tokens = total(
            lambda value: value.policy_action is CompileAction.KEEP_MUTABLE
        )
        promotions, promotion_tokens = total(
            lambda value: value.adjustment
            is LayoutAdjustment.FORCE_COMPILE_FOR_PREFIX_CLOSURE
        )
        demotions, demotion_tokens = total(
            lambda value: value.adjustment
            is LayoutAdjustment.FORCE_MUTABLE_FOR_SUFFIX_CLOSURE
        )
        effective, effective_tokens = total(
            lambda value: value.effective_action is CompileAction.COMPILE
        )
        forced_bad, _ = total(lambda value: value.forced_negative_benefit_compile)
        suppressed_good, _ = total(
            lambda value: value.suppressed_positive_benefit_compile
        )
        return cls(
            total_block_decisions=len(decisions),
            requested_reuse_count=sum(value.policy_action is CompileAction.COMPILE and value.cache_key in cached_cache_keys for value in decisions),
            non_prefix_reuse_opportunities=sum(
                value.policy_action is CompileAction.COMPILE and value.cache_key in cached_cache_keys
                and any(previous.cache_key not in cached_cache_keys for previous in decisions[:index])
                for index, value in enumerate(decisions)
            ),
            policy_compile_count=policy_compiles,
            policy_decline_count=declines,
            layout_promotion_count=promotions,
            layout_demotion_count=demotions,
            effective_compile_count=effective,
            forced_negative_benefit_compile_count=forced_bad,
            suppressed_positive_benefit_compile_count=suppressed_good,
            policy_compile_model_tokens=policy_compile_tokens,
            policy_decline_model_tokens=decline_tokens,
            layout_promotion_model_tokens=promotion_tokens,
            layout_demotion_model_tokens=demotion_tokens,
            effective_compile_model_tokens=effective_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_layout_version": LINK_LAYOUT_POLICY_VERSION,
            "total_block_decisions": self.total_block_decisions,
            "requested_reuse_count": self.requested_reuse_count,
            "non_prefix_reuse_opportunities": self.non_prefix_reuse_opportunities,
            "layout_adjustment_count": self.layout_promotion_count + self.layout_demotion_count,
            "layout_adjustment_rate": (self.layout_promotion_count + self.layout_demotion_count) / self.total_block_decisions if self.total_block_decisions else 0.0,
            "executable_linked_topology": "optional_prefix_immutable_blocks_mutable_suffix_v1",
            "policy_compile_count": self.policy_compile_count,
            "policy_decline_count": self.policy_decline_count,
            "layout_promotion_count": self.layout_promotion_count,
            "layout_demotion_count": self.layout_demotion_count,
            "effective_compile_count": self.effective_compile_count,
            "forced_negative_benefit_compile_count": (
                self.forced_negative_benefit_compile_count
            ),
            "suppressed_positive_benefit_compile_count": (
                self.suppressed_positive_benefit_compile_count
            ),
            "policy_compile_model_tokens": self.policy_compile_model_tokens,
            "policy_decline_model_tokens": self.policy_decline_model_tokens,
            "layout_promotion_model_tokens": self.layout_promotion_model_tokens,
            "layout_demotion_model_tokens": self.layout_demotion_model_tokens,
            "effective_compile_model_tokens": self.effective_compile_model_tokens,
        }


class LinkLayoutFeasibility:
    """Adjust a set of policy decisions until the runtime can represent it.

    Stateless and policy-agnostic by construction: `apply` receives decisions
    and never learns which policy produced them. That is what lets the upcoming
    A/B vary the policy while holding planner behaviour fixed -- otherwise the
    comparison would confound the compile policy with the planner.

    The algorithm is unchanged from `NOCRequestPlanner._preserve_link_layout`;
    what changed is that it now returns `LayoutDecision` records instead of
    overwriting the policy's decision with its own.
    """

    version = LINK_LAYOUT_POLICY_VERSION

    def apply(
        self,
        ordered: Sequence[SemanticSubContext],
        decisions: Sequence[CompileDecision],
        *,
        link_permitted: Sequence[bool] | None = None,
    ) -> tuple[LayoutDecision, ...]:
        """Adjust `decisions` until the LinkPlan shape can represent them.

        `link_permitted` is Gate 1's answer, one per block: may this block's state be linked at all?
        Feasibility never asks *why*; it only refuses to promote a block Gate 1 declined, so a prefix
        closure can never turn an ineligible state back into a Link. The later linked blocks are demoted
        instead, which is the physically honest consequence. `None` keeps the pre-Gate-1 behaviour for
        callers that have no eligibility (every block promotable).
        """

        if len(ordered) != len(decisions):
            raise ValueError("layout feasibility needs one decision per block")
        if link_permitted is None:
            link_permitted = [True] * len(ordered)
        elif len(link_permitted) != len(ordered):
            raise ValueError("layout feasibility needs one link permission per block")

        mutable_suffix_started = False
        results: list[LayoutDecision] = []
        for index, (value, decision) in enumerate(zip(ordered, decisions)):
            tokens = int(value.token_count)
            if decision.action == CompileAction.DO_NOT_INCLUDE:
                # An excluded block occupies no physical range at all, so it
                # neither opens the mutable suffix nor needs closing over.
                results.append(
                    LayoutDecision(decision, decision, model_tokens=tokens)
                )
                continue

            if decision.action != CompileAction.COMPILE:
                later_compile = any(
                    candidate.action == CompileAction.COMPILE
                    for candidate in decisions[index + 1 :]
                )
                if (
                    later_compile
                    and decision.action == CompileAction.KEEP_MUTABLE
                    and value.reuse_candidate
                    and not value.is_mutable
                    and link_permitted[index]
                ):
                    results.append(
                        self._adjust(
                            decision,
                            LayoutAdjustment.FORCE_COMPILE_FOR_PREFIX_CLOSURE,
                            CompileAction.COMPILE,
                            saved_tokens=tokens,
                            model_tokens=tokens,
                        )
                    )
                    continue
                mutable_suffix_started = True
                results.append(
                    LayoutDecision(decision, decision, model_tokens=tokens)
                )
                continue

            if mutable_suffix_started:
                results.append(
                    self._adjust(
                        decision,
                        LayoutAdjustment.FORCE_MUTABLE_FOR_SUFFIX_CLOSURE,
                        CompileAction.KEEP_MUTABLE,
                        saved_tokens=0,
                        model_tokens=tokens,
                    )
                )
                continue

            results.append(LayoutDecision(decision, decision, model_tokens=tokens))
        return tuple(results)

    @staticmethod
    def _adjust(
        policy_decision: CompileDecision,
        adjustment: LayoutAdjustment,
        effective_action: CompileAction,
        *,
        saved_tokens: int,
        model_tokens: int,
    ) -> LayoutDecision:
        """Build the effective decision without touching the policy's.

        The provenance object is carried across unchanged -- the same profile
        id, the same milliseconds, the same sign on `net_benefit_ms`. Layout
        feasibility has no opinion about economics and must not appear to
        have recalculated any.
        """

        effective = CompileDecision(
            sub_context_id=policy_decision.sub_context_id,
            cache_key=policy_decision.cache_key,
            action=effective_action,
            reason=ADJUSTMENT_REASONS[adjustment],
            estimated_saved_prefill_tokens=saved_tokens,
            provenance=policy_decision.provenance,
        )
        return LayoutDecision(
            policy_decision=policy_decision,
            effective_decision=effective,
            adjustment=adjustment,
            model_tokens=model_tokens,
        )


__all__ = [
    "ADJUSTMENT_REASONS",
    "LINK_LAYOUT_POLICY_VERSION",
    "POSITIONAL_LAYOUT_DEMOTION",
    "POSITIONAL_LAYOUT_PROMOTION",
    "POSITIONAL_LAYOUT_REASONS",
    "LayoutAdjustment",
    "LayoutDecision",
    "LayoutPlanningSummary",
    "LinkLayoutFeasibility",
]
