"""Client-side compile policy and local compiled-cache catalog.

Two policies live here, and the difference between them is the point.

`CompilePolicy` is the historical one: compile a stable reusable block iff it
is at least `min_compile_model_tokens` -- 64 -- model tokens long. It is kept,
unchanged and under regression test, because the next experiments have to be
able to ask whether the replacement is actually better, and that question needs
the old behaviour to still exist and to still mean exactly what it meant.

`CostBenefitCompilePolicy` replaces the constant with the trade it was standing
in for::

    compile once  +  N x link        versus        N x native prefill

The terms come from `compile_cost`: a measured latency profile carrying the
hardware and model it was measured on, an explicit reuse expectation, and a
residency cost model. The algorithm here is hardware-independent; every number
it reads is not, and arrives through the profile with provenance attached.

ARCHITECTURE_STAGE: 5 -- Compile Policy + Link Feasibility
CANONICAL_ENTRYPOINT: CompileAndLinkFeasibilityPlanner (provisional decisions ->
    feasibility -> effective decisions), defined near the bottom of this module.
DELEGATED_HELPERS: CompilePolicy / CostBenefitCompilePolicy / CacheCatalog (here),
    link_layout.py (LinkLayoutFeasibility, imported lazily: it imports this
    module), compile_cost.py (latency profile and economics).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .semantic_cache_index import SemanticCacheIndex
from .compile_cost import (
    COMPILE_COST_POLICY_VERSION,
    PROFILE_UNCHECKED,
    REPAIR_WINDOW_CALLER_SUPPLIED,
    REPAIR_WINDOW_UNSPECIFIED,
    CompileCandidate,
    CompileCostBreakdown,
    CompileCostError,
    CompileCostProfile,
    CompileCostUnavailableError,
    DeploymentIdentity,
    DisabledResidencyCost,
    ResidencyCostModel,
    ReuseExpectation,
    evaluate_compile_economics,
)
from .subcontext_store import SemanticSubContext
from .token_accounting import (
    ModelTokenCount,
    TokenAccountingError,
    require_model_tokens,
)


#: The block is smaller than `min_compile_tokens`, so a compile round trip
#: buys less prefill than it costs. Measured cause of every unlinked
#: non-prefix block in the budget-2000 runs: blocks of 12, 41, 44, 45 and 48
#: tokens against a floor of 64.
BELOW_COMPILE_FLOOR = "block is smaller than the compile floor"

#: Caller stated the block will not be needed again.
NO_EXPECTED_REUSE = "no expected future reuse"

#: What the two conditions above were reported as before they were split.
#: Kept so artifacts recorded before 2026-09-03 still audit.
LEGACY_COMPILE_COST_REASON = "compile cost exceeds expected reuse"

#: The cost-benefit policy's two verdicts. Unlike the floor, these name a
#: comparison that was actually evaluated, and the milliseconds behind them
#: travel on the decision.
NET_BENEFIT_NEGATIVE = (
    "expected reuse savings do not cover compile, link and residency cost"
)
NET_BENEFIT_POSITIVE = (
    "expected reuse savings exceed compile, link and residency cost"
)


class CompilePolicyMode(str, Enum):
    """Which compile policy an experiment arm is running.

    Both are first-class. The legacy mode is not deprecated-and-tolerated; it
    is the baseline the cost-benefit policy has to beat, so it keeps a name, a
    version, and its own regression tests.
    """

    LEGACY_FIXED_THRESHOLD = "legacy_fixed_64"
    COST_BENEFIT = "cost_benefit"


LEGACY_COMPILE_POLICY_VERSION = "legacy-fixed-64-v1"


class CompileAction(str, Enum):
    COMPILE = "compile"
    KEEP_MUTABLE = "keep_mutable"
    DO_NOT_INCLUDE = "do_not_include"
    SUMMARIZE_THEN_COMPILE = "summarize_then_compile"


@dataclass(frozen=True)
class CompileDecisionProvenance:
    """Why a compile decision came out the way it did.

    A decision used to be an action and a sentence. That was enough while the
    rule was a constant, because "below the compile floor" fully describes
    `token_count < 64`. It is not enough for an economic decision: the same
    block compiles or does not depending on four measured latencies, a reuse
    expectation and which GPU the profile came from, and none of that is
    recoverable from the word "compile" months later.

    The profile is referenced by id, not embedded. Serializing a whole latency
    profile onto every block would bloat every artifact with the same payload
    repeated hundreds of times.
    """

    compile_policy: str
    compile_policy_version: str
    profile_id: str | None = None
    profile_hardware: str | None = None
    profile_model: str | None = None
    profile_compatibility: str | None = None
    #: Set when a cost-benefit policy could not produce a decision and the
    #: caller had explicitly authorised falling back, so an artifact never has
    #: to guess whether a run's decisions were economic or thresholded.
    fallback_from: str | None = None
    fallback_reason: str | None = None
    cost: CompileCostBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "compile_policy": self.compile_policy,
            "compile_policy_version": self.compile_policy_version,
            "profile_id": self.profile_id,
            "profile_hardware": self.profile_hardware,
            "profile_model": self.profile_model,
            "profile_compatibility": self.profile_compatibility,
            "fallback_from": self.fallback_from,
            "fallback_reason": self.fallback_reason,
        }
        if self.cost is not None:
            payload.update(self.cost.to_dict())
        return payload


@dataclass(frozen=True)
class CompileDecision:
    sub_context_id: str
    cache_key: str
    action: CompileAction
    reason: str
    estimated_saved_prefill_tokens: int = 0
    #: Attached by `CompilePolicy.decide` for every policy, so no decision
    #: reaches an artifact without saying which policy made it.
    provenance: CompileDecisionProvenance | None = None


@dataclass(frozen=True)
class CompilePolicyContext:
    """Per-request and per-block inputs to the compile decision.

    Three kinds of input, kept apart on purpose:

    * **Correctness** -- the block's own immutability, reuse eligibility and
      authoritative `ModelTokenCount`. These come from the block, not from here,
      and never depend on hardware. `require_model_tokens` is the fail-closed
      switch over them, and `NOCRequestPlanner` turns it on for every plan it
      builds: a plan is what reaches the runtime, so its counts must be in the
      unit the policy is expressed in. It is off on a bare
      `CompilePolicyContext()` because `decide` is also called directly to
      inspect the shape of a decision, and that is not a runtime input.
    * **Workload expectation** -- `expected_future_reuse` is what the caller
      declares about this request. Note that it is a *declaration*, not an
      observation: the agent passes `max(1, turn)` and the loop passes
      `max_rounds - index + 1`. The cost-benefit policy therefore runs it
      through a `ReuseExpectation` that can discount or ignore it, rather than
      treating it as a count.
    * **Cost shape** -- `planned_repair_model_tokens` and
      `preceding_model_tokens`. These are the only position-derived inputs, and
      they reach the decision only through the repair cost term.

    `min_compile_model_tokens` is 64 *model* tokens, and belongs to the legacy
    policy alone. The number and the branch are unchanged; measured
    regex-to-BPE ratios on this workload range from 1.19 to 7.63, so before the
    unit was named the floor was somewhere between 8 and 54 model tokens
    depending on which path built the block.
    """

    expected_future_reuse: int = 1
    memory_pressure: float = 0.0
    cache_present: bool = False
    min_compile_model_tokens: int = 64
    summarize_threshold_tokens: int = 8192
    require_model_tokens: bool = False
    #: Recompute window this block is planned to run, in model tokens. Needed
    #: only when the cost profile prices sparse repair separately from link.
    planned_repair_model_tokens: int | None = None
    #: Context preceding this block in the link, in model tokens. A compiled
    #: block is prefilled standalone from local position 0, so this is how much
    #: context its leading tokens never attended to. It is carried as
    #: provenance for a caller that sizes a repair window from it; it prices
    #: nothing on its own, and it is never a threshold.
    preceding_model_tokens: int = 0
    #: Where `planned_repair_model_tokens` came from. Unvalidated by default,
    #: because the position -> window derivation is not yet calibrated.
    repair_window_source: str = REPAIR_WINDOW_UNSPECIFIED

    @property
    def min_compile_tokens(self) -> int:
        """Compatibility alias for the pre-unit name."""

        return self.min_compile_model_tokens


class CompilePolicy:
    """The legacy fixed 64-model-token floor.

    Kept verbatim as `CompilePolicyMode.LEGACY_FIXED_THRESHOLD`. Experiments
    comparing the cost-benefit policy against it need it to keep meaning
    exactly `>= 64 model tokens`, so its branches are not to be tuned, and
    `tests/test_compile_cost_benefit_policy.py` pins its behaviour at 63, 64
    and 65.
    """

    policy_name = CompilePolicyMode.LEGACY_FIXED_THRESHOLD.value
    policy_version = LEGACY_COMPILE_POLICY_VERSION

    def decide(
        self, value: SemanticSubContext, context: CompilePolicyContext | None = None
    ) -> CompileDecision:
        """The unit boundary. Subclasses override `_decide`, never this.

        `CompileAllImmutablePolicy` used to override `decide` outright, so the
        ablation policy the agent selects with
        `compile_policy_mode="compile_all_immutable"` skipped the guard
        entirely and could turn an estimated block into an executable compile
        decision. Making this a template method means a policy cannot opt out
        of the boundary by existing.

        It is also where provenance is stamped, for the same reason: a policy
        cannot ship an unattributed decision by forgetting to fill one in.
        """

        context = context or CompilePolicyContext()
        if context.require_model_tokens:
            # Fail closed at the decision boundary rather than at every
            # construction site: a manager may honestly produce estimates for
            # an offline utility, but nothing may compare one against a
            # model-token floor and call the result a compile decision.
            require_model_tokens(
                value.token_accounting(),
                purpose=(
                    f"{type(self).__name__}.decide({value.sub_context_id!r}) "
                    f"against a {context.min_compile_model_tokens}-model-token "
                    "floor"
                ),
            )
        decision = self._decide(value, context)
        if decision.provenance is None:
            return replace(decision, provenance=self.default_provenance())
        return decision

    def default_provenance(self) -> CompileDecisionProvenance:
        return CompileDecisionProvenance(
            compile_policy=self.policy_name,
            compile_policy_version=self.policy_version,
        )

    def _decide(
        self, value: SemanticSubContext, context: CompilePolicyContext
    ) -> CompileDecision:
        if value.invalidated:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.DO_NOT_INCLUDE,
                "invalidated version",
            )
        if value.is_mutable:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                "content is mutable",
            )
        if not value.reuse_candidate:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                "not a reuse candidate",
            )
        if context.cache_present:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.COMPILE,
                "already compiled; reuse catalog entry",
                value.token_count,
            )
        # Split from one branch reading "compile cost exceeds expected reuse",
        # which described a cost model this code never evaluated: the test is a
        # fixed token floor, and `expected_future_reuse` is only ever checked
        # for being absent. Two conditions, two reasons, because the reason is
        # the whole audit trail once a block is left out of the link.
        if value.token_count < context.min_compile_model_tokens:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                BELOW_COMPILE_FLOOR,
            )
        if context.expected_future_reuse < 1:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                NO_EXPECTED_REUSE,
            )
        if context.memory_pressure >= 0.9:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                "high KV memory pressure",
            )
        if (
            value.token_count >= context.summarize_threshold_tokens
            and value.context_field not in {"workspace_evidence", "tool_definitions"}
        ):
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.SUMMARIZE_THEN_COMPILE,
                "large compressible stable context",
            )
        return CompileDecision(
            value.sub_context_id,
            value.cache_key,
            CompileAction.COMPILE,
            "stable reusable block has positive expected prefill savings",
            value.token_count * context.expected_future_reuse,
        )


class CompileAllImmutablePolicy(CompilePolicy):
    """Ablation policy that compiles every selected immutable block."""

    policy_name = "compile_all_immutable"
    policy_version = "compile-all-immutable-v1"

    def _decide(
        self,
        value: SemanticSubContext,
        context: CompilePolicyContext,
    ) -> CompileDecision:
        if value.invalidated:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.DO_NOT_INCLUDE,
                "invalidated version",
            )
        if value.is_mutable:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                "content is mutable",
            )
        return CompileDecision(
            value.sub_context_id,
            value.cache_key,
            CompileAction.COMPILE,
            "compile-all immutable ablation",
            value.token_count * max(context.expected_future_reuse, 1),
        )


class CostBenefitCompilePolicy(CompilePolicy):
    """Compile a block iff its expected reuse savings exceed its cost.

    Structurally this is the same policy object as the legacy one -- same
    correctness pre-checks, same template method, same model-token boundary --
    with the single `token_count < 64` branch replaced by arithmetic over
    measured latencies. Everything before that branch is a correctness or
    eligibility question and stays exactly where it was.

    What it needs, and refuses to proceed without:

    * a `CompileCostProfile` that covers the block's length and matches the
      deployment, and
    * a `ReuseExpectation`, which is a required argument rather than a default,
      because an implicit "assume it will be reused once" is precisely the
      hidden assumption this change exists to remove.

    Without them there is no decision. `fallback_policy` is the only way past
    that, it has to be named by the caller, and it is recorded on every
    decision it produces.
    """

    policy_name = CompilePolicyMode.COST_BENEFIT.value
    policy_version = COMPILE_COST_POLICY_VERSION

    #: Above this, KV pressure is treated as a hard stop rather than an
    #: economic term. This is a v1 safety rail carried over from the legacy
    #: policy, not a measured residency cost: the residency term exists
    #: (`residency_cost_model`) and is disabled pending measurement, so
    #: removing the rail as well would leave pressure priced at nothing.
    memory_pressure_stop = 0.9

    def __init__(
        self,
        *,
        reuse_expectation: ReuseExpectation,
        profile: CompileCostProfile | None = None,
        residency_cost_model: ResidencyCostModel | None = None,
        deployment: DeploymentIdentity | None = None,
        strict_profile_compatibility: bool = True,
        fallback_policy: CompilePolicyMode | None = None,
    ) -> None:
        if reuse_expectation is None:
            raise CompileCostError(
                "a cost-benefit compile policy must be given a ReuseExpectation; "
                "there is no safe default, and assuming one reuse is the "
                "assumption this policy replaces"
            )
        if fallback_policy is not None and fallback_policy is not (
            CompilePolicyMode.LEGACY_FIXED_THRESHOLD
        ):
            raise CompileCostError(
                "the only supported cost-benefit fallback is "
                "CompilePolicyMode.LEGACY_FIXED_THRESHOLD"
            )
        self.reuse_expectation = reuse_expectation
        self.profile = profile
        self.residency_cost_model = residency_cost_model or DisabledResidencyCost()
        self.deployment = deployment
        self.strict_profile_compatibility = strict_profile_compatibility
        self.fallback_policy = fallback_policy
        self._legacy = CompilePolicy()

    # -- profile resolution --------------------------------------------------

    def _require_profile(self) -> tuple[CompileCostProfile, str]:
        if self.profile is None:
            raise CompileCostUnavailableError(
                "cost-benefit compile decisions require a CompileCostProfile "
                "and none was supplied. Build one with "
                "scripts/build_compile_cost_profile.py, or ask for "
                "CompilePolicyMode.LEGACY_FIXED_THRESHOLD by name."
            )
        compatibility = self.profile.check_compatibility(
            self.deployment, strict=self.strict_profile_compatibility
        )
        return self.profile, compatibility

    def _profile_provenance(
        self,
        profile: CompileCostProfile | None,
        compatibility: str,
        cost: CompileCostBreakdown | None = None,
    ) -> CompileDecisionProvenance:
        return CompileDecisionProvenance(
            compile_policy=self.policy_name,
            compile_policy_version=self.policy_version,
            profile_id=None if profile is None else profile.profile_id,
            profile_hardware=None if profile is None else profile.provenance.hardware_id,
            profile_model=None if profile is None else profile.provenance.model_id,
            profile_compatibility=compatibility,
            cost=cost,
        )

    def _fallback(
        self,
        value: SemanticSubContext,
        context: CompilePolicyContext,
        error: CompileCostUnavailableError,
    ) -> CompileDecision:
        """Delegate to the legacy floor, and say so on the decision.

        Only reachable when the caller passed `fallback_policy` explicitly.
        The alternative -- quietly falling back whenever a profile is missing --
        would make an artifact's decisions unattributable, because a run with a
        broken profile would look exactly like a run with a good one.
        """

        decision = self._legacy._decide(value, context)
        return replace(
            decision,
            provenance=CompileDecisionProvenance(
                compile_policy=self._legacy.policy_name,
                compile_policy_version=self._legacy.policy_version,
                fallback_from=self.policy_name,
                fallback_reason=str(error),
            ),
        )

    # -- the decision --------------------------------------------------------

    def _decide(
        self, value: SemanticSubContext, context: CompilePolicyContext
    ) -> CompileDecision:
        # Correctness and eligibility first, and hardware-independent. A
        # mutable or invalidated block is not an economic question, and must
        # not fail for want of a latency profile.
        if value.invalidated:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.DO_NOT_INCLUDE,
                "invalidated version",
            )
        if value.is_mutable:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                "content is mutable",
            )
        if not value.reuse_candidate:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                "not a reuse candidate",
            )
        if context.memory_pressure >= self.memory_pressure_stop:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                "high KV memory pressure",
            )
        if (
            value.token_count >= context.summarize_threshold_tokens
            and value.context_field not in {"workspace_evidence", "tool_definitions"}
        ):
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.SUMMARIZE_THEN_COMPILE,
                "large compressible stable context",
            )

        try:
            candidate = self.build_candidate(value, context)
            profile, compatibility = self._require_profile()
            cost = evaluate_compile_economics(
                candidate,
                profile=profile,
                reuse_expectation=self.reuse_expectation,
                residency_cost_model=self.residency_cost_model,
                planned_future_reuse=context.expected_future_reuse,
                memory_pressure=context.memory_pressure,
            )
        except CompileCostUnavailableError as error:
            if self.fallback_policy is None:
                raise
            return self._fallback(value, context, error)

        provenance = self._profile_provenance(profile, compatibility, cost)
        if cost.effective_reuses <= 0.0:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                NO_EXPECTED_REUSE,
                0,
                provenance,
            )
        if not cost.is_profitable:
            return CompileDecision(
                value.sub_context_id,
                value.cache_key,
                CompileAction.KEEP_MUTABLE,
                NET_BENEFIT_NEGATIVE,
                0,
                provenance,
            )
        return CompileDecision(
            value.sub_context_id,
            value.cache_key,
            CompileAction.COMPILE,
            NET_BENEFIT_POSITIVE,
            int(round(cost.model_tokens * cost.effective_reuses)),
            provenance,
        )

    def build_candidate(
        self, value: SemanticSubContext, context: CompilePolicyContext
    ) -> CompileCandidate:
        """Assemble the correctness inputs, refusing an estimated count.

        The check is unconditional here, not gated on
        `context.require_model_tokens`. A cost-benefit decision reads a latency
        curve indexed by model tokens, so an estimated count would not merely
        be imprecise -- it would look up the wrong point on a real curve and
        return a confident number. The legacy floor's gate stays as it is,
        because a bare `CompilePolicyContext()` there is a shape inspection
        rather than a measurement.
        """

        counted: ModelTokenCount = require_model_tokens(
            value.token_accounting(),
            purpose=(
                f"cost-benefit compile decision for {value.sub_context_id!r} "
                "against a measured latency profile"
            ),
        )
        return CompileCandidate(
            sub_context_id=value.sub_context_id,
            cache_key=value.cache_key,
            context_field=value.context_field,
            model_tokens=counted,
            immutable=not value.is_mutable,
            reuse_candidate=value.reuse_candidate,
            invalidated=value.invalidated,
            preceding_model_tokens=int(context.preceding_model_tokens),
            planned_repair_model_tokens=context.planned_repair_model_tokens,
            repair_window_source=context.repair_window_source,
            cache_present=context.cache_present,
        )


def build_compile_policy(
    mode: CompilePolicyMode | str = CompilePolicyMode.LEGACY_FIXED_THRESHOLD,
    *,
    reuse_expectation: ReuseExpectation | None = None,
    profile: CompileCostProfile | None = None,
    residency_cost_model: ResidencyCostModel | None = None,
    deployment: DeploymentIdentity | None = None,
    strict_profile_compatibility: bool = True,
    fallback_policy: CompilePolicyMode | None = None,
) -> CompilePolicy:
    """Select a compile policy by name, so an experiment arm can record one.

    The legacy mode takes no performance profile at all -- that is the point of
    keeping it: a fixed token floor is hardware-independent, needs no
    measurement, and therefore stays runnable on any machine including this
    Windows checkout with no GPU.
    """

    selected = CompilePolicyMode(mode)
    if selected is CompilePolicyMode.LEGACY_FIXED_THRESHOLD:
        if profile is not None or reuse_expectation is not None:
            raise CompileCostError(
                "the legacy fixed-64 policy evaluates no cost model; passing a "
                "profile or reuse expectation to it would imply one was used"
            )
        return CompilePolicy()
    if reuse_expectation is None:
        raise CompileCostError(
            "cost-benefit mode requires an explicit ReuseExpectation"
        )
    return CostBenefitCompilePolicy(
        reuse_expectation=reuse_expectation,
        profile=profile,
        residency_cost_model=residency_cost_model,
        deployment=deployment,
        strict_profile_compatibility=strict_profile_compatibility,
        fallback_policy=fallback_policy,
    )


class CacheCatalog:
    """Process-local cache status and per-key compile serialization."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, object]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.RLock()
        self.removals_by_namespace: dict[str, int] = {}
        #: Logical roots over this catalog's rows (see semantic_cache_index). Not a second cache: primary
        #: entries are these rows, and an alias is valid only while its target row is.
        self.semantic_index = SemanticCacheIndex(self)

    def contains(self, cache_key: str) -> bool:
        with self._guard:
            return cache_key in self._entries and not bool(
                self._entries[cache_key].get("stale")
            )

    def mark_compiled(self, cache_key: str, **metadata: object) -> dict[str, object]:
        """Record a Compile the host issued; returns a copy of the row it wrote.

        `state_origin` defaults to a standalone compile because that is the only state the host can
        create: `NOCSGLangAdapter.compile` sends the block's own tokens with no preceding context. A caller
        registering any other kind of state -- a promoted or context-conditioned one -- must say so, and
        `reuse_eligibility` will then refuse it on the canonical path.
        """

        with self._guard:
            self._entries[cache_key] = {
                "state_origin": "standalone_compile",
                **metadata,
                "stale": False,
            }
            return dict(self._entries[cache_key])

    def mark_stale(self, cache_key: str) -> None:
        with self._guard:
            if cache_key in self._entries:
                self._entries[cache_key]["stale"] = True
                self._count_removal(self._entries[cache_key])
                self.semantic_index.on_physical_removed(
                    cache_key, self._entries[cache_key], reason="marked_stale"
                )

    def metadata(self, cache_key: str) -> dict[str, object] | None:
        with self._guard:
            value = self._entries.get(cache_key)
            return dict(value) if value is not None else None

    def invalidate(self, cache_key: str) -> None:
        with self._guard:
            removed = self._entries.pop(cache_key, None)
            if removed is not None:
                self._count_removal(removed)
                self.semantic_index.on_physical_removed(cache_key, removed, reason="invalidated")

    def _count_removal(self, entry: dict[str, object]) -> None:
        """Host-observed loss of a compiled block, by namespace. The runtime's own
        evictions are not visible here; only invalidations the host performed or
        was told about (e.g. residency recovery) are counted."""

        namespace = str(entry.get("namespace") or "unknown")
        self.removals_by_namespace[namespace] = self.removals_by_namespace.get(namespace, 0) + 1

    def find_by_physical_identity(self, digest: str) -> list[tuple[str, dict[str, object]]]:
        """Live entries compiled under physical KV identity ``digest`` (see cache_namespace)."""

        with self._guard:
            return [
                (key, dict(entry))
                for key, entry in self._entries.items()
                if entry.get("physical_kv_identity") == digest and not entry.get("stale")
            ]

    def live_entries(self) -> list[tuple[str, dict[str, object]]]:
        with self._guard:
            return [
                (key, dict(entry)) for key, entry in self._entries.items() if not entry.get("stale")
            ]

    def active_by_namespace(self) -> dict[str, int]:
        with self._guard:
            counts: dict[str, int] = {}
            for entry in self._entries.values():
                if entry.get("stale"):
                    continue
                namespace = str(entry.get("namespace") or "unknown")
                counts[namespace] = counts.get(namespace, 0) + 1
            return counts

    def lock_for(self, cache_key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(cache_key, threading.Lock())


class CompileAndLinkFeasibilityPlanner:
    """Stage 5 entrypoint: provisional -> feasibility -> effective decisions.

    ARCHITECTURE_STAGE: 5
    CANONICAL_ENTRYPOINT: CompileAndLinkFeasibilityPlanner

    The block-local economic decision (the compile policy) and the
    representability adjustment (``LinkLayoutFeasibility``) stay separate steps,
    in this order, so a report can still tell "did the policy choose better
    blocks?" from "did the LinkPlan shape let it?". Nothing here is new logic:
    ``provisional_decisions`` is the per-block loop ``NOCRequestPlanner.plan``
    used to hold, and ``apply_feasibility`` is ``LinkLayoutFeasibility.apply``.

    The Gate 1 reuse-eligibility check runs *between* the two steps, and is
    still taken by ``NOCRequestPlanner._assemble`` (Stage 6). This class takes
    its ``link_permitted`` answer as an argument and never learns why.
    """

    def __init__(
        self,
        *,
        compile_policy: "CompilePolicy | None" = None,
        cache_catalog: "CacheCatalog | None" = None,
        layout_feasibility: Any = None,
    ) -> None:
        self.compile_policy = CompilePolicy() if compile_policy is None else compile_policy
        self.cache_catalog = CacheCatalog() if cache_catalog is None else cache_catalog
        if layout_feasibility is None:
            from .link_layout import LinkLayoutFeasibility

            layout_feasibility = LinkLayoutFeasibility()
        self.layout_feasibility = layout_feasibility

    def provisional_decisions(
        self,
        ordered: Iterable[SemanticSubContext],
        *,
        expected_future_reuse: int = 1,
        memory_pressure: float = 0.0,
        require_model_tokens: bool = True,
        repair_model_tokens_by_block: Mapping[str, int] | None = None,
        repair_window_source: str = REPAIR_WINDOW_CALLER_SUPPLIED,
    ) -> tuple[CompileDecision, ...]:
        repair_by_block = dict(repair_model_tokens_by_block or {})
        decisions: list[CompileDecision] = []
        # Context preceding each block in selected order. A compiled block was
        # prefilled standalone from local position 0, so this is exactly the
        # context its leading tokens never attended to -- the quantity a
        # content-aware repair window is sized from. It is passed as a cost
        # input and nothing compares it to a threshold.
        preceding = 0
        for value in ordered:
            decisions.append(
                self.compile_policy.decide(
                    value,
                    CompilePolicyContext(
                        expected_future_reuse=expected_future_reuse,
                        memory_pressure=memory_pressure,
                        cache_present=self.cache_catalog.contains(value.cache_key),
                        require_model_tokens=require_model_tokens,
                        preceding_model_tokens=preceding,
                        planned_repair_model_tokens=repair_by_block.get(
                            value.sub_context_id
                        ),
                        repair_window_source=repair_window_source,
                    ),
                )
            )
            preceding += int(value.token_count)
        return tuple(decisions)

    def apply_feasibility(
        self,
        ordered: Sequence[SemanticSubContext],
        decisions: Sequence[CompileDecision],
        *,
        link_permitted: Sequence[bool] | None = None,
    ) -> tuple[Any, ...]:
        """The ``LayoutDecision`` records: policy, adjustment and effective action."""

        return self.layout_feasibility.apply(
            ordered, decisions, link_permitted=link_permitted
        )

    @staticmethod
    def effective_decisions(layout_decisions: Iterable[Any]) -> tuple[CompileDecision, ...]:
        return tuple(value.effective_decision for value in layout_decisions)


__all__ = [
    "BELOW_COMPILE_FLOOR",
    "LEGACY_COMPILE_COST_REASON",
    "LEGACY_COMPILE_POLICY_VERSION",
    "NET_BENEFIT_NEGATIVE",
    "NET_BENEFIT_POSITIVE",
    "NO_EXPECTED_REUSE",
    "PROFILE_UNCHECKED",
    "CacheCatalog",
    "CompileAndLinkFeasibilityPlanner",
    "CompileAction",
    "CompileAllImmutablePolicy",
    "CompileDecision",
    "CompileDecisionProvenance",
    "CompilePolicy",
    "CompilePolicyContext",
    "CompilePolicyMode",
    "CostBenefitCompilePolicy",
    "TokenAccountingError",
    "build_compile_policy",
]
