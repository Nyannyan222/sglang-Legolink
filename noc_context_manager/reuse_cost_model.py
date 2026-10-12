"""Gate 3: reuse profitability -- the interface and the accounting, not a policy.

A cache hit is not a profitable Link. Finer segmentation can raise both the reusable-block count and the
repair work: on the agent smoke, S0K recomputed 1196 tokens (1024 of them repair) and S1K 2634 (2462
repair). Whether extra reuse opportunity pays for the extra contextual boundaries is an empirical question
the formal S0/S1 x N/K matrix has to answer. So this module does three things and deliberately no fourth:

1. **Structure.** The cost decomposition the future model will be fitted to::

       CompileCost = C_compile_fixed + compile_tokens * C_prefill_token
       LinkCost    = C_link_fixed + linked_blocks * C_block + repair_tokens * C_repair_token
                     + repair_runs * C_repair_run + boundary_count * C_boundary

   `ReuseCostCoefficients` holds those symbols. Every one is `None` until a measured, provenance-carrying
   profile supplies it, and a prediction with any missing coefficient is `None` -- never an estimate.

2. **Accounting.** `RequestReuseAccounting` records, per request, the counts the equations need, the
   boundary structure segmentation creates, and the measured times the runtime exposes. A quantity nothing
   measures is recorded as unavailable with a reason, never filled in.

3. **Interface.** `ReuseProfitabilityPolicy`, with one implementation: `ObserveOnlyProfitabilityPolicy`. It
   computes what it can, records it, and changes nothing. An enforcing mode is refused at construction:
   there is no validated cost model, and inventing `if repair_ratio < Z` thresholds is the thing this
   module exists to prevent.

The fourth thing -- deciding LINK vs COMPILE on predicted cost -- waits for formal measurements.

Token accounting, stated once because it is easy to get wrong: **recomputed tokens already include repair
tokens.** The runtime's `recompute_positions` is chat prefix + per-block repair windows + inter-block
fresh gaps + mutable suffix (`pic_runtime._prepare_link`). The agent row's `recompute_tokens` is
`min(linked, mutable + sum(min(K, n_b)))`, the same decomposition from the host. `recomputed + repair`
double-counts every repair token; `RecomputeTokenDecomposition` is the only supported way to split them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .compile_policy import CompileAction


REUSE_COST_ACCOUNTING_VERSION = "reuse-cost-accounting-v1"

#: The relationship every consumer must respect. Pinned by a regression test against the agent formula.
RECOMPUTED_INCLUDES_REPAIR = True


class Availability(str, Enum):
    MEASURED = "measured"
    #: Derived exactly from the host plan (counts, not estimates of runtime behaviour).
    HOST_PLAN = "host_plan"
    #: Host-side estimate of a runtime quantity; the runtime trace is authoritative.
    HOST_ESTIMATE = "host_estimate"
    UNAVAILABLE = "unavailable"


#: Why a quantity is missing. Stable strings; artifacts key on them.
NOT_EXPOSED_BY_RUNTIME = "not_exposed_by_runtime"
NOT_KNOWN_AT_PLAN_TIME = "not_known_at_plan_time"
NOT_APPLICABLE_NATIVE_ROUTE = "not_applicable_native_route"
NOT_MEASURED_BY_CALLER = "not_measured_by_caller"
#: The server reported no prompt-token usage, so the host has no recomputed-token observation at all.
#: Distinct from a measured zero, and never substituted by one.
RECOMPUTED_TOKENS_NOT_OBSERVED = "recomputed_tokens_not_observed"
#: Both values were observed and they contradict the invariant (repair ⊆ recomputed). An audit failure,
#: deliberately not a serving failure.
INCONSISTENT_MEASURED_EVIDENCE = "inconsistent_measured_evidence"


@dataclass(frozen=True)
class Quantity:
    """A value with the evidence it rests on. `value is None` iff unavailable."""

    value: float | int | None
    availability: Availability
    reason: str = ""

    @classmethod
    def unavailable(cls, reason: str) -> "Quantity":
        return cls(None, Availability.UNAVAILABLE, reason)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "availability": self.availability.value}
        if self.reason:
            out["reason"] = self.reason
        return out


# --- 1. recomputed vs repair --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecomputeTokenDecomposition:
    """`recomputed = repair + fresh`, never `recomputed + repair`.

    `fresh` is every recomputed token that is not a repair row: chat prefix, mutable suffix, blocks kept
    native, inter-block gaps.
    """

    recomputed_tokens: int
    repair_tokens: int

    def __post_init__(self) -> None:
        if self.repair_tokens < 0 or self.recomputed_tokens < 0:
            raise ValueError("token counts are non-negative")
        if self.repair_tokens > self.recomputed_tokens:
            raise ValueError(
                "repair tokens exceed recomputed tokens: recomputed already includes repair, so a "
                "larger repair count means the two came from different accounting paths"
            )

    @property
    def fresh_tokens(self) -> int:
        return self.recomputed_tokens - self.repair_tokens

    @property
    def total_recomputed_tokens(self) -> int:
        """The total. Deliberately not `recomputed + repair`."""

        return self.recomputed_tokens

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "recomputed_tokens": self.recomputed_tokens,
            "repair_tokens": self.repair_tokens,
            "fresh_non_repair_tokens": self.fresh_tokens,
            "recomputed_includes_repair": RECOMPUTED_INCLUDES_REPAIR,
        }


# --- 2. boundaries ------------------------------------------------------------------------------------


def _source_ids(value: Any) -> frozenset[str]:
    return frozenset(span.source_id for span in getattr(value, "source_spans", ()) or ())


@dataclass(frozen=True)
class BoundaryAccounting:
    """How many contextual boundaries a request has, and which ones segmentation introduced.

    A boundary sits between two consecutive included blocks. An *intra-source* boundary separates two
    blocks cut from the same source (one tool output, one file): segmentation made it, and a finer
    segmenter makes more of them. An *inter-source* boundary exists whatever the segmenter does. A
    *linked* boundary is the leading edge of a linked block: its standalone state never attended to what
    precedes it, which is where sparse repair is spent.
    """

    semantic_block_count: int
    boundary_count: int
    intra_source_boundary_count: int
    inter_source_boundary_count: int
    linked_block_count: int
    linked_boundary_count: int
    linked_intra_source_boundary_count: int
    source_count: int

    @classmethod
    def from_plan(cls, plan: Any) -> "BoundaryAccounting":
        decisions = {value.sub_context_id: value for value in plan.decisions}
        included = [
            value
            for value in plan.ordered_contexts
            if decisions[value.sub_context_id].action is not CompileAction.DO_NOT_INCLUDE
        ]
        linked = {
            value.sub_context_id
            for value in included
            if decisions[value.sub_context_id].action is CompileAction.COMPILE
        }
        intra = inter = linked_intra = 0
        for previous, current in zip(included, included[1:]):
            same_source = bool(_source_ids(previous) & _source_ids(current))
            intra += same_source
            inter += not same_source
            linked_intra += same_source and current.sub_context_id in linked
        sources: set[str] = set()
        for value in included:
            sources |= _source_ids(value)
        return cls(
            semantic_block_count=len(included),
            boundary_count=max(len(included) - 1, 0),
            intra_source_boundary_count=intra,
            inter_source_boundary_count=inter,
            linked_block_count=len(linked),
            # Every linked block has a leading edge: after another block, or after the chat prefix.
            linked_boundary_count=len(linked),
            linked_intra_source_boundary_count=linked_intra,
            source_count=len(sources),
        )

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


# --- 3. per-request accounting -----------------------------------------------------------------------


@dataclass(frozen=True)
class RequestReuseAccounting:
    """Everything the future cost model needs about one request, with provenance per quantity."""

    counts: Mapping[str, Quantity]
    boundaries: BoundaryAccounting
    blocks: tuple[Mapping[str, Any], ...] = ()
    route: str = ""

    @classmethod
    def from_plan(
        cls,
        plan: Any,
        *,
        linked_route: bool = True,
        repair_tokens_by_cache_key: Mapping[str, int] | None = None,
        linked_cache_keys: Sequence[str] | None = None,
        prompt_tokens: int | None = None,
        recomputed_tokens: int | None = None,
        compile_latency_s: float | None = None,
        request_latency_s: float | None = None,
        ttft_s: float | None = None,
        route: str = "",
        recovered_cache_keys: Sequence[str] | None = None,
    ) -> "RequestReuseAccounting":
        """Build from a plan plus whatever the caller measured. Nothing missing is inferred.

        `linked_cache_keys` is the Link request's `cache_keys` when the caller has built one: token-stable
        framing can carry a planned-for-link block as fresh tokens, and such a block is not linked.
        """

        records = plan.planning_records
        boundaries = BoundaryAccounting.from_plan(plan)
        planned = [r for r in records if r.effective_action == CompileAction.COMPILE.value]
        physical = None if linked_cache_keys is None else set(linked_cache_keys)
        linked = [r for r in planned if physical is None or r.cache_key in physical]
        compiled = [r for r in records if r.planner_category == "COMPILE_ON_MISS"]
        # Blocks the runtime reported gone and the host compiled again were planned as cached from a catalog that records
        # intent, not residency. They were compiled: counted as compiled, never as cached, and named.
        recovered_keys = {str(key) for key in (recovered_cache_keys or ())}
        recovered = [r for r in planned if r.cache_key in recovered_keys]
        cached = [r for r in records if r.planner_category == "REUSE_CACHED" and r.cache_key not in recovered_keys]
        native = [r for r in records if r.planner_category == "KEEP_MUTABLE"]
        plan_q = lambda value: Quantity(value, Availability.HOST_PLAN)  # noqa: E731

        def measured(value, reason=NOT_MEASURED_BY_CALLER) -> Quantity:
            return Quantity.unavailable(reason) if value is None else Quantity(value, Availability.MEASURED)

        counts: dict[str, Quantity] = {
            "selected_block_count": plan_q(len(records)),
            "semantic_candidate_count": plan_q(sum(bool(r.semantic_candidate) for r in records)),
            "eligible_reuse_block_count": plan_q(
                sum(r.reuse_eligibility == "exact_safe" for r in records)
            ),
            "eligibility_declined_block_count": plan_q(sum(r.eligibility_adjusted for r in records)),
            "layout_adjusted_block_count": plan_q(sum(r.layout_forced for r in records)),
            "linked_block_count": plan_q(len(linked) if linked_route else 0),
            "framing_fresh_block_count": plan_q(len(planned) - len(linked) if linked_route else 0),
            "linked_cached_block_count": plan_q(len(cached) if linked_route else 0),
            "compiled_block_count": plan_q((len(compiled) + len(recovered)) if linked_route else 0),
            "residency_recovered_block_count": Quantity(len(recovered) if linked_route else 0, Availability.MEASURED),
            "residency_recovered_tokens": Quantity(sum(r.token_count for r in recovered) if linked_route else 0, Availability.MEASURED),
            "native_block_count": plan_q(len(native) if linked_route else len(records)),
            "linked_tokens": plan_q(sum(r.token_count for r in linked) if linked_route else 0),
            "compile_tokens": plan_q((sum(r.token_count for r in compiled) + sum(r.token_count for r in recovered)) if linked_route else 0),
            "native_block_tokens": plan_q(
                sum(r.token_count for r in (native if linked_route else records))
            ),
            "boundary_count": plan_q(boundaries.boundary_count),
            "linked_boundary_count": plan_q(boundaries.linked_boundary_count if linked_route else 0),
        }
        if not linked_route:
            for name in ("repair_tokens", "repair_run_count", "recomputed_tokens"):
                counts[name] = Quantity.unavailable(NOT_APPLICABLE_NATIVE_ROUTE)
        else:
            if repair_tokens_by_cache_key is None:
                counts["repair_tokens"] = Quantity.unavailable(NOT_KNOWN_AT_PLAN_TIME)
                counts["repair_run_count"] = Quantity.unavailable(NOT_KNOWN_AT_PLAN_TIME)
            else:
                windows = [int(repair_tokens_by_cache_key.get(r.cache_key, 0)) for r in linked]
                # The host's request-level windows. The runtime clips each to its block and runs them in
                # one batched forward, so "runs" here are windows, not forward passes.
                counts["repair_tokens"] = Quantity(
                    sum(min(w, r.token_count) for w, r in zip(windows, linked)), Availability.HOST_PLAN
                )
                counts["repair_run_count"] = Quantity(sum(w > 0 for w in windows), Availability.HOST_PLAN)
            counts["recomputed_tokens"] = (
                Quantity(recomputed_tokens, Availability.HOST_ESTIMATE)
                if recomputed_tokens is not None
                else Quantity.unavailable(RECOMPUTED_TOKENS_NOT_OBSERVED)
            )
        counts["prompt_tokens"] = measured(prompt_tokens)
        counts["compile_prefill_time_s"] = measured(compile_latency_s)
        counts["request_latency_s"] = measured(request_latency_s)
        counts["ttft_s"] = measured(ttft_s)
        # Link setup and repair are one fused runtime step behind the Link request; neither is reported to
        # the host separately. The runtime trace's host-timing spans are the place to recover them.
        counts["link_setup_time_s"] = Quantity.unavailable(NOT_EXPOSED_BY_RUNTIME)
        counts["repair_time_s"] = Quantity.unavailable(NOT_EXPOSED_BY_RUNTIME)
        # Unknown is not zero. A request whose server reported no usage has *no* recomputed-token
        # observation; recording 0 there would invent a measurement and, worse, make repair look larger
        # than the total it is part of. Both quantities stay unavailable and the request is unaffected.
        repair = counts["repair_tokens"].value
        recomputed = counts.get("recomputed_tokens", Quantity.unavailable(NOT_MEASURED_BY_CALLER)).value
        if recomputed is None or repair is None:
            counts["nonrepair_recomputed_tokens"] = Quantity.unavailable(
                RECOMPUTED_TOKENS_NOT_OBSERVED if recomputed is None else NOT_KNOWN_AT_PLAN_TIME
            )
        elif repair > recomputed:
            counts["nonrepair_recomputed_tokens"] = Quantity.unavailable(INCONSISTENT_MEASURED_EVIDENCE)
        else:
            counts["nonrepair_recomputed_tokens"] = Quantity(
                recomputed - repair, Availability.HOST_ESTIMATE
            )
        counts["repair_tokens_per_linked_boundary"] = (
            Quantity(repair / boundaries.linked_boundary_count, Availability.HOST_PLAN)
            if repair is not None and boundaries.linked_boundary_count
            else Quantity.unavailable(
                NOT_KNOWN_AT_PLAN_TIME if repair is None else "no_linked_boundary"
            )
        )
        blocks = tuple(
            {
                "block_id": r.block_id,
                "block_token_count": r.token_count,
                "effective_decision": r.effective_decision,
                "reuse_eligibility": r.reuse_eligibility,
                "eligibility_reason": r.eligibility_reason,
                "feasibility_reason": r.feasibility_reason,
                # Host estimate: selected-body tokens before the block, chat prefix excluded.
                "position_shift_estimate": r.preceding_context_tokens,
                "preceding_context_tokens": r.preceding_context_tokens,
                "repair_tokens": (
                    None
                    if repair_tokens_by_cache_key is None or r.effective_action != CompileAction.COMPILE.value
                    else min(int(repair_tokens_by_cache_key.get(r.cache_key, 0)), r.token_count)
                ),
            }
            for r in records
        )
        return cls(counts=counts, boundaries=boundaries, blocks=blocks, route=route)

    def value(self, name: str) -> float | int | None:
        return self.counts[name].value

    def recompute_decomposition(self) -> RecomputeTokenDecomposition | None:
        """The split, or `None` when it cannot be formed. Never raises.

        This is read while a request is being served, so an unobservable or self-contradictory pair of
        numbers must degrade to "no decomposition recorded", never to an exception. The strictness lives
        in `RecomputeTokenDecomposition` itself, which is still the only way to build one, and in
        `token_accounting_status`, which names what went wrong for the audit to act on.
        """

        recomputed, repair = self.value("recomputed_tokens"), self.value("repair_tokens")
        if recomputed is None or repair is None or repair > recomputed:
            return None
        return RecomputeTokenDecomposition(int(recomputed), int(repair))

    def token_accounting_status(self) -> str:
        """Why a decomposition is or is not available. A measurement verdict, not a serving one."""

        recomputed, repair = self.value("recomputed_tokens"), self.value("repair_tokens")
        if repair is None:
            return "repair_tokens_unavailable"
        if recomputed is None:
            return "recomputed_tokens_unavailable"
        if repair > recomputed:
            return INCONSISTENT_MEASURED_EVIDENCE
        return "consistent"

    def to_dict(self) -> dict[str, Any]:
        decomposition = self.recompute_decomposition()
        return {
            "reuse_cost_accounting_version": REUSE_COST_ACCOUNTING_VERSION,
            "route": self.route,
            "counts": {name: quantity.to_dict() for name, quantity in self.counts.items()},
            "boundaries": self.boundaries.to_dict(),
            "recompute_decomposition": None if decomposition is None else decomposition.to_dict(),
            "token_accounting_status": self.token_accounting_status(),
            "blocks": [dict(value) for value in self.blocks],
        }


def aggregate_accounting(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum serialized per-request accounting over an arm, keeping unavailability visible.

    A count is summed only over requests where it was available; `requests_missing` says how many were
    not. `recomputed_tokens` and `repair_tokens` are summed separately and never added together.
    """

    totals: dict[str, float] = {}
    missing: dict[str, int] = {}
    statuses: dict[str, int] = {}
    n = 0
    for row in rows:
        n += 1
        status = str(row.get("token_accounting_status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        for name, quantity in (row.get("counts") or {}).items():
            value = quantity.get("value")
            if value is None:
                missing[name] = missing.get(name, 0) + 1
            elif name not in {"repair_tokens_per_linked_boundary"}:
                totals[name] = totals.get(name, 0) + value
    out: dict[str, Any] = {
        "requests": n,
        "totals": totals,
        "requests_missing": missing,
        "token_accounting_status_counts": statuses,
    }
    if "recomputed_tokens" in totals and "repair_tokens" in totals and not (
        missing.get("recomputed_tokens") or missing.get("repair_tokens")
    ):
        out["recompute_decomposition"] = RecomputeTokenDecomposition(
            int(totals["recomputed_tokens"]), int(totals["repair_tokens"])
        ).to_dict()
    if totals.get("linked_boundary_count") and "repair_tokens" in totals and not missing.get("repair_tokens"):
        out["repair_tokens_per_linked_boundary"] = totals["repair_tokens"] / totals["linked_boundary_count"]
    return out


# --- 4. the cost model structure and the profitability interface --------------------------------------


@dataclass(frozen=True)
class ReuseCostCoefficients:
    """The symbols of the decomposition. `None` means not measured; no defaults exist on purpose."""

    compile_fixed_ms: float | None = None
    prefill_ms_per_token: float | None = None
    link_fixed_ms: float | None = None
    per_linked_block_ms: float | None = None
    repair_ms_per_token: float | None = None
    per_repair_run_ms: float | None = None
    per_boundary_ms: float | None = None
    #: Where the numbers came from (hardware, model, artifact hash). Required once any value is set.
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        populated = [v for k, v in self.__dict__.items() if k != "provenance" and v is not None]
        if populated and not self.provenance:
            raise ValueError("cost coefficients need provenance: which deployment measured them")


class ReuseCostModel:
    """Evaluate the decomposition. Returns `None` whenever a needed coefficient or count is missing."""

    def __init__(self, coefficients: ReuseCostCoefficients | None = None) -> None:
        self.coefficients = coefficients or ReuseCostCoefficients()

    @staticmethod
    def _terms(pairs: Sequence[tuple[float | None, float | int | None]], fixed: float | None) -> float | None:
        if fixed is None or any(c is None or n is None for c, n in pairs):
            return None
        return float(fixed) + sum(float(c) * float(n) for c, n in pairs)

    def compile_cost_ms(self, accounting: RequestReuseAccounting) -> float | None:
        """The counterfactual: prefilling the linked blocks' tokens natively instead of linking them.

        The architecture's "compile_tokens" in this equation is therefore `linked_tokens`. The accounting
        field named `compile_tokens` is the repository's standalone Compile work (compile-on-miss), which
        the Link path pays in addition and a future model has to charge to the Link side.
        """

        c = self.coefficients
        return self._terms([(c.prefill_ms_per_token, accounting.value("linked_tokens"))], c.compile_fixed_ms)

    def link_cost_ms(self, accounting: RequestReuseAccounting) -> float | None:
        c = self.coefficients
        return self._terms(
            [
                (c.per_linked_block_ms, accounting.value("linked_block_count")),
                (c.repair_ms_per_token, accounting.value("repair_tokens")),
                (c.per_repair_run_ms, accounting.value("repair_run_count")),
                # All semantic boundaries, not only linked ones: linked boundaries equal linked blocks
                # by construction, so using them here would make C_boundary and C_block collinear.
                (c.per_boundary_ms, accounting.value("boundary_count")),
            ],
            c.link_fixed_ms,
        )


class ProfitabilityMode(str, Enum):
    OBSERVE_ONLY = "observe_only"
    ENFORCED = "enforced"


@dataclass(frozen=True)
class ProfitabilityVerdict:
    mode: ProfitabilityMode
    enforced: bool
    predicted_link_cost_ms: float | None
    predicted_compile_cost_ms: float | None
    #: What an enforcing policy *would* prefer; recorded, never applied in observe-only mode.
    observed_preference: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profitability_mode": self.mode.value,
            "enforced": self.enforced,
            "predicted_link_cost_ms": self.predicted_link_cost_ms,
            "predicted_compile_cost_ms": self.predicted_compile_cost_ms,
            "observed_preference": self.observed_preference,
            "status": self.status,
        }


class ReuseProfitabilityPolicy:
    """Gate 3 interface. Subclasses decide; the planner applies only enforced verdicts."""

    mode = ProfitabilityMode.OBSERVE_ONLY

    def evaluate(self, accounting: RequestReuseAccounting) -> ProfitabilityVerdict:  # pragma: no cover
        raise NotImplementedError


class ObserveOnlyProfitabilityPolicy(ReuseProfitabilityPolicy):
    """Records what the cost model can say. Changes no decision, under any input."""

    mode = ProfitabilityMode.OBSERVE_ONLY

    def __init__(self, cost_model: ReuseCostModel | None = None) -> None:
        self.cost_model = cost_model or ReuseCostModel()

    def evaluate(self, accounting: RequestReuseAccounting) -> ProfitabilityVerdict:
        link = self.cost_model.link_cost_ms(accounting)
        native = self.cost_model.compile_cost_ms(accounting)
        if link is None or native is None:
            preference, status = "unknown", "coefficients_or_counts_unavailable"
        else:
            preference = "link" if link < native else "compile"
            status = "predicted_not_enforced"
        return ProfitabilityVerdict(
            mode=self.mode,
            enforced=False,
            predicted_link_cost_ms=link,
            predicted_compile_cost_ms=native,
            observed_preference=preference,
            status=status,
        )


def require_observe_only(policy: ReuseProfitabilityPolicy) -> ReuseProfitabilityPolicy:
    """The only mode this repository may run until a cost model is validated on formal data."""

    if policy.mode is not ProfitabilityMode.OBSERVE_ONLY:
        raise NotImplementedError(
            "enforced reuse profitability needs a cost model validated on formal Agent measurements; "
            "none exists, and no threshold may stand in for one"
        )
    return policy


__all__ = [
    "INCONSISTENT_MEASURED_EVIDENCE",
    "NOT_APPLICABLE_NATIVE_ROUTE",
    "NOT_EXPOSED_BY_RUNTIME",
    "NOT_KNOWN_AT_PLAN_TIME",
    "NOT_MEASURED_BY_CALLER",
    "RECOMPUTED_INCLUDES_REPAIR",
    "RECOMPUTED_TOKENS_NOT_OBSERVED",
    "REUSE_COST_ACCOUNTING_VERSION",
    "Availability",
    "BoundaryAccounting",
    "ObserveOnlyProfitabilityPolicy",
    "ProfitabilityMode",
    "ProfitabilityVerdict",
    "Quantity",
    "RecomputeTokenDecomposition",
    "RequestReuseAccounting",
    "ReuseCostCoefficients",
    "ReuseCostModel",
    "ReuseProfitabilityPolicy",
    "aggregate_accounting",
    "require_observe_only",
]
