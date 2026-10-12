"""Compile economics: the measured half of "should this block be compiled?".

The old answer was a constant. `CompilePolicyContext.min_compile_model_tokens
= 64` decided which blocks were eligible to be cached at all, and 64 was not
derived from anything: not from prefill cost, not from compile cost, not from
link or sparse-repair cost, not from how often the block was likely to come
back. The unit was made explicit in the previous change -- 64 *model* tokens,
not 64 tokens of whatever tokenizer happened to build the block -- which fixed
the ambiguity without touching the fact that the number itself was a guess.

What a fixed floor cannot express is that the trade is a break-even:

    compile once  +  N x link        versus        N x native prefill

A 40-token block that returns twelve times can be worth compiling; a
200-token block that is never seen again is not. The floor answers neither
question, because block size is only one of the four terms.

This module holds the terms. It deliberately holds *only* the terms:

  * `CompileCostProfile` -- latency measurements, which are hardware, model and
    configuration specific, carrying the provenance that says which.
  * `ReuseExpectation` -- how often the block is expected back, which is a
    property of the workload.
  * `ResidencyCostModel` -- what holding the KV costs, which is currently
    unmeasured and therefore explicitly disabled rather than quietly zero.
  * `CompileCandidate` / `CompileCostBreakdown` -- one block's inputs and the
    arithmetic over them, so every decision can be re-derived from its record.

The policy that consumes them lives in `compile_policy.py`, and imports this
module rather than the other way round. That direction is the point: the
decision algorithm is hardware-independent and the numbers it reads are not.

Nothing here invents a number. A curve asked for a length it never measured
raises rather than extrapolating, and a profile from the wrong model or the
wrong GPU is refused under a strict compatibility check rather than silently
reused. Missing evidence produces no decision, in keeping with the rest of this
repository: a run that never measured its prefill curve did not thereby
establish anything about it.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

COMPILE_COST_SCHEMA_VERSION = 1

#: Bumped whenever the arithmetic in `evaluate_compile_economics` changes, so a
#: recorded decision can be matched to the formula that produced it.
COMPILE_COST_POLICY_VERSION = "compile-cost-benefit-v1"


class CompileCostError(RuntimeError):
    """Base for every failure in the compile cost model."""


class CompileCostProfileError(CompileCostError):
    """A profile is malformed or internally inconsistent."""


class CompileCostUnavailableError(CompileCostError):
    """No honest estimate exists, so no cost-benefit decision exists.

    Raised for a missing profile, a length outside the measured range, a
    profile that does not match the deployment, and an unstated repair window
    that the profile needs. Every one of these is "we do not know", and the
    caller either supplies the evidence or asks for the legacy fixed threshold
    by name.
    """


# --- B. performance profile inputs: hardware, model and config specific -----


@dataclass(frozen=True)
class LatencyCurve:
    """Milliseconds as a function of model tokens, over measured points only.

    Piecewise linear between the points that were actually measured, and an
    error outside them. Clamping to the nearest endpoint would be the tempting
    alternative and it is the same defect in a friendlier shape: a 16,000-token
    block would silently be costed at the 4,096-token measurement.
    """

    name: str
    points: tuple[tuple[int, float], ...]
    #: Which statistic the milliseconds are. A median and a p95 curve support
    #: different claims, so the curve says which one it is rather than leaving
    #: the reader to assume.
    statistic: str = "median"
    samples_per_point: int = 0

    def __post_init__(self) -> None:
        if not self.points:
            raise CompileCostProfileError(f"{self.name} curve has no measured points")
        xs = [int(x) for x, _ in self.points]
        if any(x < 0 for x in xs):
            raise CompileCostProfileError(f"{self.name} curve has a negative length")
        if xs != sorted(xs) or len(set(xs)) != len(xs):
            raise CompileCostProfileError(
                f"{self.name} curve points must be strictly increasing in tokens"
            )
        if any(float(y) < 0.0 for _, y in self.points):
            raise CompileCostProfileError(f"{self.name} curve has a negative latency")

    @property
    def measured_range(self) -> tuple[int, int]:
        return int(self.points[0][0]), int(self.points[-1][0])

    def covers(self, model_tokens: int) -> bool:
        low, high = self.measured_range
        return low <= int(model_tokens) <= high

    def estimate_ms(self, model_tokens: int) -> float:
        value = int(model_tokens)
        low, high = self.measured_range
        if not self.covers(value):
            raise CompileCostUnavailableError(
                f"{self.name} was measured over {low}..{high} model tokens and "
                f"cannot estimate {value}. Extrapolating would invent a number; "
                "extend the profile instead."
            )
        previous_x, previous_y = self.points[0]
        for point_x, point_y in self.points:
            if value == point_x:
                return float(point_y)
            if value < point_x:
                span = point_x - previous_x
                if span <= 0:
                    return float(point_y)
                weight = (value - previous_x) / span
                return float(previous_y) + weight * (float(point_y) - float(previous_y))
            previous_x, previous_y = point_x, point_y
        return float(self.points[-1][1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": [[int(x), float(y)] for x, y in self.points],
            "statistic": self.statistic,
            "samples_per_point": int(self.samples_per_point),
        }

    @classmethod
    def from_payload(cls, name: str, payload: Mapping[str, Any]) -> "LatencyCurve":
        points = tuple(
            (int(entry[0]), float(entry[1])) for entry in payload.get("points", ())
        )
        return cls(
            name=str(payload.get("name") or name),
            points=points,
            statistic=str(payload.get("statistic") or "median"),
            samples_per_point=int(payload.get("samples_per_point") or 0),
        )


@dataclass(frozen=True)
class DeploymentIdentity:
    """What is actually being served, to check a profile against."""

    hardware_id: str
    model_id: str
    dtype: str = ""
    tensor_parallel_size: int | None = None
    attention_backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_id": self.hardware_id,
            "model_id": self.model_id,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "attention_backend": self.attention_backend,
        }


@dataclass(frozen=True)
class ProfileProvenance:
    """Where the numbers came from, in enough detail to reject them.

    A latency curve is meaningless without this. The same block costs a
    different number of milliseconds on an RTX 5070 and an H100, at bf16 and
    fp8, at TP1 and TP4, on FlashInfer and Triton -- so a profile that does not
    say which it measured cannot be checked against the deployment it is used
    on, and an unlabelled curve is indistinguishable from a made-up one.
    """

    profile_version: str
    hardware_id: str
    model_id: str
    dtype: str = ""
    tensor_parallel_size: int = 1
    sglang_version: str = ""
    attention_backend: str = ""
    measured_at: str = ""
    #: The run directory or artifact the curves were fitted from. A profile
    #: built by hand for a test says so here; it does not pretend to a run.
    source_artifact: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in ("profile_version", "hardware_id", "model_id"):
            if not str(getattr(self, field_name) or "").strip():
                raise CompileCostProfileError(
                    f"a compile cost profile must state its {field_name}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "hardware_id": self.hardware_id,
            "model_id": self.model_id,
            "dtype": self.dtype,
            "tensor_parallel_size": int(self.tensor_parallel_size),
            "sglang_version": self.sglang_version,
            "attention_backend": self.attention_backend,
            "measured_at": self.measured_at,
            "source_artifact": self.source_artifact,
            "notes": self.notes,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProfileProvenance":
        return cls(
            profile_version=str(payload.get("profile_version") or ""),
            hardware_id=str(payload.get("hardware_id") or ""),
            model_id=str(payload.get("model_id") or ""),
            dtype=str(payload.get("dtype") or ""),
            tensor_parallel_size=int(payload.get("tensor_parallel_size") or 1),
            sglang_version=str(payload.get("sglang_version") or ""),
            attention_backend=str(payload.get("attention_backend") or ""),
            measured_at=str(payload.get("measured_at") or ""),
            source_artifact=str(payload.get("source_artifact") or ""),
            notes=str(payload.get("notes") or ""),
        )


#: What a compatibility check concluded. `UNCHECKED` is not `MATCHED`: it means
#: the caller never said what it was serving, so nothing was compared.
PROFILE_MATCHED = "matched"
PROFILE_MISMATCHED = "mismatched"
PROFILE_UNCHECKED = "unchecked"


@dataclass(frozen=True)
class CompileCostProfile:
    """Measured latency for one model on one hardware configuration.

    Three curves are required because the break-even needs all three: what a
    native prefill of this block would have cost, what compiling it once costs,
    and what linking it costs on every reuse. A fourth, `repair_ms`, is optional
    and its absence is a claim in itself -- see `repair_is_folded_into_link`.
    """

    provenance: ProfileProvenance
    prefill_ms: LatencyCurve
    compile_ms: LatencyCurve
    link_ms: LatencyCurve
    #: Sparse boundary repair, as a function of recomputed model tokens. When
    #: absent, the link curve is taken to already contain whatever repair the
    #: measured runs performed, and the policy records that rather than costing
    #: repair at zero behind the reader's back.
    repair_ms: LatencyCurve | None = None

    @property
    def repair_is_folded_into_link(self) -> bool:
        return self.repair_ms is None

    @property
    def profile_id(self) -> str:
        """Content hash. Decisions reference this, never the whole profile."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"sha256:{digest[:16]}"

    def estimate_prefill_ms(self, model_tokens: int) -> float:
        return self.prefill_ms.estimate_ms(model_tokens)

    def estimate_compile_ms(self, model_tokens: int) -> float:
        return self.compile_ms.estimate_ms(model_tokens)

    def estimate_link_ms(self, model_tokens: int) -> float:
        return self.link_ms.estimate_ms(model_tokens)

    def estimate_repair_ms(self, repair_model_tokens: int) -> float:
        if self.repair_ms is None:
            raise CompileCostUnavailableError(
                f"profile {self.profile_id} has no repair curve; repair is "
                "folded into the link curve and cannot be costed separately"
            )
        return self.repair_ms.estimate_ms(repair_model_tokens)

    def check_compatibility(
        self,
        deployment: DeploymentIdentity | None,
        *,
        strict: bool = True,
    ) -> str:
        """Compare the profile against what is being served.

        `strict` is the default because the failure it prevents is silent: an
        H100 curve used to plan an RTX 5070 run produces plausible-looking
        milliseconds that are wrong by an order of magnitude, and nothing
        downstream would say so. Fields the deployment leaves blank are not
        compared -- an unstated dtype is not a mismatch, it is an unstated
        dtype -- but hardware and model are always compared.
        """

        if deployment is None:
            return PROFILE_UNCHECKED
        mismatches: list[str] = []
        if deployment.hardware_id != self.provenance.hardware_id:
            mismatches.append(
                f"hardware {self.provenance.hardware_id!r} != {deployment.hardware_id!r}"
            )
        if deployment.model_id != self.provenance.model_id:
            mismatches.append(
                f"model {self.provenance.model_id!r} != {deployment.model_id!r}"
            )
        if (
            deployment.dtype
            and self.provenance.dtype
            and deployment.dtype != self.provenance.dtype
        ):
            mismatches.append(
                f"dtype {self.provenance.dtype!r} != {deployment.dtype!r}"
            )
        if (
            deployment.tensor_parallel_size is not None
            and int(deployment.tensor_parallel_size)
            != int(self.provenance.tensor_parallel_size)
        ):
            mismatches.append(
                f"TP {self.provenance.tensor_parallel_size} != "
                f"{deployment.tensor_parallel_size}"
            )
        if (
            deployment.attention_backend
            and self.provenance.attention_backend
            and deployment.attention_backend != self.provenance.attention_backend
        ):
            mismatches.append(
                f"attention backend {self.provenance.attention_backend!r} != "
                f"{deployment.attention_backend!r}"
            )
        if not mismatches:
            return PROFILE_MATCHED
        if strict:
            raise CompileCostUnavailableError(
                f"compile cost profile {self.profile_id} does not match the "
                f"deployment: {'; '.join(mismatches)}. Latency measured on one "
                "configuration is not evidence about another."
            )
        return PROFILE_MISMATCHED

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": COMPILE_COST_SCHEMA_VERSION,
            "provenance": self.provenance.to_dict(),
            "prefill_ms": self.prefill_ms.to_dict(),
            "compile_ms": self.compile_ms.to_dict(),
            "link_ms": self.link_ms.to_dict(),
        }
        payload["repair_ms"] = (
            self.repair_ms.to_dict() if self.repair_ms is not None else None
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CompileCostProfile":
        missing = [
            name
            for name in ("prefill_ms", "compile_ms", "link_ms")
            if not payload.get(name)
        ]
        if missing:
            raise CompileCostProfileError(
                f"compile cost profile is missing required curves: {missing}"
            )
        repair = payload.get("repair_ms")
        return cls(
            provenance=ProfileProvenance.from_payload(payload.get("provenance") or {}),
            prefill_ms=LatencyCurve.from_payload("prefill_ms", payload["prefill_ms"]),
            compile_ms=LatencyCurve.from_payload("compile_ms", payload["compile_ms"]),
            link_ms=LatencyCurve.from_payload("link_ms", payload["link_ms"]),
            repair_ms=(
                LatencyCurve.from_payload("repair_ms", repair) if repair else None
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CompileCostProfile":
        return cls.from_payload(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.to_dict(), "profile_id": self.profile_id}
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target


# --- C. workload expectation inputs -----------------------------------------


@dataclass(frozen=True)
class ReuseEstimate:
    """How often a block is expected back, and on whose authority.

    Two numbers rather than one, because "this block will be needed four more
    times" and "this block has a 60% chance of being needed again" are
    different claims and the policy multiplies them. `source` is mandatory:
    every expected-reuse number in this repository so far has been a caller's
    guess (`max(1, turn)`, `max_rounds - index + 1`), and a guess that says it
    is one is usable evidence in a way that an anonymous float is not.
    """

    expected_future_reuses: float
    reuse_probability: float
    source: str
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if self.expected_future_reuses < 0:
            raise CompileCostError("expected_future_reuses must be non-negative")
        if not 0.0 <= self.reuse_probability <= 1.0:
            raise CompileCostError("reuse_probability must be in [0, 1]")
        if not str(self.source or "").strip():
            raise CompileCostError("a reuse estimate must name its source")

    @property
    def effective_reuses(self) -> float:
        return float(self.expected_future_reuses) * float(self.reuse_probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_future_reuses": float(self.expected_future_reuses),
            "reuse_probability": float(self.reuse_probability),
            "effective_reuses": self.effective_reuses,
            "source": self.source,
            "evidence_id": self.evidence_id,
        }


class ReuseExpectation(ABC):
    """How many future reuses to expect for one block."""

    name = "abstract"

    @abstractmethod
    def estimate(
        self,
        *,
        cache_key: str,
        context_field: str,
        planned_future_reuse: int,
    ) -> ReuseEstimate:
        """`planned_future_reuse` is what the caller declared for the request."""


class FixedReuseExpectation(ReuseExpectation):
    """A stated constant. Honest when the workload really is fixed-length."""

    name = "fixed"

    def __init__(self, expected_future_reuses: float, *, probability: float = 1.0):
        self.expected_future_reuses = float(expected_future_reuses)
        self.probability = float(probability)

    def estimate(
        self, *, cache_key: str, context_field: str, planned_future_reuse: int
    ) -> ReuseEstimate:
        return ReuseEstimate(
            expected_future_reuses=self.expected_future_reuses,
            reuse_probability=self.probability,
            source=f"fixed:{self.expected_future_reuses:g}",
        )


class PlannedTurnsReuseExpectation(ReuseExpectation):
    """The caller's own `expected_future_reuse`, discounted and labelled.

    This is what the repository already does -- the agent passes
    `max(1, turn)`, the loop passes `max_rounds - index + 1` -- and neither is
    an observation. `probability` is the discount that says so, and it defaults
    below 1.0 so that "the session has eight turns left" is not silently read
    as "this block will be used eight more times".
    """

    name = "planned_turns"

    def __init__(self, *, probability: float = 0.5):
        if not 0.0 <= probability <= 1.0:
            raise CompileCostError("probability must be in [0, 1]")
        self.probability = float(probability)

    def estimate(
        self, *, cache_key: str, context_field: str, planned_future_reuse: int
    ) -> ReuseEstimate:
        return ReuseEstimate(
            expected_future_reuses=float(max(int(planned_future_reuse), 0)),
            reuse_probability=self.probability,
            source=f"planned_turns:{planned_future_reuse}",
        )


class ObservedReuseExpectation(ReuseExpectation):
    """Reuse counts actually recorded for a cache key in earlier turns.

    Falls back to `default` for a key never seen before, which is the honest
    answer for a first appearance: a block that has never been reused is not
    evidence that it will be, and it is not evidence that it will not be.
    """

    name = "observed"

    def __init__(
        self,
        observations: Mapping[str, int] | None = None,
        *,
        default: float = 0.0,
        probability: float = 1.0,
        evidence_id: str | None = None,
    ):
        self._observations = dict(observations or {})
        self.default = float(default)
        self.probability = float(probability)
        self.evidence_id = evidence_id

    def record_reuse(self, cache_key: str, count: int = 1) -> None:
        self._observations[cache_key] = self._observations.get(cache_key, 0) + int(count)

    def estimate(
        self, *, cache_key: str, context_field: str, planned_future_reuse: int
    ) -> ReuseEstimate:
        observed = self._observations.get(cache_key)
        if observed is None:
            return ReuseEstimate(
                expected_future_reuses=self.default,
                reuse_probability=self.probability,
                source="observed:unseen",
                evidence_id=self.evidence_id,
            )
        return ReuseEstimate(
            expected_future_reuses=float(observed),
            reuse_probability=self.probability,
            source=f"observed:{observed}",
            evidence_id=self.evidence_id,
        )


class NoReuseExpectation(ReuseExpectation):
    """Expect nothing. The conservative default when no evidence exists.

    A policy configured with this compiles nothing, which is the correct
    behaviour for "we have no idea": it costs the run its reuse benefit and
    costs it no correctness, and it is loudly visible in the decision record
    rather than showing up as an unexplained absence of links.
    """

    name = "none"

    def estimate(
        self, *, cache_key: str, context_field: str, planned_future_reuse: int
    ) -> ReuseEstimate:
        return ReuseEstimate(
            expected_future_reuses=0.0,
            reuse_probability=0.0,
            source="none:no_reuse_evidence",
        )


# --- Residency opportunity cost ---------------------------------------------


class ResidencyCostModel(ABC):
    """What it costs to keep a compiled block's KV resident.

    Separate from the compile decision's other terms because it is the one this
    repository has never measured. It is not zero -- held KV is KV another
    request cannot have, and the cache-pressure work exists precisely because
    that matters -- but no artifact here supports a coefficient, so v1 ships the
    interface and an explicitly disabled implementation instead of a number
    someone would later have to defend.
    """

    name = "abstract"

    @abstractmethod
    def estimate_ms(
        self,
        *,
        model_tokens: int,
        expected_hold_reuses: float,
        memory_pressure: float,
    ) -> float:
        """Opportunity cost in milliseconds for holding this block."""


class DisabledResidencyCost(ResidencyCostModel):
    """Zero, and says so. The decision record carries `disabled`."""

    name = "disabled"

    def estimate_ms(
        self,
        *,
        model_tokens: int,
        expected_hold_reuses: float,
        memory_pressure: float,
    ) -> float:
        return 0.0


class LinearPressureResidencyCost(ResidencyCostModel):
    """A pluggable proxy, off by default and never used without a coefficient.

    Provided so the term has a working implementation to test the plumbing
    against, not because the coefficient is known. `ms_per_model_token_hold` has
    no measured value in this repository; a caller supplying one is asserting
    it, and the decision record names this model so that assertion is visible.
    """

    name = "linear_pressure"

    def __init__(self, *, ms_per_model_token_hold: float):
        if ms_per_model_token_hold < 0:
            raise CompileCostError("ms_per_model_token_hold must be non-negative")
        self.ms_per_model_token_hold = float(ms_per_model_token_hold)

    def estimate_ms(
        self,
        *,
        model_tokens: int,
        expected_hold_reuses: float,
        memory_pressure: float,
    ) -> float:
        return (
            self.ms_per_model_token_hold
            * float(max(int(model_tokens), 0))
            * float(max(expected_hold_reuses, 0.0))
            * float(max(memory_pressure, 0.0))
        )


# --- A. correctness inputs, plus the one legitimate cost-shape input --------


#: How the repair term was obtained, recorded on every decision.
REPAIR_FROM_PLAN = "planned_repair_tokens"
REPAIR_FOLDED_INTO_LINK = "folded_into_link_curve"

#: Where the *window* came from, which is a separate question from where its
#: price came from, and currently the weaker of the two.
#:
#: The chain `preceding_model_tokens -> repair window -> repair ms` has two
#: legs and neither is validated yet. `repair_window.ContextGainRepairWindow`
#: models the first leg from `global_start`, but its own docstring records that
#: `saturation_tokens` "must be supplied explicitly -- there is no defensible
#: default before calibration", and every reduction below the validated K stays
#: behind the exact-output evidence gate, so the policy is inert until a K
#: screen has run. The second leg needs a measured `repair_ms` curve, and no
#: profile is populated. So a caller-supplied window is labelled unvalidated,
#: and no position-dependent cost claim rests on it before profiling.
REPAIR_WINDOW_UNSPECIFIED = "unspecified"
REPAIR_WINDOW_CALLER_SUPPLIED = "caller_supplied_unvalidated"
REPAIR_WINDOW_EVIDENCE_GATED = "repair_window_policy_evidence_gated"


@dataclass(frozen=True)
class CompileCandidate:
    """One block, described by what may legitimately decide its fate.

    The correctness fields are hardware-independent by construction:
    `model_tokens` is an authoritative `ModelTokenCount`, and immutability,
    reuse eligibility and invalidation are properties of the block itself.

    `preceding_model_tokens` is the single position-derived field, and it is
    worth being exact about what it does: **nothing, by itself**. No term of
    `evaluate_compile_economics` reads it. It is carried so that a caller which
    sizes a repair window from block position -- `repair_window.py` owns that --
    can be seen to have done so, and the resulting window arrives separately as
    `planned_repair_model_tokens` with a `repair_window_source` saying where it
    came from.

    The underlying effect is real: a compiled block is prefilled standalone
    from local position 0, so at link time its leading tokens must attend to
    context they never saw, and how much depends on where the block sits. What
    is not yet established is the size of that effect in milliseconds. Until a
    repair curve is measured and the window derivation calibrated, this field
    is provenance, not a price. It is never compared against a threshold, and
    no branch reads a block index.
    """

    sub_context_id: str
    cache_key: str
    context_field: str
    model_tokens: ModelTokenCount
    immutable: bool
    reuse_candidate: bool
    invalidated: bool = False
    preceding_model_tokens: int = 0
    planned_repair_model_tokens: int | None = None
    #: How `planned_repair_model_tokens` was arrived at. Recorded rather than
    #: assumed, because the position -> window derivation is not calibrated.
    repair_window_source: str = REPAIR_WINDOW_UNSPECIFIED
    cache_present: bool = False

    @property
    def token_count(self) -> int:
        return int(self.model_tokens.value)


@dataclass(frozen=True)
class CompileCostBreakdown:
    """Every term of one decision, in milliseconds, with its provenance.

    Serialized onto the decision so a compile choice can be re-derived months
    later without the profile object, only its id.
    """

    model_tokens: int
    expected_future_reuses: float
    expected_reuse_probability: float
    effective_reuses: float
    reuse_source: str
    estimated_prefill_ms: float
    estimated_compile_ms: float
    estimated_link_ms: float
    estimated_repair_ms: float
    estimated_residency_cost_ms: float
    per_reuse_saved_ms: float
    expected_saved_ms: float
    total_cost_ms: float
    net_benefit_ms: float
    repair_model_tokens: int
    repair_source: str
    residency_cost_model: str
    #: Provenance of the *window*, not of its price. See the constants above:
    #: `caller_supplied_unvalidated` means a number was supplied and its
    #: derivation from block position has not been calibrated.
    repair_window_source: str = REPAIR_WINDOW_UNSPECIFIED
    compile_cost_sunk: bool = False

    @property
    def is_profitable(self) -> bool:
        """Strictly positive. Break-even is not a reason to change the plan."""

        return self.net_benefit_ms > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_token_count": int(self.model_tokens),
            "expected_future_reuses": float(self.expected_future_reuses),
            "expected_reuse_probability": float(self.expected_reuse_probability),
            "effective_reuses": float(self.effective_reuses),
            "reuse_source": self.reuse_source,
            "estimated_prefill_ms": float(self.estimated_prefill_ms),
            "estimated_compile_ms": float(self.estimated_compile_ms),
            "estimated_link_ms": float(self.estimated_link_ms),
            "estimated_repair_ms": float(self.estimated_repair_ms),
            "estimated_residency_cost_ms": float(self.estimated_residency_cost_ms),
            "per_reuse_saved_ms": float(self.per_reuse_saved_ms),
            "expected_saved_ms": float(self.expected_saved_ms),
            "total_cost_ms": float(self.total_cost_ms),
            "net_benefit_ms": float(self.net_benefit_ms),
            "repair_model_token_count": int(self.repair_model_tokens),
            "repair_source": self.repair_source,
            "repair_window_source": self.repair_window_source,
            "residency_cost_model": self.residency_cost_model,
            "compile_cost_sunk": bool(self.compile_cost_sunk),
        }


def evaluate_compile_economics(
    candidate: CompileCandidate,
    *,
    profile: CompileCostProfile,
    reuse_expectation: ReuseExpectation,
    residency_cost_model: ResidencyCostModel,
    planned_future_reuse: int,
    memory_pressure: float = 0.0,
) -> CompileCostBreakdown:
    """The arithmetic, and nothing else.

    ::

        per_reuse_saved_ms = max(prefill_ms - link_ms - repair_ms, 0)
        expected_saved_ms  = effective_reuses * per_reuse_saved_ms
        total_cost_ms      = compile_ms + residency_ms
        net_benefit_ms     = expected_saved_ms - total_cost_ms

    Compile iff `net_benefit_ms > 0`. Strictly, so a block that exactly breaks
    even stays on the native path: a plan change has to pay for itself before
    it is worth making, and equality is not payment.

    Compile cost is charged once and link and repair on every reuse, which is
    the whole structure of the trade and the thing a fixed token floor could
    not express. When the block is already in the catalog that compile cost is
    sunk and charged at zero -- the question then is only whether linking it
    beats prefilling it -- and the breakdown records that it was sunk rather
    than reporting a compile that costs nothing.
    """

    tokens = candidate.token_count
    reuse = reuse_expectation.estimate(
        cache_key=candidate.cache_key,
        context_field=candidate.context_field,
        planned_future_reuse=planned_future_reuse,
    )

    prefill_ms = profile.estimate_prefill_ms(tokens)
    link_ms = profile.estimate_link_ms(tokens)
    compile_ms = 0.0 if candidate.cache_present else profile.estimate_compile_ms(tokens)

    if profile.repair_is_folded_into_link:
        repair_ms = 0.0
        repair_tokens = 0
        repair_source = REPAIR_FOLDED_INTO_LINK
        window_source = REPAIR_FOLDED_INTO_LINK
    else:
        if candidate.planned_repair_model_tokens is None:
            raise CompileCostUnavailableError(
                f"profile {profile.profile_id} costs sparse repair separately, "
                f"but block {candidate.sub_context_id} has no planned repair "
                "window. Supply planned_repair_model_tokens, or use a profile "
                "whose link curve already contains repair."
            )
        repair_tokens = int(candidate.planned_repair_model_tokens)
        repair_ms = profile.estimate_repair_ms(repair_tokens)
        repair_source = REPAIR_FROM_PLAN
        window_source = candidate.repair_window_source

    residency_ms = residency_cost_model.estimate_ms(
        model_tokens=tokens,
        expected_hold_reuses=reuse.effective_reuses,
        memory_pressure=memory_pressure,
    )

    per_reuse_saved_ms = max(prefill_ms - link_ms - repair_ms, 0.0)
    expected_saved_ms = reuse.effective_reuses * per_reuse_saved_ms
    total_cost_ms = compile_ms + residency_ms

    return CompileCostBreakdown(
        model_tokens=tokens,
        expected_future_reuses=reuse.expected_future_reuses,
        expected_reuse_probability=reuse.reuse_probability,
        effective_reuses=reuse.effective_reuses,
        reuse_source=reuse.source,
        estimated_prefill_ms=prefill_ms,
        estimated_compile_ms=compile_ms,
        estimated_link_ms=link_ms,
        estimated_repair_ms=repair_ms,
        estimated_residency_cost_ms=residency_ms,
        per_reuse_saved_ms=per_reuse_saved_ms,
        expected_saved_ms=expected_saved_ms,
        total_cost_ms=total_cost_ms,
        net_benefit_ms=expected_saved_ms - total_cost_ms,
        repair_model_tokens=repair_tokens,
        repair_source=repair_source,
        repair_window_source=window_source,
        residency_cost_model=residency_cost_model.name,
        compile_cost_sunk=candidate.cache_present,
    )


def build_curve(points: Sequence[Sequence[float]], *, name: str, **kwargs: Any) -> LatencyCurve:
    """Small helper for profile builders and tests."""

    return LatencyCurve(
        name=name,
        points=tuple((int(x), float(y)) for x, y in points),
        **kwargs,
    )
