"""Online eligibility for reconstruct-and-promote of an in-order execution group.

EXPERIMENTAL governance, off by default, not wired into the production planner.

A `GroupRepairQualityProfile` turns a characterized domain into a contract a
planner can check *before* execution: is this execution group inside the
domain where `PREFIX_ABSORB_REFRESH_1` reconstruction and promotion was
measured correct? If yes, the experimental path may run; if not, the request
is still served, through `FULL_GROUP_RECOMPILE`.

Two things are kept apart on purpose:

* **Offline certification** is why a profile exists. It used native references,
  margins and oracle screens, and it lives in an evidence manifest that nothing
  here reads.
* **Online eligibility** is what a request can show before it runs: token
  counts, member order, residency, the declared reconstruction policy and the
  deployment's compatibility identity. `GroupRequestMetadata` has no field for
  an answer, an oracle, a margin or a logit, and `evaluate_repair_profile`
  cannot be handed one.

This is a sibling of `execution_governance.RepairQualityProfile`, not a
replacement: that one certifies one exact window vector for formal execution;
this one certifies a reconstruction *method* over a structural domain. Neither
owns physical layout -- the only numbers here are the policy-level budget
(member 0's leading `K0`, later members in full); slots, positions, ranges and
repair indices stay runtime-owned.

Layering::

    semantic blocks -> P3 execution grouping -> this profile -> route -> runtime

P3 decides which blocks form a group. The profile decides whether *that* group
may be reconstructed from its members. The runtime reconstructs, promotes,
links, repairs and decodes.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

GROUP_PROFILE_SCHEMA_VERSION = "group-repair-quality-profile-v1"
PROFILE_TRACE_SCHEMA_VERSION = 1

#: Request fields the profile must never see. Offline certification used some
#: of these; online eligibility must not be able to.
FORBIDDEN_ONLINE_FIELDS = frozenset(
    {
        "answer",
        "expected",
        "expected_answer",
        "target_code",
        "oracle_id",
        "native_answer",
        "native_margin",
        "margin",
        "logits",
        "logprobs",
        "top_logprobs",
        "correct",
        "matches_native",
    }
)

#: Physical layout the runtime owns. A profile carrying any of these would move
#: layout decisions back onto the host.
RUNTIME_OWNED_FIELDS = frozenset(
    {
        "imp_indices",
        "kvlink",
        "recompute_positions",
        "kv_slots",
        "block_ranges",
        "global_positions",
    }
)

#: Provenance never belongs in a profile's identity.
PROVENANCE_FIELDS = frozenset({"request_id", "task_id", "agent_id", "session_id"})


class ReasonCode(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    GROUP_ALREADY_PRESENT = "GROUP_ALREADY_PRESENT"
    PROFILE_VERSION_MISMATCH = "PROFILE_VERSION_MISMATCH"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    TOKENIZER_MISMATCH = "TOKENIZER_MISMATCH"
    COMPATIBILITY_MISMATCH = "COMPATIBILITY_MISMATCH"
    TOKEN_UNIT_NOT_MODEL = "TOKEN_UNIT_NOT_MODEL"
    REORDERED = "REORDERED"
    UNSUPPORTED_BLOCK_COUNT = "UNSUPPORTED_BLOCK_COUNT"
    UNSUPPORTED_GROUP_SIZE = "UNSUPPORTED_GROUP_SIZE"
    OUT_OF_CONTEXT_RANGE = "OUT_OF_CONTEXT_RANGE"
    UNSUPPORTED_LAYOUT = "UNSUPPORTED_LAYOUT"
    WRONG_RECONSTRUCTION_POLICY = "WRONG_RECONSTRUCTION_POLICY"
    MEMBERS_NOT_RESIDENT = "MEMBERS_NOT_RESIDENT"
    PARTIAL_MEMBER_RESIDENCY = "PARTIAL_MEMBER_RESIDENCY"


class Route(str, Enum):
    PREFIX_ABSORB_RECONSTRUCT_AND_PROMOTE = "PREFIX_ABSORB_RECONSTRUCT_AND_PROMOTE"
    GROUP_CACHE_HIT = "GROUP_CACHE_HIT"
    FULL_GROUP_RECOMPILE = "FULL_GROUP_RECOMPILE"


# --- the profile -------------------------------------------------------------


@dataclass(frozen=True)
class MemberShapeClassifier:
    """The member-length shapes that were measured, and nothing else.

    The characterization's layout names (balanced / front-heavy / back-heavy)
    are experiment labels; a live request has only member lengths. This
    classifies from lengths alone, with bands taken from the measured shapes:
    `balanced` if every member is within `balanced_max_ratio` of every other;
    `front_heavy` / `back_heavy` if the largest member is first / last, is
    `heavy_ratio_min..heavy_ratio_max` times the mean of the others, and the
    others are within `others_max_ratio` of each other. Anything else --
    middle-heavy, two heavy members, a 1.5x skew -- was never measured.
    """

    balanced_max_ratio: float
    heavy_ratio_min: float
    heavy_ratio_max: float
    others_max_ratio: float

    def classify(self, lengths: Sequence[int]) -> str | None:
        lengths = [int(v) for v in lengths]
        if len(lengths) < 2 or min(lengths) <= 0:
            return None
        if max(lengths) / min(lengths) <= self.balanced_max_ratio:
            return "balanced"
        heavy = lengths.index(max(lengths))
        others = lengths[:heavy] + lengths[heavy + 1 :]
        if max(others) / min(others) > self.others_max_ratio:
            return None
        ratio = lengths[heavy] / (sum(others) / len(others))
        if not self.heavy_ratio_min <= ratio <= self.heavy_ratio_max:
            return None
        if heavy == 0:
            return "front_heavy"
        if heavy == len(lengths) - 1:
            return "back_heavy"
        return None


@dataclass(frozen=True)
class GroupRepairQualityProfile:
    profile_id: str
    profile_version: str
    schema_version: str
    method: str
    supported_ordering: str
    #: Tested context buckets, in model tokens of the whole request
    #: (chat prefix + group + mutable suffix). No interpolation between them.
    context_token_buckets: tuple[tuple[int, int], ...]
    semantic_block_count_min: int
    semantic_block_count_max: int
    execution_group_member_min: int
    execution_group_member_max: int
    reconstruction_policy: str
    member0_prefix_refresh_tokens: int
    later_member_refresh: str
    requires_all_members_resident: bool
    requires_group_absent_for_reconstruction: bool
    promotion: str
    fallback_policy: str
    supported_member_shapes: tuple[str, ...]
    member_shape_classifier: MemberShapeClassifier
    #: Every key must match the request's declared deployment identity exactly.
    compatibility: Mapping[str, str]
    #: Pointer to the offline evidence, by path and content hash. Informational:
    #: `evaluate_repair_profile` never reads it.
    certification: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        names = set(asdict(self)) | set(self.compatibility)
        leaked = names & (RUNTIME_OWNED_FIELDS | PROVENANCE_FIELDS | FORBIDDEN_ONLINE_FIELDS)
        if leaked:
            raise ValueError("profile carries forbidden fields: %s" % sorted(leaked))
        if self.supported_ordering != "in_order_only":
            raise ValueError("v1 profiles are in-order only")
        if self.later_member_refresh != "full":
            raise ValueError("later members must be refreshed in full")
        if type(self.member0_prefix_refresh_tokens) is not int or self.member0_prefix_refresh_tokens <= 0:
            raise ValueError("member0_prefix_refresh_tokens must be a positive integer")
        if self.fallback_policy != Route.FULL_GROUP_RECOMPILE.value:
            raise ValueError("the fallback must be FULL_GROUP_RECOMPILE")
        for low, high in self.context_token_buckets:
            if not 0 < low <= high:
                raise ValueError("invalid context bucket")
        if not 2 <= self.execution_group_member_min <= self.execution_group_member_max:
            raise ValueError("a reconstructed group has at least two members")

    # --- identity ---------------------------------------------------------

    def contract_dict(self) -> dict[str, Any]:
        """Everything that decides eligibility -- the contract, not the evidence."""

        out = asdict(self)
        out.pop("certification")
        out["context_token_buckets"] = [list(b) for b in self.context_token_buckets]
        out["supported_member_shapes"] = list(self.supported_member_shapes)
        out["compatibility"] = dict(sorted(self.compatibility.items()))
        return out

    @property
    def contract_digest(self) -> str:
        payload = json.dumps(self.contract_dict(), sort_keys=True, separators=(",", ":"))
        return "grqp-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        out = self.contract_dict()
        out["certification"] = dict(self.certification)
        out["contract_digest"] = self.contract_digest
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroupRepairQualityProfile":
        data = dict(data)
        declared_digest = data.pop("contract_digest", None)
        if data.get("schema_version") != GROUP_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported group repair profile schema")
        profile = cls(
            profile_id=str(data["profile_id"]),
            profile_version=str(data["profile_version"]),
            schema_version=str(data["schema_version"]),
            method=str(data["method"]),
            supported_ordering=str(data["supported_ordering"]),
            context_token_buckets=tuple(
                (int(low), int(high)) for low, high in data["context_token_buckets"]
            ),
            semantic_block_count_min=int(data["semantic_block_count_min"]),
            semantic_block_count_max=int(data["semantic_block_count_max"]),
            execution_group_member_min=int(data["execution_group_member_min"]),
            execution_group_member_max=int(data["execution_group_member_max"]),
            reconstruction_policy=str(data["reconstruction_policy"]),
            member0_prefix_refresh_tokens=data["member0_prefix_refresh_tokens"],
            later_member_refresh=str(data["later_member_refresh"]),
            requires_all_members_resident=bool(data["requires_all_members_resident"]),
            requires_group_absent_for_reconstruction=bool(
                data["requires_group_absent_for_reconstruction"]
            ),
            promotion=str(data["promotion"]),
            fallback_policy=str(data["fallback_policy"]),
            supported_member_shapes=tuple(data["supported_member_shapes"]),
            member_shape_classifier=MemberShapeClassifier(**data["member_shape_classifier"]),
            compatibility={str(k): str(v) for k, v in data["compatibility"].items()},
            certification=dict(data.get("certification") or {}),
        )
        if declared_digest is not None and declared_digest != profile.contract_digest:
            raise ValueError("profile contract digest does not match its contents")
        return profile


def load_profile(path: str | Path) -> GroupRepairQualityProfile:
    return GroupRepairQualityProfile.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


# --- the request ---------------------------------------------------------------


@dataclass(frozen=True)
class GroupRequestMetadata:
    """What a planner can know about an execution group before it runs.

    `request_member_ids` is the members' order in the request;
    `group_member_ids` is the order P3 grouped them in. Residency and group
    presence are the host's cache bookkeeping; the runtime stays authoritative
    and fails closed if they are wrong.
    """

    context_tokens: int
    token_unit: str
    semantic_block_count: int
    request_member_ids: tuple[str, ...]
    group_member_ids: tuple[str, ...]
    member_token_counts: tuple[int, ...]
    members_resident: tuple[bool, ...]
    group_present: bool
    reconstruction_policy: str
    member0_prefix_refresh_tokens: int
    later_member_refresh: str
    compatibility: Mapping[str, str]
    requested_profile_id: str | None = None
    requested_profile_version: str | None = None

    def __post_init__(self):
        leaked = set(asdict(self)) & FORBIDDEN_ONLINE_FIELDS
        if leaked:  # pragma: no cover - the dataclass has no such field
            raise ValueError("online metadata carries forbidden fields")
        n = len(self.group_member_ids)
        if len(self.member_token_counts) != n or len(self.members_resident) != n:
            raise ValueError("member counts and residency must match the group")


@dataclass(frozen=True)
class ProfileDecision:
    eligible: bool
    reason_code: str
    profile_id: str
    profile_version: str
    contract_digest: str
    features: Mapping[str, Any]
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reason_code: str
    promote: bool
    reconstruction_windows: tuple[int, ...] | None
    profile_decision: ProfileDecision

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reconstruction_windows"] = (
            list(self.reconstruction_windows) if self.reconstruction_windows else None
        )
        return out


def request_features(request: GroupRequestMetadata, profile: GroupRepairQualityProfile) -> dict:
    lengths = [int(v) for v in request.member_token_counts]
    group_tokens = sum(lengths)
    return {
        "context_tokens": int(request.context_tokens),
        "token_unit": request.token_unit,
        "semantic_block_count": int(request.semantic_block_count),
        "execution_group_member_count": len(request.group_member_ids),
        "ordering": (
            "in_order"
            if tuple(request.request_member_ids) == tuple(request.group_member_ids)
            else "reordered"
        ),
        "member_token_counts": lengths,
        "member0_tokens": lengths[0] if lengths else 0,
        "member0_ratio": round(lengths[0] / group_tokens, 4) if group_tokens else None,
        "member_shape": profile.member_shape_classifier.classify(lengths),
        "all_members_resident": bool(request.members_resident) and all(request.members_resident),
        "resident_member_count": sum(1 for v in request.members_resident if v),
        "group_present": bool(request.group_present),
        "reconstruction_policy": request.reconstruction_policy,
        "member0_prefix_refresh_tokens": request.member0_prefix_refresh_tokens,
        "later_member_refresh": request.later_member_refresh,
    }


def evaluate_repair_profile(
    request: GroupRequestMetadata, profile: GroupRepairQualityProfile
) -> ProfileDecision:
    """Deterministic, structural, answer-free. The first failing check names it."""

    features = request_features(request, profile)

    def decide(reason: ReasonCode, details: str = "") -> ProfileDecision:
        return ProfileDecision(
            eligible=reason is ReasonCode.ELIGIBLE,
            reason_code=reason.value,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            contract_digest=profile.contract_digest,
            features=features,
            details=details,
        )

    # A group that already exists is served from the cache. The profile governs
    # whether to *build* one, not whether to reuse one.
    if request.group_present:
        return decide(ReasonCode.GROUP_ALREADY_PRESENT)

    if (request.requested_profile_id not in (None, profile.profile_id)
            or request.requested_profile_version not in (None, profile.profile_version)):
        return decide(ReasonCode.PROFILE_VERSION_MISMATCH)
    declared = dict(request.compatibility)
    if declared.get("model_id") != profile.compatibility.get("model_id"):
        return decide(ReasonCode.MODEL_MISMATCH)
    if declared.get("tokenizer_fingerprint") != profile.compatibility.get("tokenizer_fingerprint"):
        return decide(ReasonCode.TOKENIZER_MISMATCH)
    mismatched = sorted(
        key for key, value in profile.compatibility.items() if declared.get(key) != value
    )
    if mismatched:
        return decide(ReasonCode.COMPATIBILITY_MISMATCH, ",".join(mismatched))
    if request.token_unit != "model":
        return decide(ReasonCode.TOKEN_UNIT_NOT_MODEL)

    if features["ordering"] != "in_order":
        return decide(ReasonCode.REORDERED)
    if not (profile.semantic_block_count_min
            <= features["semantic_block_count"]
            <= profile.semantic_block_count_max):
        return decide(ReasonCode.UNSUPPORTED_BLOCK_COUNT)
    if not (profile.execution_group_member_min
            <= features["execution_group_member_count"]
            <= profile.execution_group_member_max):
        return decide(ReasonCode.UNSUPPORTED_GROUP_SIZE)
    if not any(low <= features["context_tokens"] <= high
               for low, high in profile.context_token_buckets):
        return decide(ReasonCode.OUT_OF_CONTEXT_RANGE)
    if features["member_shape"] not in profile.supported_member_shapes:
        return decide(ReasonCode.UNSUPPORTED_LAYOUT)
    if (request.reconstruction_policy != profile.reconstruction_policy
            or request.member0_prefix_refresh_tokens != profile.member0_prefix_refresh_tokens
            or request.later_member_refresh != profile.later_member_refresh):
        return decide(ReasonCode.WRONG_RECONSTRUCTION_POLICY)

    resident = features["resident_member_count"]
    if resident == 0:
        return decide(ReasonCode.MEMBERS_NOT_RESIDENT)
    if resident < features["execution_group_member_count"]:
        return decide(ReasonCode.PARTIAL_MEMBER_RESIDENCY)
    return decide(ReasonCode.ELIGIBLE)


def reconstruction_windows(member_token_counts: Sequence[int], profile) -> tuple[int, ...]:
    """The policy-level budget only: `[min(K0, L0), L1, ..., L(N-1)]`.

    A per-member count, not a position: which rows it becomes is the runtime's.
    """

    lengths = [int(v) for v in member_token_counts]
    return (min(profile.member0_prefix_refresh_tokens, lengths[0]),) + tuple(lengths[1:])


def route_execution_group(
    request: GroupRequestMetadata,
    profile: GroupRepairQualityProfile,
    *,
    promotion_feature_enabled: bool,
) -> RoutingDecision:
    """Eligible *and* the experimental feature enabled, or the safe path.

    An ineligible request is never refused: it is compiled as a group, which is
    the path the characterization used as its correctness baseline. Profile
    eligibility cannot turn promotion on; the deployment flag has to say yes
    independently.
    """

    decision = evaluate_repair_profile(request, profile)
    if decision.reason_code == ReasonCode.GROUP_ALREADY_PRESENT.value:
        return RoutingDecision(Route.GROUP_CACHE_HIT.value, decision.reason_code,
                               False, None, decision)
    if decision.eligible and promotion_feature_enabled:
        return RoutingDecision(
            Route.PREFIX_ABSORB_RECONSTRUCT_AND_PROMOTE.value, decision.reason_code, True,
            reconstruction_windows(request.member_token_counts, profile), decision,
        )
    reason = decision.reason_code if not decision.eligible else "PROMOTION_FEATURE_DISABLED"
    return RoutingDecision(Route.FULL_GROUP_RECOMPILE.value, reason, False, None, decision)


# --- observability -----------------------------------------------------------


def profile_trace_rows(routing: RoutingDecision, **context: Any) -> list[dict[str, Any]]:
    """Trace rows for one routing decision. No answer, logit or margin fields."""

    leaked = set(context) & FORBIDDEN_ONLINE_FIELDS
    if leaked:
        raise ValueError("trace context carries forbidden fields: %s" % sorted(leaked))
    decision = routing.profile_decision
    features = decision.features
    base = {
        "schema_version": PROFILE_TRACE_SCHEMA_VERSION,
        "timestamp_unix_s": time.time(),
        "profile_id": decision.profile_id,
        "profile_version": decision.profile_version,
        "contract_digest": decision.contract_digest,
        **context,
    }
    checked = {
        **base,
        "event": "repair_profile_checked",
        "eligible": decision.eligible,
        "reason": decision.reason_code,
        "context_tokens": features["context_tokens"],
        "block_count": features["semantic_block_count"],
        "group_member_count": features["execution_group_member_count"],
        "member_shape": features["member_shape"],
        "ordering": features["ordering"],
        "resident_member_count": features["resident_member_count"],
        "group_present": features["group_present"],
    }
    outcome = {
        **base,
        "event": (
            "repair_profile_eligible"
            if routing.route == Route.PREFIX_ABSORB_RECONSTRUCT_AND_PROMOTE.value
            else "repair_profile_fallback"
        ),
        "route": routing.route,
        "reason": routing.reason_code,
    }
    return [checked, outcome]


def append_profile_trace(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
