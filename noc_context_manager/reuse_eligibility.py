"""Gate 1: is this *cached KV state* permitted for reuse?

A semantic match is not a reuse decision. The same block B has different transformer states depending on
what preceded it when it was computed -- KV(B | C1) is not KV(B | C2) -- and four studies
(`results/contextual_*`) tried and failed to find a cheap signal that tells a safe foreign state from an
unsafe one. So eligibility is not a classifier here. It is a conservative rule over *recorded provenance*:

    positive evidence the state is the canonical one   -> EXACT_SAFE        (may continue toward Link)
    no such evidence                                   -> RECONSTRUCT_REQUIRED (computed natively)
    a foreign contextual state                         -> EXPERIMENTAL_FOREIGN (explicit flag only)

and, for blocks that were never offered for reuse at all, NO_REUSE_CANDIDATE -- so that "not a candidate"
is never counted as "a candidate we refused".

Three concepts are kept apart, and each has its own field on the planning record:

* **semantic candidate** -- selection offered the block for reuse (settled, immutable, not invalidated).
  A property of the *content*.
* **cache availability** -- the host catalog holds a non-stale entry under the block's key. A property of
  host bookkeeping; runtime residency is only known when the Link runs.
* **reuse eligibility** -- the state behind that key may be reused on the canonical path. A property of
  the *state*.

The canonical state is a standalone Compile: the block's own tokens, prefilled from local position 0 with
no preceding context. It is context-free by construction, and its use under a request's context is what
LegoLink's re-rotation and sparse repair were validated for. EXACT_SAFE means "legal for that canonical
path", not "proven to reproduce native output"; the latter is the repair-quality gate's claim.

What this module deliberately does not read: LLM semantic role or topic, titles, similarity scores, or any
answer-side signal. None of them is evidence about a KV state.

Ownership: host-side logical policy only. No KV slot, position, repair index or residency handle is read
or produced here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

from .compile_policy import CompileAction, CompileDecision
from .subcontext_store import SemanticSubContext


REUSE_ELIGIBILITY_POLICY_VERSION = "reuse-eligibility-v1"


class ReuseEligibility(str, Enum):
    #: Positive evidence that the specific state is legal for the canonical reuse path.
    EXACT_SAFE = "exact_safe"
    #: Content may match, but reuse of this state is not established; the block is computed natively.
    RECONSTRUCT_REQUIRED = "reconstruct_required"
    #: A foreign contextual state. Only explicit experiment machinery may exercise it; default off.
    EXPERIMENTAL_FOREIGN = "experimental_foreign"
    #: Selection never offered the block for reuse (mutable, invalidated, or not a reuse candidate), so no
    #: state question arises. Kept separate so these blocks never inflate the RECONSTRUCT_REQUIRED count.
    NO_REUSE_CANDIDATE = "no_reuse_candidate"

    @property
    def permits_canonical_link(self) -> bool:
        return self is ReuseEligibility.EXACT_SAFE


class StateOrigin(str, Enum):
    """How the cached state was produced. Recorded by whoever registers the state."""

    #: The host's Compile request: block tokens alone, local position 0. The only state the host creates.
    STANDALONE_COMPILE = "standalone_compile"
    #: KV(B | C) computed under some causal context C: a promoted reconstructed group member, an in-context
    #: prefill harvested into the block cache, a context-conditioned variant.
    CONTEXTUAL_FOREIGN = "contextual_foreign"
    UNKNOWN = "unknown"


class EligibilityReason(str, Enum):
    """Stable machine-readable reason codes. Artifacts key on these strings; do not rename."""

    EXACT_SAFE_STATE = "exact_safe_state"
    CANONICAL_COMPILE_ON_MISS = "canonical_standalone_compile_on_miss"
    NOT_A_SEMANTIC_CANDIDATE = "not_a_semantic_candidate"
    CACHED_STATE_IDENTITY_MISMATCH = "cached_state_identity_mismatch"
    LEGACY_IDENTITY_UNVERIFIED = "legacy_identity_unverified"
    STATE_ORIGIN_UNKNOWN = "state_origin_unknown"
    CONTEXTUAL_STATE_NOT_PROVEN = "contextual_state_not_proven"
    EXPERIMENTAL_FOREIGN_DISABLED = "experimental_foreign_disabled"
    EXPERIMENTAL_FOREIGN_ENABLED = "experimental_foreign_enabled"


class CacheAvailability(str, Enum):
    """What the host catalog says. Not runtime residency, which only the Link observes."""

    HOST_CATALOG_ENTRY = "host_catalog_entry"
    NO_CACHED_STATE = "no_cached_state"
    STALE_CACHED_STATE = "stale_cached_state"


class IdentityBasis(str, Enum):
    """What the block's content identity is a function of (`semantic_identity` prefix)."""

    TOKEN_IDS = "token_ids"
    TEXT = "text"
    #: A row written before the content-identity split: identity is the provenance-scoped block id, so two
    #: different contents from the same span can share a key.
    LEGACY_PROVENANCE = "legacy_provenance"


#: Human-readable decline reasons for the effective decision. They are legitimate declines, so they are
#: on `scheduler.LEGITIMATE_KEEP_MUTABLE_REASONS`; the machine-readable code travels on the record.
RECONSTRUCT_REQUIRED_DECLINE = (
    "reuse of the cached state is not established; block is computed natively"
)
EXPERIMENTAL_FOREIGN_DISABLED_DECLINE = (
    "foreign contextual state is experiment-only and disabled; block is computed natively"
)
ELIGIBILITY_DECLINE_REASONS = frozenset(
    {RECONSTRUCT_REQUIRED_DECLINE, EXPERIMENTAL_FOREIGN_DISABLED_DECLINE}
)

#: Decisions that ask for the block's state to be placed by a Link.
_LINK_ACTIONS = frozenset({CompileAction.COMPILE, CompileAction.SUMMARIZE_THEN_COMPILE})


def identity_basis(value: SemanticSubContext) -> IdentityBasis:
    identity = value.semantic_identity or ""
    if identity.startswith("si-tok-"):
        return IdentityBasis.TOKEN_IDS
    if identity.startswith("si-txt-"):
        return IdentityBasis.TEXT
    return IdentityBasis.LEGACY_PROVENANCE


def is_semantic_candidate(value: SemanticSubContext) -> bool:
    """Selection's own filter (`context_selection`): settled, immutable, live."""

    return bool(value.reuse_candidate and not value.is_mutable and not value.invalidated)


@dataclass(frozen=True)
class ReuseEligibilityAssessment:
    """One block's Gate-1 verdict with the evidence it rests on."""

    sub_context_id: str
    cache_key: str
    semantic_candidate: bool
    cache_available: bool
    cache_availability: CacheAvailability
    eligibility: ReuseEligibility
    reason: EligibilityReason
    state_origin: str
    identity_basis: IdentityBasis
    experimental_foreign_enabled: bool = False
    detail: str = ""

    @property
    def permits_link(self) -> bool:
        """May this block's state continue toward layout feasibility as a Link?

        EXACT_SAFE always; EXPERIMENTAL_FOREIGN only when the experiment flag was set explicitly.
        """

        if self.eligibility is ReuseEligibility.EXACT_SAFE:
            return True
        return (
            self.eligibility is ReuseEligibility.EXPERIMENTAL_FOREIGN
            and self.experimental_foreign_enabled
        )

    @property
    def experimental(self) -> bool:
        return self.eligibility is ReuseEligibility.EXPERIMENTAL_FOREIGN and self.permits_link

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_context_id": self.sub_context_id,
            "cache_key": self.cache_key,
            "semantic_candidate": self.semantic_candidate,
            "cache_available": self.cache_available,
            "cache_availability": self.cache_availability.value,
            "reuse_eligibility": self.eligibility.value,
            "eligibility_reason": self.reason.value,
            "state_origin": self.state_origin,
            "identity_basis": self.identity_basis.value,
            "experimental_foreign_enabled": self.experimental_foreign_enabled,
            "eligibility_detail": self.detail,
        }


@dataclass(frozen=True)
class EligibilityDecision:
    """The provisional (compile-policy) decision, the Gate-1 verdict, and what enters layout."""

    assessment: ReuseEligibilityAssessment
    provisional_decision: CompileDecision
    gated_decision: CompileDecision

    @property
    def sub_context_id(self) -> str:
        return self.provisional_decision.sub_context_id

    @property
    def eligibility_adjusted(self) -> bool:
        return self.gated_decision.action is not self.provisional_decision.action


class ReuseEligibilityPolicy:
    """Conservative Gate 1. No classifier, no similarity, no learned compatibility.

    `allow_experimental_foreign` is the only switch, it is off by default, and turning it on marks every
    plan that uses it as non-canonical (see `NOCRequestExecutionPlan.experimental_links`).
    """

    version = REUSE_ELIGIBILITY_POLICY_VERSION

    def __init__(self, *, allow_experimental_foreign: bool = False) -> None:
        self.allow_experimental_foreign = bool(allow_experimental_foreign)

    def assess(
        self,
        value: SemanticSubContext,
        catalog_entry: Mapping[str, Any] | None,
    ) -> ReuseEligibilityAssessment:
        """`catalog_entry` is the host catalog's metadata for `value.cache_key` (None if absent)."""

        basis = identity_basis(value)
        stale = bool(catalog_entry and catalog_entry.get("stale"))
        available = catalog_entry is not None and not stale
        availability = (
            CacheAvailability.HOST_CATALOG_ENTRY
            if available
            else CacheAvailability.STALE_CACHED_STATE
            if stale
            else CacheAvailability.NO_CACHED_STATE
        )
        origin = (
            str(catalog_entry.get("state_origin") or StateOrigin.UNKNOWN.value)
            if available
            else StateOrigin.STANDALONE_COMPILE.value
        )

        def verdict(
            eligibility: ReuseEligibility, reason: EligibilityReason, detail: str = ""
        ) -> ReuseEligibilityAssessment:
            return ReuseEligibilityAssessment(
                sub_context_id=value.sub_context_id,
                cache_key=value.cache_key,
                semantic_candidate=is_semantic_candidate(value),
                cache_available=available,
                cache_availability=availability,
                eligibility=eligibility,
                reason=reason,
                state_origin=origin,
                identity_basis=basis,
                experimental_foreign_enabled=self.allow_experimental_foreign,
                detail=detail,
            )

        if not is_semantic_candidate(value):
            return verdict(
                ReuseEligibility.NO_REUSE_CANDIDATE, EligibilityReason.NOT_A_SEMANTIC_CANDIDATE
            )
        if not available:
            # Nothing cached (or only a stale entry the Compile will overwrite): the state the Link will
            # use is the standalone Compile this request issues, i.e. the canonical state by construction.
            return verdict(ReuseEligibility.EXACT_SAFE, EligibilityReason.CANONICAL_COMPILE_ON_MISS)

        if origin == StateOrigin.CONTEXTUAL_FOREIGN.value:
            return verdict(
                ReuseEligibility.EXPERIMENTAL_FOREIGN,
                EligibilityReason.EXPERIMENTAL_FOREIGN_ENABLED
                if self.allow_experimental_foreign
                else EligibilityReason.EXPERIMENTAL_FOREIGN_DISABLED,
                str(catalog_entry.get("source_context_identity") or ""),
            )
        if origin != StateOrigin.STANDALONE_COMPILE.value:
            return verdict(
                ReuseEligibility.RECONSTRUCT_REQUIRED,
                EligibilityReason.STATE_ORIGIN_UNKNOWN
                if origin == StateOrigin.UNKNOWN.value
                else EligibilityReason.CONTEXTUAL_STATE_NOT_PROVEN,
                origin,
            )

        # A standalone state: it is the canonical one only if it is a state *of this content*. The key is
        # content-addressed for identity v2, so these checks can only fail on a legacy provenance-scoped key
        # or a catalog written with conflicting metadata -- which is exactly when they must.
        mismatched = [
            name
            for name, expected in (
                ("content_hash", value.content_hash),
                ("tokenizer_fingerprint", value.tokenizer_fingerprint),
                ("model_fingerprint", value.model_fingerprint),
                ("segmentation_policy_version", value.policy_version),
                ("recompute_policy_version", value.recompute_policy_version),
            )
            if catalog_entry.get(name) not in (None, expected)
        ]
        if mismatched:
            return verdict(
                ReuseEligibility.RECONSTRUCT_REQUIRED,
                EligibilityReason.CACHED_STATE_IDENTITY_MISMATCH,
                ",".join(mismatched),
            )
        if basis is IdentityBasis.LEGACY_PROVENANCE and catalog_entry.get("content_hash") is None:
            return verdict(
                ReuseEligibility.RECONSTRUCT_REQUIRED, EligibilityReason.LEGACY_IDENTITY_UNVERIFIED
            )
        return verdict(ReuseEligibility.EXACT_SAFE, EligibilityReason.EXACT_SAFE_STATE)

    def gate(
        self, provisional: CompileDecision, assessment: ReuseEligibilityAssessment
    ) -> EligibilityDecision:
        """Decline a requested Link whose state is not permitted. Never promotes anything."""

        if provisional.action not in _LINK_ACTIONS or assessment.permits_link:
            return EligibilityDecision(assessment, provisional, provisional)
        reason = (
            EXPERIMENTAL_FOREIGN_DISABLED_DECLINE
            if assessment.eligibility is ReuseEligibility.EXPERIMENTAL_FOREIGN
            else RECONSTRUCT_REQUIRED_DECLINE
        )
        gated = replace(
            provisional,
            action=CompileAction.KEEP_MUTABLE,
            reason=reason,
            estimated_saved_prefill_tokens=0,
        )
        return EligibilityDecision(assessment, provisional, gated)


__all__ = [
    "ELIGIBILITY_DECLINE_REASONS",
    "EXPERIMENTAL_FOREIGN_DISABLED_DECLINE",
    "RECONSTRUCT_REQUIRED_DECLINE",
    "REUSE_ELIGIBILITY_POLICY_VERSION",
    "CacheAvailability",
    "EligibilityDecision",
    "EligibilityReason",
    "IdentityBasis",
    "ReuseEligibility",
    "ReuseEligibilityAssessment",
    "ReuseEligibilityPolicy",
    "StateOrigin",
    "identity_basis",
    "is_semantic_candidate",
]
