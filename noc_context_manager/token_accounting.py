"""Two token units, and the refusal to confuse them.

`count_tokens(text)` used to mean "whatever tokenizer happened to be
available", and the tokenizer that happened to be available was
`RegexTokenizer`. The same integer then reached the compile floor, the context
budget, the LegoLink recompute ranges and the benchmark tables, and nothing in
its shape said which unit it was in.

The cost is measured. Regex-to-BPE ratios across candidate Terminal-Bench tasks
range from 1.19 to 7.63, so a "30,013-token" prompt built on regex counts was
refused by the server at 36,573 model tokens. That failure was loud. The quiet
version is a 64-token compile floor that means 64 model tokens on one path and
anywhere from 8 to 54 on another, deciding which blocks are eligible to be
cached at all.

So the unit is part of the value:

    EstimatedTokenCount   a cheap approximation. Fine for budgeting a
                          segmentation pass, previewing a report, or sizing a
                          preflight. Never a runtime or cache-planning input.
    ModelTokenCount       what the serving model's own tokenizer produced,
                          carrying the identity of the tokenizer that produced
                          it.

There is no implicit conversion between them, in either direction. Going from
text to `ModelTokenCount` requires an authoritative tokenizer and nothing else
will do -- in particular, two `RegexTokenizer` instances agreeing with each
other is not evidence about the model, which is the specific hole this module
closes: the previous fingerprint check proved `tokenizer A == tokenizer B` and
both were regex.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class TokenAccountingError(RuntimeError):
    """Raised when a decision path is handed the wrong token unit."""


class TokenUnit(str, Enum):
    """What a token count is a count of."""

    ESTIMATED = "estimated_tokens"
    MODEL = "model_tokens"
    #: A recorded integer whose provenance was never written down. Legacy
    #: artifacts land here, and they are treated as estimates: an old number is
    #: not authoritative for being old.
    UNKNOWN = "unknown_tokens"

    @property
    def is_authoritative(self) -> bool:
        return self is TokenUnit.MODEL


@dataclass(frozen=True)
class EstimatedTokenCount:
    """An approximation. Safe to display, never safe to decide on."""

    value: int
    tokenizer_fingerprint: str = "unknown-tokenizer"

    unit = TokenUnit.ESTIMATED

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("token counts are non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": TokenUnit.ESTIMATED.value,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }


@dataclass(frozen=True)
class ModelTokenCount:
    """What the serving model's tokenizer counted, and which tokenizer it was.

    `tokenizer_fingerprint` and `model_id` travel with the number because a
    count is only authoritative relative to a specific tokenizer. A count
    carried over from a different model is not a model-token count for this one.
    """

    value: int
    tokenizer_fingerprint: str
    model_id: str

    unit = TokenUnit.MODEL

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("token counts are non-negative")
        if not str(self.tokenizer_fingerprint).strip():
            raise TokenAccountingError(
                "a model token count must name the tokenizer that produced it"
            )
        if not str(self.model_id).strip():
            raise TokenAccountingError(
                "a model token count must name the model it belongs to"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": TokenUnit.MODEL.value,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "model_id": self.model_id,
        }


TokenCount = EstimatedTokenCount | ModelTokenCount


@dataclass(frozen=True)
class AuthoritativeTokenizerIdentity:
    """Enough to argue that this tokenizer is the serving model's.

    A fingerprint alone cannot make that argument. `RegexTokenizer` has a
    perfectly stable fingerprint and it is not a model tokenizer, so the guard
    asks for the things only a real tokenizer can supply: which model it is for,
    which implementation produced it, and the content hash of its vocabulary or
    configuration.
    """

    model_id: str
    tokenizer_class: str
    fingerprint: str
    backend: str = ""
    revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tokenizer_class": self.tokenizer_class,
            "fingerprint": self.fingerprint,
            "backend": self.backend,
            "revision": self.revision,
        }


def token_unit_of(tokenizer: Any) -> TokenUnit:
    """The unit a tokenizer produces, declared by the tokenizer itself.

    Read from a `token_unit` class attribute rather than inferred from the type
    name, so a new backend states its own nature and an unmarked object is
    UNKNOWN rather than assumed authoritative.
    """

    declared = getattr(tokenizer, "token_unit", None)
    if isinstance(declared, TokenUnit):
        return declared
    if isinstance(declared, str):
        try:
            return TokenUnit(declared)
        except ValueError:
            return TokenUnit.UNKNOWN
    return TokenUnit.UNKNOWN


def authoritative_identity(tokenizer: Any) -> AuthoritativeTokenizerIdentity | None:
    """The tokenizer's authoritative identity, or `None` if it has no claim."""

    if token_unit_of(tokenizer) is not TokenUnit.MODEL:
        return None
    model_id = str(getattr(tokenizer, "model_id", "") or "")
    fingerprint = str(getattr(tokenizer, "fingerprint", "") or "")
    if not model_id.strip() or not fingerprint.strip():
        return None
    return AuthoritativeTokenizerIdentity(
        model_id=model_id,
        tokenizer_class=f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        fingerprint=fingerprint,
        backend=str(getattr(tokenizer, "backend_name", "") or ""),
        revision=str(getattr(tokenizer, "revision", "") or ""),
    )


def is_authoritative_tokenizer(tokenizer: Any) -> bool:
    return authoritative_identity(tokenizer) is not None


def require_authoritative_tokenizer(
    tokenizer: Any, *, purpose: str
) -> AuthoritativeTokenizerIdentity:
    """Fail closed unless this is demonstrably the serving model's tokenizer.

    `purpose` names the decision being protected, because the remedy differs:
    a formal benchmark needs the run's tokenizer, an agent loop needs the
    deployed model's, and an offline utility should probably have asked for an
    estimate instead.
    """

    if tokenizer is None:
        raise TokenAccountingError(
            f"{purpose} requires the serving model's tokenizer and none was "
            "supplied. Pass one explicitly, or ask for an estimate through "
            "estimate_tokens() if an approximation is what you want."
        )
    identity = authoritative_identity(tokenizer)
    if identity is None:
        unit = token_unit_of(tokenizer)
        raise TokenAccountingError(
            f"{purpose} requires an authoritative model tokenizer; "
            f"{type(tokenizer).__name__} declares unit {unit.value} and cannot "
            "establish a model identity. Two matching estimating tokenizers "
            "agree with each other and say nothing about the model."
        )
    return identity


def count_from_provenance(
    value: Any,
    *,
    tokenizer_fingerprint: str | None,
    model_id: str | None = None,
    authoritative_fingerprints: frozenset[str] | set[str] | None = None,
) -> TokenCount:
    """Interpret a recorded integer using whatever provenance came with it.

    Conservative on purpose. A historical artifact carrying a bare
    `token_count` and no tokenizer fingerprint becomes an
    `EstimatedTokenCount`, because "we do not know" and "it was the model's"
    are different claims and only one of them is safe to act on.
    """

    count = int(value or 0)
    fingerprint = str(tokenizer_fingerprint or "").strip()
    known = frozenset(authoritative_fingerprints or ())
    if fingerprint and fingerprint in known and str(model_id or "").strip():
        return ModelTokenCount(
            value=count,
            tokenizer_fingerprint=fingerprint,
            model_id=str(model_id),
        )
    return EstimatedTokenCount(
        value=count, tokenizer_fingerprint=fingerprint or "unknown-tokenizer"
    )


def require_model_tokens(count: Any, *, purpose: str) -> ModelTokenCount:
    """The decision boundary. Anything but a `ModelTokenCount` is refused.

    Including a bare `int`, which is the shape every one of these values had
    before this module existed and which carries no unit at all.
    """

    if isinstance(count, ModelTokenCount):
        return count
    if isinstance(count, EstimatedTokenCount):
        raise TokenAccountingError(
            f"{purpose} requires model tokens and was given an estimate of "
            f"{count.value} from {count.tokenizer_fingerprint}. Estimates do "
            "not convert; re-count the text with the serving model's tokenizer."
        )
    raise TokenAccountingError(
        f"{purpose} requires a ModelTokenCount, not {type(count).__name__}. A "
        "bare integer has no unit, and this decision depends on the unit."
    )


def unit_metadata(count: TokenCount) -> Mapping[str, Any]:
    """Serialization helper: the unit travels with the number."""

    return count.to_dict()
