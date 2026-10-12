"""Validate LLM source spans and materialize immutable source-backed blocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .semantic_segmentation import (
    DeterministicFieldSegmenter,
    SegmentProposal,
    SemanticSegmenter,
)
from .atomic_units import UnitGroupingError
from .session import FieldExtraction, SourceSpan
from .subcontext_store import SemanticSubContext
from .token_accounting import (
    EstimatedTokenCount,
    ModelTokenCount,
    authoritative_identity,
    token_unit_of,
)
from .tokenizer import ESTIMATE_TOKENIZER, RegexTokenizer, TextTokenizer


class ValidationStatus(str, Enum):
    VALID = "valid"
    REPAIRABLE = "repairable"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    proposal_index: int | None = None


@dataclass(frozen=True)
class SegmentationValidationReport:
    status: ValidationStatus
    issues: tuple[ValidationIssue, ...]
    proposal_count: int
    valid_count: int
    repair_count: int = 0
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row


@dataclass(frozen=True)
class ValidatedSegmentation:
    sub_contexts: tuple[SemanticSubContext, ...]
    report: SegmentationValidationReport
    preliminary_token_count: "PreliminaryTokenCountReport | None" = None


@dataclass(frozen=True)
class PreliminaryTokenCountReport:
    """Cheap pre-segmentation sizing; never reused as final block accounting."""

    tokenizer_fingerprint: str
    segmentation_input_token_budget: int
    total_tokens: int
    field_token_counts: dict[str, int]
    source_token_counts: dict[str, int]
    oversized_source_ids: tuple[str, ...]
    field_internal_segmentation: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticSegmentationValidator:
    def __init__(
        self,
        *,
        max_model_tokens_per_block: int = 131_072,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        # A hard ceiling that DROPS the block, so it is material omission from
        # the runtime request rather than a shape hint. Counted with
        # `self.tokenizer`, which is the serving model's in AUTHORITATIVE mode,
        # so this is a model-token limit and now says so.
        self.max_model_tokens_per_block = max_model_tokens_per_block
        self.tokenizer = tokenizer or ESTIMATE_TOKENIZER

    @property
    def max_tokens_per_block(self) -> int:
        """Compatibility alias for the pre-unit name."""

        return self.max_model_tokens_per_block

    @staticmethod
    def reconstruct(proposal: SegmentProposal, extraction: FieldExtraction) -> str:
        sources = extraction.source_index()
        return "\n\n".join(
            sources[span.source_id].raw_text[span.start : span.end]
            for span in proposal.source_spans
        )

    def validate(
        self,
        proposals: list[SegmentProposal],
        extraction: FieldExtraction,
        *,
        model_fingerprint: str = "unknown-model",
        tokenizer_fingerprint: str | None = None,
        policy_version: str = "semantic-v1",
        recompute_policy_version: str = "batched-v1",
        expected_source_ids: set[str] | None = None,
    ) -> ValidatedSegmentation:
        source_index = extraction.source_index()
        issues: list[ValidationIssue] = []
        intervals: dict[str, list[tuple[int, int, int]]] = {}
        contexts: list[SemanticSubContext] = []
        seen: set[tuple[str, tuple[tuple[str, int, int], ...]]] = set()
        malformed_sources: set[str] = set()
        for index, proposal in enumerate(proposals):
            proposal_ok = True
            if not proposal.source_spans:
                issues.append(ValidationIssue("empty_spans", "block has no source spans", index))
                proposal_ok = False
            if proposal.is_mutable and proposal.reuse_candidate:
                issues.append(ValidationIssue("mutable_reuse", "mutable block cannot be a reuse candidate", index))
                proposal_ok = False
            key = (
                proposal.context_field,
                tuple((span.source_id, span.start, span.end) for span in proposal.source_spans),
            )
            if key in seen:
                issues.append(ValidationIssue("duplicate", "duplicate semantic block", index))
                proposal_ok = False
            seen.add(key)
            for span in proposal.source_spans:
                source = source_index.get(span.source_id)
                if source is None:
                    issues.append(ValidationIssue("missing_source", f"unknown source_id {span.source_id}", index))
                    proposal_ok = False
                    continue
                if source.context_field != proposal.context_field:
                    issues.append(ValidationIssue("field_crossing", "span crosses context field", index))
                    proposal_ok = False
                if span.start < 0 or span.end > len(source.raw_text) or span.start >= span.end:
                    malformed_sources.add(span.source_id)
                    issues.append(ValidationIssue("invalid_range", f"invalid span [{span.start},{span.end})", index))
                    proposal_ok = False
                    continue
                for old_start, old_end, old_index in intervals.setdefault(span.source_id, []):
                    if max(span.start, old_start) < min(span.end, old_end):
                        issues.append(ValidationIssue("overlap", f"span overlaps proposal {old_index}", index))
                        proposal_ok = False
                intervals[span.source_id].append((span.start, span.end, index))
            if not proposal_ok:
                continue
        # Validate the complete intended region before reconstruction/tokenization.
        # Incremental callers specify changed sources, including omitted sources.
        expected = set(source_index) if expected_source_ids is None else expected_source_ids
        for source_id in sorted(expected):
            source = source_index[source_id]
            if source_id in malformed_sources:
                continue  # The existing invalid_range diagnostic already requests repair.
            cursor = 0
            for start, end, _ in sorted(intervals.get(source_id, [])):
                if start > cursor:
                    issues.append(ValidationIssue("coverage_gap", (
                        f"Your spans do not completely partition source {source_id}. "
                        f"Missing character range [{cursor},{start}). Return ordered, "
                        f"non-overlapping, end-exclusive spans whose union exactly covers "
                        f"[0,{len(source.raw_text)}). Do not rewrite source text."
                    )))
                cursor = max(cursor, end)
            if cursor < len(source.raw_text):
                issues.append(ValidationIssue("coverage_gap", (
                    f"Your spans do not completely partition source {source_id}. "
                    f"Missing character range [{cursor},{len(source.raw_text)}). Return ordered, "
                    f"non-overlapping, end-exclusive spans whose union exactly covers "
                    f"[0,{len(source.raw_text)}). Do not rewrite source text."
                )))
        for index, proposal in enumerate(proposals if not issues else []):
            raw_text = self.reconstruct(proposal, extraction)
            if not raw_text:
                issues.append(ValidationIssue("empty_text", "reconstructed block is empty", index))
                continue
            token_ids = tuple(self.tokenizer.tokenize(raw_text))
            token_count = len(token_ids)
            if token_count > self.max_model_tokens_per_block:
                issues.append(ValidationIssue("token_limit", "block exceeds token limit", index))
                continue
            authoritative_fingerprint = self.tokenizer.fingerprint
            tokenizer_unit = token_unit_of(self.tokenizer)
            tokenizer_identity = authoritative_identity(self.tokenizer)
            if (
                tokenizer_fingerprint is not None
                and tokenizer_fingerprint != authoritative_fingerprint
            ):
                raise ValueError(
                    "authoritative tokenizer fingerprint does not match the "
                    "segmentation validation request"
                )
            token_ids_hash = hashlib.sha256(
                json.dumps(
                    token_ids,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            contexts.append(
                SemanticSubContext.create(
                    task_id=extraction.task_id,
                    session_revision=extraction.session_revision,
                    context_field=proposal.context_field,
                    title=proposal.title,
                    question_or_objective=proposal.question_or_objective,
                    raw_text=raw_text,
                    source_spans=proposal.source_spans,
                    token_count=token_count,
                    token_accounting=(
                        ModelTokenCount(
                            value=token_count,
                            tokenizer_fingerprint=tokenizer_identity.fingerprint,
                            model_id=tokenizer_identity.model_id,
                        )
                        if tokenizer_identity is not None
                        else EstimatedTokenCount(
                            value=token_count,
                            tokenizer_fingerprint=authoritative_fingerprint,
                        )
                    ),
                    is_mutable=proposal.is_mutable,
                    reuse_candidate=proposal.reuse_candidate,
                    semantic_independence_reason=proposal.semantic_independence_reason,
                    segmentation_confidence=proposal.confidence,
                    policy_version=policy_version,
                    recompute_policy_version=recompute_policy_version,
                    model_fingerprint=model_fingerprint,
                    tokenizer_fingerprint=authoritative_fingerprint,
                    token_ids=token_ids,
                    metadata={
                        "authoritative_tokenizer_fingerprint": (
                            authoritative_fingerprint
                        ),
                        "authoritative_token_ids_sha256": token_ids_hash,
                        "tokenization_stage": (
                            "after_validation_and_exact_reconstruction"
                        ),
                        # Which unit `token_count` is in. The fingerprint above
                        # proves the two sides of the validation agreed with
                        # each other; it does not prove either of them was the
                        # model's tokenizer, and two regex estimators agree
                        # perfectly. This field is what a consumer reads to
                        # find out, and the identity below is what backs it.
                        "token_unit": tokenizer_unit.value,
                        "authoritative_tokenizer_identity": (
                            dict(tokenizer_identity.to_dict())
                            if tokenizer_identity is not None
                            else None
                        ),
                        **(
                            {
                                "segmentation_decision_source": proposal.segmentation_decision_source,
                                "classification_decision_source": proposal.classification_decision_source,
                                "classification_source_id": proposal.source_spans[0].source_id,
                                "classification_source_revision": extraction.session_revision,
                                "classification_segment_index": index,
                                "classification_unit_start_id": proposal.unit_start_id,
                                "classification_unit_end_id": proposal.unit_end_id,
                                "classification_context_field": proposal.context_field,
                                "classification_model": proposal.classification_model,
                                "classification_prompt_version": proposal.classification_prompt_version,
                                "classification_repair_count": proposal.classification_repair_count,
                                "classification_fallback": proposal.classification_fallback,
                                "classification_status": proposal.classification_status,
                            }
                            if proposal.classification_status is not None
                            else {}
                        ),
                        **(
                            {
                                "llm_semantic_role": proposal.llm_semantic_role,
                                "llm_semantic_topic": proposal.llm_semantic_topic,
                            }
                            if proposal.llm_semantic_role is not None
                            else {}
                        ),
                    },
                )
            )
        hard = {"missing_source", "field_crossing"}
        codes = {issue.code for issue in issues}
        status = (
            ValidationStatus.VALID
            if not issues
            else ValidationStatus.REJECTED
            if codes & hard
            else ValidationStatus.REPAIRABLE
        )
        return ValidatedSegmentation(
            sub_contexts=tuple(contexts),
            report=SegmentationValidationReport(
                status=status,
                issues=tuple(issues),
                proposal_count=len(proposals),
                valid_count=len(contexts),
            ),
        )


@dataclass(frozen=True)
class FrozenSourcePrefix:
    """The finalized leading blocks of a source, whose boundaries must not move.

    Built from blocks already stored for the source's previous revision. When the
    source grows by appending, these blocks are replayed as proposals exactly as
    they were stored, and only the text after them is segmented -- so a later
    segmentation call (an LLM, possibly nondeterministic across a context change)
    cannot re-cut a block a Link may already depend on.
    """

    source_id: str
    end: int
    text: str
    proposals: tuple[SegmentProposal, ...]

    #: Block metadata keys that are copied back onto a replayed proposal, so a
    #: frozen block carries the labels it was stored with.
    _CARRIED = {
        "llm_semantic_role": "llm_semantic_role",
        "llm_semantic_topic": "llm_semantic_topic",
        "segmentation_decision_source": "segmentation_decision_source",
        "classification_decision_source": "classification_decision_source",
        "classification_unit_start_id": "unit_start_id",
        "classification_unit_end_id": "unit_end_id",
        "classification_model": "classification_model",
        "classification_prompt_version": "classification_prompt_version",
        "classification_repair_count": "classification_repair_count",
        "classification_fallback": "classification_fallback",
        "classification_status": "classification_status",
    }

    @classmethod
    def from_blocks(
        cls, source_id: str, blocks: list[SemanticSubContext]
    ) -> "FrozenSourcePrefix | None":
        """The longest finalized prefix of ``blocks``, or ``None`` if there is none.

        A block is frozen only if it is immutable and a reuse candidate and it
        starts exactly where the previous frozen block ended; the first mutable,
        non-reusable or non-contiguous block ends the frozen run.
        """

        owned: list[tuple[int, int, SemanticSubContext]] = []
        for block in blocks:
            spans = [span for span in block.source_spans if span.source_id == source_id]
            if len(spans) != len(block.source_spans) or len(spans) != 1:
                return None
            owned.append((spans[0].start, spans[0].end, block))
        owned.sort(key=lambda row: (row[0], row[1], row[2].sub_context_id))
        proposals: list[SegmentProposal] = []
        text: list[str] = []
        cursor = 0
        for start, end, block in owned:
            if block.is_mutable or not block.reuse_candidate or start != cursor:
                break
            metadata = block.metadata or {}
            carried = {
                target: metadata.get(key)
                for key, target in cls._CARRIED.items()
                if metadata.get(key) is not None
            }
            proposals.append(
                SegmentProposal(
                    title=block.title,
                    context_field=block.context_field,
                    source_spans=list(block.source_spans),
                    question_or_objective=block.question_or_objective,
                    semantic_independence_reason=block.semantic_independence_reason,
                    is_mutable=False,
                    reuse_candidate=True,
                    confidence=block.segmentation_confidence,
                    **carried,
                )
            )
            text.append(block.raw_text)
            cursor = end
        if not proposals:
            return None
        return cls(source_id, cursor, "".join(text), tuple(proposals))


class SemanticSegmentationPipeline:
    """Segment, validate, retry once, and always retain deterministic fallback."""

    def __init__(
        self,
        segmenter: SemanticSegmenter,
        *,
        validator: SemanticSegmentationValidator | None = None,
        fallback: SemanticSegmenter | None = None,
        report_dir: str | Path | None = None,
        preliminary_tokenizer: TextTokenizer | None = None,
        segmentation_estimate_token_budget: int = 32_000,
    ) -> None:
        if segmentation_estimate_token_budget <= 0:
            raise ValueError("segmentation_estimate_token_budget must be positive")
        self.segmenter = segmenter
        self.validator = validator or SemanticSegmentationValidator()
        self.fallback = fallback or DeterministicFieldSegmenter()
        self.preliminary_tokenizer = preliminary_tokenizer or RegexTokenizer()
        # Counted with the *preliminary estimator*, and it is not decoration:
        # this budget batches sources into segmenter calls and splits an
        # oversized source into windows, so changing it changes where block
        # boundaries fall and therefore which blocks exist and what their cache
        # keys are. That is a deliberate estimate-unit decision -- the segmenter
        # is an approximate algorithm and its input budget is approximate to
        # match -- but it must not be read as model-context capacity, which is
        # what the old name `segmentation_input_token_budget` invited.
        #
        # What it does NOT do: omit text. Every source is still segmented, and
        # every resulting block is then counted authoritatively.
        self.segmentation_estimate_token_budget = segmentation_estimate_token_budget
        self.report_dir = Path(report_dir) if report_dir else None
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)

    def preliminary_count(
        self,
        extraction: FieldExtraction,
    ) -> PreliminaryTokenCountReport:
        source_counts: dict[str, int] = {}
        field_counts: dict[str, int] = {}
        oversized: list[str] = []
        internal_fields: list[str] = []
        for context_field, sources in extraction.fields.items():
            field_total = 0
            for source in sources:
                count = self.preliminary_tokenizer.count(source.raw_text)
                source_counts[source.source_id] = count
                field_total += count
                if count > self.segmentation_estimate_token_budget:
                    oversized.append(source.source_id)
            field_counts[context_field] = field_total
            if field_total > self.segmentation_estimate_token_budget:
                internal_fields.append(context_field)
        return PreliminaryTokenCountReport(
            tokenizer_fingerprint=self.preliminary_tokenizer.fingerprint,
            segmentation_input_token_budget=self.segmentation_estimate_token_budget,
            total_tokens=sum(field_counts.values()),
            field_token_counts=field_counts,
            source_token_counts=source_counts,
            oversized_source_ids=tuple(oversized),
            field_internal_segmentation=tuple(internal_fields),
        )

    def _segment_with_budget(
        self,
        segmenter: SemanticSegmenter,
        extraction: FieldExtraction,
        changed_source_ids: set[str] | None,
        preliminary: PreliminaryTokenCountReport,
        *,
        validation_errors: list[str] | None = None,
        frozen_prefixes: dict[str, FrozenSourcePrefix] | None = None,
    ) -> list[SegmentProposal]:
        """Segment the changed sources; replay frozen prefixes instead of re-cutting them."""

        changed = changed_source_ids or set(extraction.source_index())
        index = extraction.source_index()
        active = {
            source_id: prefix
            for source_id, prefix in (frozen_prefixes or {}).items()
            if source_id in changed
            and source_id in index
            and index[source_id].raw_text.startswith(prefix.text)
        }
        if not active:
            return self._segment_unlocked(
                segmenter,
                extraction,
                changed_source_ids,
                preliminary,
                validation_errors=validation_errors,
            )
        offsets: dict[str, tuple[int, int]] = {}
        fields: dict[str, tuple[Any, ...]] = {}
        for context_field, sources in extraction.fields.items():
            kept = []
            for source in sources:
                prefix = active.get(source.source_id)
                if prefix is None:
                    kept.append(source)
                    continue
                suffix = source.raw_text[prefix.end :]
                if suffix:
                    kept.append(replace(source, raw_text=suffix))
                    offsets[source.source_id] = (prefix.end, prefix.text.count("\n"))
            fields[context_field] = tuple(kept)
        sliced = replace(extraction, fields=fields)
        fresh = self._segment_unlocked(
            segmenter,
            sliced,
            changed_source_ids,
            self.preliminary_count(sliced),
            validation_errors=validation_errors,
        )
        replayed: list[SegmentProposal] = []
        for prefix in active.values():
            replayed.extend(prefix.proposals)
        shifted: list[SegmentProposal] = []
        for proposal in fresh:
            spans = []
            for span in proposal.source_spans:
                offset = offsets.get(span.source_id)
                if offset is None:
                    spans.append(span)
                    continue
                chars, lines = offset
                spans.append(
                    SourceSpan(
                        source_id=span.source_id,
                        start=span.start + chars,
                        end=span.end + chars,
                        source_path=span.source_path,
                        start_line=None if span.start_line is None else span.start_line + lines,
                        end_line=None if span.end_line is None else span.end_line + lines,
                    )
                )
            shifted.append(replace(proposal, source_spans=spans))
        return replayed + shifted

    def _segment_unlocked(
        self,
        segmenter: SemanticSegmenter,
        extraction: FieldExtraction,
        changed_source_ids: set[str] | None,
        preliminary: PreliminaryTokenCountReport,
        *,
        validation_errors: list[str] | None = None,
    ) -> list[SegmentProposal]:
        changed = changed_source_ids or set(extraction.source_index())
        deterministic = (
            segmenter
            if isinstance(segmenter, DeterministicFieldSegmenter)
            else DeterministicFieldSegmenter()
        )
        proposals: list[SegmentProposal] = []
        budget = self.segmentation_estimate_token_budget
        for context_field, values in extraction.fields.items():
            selected = [source for source in values if source.source_id in changed]
            if not selected:
                continue
            batch: list[Any] = []
            batch_tokens = 0

            def flush_batch() -> None:
                nonlocal batch, batch_tokens
                if batch:
                    proposals.extend(
                        segmenter.segment_field_with_feedback(
                            context_field,
                            batch,
                            validation_errors,
                        )
                    )
                    batch = []
                    batch_tokens = 0

            for source in selected:
                source_tokens = preliminary.source_token_counts[source.source_id]
                if source_tokens > budget:
                    flush_batch()
                    proposals.extend(
                        deterministic.segment_source_with_token_budget(
                            context_field,
                            source,
                            tokenizer=self.preliminary_tokenizer,
                            max_tokens=budget,
                        )
                    )
                    continue
                if batch and batch_tokens + source_tokens > budget:
                    flush_batch()
                batch.append(source)
                batch_tokens += source_tokens
            flush_batch()
        return proposals

    def run(
        self,
        extraction: FieldExtraction,
        *,
        changed_source_ids: set[str] | None = None,
        frozen_prefixes: dict[str, FrozenSourcePrefix] | None = None,
        **validation_kwargs: Any,
    ) -> ValidatedSegmentation:
        repair_count = 0
        fallback_used = False
        preliminary = self.preliminary_count(extraction)
        source_ids = set(extraction.source_index())
        validation_kwargs["expected_source_ids"] = (changed_source_ids or source_ids) & source_ids
        if hasattr(self.segmenter, "atomic_tokenizer"):
            self.segmenter.atomic_tokenizer = self.validator.tokenizer
        self.format_validation_attempts = []
        def attempt(validation_errors=None):
            try:
                proposals = self._segment_with_budget(self.segmenter, extraction,
                    changed_source_ids, preliminary, validation_errors=validation_errors,
                    frozen_prefixes=frozen_prefixes)
            except UnitGroupingError as exc:
                report = SegmentationValidationReport(ValidationStatus.REPAIRABLE,
                    (ValidationIssue(exc.code, str(exc)),), 0, 0)
                self.format_validation_attempts.append(report.to_dict())
                return ValidatedSegmentation(tuple(), report)
            return self.validator.validate(proposals, extraction, **validation_kwargs)
        try:
            result = attempt()
            if result.report.status == ValidationStatus.REPAIRABLE:
                repair_count = 1
                result = attempt([issue.message for issue in result.report.issues])
            if result.report.status != ValidationStatus.VALID:
                raise ValueError("semantic segmentation remained invalid after repair")
        except Exception:
            fallback_used = True
            proposals = self._segment_with_budget(
                self.fallback,
                extraction,
                changed_source_ids,
                preliminary,
                frozen_prefixes=frozen_prefixes,
            )
            result = self.validator.validate(proposals, extraction, **validation_kwargs)
        report = SegmentationValidationReport(
            status=result.report.status,
            issues=result.report.issues,
            proposal_count=result.report.proposal_count,
            valid_count=result.report.valid_count,
            repair_count=repair_count,
            fallback_used=fallback_used,
        )
        final = ValidatedSegmentation(
            result.sub_contexts,
            report,
            preliminary_token_count=preliminary,
        )
        if self.report_dir:
            path = self.report_dir / f"segmentation_validation_r{extraction.session_revision}.json"
            path.write_text(
                json.dumps(
                    {
                        "preliminary_token_count": preliminary.to_dict(),
                        "validation": report.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return final
