"""Incremental upper-layer context management orchestration."""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from dataclasses import dataclass, replace
from typing import Any

from .context_selection import (
    AgentContextManifest,
    SelectionRequest,
    SemanticContextSelector,
    build_manifest,
)
from .segmentation_validation import (
    FrozenSourcePrefix,
    SemanticSegmentationPipeline,
    ValidatedSegmentation,
)
from .session import (
    EventType,
    FieldFirstExtractor,
    SessionEvent,
    SessionEventCollector,
    SessionFieldPipeline,
)
from .subcontext_store import (
    APPEND_ONLY_CONTEXT_FIELDS,
    SemanticSubContext,
    SubContextStore,
)
from .token_accounting import (
    TokenUnit,
    authoritative_identity,
    require_model_tokens,
    is_authoritative_tokenizer,
    require_authoritative_tokenizer,
)
from .tokenizer import ESTIMATE_TOKENIZER, TextTokenizer


class TokenAccountingMode(str, Enum):
    """Whether this manager's counts may decide anything.

    AUTHORITATIVE is for the agent loop, formal runs, and anything whose block
    counts reach the compile floor, the context budget or a link plan. It
    requires the serving model's tokenizer and fails at construction without
    one, because the failure it replaces was silent: the manager fell back to
    the regex estimator and every downstream consumer went on believing it had
    model tokens.

    ESTIMATE_ONLY is for offline utilities -- corpus inspection, rough sizing,
    fixtures. Its counts are approximations and are marked as such on every
    block it stores.
    """

    AUTHORITATIVE = "authoritative"
    ESTIMATE_ONLY = "estimate_only"

    @classmethod
    def infer(cls, tokenizer: Any) -> "TokenAccountingMode":
        """What this tokenizer can honestly claim, with no room to flatter it.

        The default, because the alternative -- demanding an explicit mode at
        every construction site -- would break offline fixtures that never
        decide anything, while the mode that actually matters is enforced where
        a count reaches a decision. A manager cannot overstate its unit here;
        it can only be asked to prove it, which is what passing AUTHORITATIVE
        explicitly does.
        """

        return (
            cls.AUTHORITATIVE
            if is_authoritative_tokenizer(tokenizer)
            else cls.ESTIMATE_ONLY
        )


class SourceCoverageError(RuntimeError):
    """An active session source reaches no stored block of its current revision.

    Raised before a manifest is built, so a request cannot be assembled without
    a source the session holds. `failures` lists one coverage record per
    uncovered source (source_id, event_id, session_revision, context_field,
    expected chars and sha256, covered chars, block count, status).
    """

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = failures
        summary = ", ".join(
            f"{row['source_id']}@r{row['session_revision']}:{row['status']}"
            for row in failures
        )
        super().__init__(f"active session sources not covered by stored blocks: {summary}")


#: Exclusion reason recorded when a source revision's siblings cannot all fit.
SOURCE_ATOMICITY_EXCLUSION = "source revision atomicity: whole source exceeds token budget"
#: Selection reason for a sibling added so a source revision is presented whole.
SOURCE_ATOMICITY_INCLUSION = "source revision sibling (reconstruction atomicity)"


@dataclass(frozen=True)
class IngestResult:
    revision: int
    events: tuple[SessionEvent, ...]
    segmentation: ValidatedSegmentation
    stored: tuple[SemanticSubContext, ...]
    segmentation_latency_s: float
    # Entries of append-only fields that this turn proved can no longer change,
    # and are therefore reusable from now on.
    settled: tuple[SemanticSubContext, ...] = ()


class IncrementalContextManager:
    def __init__(
        self,
        *,
        task_id: str,
        segment_pipeline: SemanticSegmentationPipeline,
        store: SubContextStore,
        selector: SemanticContextSelector,
        collector: SessionEventCollector | None = None,
        extractor: FieldFirstExtractor | None = None,
        tokenizer: TextTokenizer | None = None,
        preliminary_tokenizer: TextTokenizer | None = None,
        authoritative_tokenizer: TextTokenizer | None = None,
        model_fingerprint: str = "unknown-model",
        preserve_full_source_content: bool = False,
        token_accounting_mode: TokenAccountingMode | str | None = None,
    ) -> None:
        self.task_id = task_id
        self.collector = collector or SessionEventCollector(task_id)
        self.extractor = extractor or FieldFirstExtractor()
        # Stage 2 facade over the two objects above; same instances, no new state.
        self.session_fields = SessionFieldPipeline(
            collector=self.collector, extractor=self.extractor
        )
        self.segment_pipeline = segment_pipeline
        self.store = store
        self.selector = selector
        candidate = (
            authoritative_tokenizer
            or tokenizer
            or segment_pipeline.validator.tokenizer
        )
        requested = (
            None
            if token_accounting_mode is None
            else TokenAccountingMode(token_accounting_mode)
        )
        if requested is TokenAccountingMode.AUTHORITATIVE:
            # An explicit demand, so prove it here rather than three layers down
            # when a regex count reaches the compile floor. No fallback: a
            # manager that cannot name the serving model's tokenizer cannot
            # produce counts anything may decide on.
            self.authoritative_identity = require_authoritative_tokenizer(
                candidate,
                purpose=(
                    f"IncrementalContextManager({task_id!r}) in "
                    "AUTHORITATIVE token accounting mode"
                ),
            )
            self.authoritative_tokenizer = candidate
            self.token_accounting_mode = TokenAccountingMode.AUTHORITATIVE
        else:
            self.authoritative_tokenizer = candidate or ESTIMATE_TOKENIZER
            self.token_accounting_mode = requested or TokenAccountingMode.infer(
                self.authoritative_tokenizer
            )
            self.authoritative_identity = authoritative_identity(
                self.authoritative_tokenizer
            )
        # The preliminary tokenizer budgets the segmentation input and is
        # allowed to be an estimator in either mode; nothing decides on it.
        self.preliminary_tokenizer = (
            preliminary_tokenizer
            or segment_pipeline.preliminary_tokenizer
            or ESTIMATE_TOKENIZER
        )
        # `tokenizer` remains as a compatibility attribute, but all final block
        # accounting is authoritative and occurs after span reconstruction.
        self.tokenizer = self.authoritative_tokenizer
        self.segment_pipeline.validator.tokenizer = self.authoritative_tokenizer
        self.segment_pipeline.preliminary_tokenizer = self.preliminary_tokenizer
        self.model_fingerprint = model_fingerprint
        self.preserve_full_source_content = bool(preserve_full_source_content)
        self.source_reconstruction_evidence: list[dict[str, Any]] = []
        # Per request: the source-coverage records `select` checked, and any
        # sibling-atomicity adjustment it made. Evidence only; nothing reads it.
        self.source_coverage_evidence: list[dict[str, Any]] = []
        self.source_atomicity_adjustments: list[dict[str, Any]] = []
        # The manifest `select` built last; an error fixture reads it when a
        # later stage of the same call raises. Evidence only.
        self.last_manifest: AgentContextManifest | None = None

    @property
    def produces_model_tokens(self) -> bool:
        """Whether this manager's block counts are model tokens."""

        return self.token_accounting_mode is TokenAccountingMode.AUTHORITATIVE and (
            is_authoritative_tokenizer(self.authoritative_tokenizer)
        )

    def ingest(
        self,
        values: list[tuple[EventType | str, str, str, dict[str, Any]]],
    ) -> IngestResult:
        # Transactional: the events count as seen only once their revision is
        # segmented, validated and stored. Anything that raises before the store
        # commit -- including the segmentation experiment gate -- rolls the
        # collector back, so a retried call offers these events again and
        # ingests them, instead of deduplicating them into an ingest that stores
        # nothing and a prompt that silently lacks them.
        checkpoint = self.collector.checkpoint()
        events = self.session_fields.collect(values)
        if not events:
            extraction = self.session_fields.extract()
            empty_validation = self.segment_pipeline.validator.validate(
                [],
                extraction,
                expected_source_ids=set(),
            )
            empty = ValidatedSegmentation(
                tuple(),
                empty_validation.report,
                self.segment_pipeline.preliminary_count(extraction),
            )
            return IngestResult(self.collector.revision, tuple(), empty, tuple(), 0.0)
        try:
            return self._ingest_accepted(events)
        except BaseException:
            self.collector.rollback(checkpoint)
            raise

    def _ingest_accepted(self, events: list[SessionEvent]) -> IngestResult:
        extraction = self.session_fields.extract()
        changed_ids = {event.source_id for event in events}
        started = time.perf_counter()
        frozen = self._frozen_prefixes(extraction, changed_ids)
        segmentation = self.segment_pipeline.run(
            extraction,
            changed_source_ids=changed_ids,
            model_fingerprint=self.model_fingerprint,
            tokenizer_fingerprint=self.authoritative_tokenizer.fingerprint,
            # Only passed when there is something to freeze, so a pipeline that
            # predates boundary locking is called exactly as before.
            **({"frozen_prefixes": frozen} if frozen else {}),
        )
        latency = time.perf_counter() - started
        source_index = extraction.source_index()
        # Field extraction groups sources by field, which is deliberately not
        # presentation order.  Preserve the collector's source sequence so
        # sibling spans can be merged into that original order later.
        source_order = {
            event.source_id: event.sequence_number
            for event in self.collector.events()
            if event.source_id in source_index
            and source_index[event.source_id].event_id == event.event_id
        }
        by_source: dict[str, list[SemanticSubContext]] = {}
        for value in segmentation.sub_contexts:
            source_ids = {span.source_id for span in value.source_spans}
            if len(source_ids) != 1:
                raise ValueError(
                    "active source ownership requires one source per semantic block"
                )
            source_id = next(iter(source_ids))
            source = source_index[source_id]
            metadata = dict(value.metadata or {})
            metadata.update(
                {
                    "source_revision_identity": source.event_id,
                    "source_order": source_order[source_id],
                }
            )
            by_source.setdefault(source_id, []).append(
                replace(value, metadata=metadata)
            )
        # Validate the revision before any of it is stored, then store every
        # changed source in one transaction: all of them or none.
        self.source_reconstruction_evidence = (
            self._validate_active_source_reconstruction(
                extraction, changed_ids, prospective=by_source
            )
        )
        stored = list(
            self.store.replace_source_revisions(
                self.task_id,
                [
                    (source_id, source_index[source_id].event_id, siblings)
                    for source_id, siblings in by_source.items()
                ],
            )
        )
        settled = self._settle_append_only_history(self.collector.revision)
        return IngestResult(
            revision=self.collector.revision,
            events=tuple(events),
            segmentation=segmentation,
            stored=tuple(stored),
            segmentation_latency_s=latency,
            settled=settled,
        )

    def _frozen_prefixes(
        self, extraction: Any, changed_ids: set[str]
    ) -> dict[str, FrozenSourcePrefix]:
        """Boundary locking: finalized blocks of a source that has only grown stay as cut.

        A changed source whose new text starts with the text its stored, finalized
        leading blocks already cover keeps those blocks (same spans, same text,
        therefore the same identity and cache key). Only the appended remainder is
        segmented. A source that was edited rather than appended to, or whose
        leading blocks are not yet finalized, is segmented as a whole.
        """

        sources = extraction.source_index()
        frozen: dict[str, FrozenSourcePrefix] = {}
        for source_id in sorted(changed_ids):
            source = sources.get(source_id)
            if source is None:
                continue
            previous = [
                value
                for value in self.store.find_by_source_id(self.task_id, source_id)
                if not value.invalidated
            ]
            if not previous:
                continue
            prefix = FrozenSourcePrefix.from_blocks(source_id, previous)
            if (
                prefix is not None
                and source.raw_text.startswith(prefix.text)
                and len(source.raw_text) > prefix.end
            ):
                frozen[source_id] = prefix
        return frozen

    @staticmethod
    def _source_position(
        value: SemanticSubContext, source_id: str
    ) -> tuple[int, int, str]:
        spans = [span for span in value.source_spans if span.source_id == source_id]
        return (
            min(span.start for span in spans),
            max(span.end for span in spans),
            value.sub_context_id,
        )

    def _validate_active_source_reconstruction(
        self,
        extraction: Any,
        source_ids: set[str],
        *,
        prospective: dict[str, list[SemanticSubContext]] | None = None,
    ) -> list[dict[str, Any]]:
        """Every changed source must be rebuilt exactly by its sibling set.

        With `prospective`, the sibling set is the one about to be stored, so a
        failure leaves the store untouched; a source it does not name is read
        from the store, which is where an idempotent re-ingest finds it.
        """

        evidence: list[dict[str, Any]] = []
        sources = extraction.source_index()
        for source_id in sorted(source_ids):
            if source_id not in sources:
                # Filtered by the extractor (hidden source): never model-visible.
                continue
            source = sources[source_id]
            if prospective is not None and source_id in prospective:
                siblings = list(prospective[source_id])
            else:
                siblings = [
                    value
                    for value in self.store.find_by_source_id(self.task_id, source_id)
                    if not value.invalidated
                    and value.source_revision_identity == source.event_id
                ]
            siblings.sort(key=lambda value: self._source_position(value, source_id))
            selected_text = "".join(value.raw_text for value in siblings)
            expected_ids = tuple(self.authoritative_tokenizer.tokenize(source.raw_text))
            selected_ids = tuple(self.authoritative_tokenizer.tokenize(selected_text))
            row = {
                "source_id": source_id,
                "source_revision_identity": source.event_id,
                "expected_chars": len(source.raw_text),
                "selected_chars": len(selected_text),
                "expected_tokens": len(expected_ids),
                "selected_tokens": len(selected_ids),
                "byte_parity": selected_text.encode("utf-8")
                == source.raw_text.encode("utf-8"),
                "token_parity": selected_ids == expected_ids,
                "active_block_count": len(siblings),
            }
            evidence.append(row)
            if not row["byte_parity"] or not row["token_parity"]:
                raise ValueError(f"active source reconstruction failed: {row}")
        return evidence

    def _settle_append_only_history(
        self, revision: int
    ) -> tuple[SemanticSubContext, ...]:
        """Promote append-only entries that this turn left behind.

        Segmentation marks an entry of an append-only field mutable because the
        field keeps growing. The entry itself stops changing as soon as a later
        turn appends after it, and its cache key is derived from content alone,
        so settling it makes it reusable without invalidating anything already
        compiled.

        Only entries from earlier revisions are settled. Everything this turn
        added stays mutable, which keeps the current query in the mutable suffix
        that a LegoLink link request requires.
        """

        settled: list[SemanticSubContext] = []
        for context_field in APPEND_ONLY_CONTEXT_FIELDS:
            for value in self.store.list_by_field(self.task_id, context_field):
                if value.invalidated or value.session_revision >= revision:
                    continue
                if not value.is_mutable and value.reuse_candidate:
                    continue
                settled.append(self.store.settle(value.sub_context_id))
        return tuple(settled)

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
        extraction = self.session_fields.extract()
        # Fail closed before anything is assembled: an active source that no
        # stored block of its current revision covers would otherwise be absent
        # from the prompt while still counted in full_session_tokens. Native
        # and LegoLink routes both select here, so both obey the same rule.
        self.source_coverage_evidence = self.source_coverage(extraction)
        uncovered = [
            row for row in self.source_coverage_evidence
            if row["status"] not in ("covered", "empty", "excluded_by_policy")
        ]
        if uncovered:
            raise SourceCoverageError(uncovered)
        candidates = self.store.list_by_task(self.task_id)
        if self.token_accounting_mode is TokenAccountingMode.AUTHORITATIVE:
            # `token_budget` here is the model's context window, and every
            # block competing for it must be counted in that unit. A block
            # restored from an earlier estimate-only session, or replayed from
            # a legacy artifact with no tokenizer provenance, would otherwise be
            # budgeted as if its regex count were model tokens -- which is how a
            # workload "cut to 30,013 tokens" arrived at the server as 36,573.
            for candidate in candidates:
                require_model_tokens(
                    candidate.token_accounting(),
                    purpose=(
                        f"context budget of {token_budget} model tokens for "
                        f"block {candidate.sub_context_id!r}"
                    ),
                )

        request = SelectionRequest(
            task_id=self.task_id,
            agent_id=agent_id,
            loop_stage=loop_stage,
            current_objective=objective,
            current_plan_step=plan_step,
            available_tools=available_tools,
            token_budget=token_budget,
            mutable_suffix_token_count=self.authoritative_tokenizer.count(
                mutable_text
            ),
        )
        selection = self.selector.select_for_request(request, candidates)
        if self.preserve_full_source_content:
            selection = self._expand_selected_source_siblings(
                request, candidates, selection
            )
        selection = self._enforce_source_revision_atomicity(
            request, candidates, selection
        )
        manifest = build_manifest(
            request=request,
            request_id=f"noc::{self.task_id}::{uuid.uuid4().hex}",
            candidates=candidates,
            selection=selection,
            mutable_text=mutable_text,
            full_session_tokens=extraction.token_count(self.authoritative_tokenizer),
            # Every token field on the manifest came from this one tokenizer,
            # so they are comparable with each other. Saying which one closes
            # the question a consumer would otherwise have to guess at.
            token_unit=(
                TokenUnit.MODEL
                if self.produces_model_tokens
                else TokenUnit.ESTIMATED
            ),
        )
        self.last_manifest = manifest
        return manifest, candidates

    def source_coverage(self, extraction: Any | None = None) -> list[dict[str, Any]]:
        """One record per active source: is it covered by stored blocks of its revision?

        `covered` means the non-invalidated blocks of the source's current
        revision rebuild its text byte for byte. `missing` means none exist,
        `partial` that they exist but do not rebuild it. Hidden sources the
        extractor filtered are recorded as `excluded_by_policy`; they are never
        model-visible.
        """

        extraction = extraction or self.session_fields.extract()
        revisions = {event.event_id: event.session_revision for event in self.collector.events()}
        records: list[dict[str, Any]] = []
        for context_field, sources in extraction.fields.items():
            for source in sources:
                siblings = sorted(
                    (
                        value
                        for value in self.store.find_by_source_id(self.task_id, source.source_id)
                        if not value.invalidated
                        and value.source_revision_identity == source.event_id
                    ),
                    key=lambda value, sid=source.source_id: self._source_position(value, sid),
                )
                rebuilt = "".join(value.raw_text for value in siblings)
                if not source.raw_text:
                    status = "empty"
                elif not siblings:
                    status = "missing"
                elif rebuilt == source.raw_text:
                    status = "covered"
                else:
                    status = "partial"
                records.append({
                    "source_id": source.source_id,
                    "event_id": source.event_id,
                    "session_revision": revisions.get(source.event_id),
                    "context_field": context_field,
                    "expected_chars": len(source.raw_text),
                    "expected_sha256": hashlib.sha256(source.raw_text.encode("utf-8")).hexdigest(),
                    "covered_chars": len(rebuilt) if siblings else 0,
                    "block_count": len(siblings),
                    "status": status,
                })
        for event_id in extraction.filtered_event_ids:
            records.append({
                "source_id": None, "event_id": event_id, "session_revision": revisions.get(event_id),
                "context_field": None, "expected_chars": None, "expected_sha256": None,
                "covered_chars": None, "block_count": 0, "status": "excluded_by_policy",
            })
        return records

    def _enforce_source_revision_atomicity(
        self, request: SelectionRequest, candidates: list[SemanticSubContext], selection: Any
    ) -> Any:
        """Present each source revision whole, or not at all.

        One source revision may be split into several sibling blocks, and the
        stitcher rebuilds the source only from a contiguous sibling run -- it
        refuses 0 and 2 without 1, correctly. Budgeted, relevance-ranked
        selection is block-local and can pick exactly that. So a partially
        selected revision is completed with its missing siblings when the budget
        still fits them, and otherwise excluded entirely with a recorded reason.
        Revisions are resolved in selection order, deterministically. A
        single-block source is never touched, so deterministic segmentation is
        unaffected.
        """

        from .context_selection import SelectionEntry, SelectionResult

        by_id = {value.sub_context_id: value for value in candidates}

        def group_key(value: SemanticSubContext) -> tuple[str, str, str] | None:
            source_ids = {span.source_id for span in value.source_spans}
            if len(source_ids) != 1 or not value.source_revision_identity:
                return None
            return (next(iter(source_ids)), value.source_revision_identity, value.context_field)

        groups: dict[tuple[str, str, str], list[SemanticSubContext]] = {}
        for value in candidates:
            key = group_key(value)
            if key is not None:
                groups.setdefault(key, []).append(value)
        budget = max(0, request.token_budget - request.mutable_suffix_token_count)
        selected = {entry.sub_context_id: entry for entry in selection.selected}
        excluded = {entry.sub_context_id: entry for entry in selection.excluded}
        used = sum(by_id[sid].token_count for sid in selected)
        order: list[str] = []
        adjustments: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for entry in selection.selected:
            if entry.sub_context_id not in selected:
                continue
            order.append(entry.sub_context_id)
            key = group_key(by_id[entry.sub_context_id])
            if key is None or key in seen:
                continue
            seen.add(key)
            members = groups.get(key, [])
            missing = [value for value in members if value.sub_context_id not in selected]
            if not missing:
                continue
            extra = sum(value.token_count for value in missing)
            if used + extra <= budget:
                for value in missing:
                    selected[value.sub_context_id] = SelectionEntry(
                        value.sub_context_id, SOURCE_ATOMICITY_INCLUSION, entry.priority
                    )
                    excluded.pop(value.sub_context_id, None)
                    order.append(value.sub_context_id)
                used += extra
                action = "completed"
            else:
                for value in members:
                    if value.sub_context_id in selected:
                        used -= value.token_count
                        del selected[value.sub_context_id]
                    excluded[value.sub_context_id] = SelectionEntry(
                        value.sub_context_id, SOURCE_ATOMICITY_EXCLUSION
                    )
                order = [sid for sid in order if sid in selected]
                action = "excluded"
            adjustments.append({
                "source_id": key[0], "source_revision_identity": key[1], "context_field": key[2],
                "action": action, "sibling_count": len(members),
                "missing_sibling_ids": [value.sub_context_id for value in missing],
            })
        self.source_atomicity_adjustments = adjustments
        if not adjustments:
            return selection
        result = SelectionResult(
            selected=tuple(selected[sid] for sid in order if sid in selected),
            excluded=tuple(
                excluded[value.sub_context_id]
                for value in candidates
                if value.sub_context_id in excluded
            ),
            fallback_used=selection.fallback_used,
        )
        self.selector.validate_selection(request, candidates, result)
        return result

    def _expand_selected_source_siblings(
        self, request: SelectionRequest, candidates: list[SemanticSubContext], selection: Any
    ) -> Any:
        from .context_selection import SelectionEntry, SelectionResult

        by_id = {value.sub_context_id: value for value in candidates}
        source_revisions = {
            (
                span.source_id,
                by_id[entry.sub_context_id].source_revision_identity,
            )
            for entry in selection.selected
            for span in by_id[entry.sub_context_id].source_spans
        }
        expanded = [
            value
            for value in candidates
            if any(
                (span.source_id, value.source_revision_identity) in source_revisions
                for span in value.source_spans
            )
        ]
        expanded_ids = {value.sub_context_id for value in expanded}
        entries = {
            entry.sub_context_id: entry for entry in selection.selected
        }
        for value in expanded:
            entries.setdefault(
                value.sub_context_id,
                SelectionEntry(
                    value.sub_context_id,
                    "full-source parity sibling",
                    0,
                ),
            )
        result = SelectionResult(
            selected=tuple(entries[value.sub_context_id] for value in expanded),
            excluded=tuple(
                entry
                for entry in selection.excluded
                if entry.sub_context_id not in expanded_ids
            ),
            fallback_used=selection.fallback_used,
        )
        self.selector.validate_selection(request, candidates, result)
        return result
