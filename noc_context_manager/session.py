"""Revisioned agent-session events and field-first source extraction.

ARCHITECTURE_STAGE: 2 -- Session Events + Field Extraction
CANONICAL_ENTRYPOINT: SessionFieldPipeline (composes SessionEventCollector and
    FieldFirstExtractor; both stay public and keep their own semantics)
DELEGATED_HELPERS: tokenizer.py (token counting for the event/field totals)
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePath
from typing import Any, Iterable

from .tokenizer import TextTokenizer, count_tokens


CONTEXT_FIELDS = (
    "user_prompt",
    "task_instruction",
    "tool_definitions",
    "tool_history_output",
    "conversation_history",
    "workspace_evidence",
    "task_metadata",
)


class EventType(str, Enum):
    USER_REQUEST = "user_request"
    TASK_INSTRUCTION = "task_instruction"
    PLANNER_OUTPUT = "planner_output"
    TOOL_DEFINITION = "tool_definition"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    CONVERSATION_MESSAGE = "conversation_message"
    WORKSPACE_FILE = "workspace_file"
    TASK_METADATA = "task_metadata"
    #: An input whose role the producer does not know. It has no entry in
    #: `FieldFirstExtractor._FIELD_MAP` on purpose: a field classifier must decide
    #: it, and without one extraction fails closed instead of guessing.
    UNCLASSIFIED = "unclassified"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class SourceSpan:
    source_id: str
    start: int
    end: int
    source_path: str = ""
    start_line: int | None = None
    end_line: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceSpan":
        return cls(
            source_id=str(value["source_id"]),
            start=int(value["start"]),
            end=int(value["end"]),
            source_path=str(value.get("source_path") or ""),
            start_line=value.get("start_line"),
            end_line=value.get("end_line"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionEvent:
    event_id: str
    task_id: str
    session_revision: int
    event_type: EventType
    source_id: str
    raw_text: str
    timestamp: str
    sequence_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["event_type"] = self.event_type.value
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SessionEvent":
        return cls(
            event_id=str(row["event_id"]),
            task_id=str(row["task_id"]),
            session_revision=int(row["session_revision"]),
            event_type=EventType(row["event_type"]),
            source_id=str(row["source_id"]),
            raw_text=str(row.get("raw_text") or ""),
            timestamp=str(row["timestamp"]),
            sequence_number=int(row["sequence_number"]),
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ExtractedSource:
    context_field: str
    event_id: str
    source_id: str
    raw_text: str
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def span(self) -> SourceSpan:
        return SourceSpan(
            source_id=self.source_id,
            start=0,
            end=len(self.raw_text),
            source_path=self.source_path,
            start_line=1 if self.raw_text else None,
            end_line=self.raw_text.count("\n") + 1 if self.raw_text else None,
        )


@dataclass(frozen=True)
class FieldExtraction:
    task_id: str
    session_revision: int
    fields: dict[str, tuple[ExtractedSource, ...]]
    filtered_event_ids: tuple[str, ...] = ()

    def source_index(self) -> dict[str, ExtractedSource]:
        return {
            source.source_id: source
            for sources in self.fields.values()
            for source in sources
        }

    def token_count(self, tokenizer: TextTokenizer | None = None) -> int:
        return sum(
            count_tokens(source.raw_text, tokenizer)
            for sources in self.fields.values()
            for source in sources
        )

    @property
    def full_session_tokens(self) -> int:
        return self.token_count()


class SessionEventCollector:
    """Collect events and assign one new revision per accepted update batch."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._events: list[SessionEvent] = []
        self._dedup: set[str] = set()
        # The dedup digest of each accepted event, aligned with `_events`.
        self._event_digests: list[str] = []
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def events(self, revision: int | None = None) -> list[SessionEvent]:
        if revision is None:
            return list(self._events)
        return [event for event in self._events if event.session_revision <= revision]

    def checkpoint(self) -> tuple[int, int]:
        """A point `rollback` can return to: (event count, revision)."""

        return len(self._events), self._revision

    def rollback(self, checkpoint: tuple[int, int]) -> list[SessionEvent]:
        """Forget every event accepted after `checkpoint`.

        Ingest is transactional: events count as seen only once their revision
        is segmented, validated and stored. A failed ingest rolls back, so a
        retried call offers the same events again and they are ingested again,
        instead of being deduplicated into an ingest that stores nothing.
        """

        count, revision = checkpoint
        if count > len(self._events) or revision > self._revision:
            raise ValueError("cannot roll back to a checkpoint from the future")
        removed = self._events[count:]
        del self._events[count:]
        self._dedup.difference_update(self._event_digests[count:])
        del self._event_digests[count:]
        self._revision = revision
        return removed

    def add_event(
        self,
        event_type: EventType | str,
        source_id: str,
        raw_text: str,
        *,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> SessionEvent | None:
        rows = self.add_events([(event_type, source_id, raw_text, metadata or {})], timestamp=timestamp)
        return rows[0] if rows else None

    def add_events(
        self,
        values: Iterable[tuple[EventType | str, str, str, dict[str, Any]]],
        *,
        timestamp: str | None = None,
    ) -> list[SessionEvent]:
        accepted: list[tuple[EventType, str, str, dict[str, Any], str]] = []
        pending_keys: set[str] = set()
        for event_type, source_id, raw_text, metadata in values:
            kind = EventType(event_type)
            digest = hashlib.sha256(
                f"{kind.value}\0{source_id}\0{raw_text}".encode("utf-8")
            ).hexdigest()
            if digest in self._dedup or digest in pending_keys:
                continue
            pending_keys.add(digest)
            accepted.append((kind, source_id, raw_text, dict(metadata), digest))
        if not accepted:
            return []
        self._revision += 1
        created: list[SessionEvent] = []
        for kind, source_id, raw_text, metadata, digest in accepted:
            sequence = len(self._events) + 1
            event = SessionEvent(
                event_id=f"evt-{digest[:20]}",
                task_id=self.task_id,
                session_revision=self._revision,
                event_type=kind,
                source_id=source_id,
                raw_text=raw_text,
                timestamp=timestamp or utc_now(),
                sequence_number=sequence,
                metadata=metadata,
            )
            self._events.append(event)
            self._dedup.add(digest)
            self._event_digests.append(digest)
            created.append(event)
        return created


class FieldFirstExtractor:
    """Classify event sources before semantic segmentation.

    The event-type table below is the deterministic classifier. An optional
    ``classifier`` (``field_classification.FunctionalFieldClassifier``) is
    consulted only for an event type the table does not cover; with none given,
    such an event fails extraction rather than being assigned a field.
    """

    def __init__(self, classifier: Any = None) -> None:
        self.classifier = classifier

    def field_for(self, event: "SessionEvent") -> tuple[str, dict[str, Any] | None]:
        """The functional field of ``event``, and how it was chosen when not by the table."""

        known = self._FIELD_MAP.get(event.event_type)
        if known is not None:
            return known, None
        if self.classifier is None:
            from .field_classification import FieldClassificationError

            raise FieldClassificationError(
                f"event {event.event_id} has type {event.event_type.value!r}, which no "
                "field table covers, and no field classifier is configured"
            )
        result = self.classifier.classify(event)
        return result.field, result.to_dict()

    _FIELD_MAP = {
        EventType.USER_REQUEST: "user_prompt",
        EventType.TASK_INSTRUCTION: "task_instruction",
        EventType.PLANNER_OUTPUT: "conversation_history",
        EventType.TOOL_DEFINITION: "tool_definitions",
        EventType.TOOL_CALL: "tool_history_output",
        EventType.TOOL_OUTPUT: "tool_history_output",
        EventType.CONVERSATION_MESSAGE: "conversation_history",
        EventType.WORKSPACE_FILE: "workspace_evidence",
        EventType.TASK_METADATA: "task_metadata",
    }
    _FORBIDDEN_PARTS = {
        "oracle",
        "grader",
        "graders",
        "solution",
        "solutions",
        "hidden_answer",
        "answer_key",
        "verifier",
    }

    @classmethod
    def is_hidden_source(cls, event: SessionEvent) -> bool:
        metadata = event.metadata
        if any(bool(metadata.get(key)) for key in ("hidden", "oracle", "grader", "answer")):
            return True
        path = str(metadata.get("path") or metadata.get("file_path") or "")
        parts = {part.lower() for part in PurePath(path.replace("\\", "/")).parts}
        name = PurePath(path.replace("\\", "/")).name.lower()
        return bool(parts & cls._FORBIDDEN_PARTS) or any(
            marker in name
            for marker in (
                "oracle",
                "grader",
                "solution",
                "answer_key",
                "test_answer",
                "expected_answer",
                "verifier",
            )
        )

    def extract(self, events: Iterable[SessionEvent]) -> FieldExtraction:
        ordered = sorted(events, key=lambda event: event.sequence_number)
        task_ids = {event.task_id for event in ordered}
        if len(task_ids) > 1:
            raise ValueError("field extraction cannot mix task ids")
        fields: dict[str, list[ExtractedSource]] = {name: [] for name in CONTEXT_FIELDS}
        filtered: list[str] = []
        # A source_id identifies one logical message/tool call/file. Keep only its
        # latest revision in the active extraction while the collector retains the
        # full event history for audit and store supersession.
        latest_by_source = {event.source_id: event for event in ordered}
        active_events = sorted(latest_by_source.values(), key=lambda event: event.sequence_number)
        for event in active_events:
            if self.is_hidden_source(event):
                filtered.append(event.event_id)
                continue
            context_field, classification = self.field_for(event)
            path = str(event.metadata.get("path") or event.metadata.get("file_path") or "")
            metadata = dict(event.metadata)
            if classification is not None:
                # Recorded only when the table did not decide, so a table-classified
                # source is byte-for-byte what it was before classifiers existed.
                metadata["field_classification"] = classification
            fields[context_field].append(
                ExtractedSource(
                    context_field=context_field,
                    event_id=event.event_id,
                    source_id=event.source_id,
                    raw_text=event.raw_text,
                    source_path=path,
                    metadata=metadata,
                )
            )
        return FieldExtraction(
            task_id=next(iter(task_ids), ""),
            session_revision=max((event.session_revision for event in ordered), default=0),
            fields={name: tuple(values) for name, values in fields.items()},
            filtered_event_ids=tuple(filtered),
        )


class SessionFieldPipeline:
    """Stage 2 facade: collect session events, then field-classify their sources.

    ARCHITECTURE_STAGE: 2
    CANONICAL_ENTRYPOINT: SessionFieldPipeline.collect / .extract

    A composition of the two stage-2 objects and nothing more. It owns no state
    of its own and adds no semantics: ``collect`` is
    ``SessionEventCollector.add_events`` and ``extract`` is
    ``FieldFirstExtractor.extract`` over the collector's current events. Callers
    that already hold a collector and an extractor (``IncrementalContextManager``
    does) pass them in, so there is exactly one collector per session.
    """

    def __init__(
        self,
        task_id: str | None = None,
        *,
        collector: SessionEventCollector | None = None,
        extractor: FieldFirstExtractor | None = None,
    ) -> None:
        if collector is None and task_id is None:
            raise ValueError("SessionFieldPipeline needs a task_id or a collector")
        self.collector = collector or SessionEventCollector(str(task_id))
        self.extractor = extractor or FieldFirstExtractor()

    def collect(
        self,
        values: Iterable[tuple[EventType | str, str, str, dict[str, Any]]],
    ) -> list[SessionEvent]:
        return self.collector.add_events(list(values))

    def extract(self) -> FieldExtraction:
        return self.extractor.extract(self.collector.events())

    def ingest(
        self,
        values: Iterable[tuple[EventType | str, str, str, dict[str, Any]]],
    ) -> tuple[list[SessionEvent], FieldExtraction]:
        events = self.collect(values)
        return events, self.extract()
