"""Versioned semantic sub-context model and SQLite repository."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .session import SourceSpan
from .token_accounting import (
    EstimatedTokenCount,
    ModelTokenCount,
    TokenCount,
    TokenUnit,
)


# Fields whose entries form an append-only log of the agent loop.
#
# The field as a whole keeps changing, which is why segmentation marks its
# entries mutable when they arrive. An individual entry does not: a tool output
# or a conversation turn is complete the moment it is recorded, and every later
# turn appends after it rather than editing it. Once a later entry exists, the
# earlier one is settled and can never change again, so it is reusable.
#
# Treating the whole field as permanently mutable is what confined LegoLink to
# the static task instruction in an agent loop: reused tokens stayed frozen at
# the first turn's blocks while recomputed tokens grew with every turn.
APPEND_ONLY_CONTEXT_FIELDS = (
    "user_prompt",
    "tool_history_output",
    "conversation_history",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def semantic_content_hash(raw_text: str) -> str:
    """Hash exact source-derived bytes; whitespace is semantically observable."""

    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def semantic_block_id(
    *, task_id: str, context_field: str, source_spans: list[SourceSpan]
) -> str:
    encoded = json.dumps(
        {
            "task_id": task_id,
            "context_field": context_field,
            "source_spans": [span.to_dict() for span in source_spans],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sc-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


#: Bumped when the identity formula changes, so a key computed under an older
#: formula can never silently collide with one computed under this formula.
SEMANTIC_IDENTITY_VERSION = "semantic-identity-v2"


def semantic_identity(
    *,
    context_field: str,
    raw_text: str,
    token_ids: Sequence[Any] = (),
) -> str:
    """Reusable-content identity: *what* the KV is of, not *where* it came from.

    Deliberately excludes `task_id`, `agent_id`, `request_id`, the session
    revision and the source spans. Two byte-identical blocks segmented out of
    two different agents' transcripts describe the same KV, and this returns
    the same string for both. Provenance keeps its own identity in
    :func:`semantic_block_id`, which is unchanged.

    The basis is part of the identity. A block whose authoritative token ids
    were captured is identified by those ids, because the compiled KV is a
    function of the ids and not of the text that produced them. A block with
    only text is identified by the text and tagged separately, so the two never
    alias on an equality that has not been shown to hold at the token level.
    """

    if token_ids:
        basis = "tok"
        payload = json.dumps(list(token_ids), separators=(",", ":"), default=str)
    else:
        basis = "txt"
        payload = raw_text
    encoded = json.dumps(
        {
            "identity_version": SEMANTIC_IDENTITY_VERSION,
            "basis": basis,
            "context_field": context_field,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"si-{basis}-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def semantic_cache_key(
    *,
    semantic_identity: str,
    context_field: str,
    model_fingerprint: str,
    tokenizer_fingerprint: str,
    segmentation_policy_version: str,
    recompute_policy_version: str,
) -> str:
    """Runtime cache identity: reusable content, qualified by compatibility.

    This used to take the `semantic_block_id`, which carries `task_id` and the
    source spans, so two byte-identical blocks from two tasks could never share
    a compiled KV block. The content identity replaces it; the compatibility
    fingerprints stay, because they are what makes a cached block *usable*
    rather than merely equal.
    """

    value = {
        "identity_version": SEMANTIC_IDENTITY_VERSION,
        "semantic_identity": semantic_identity,
        "context_field": context_field,
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "segmentation_policy_version": segmentation_policy_version,
        "recompute_policy_version": recompute_policy_version,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticSubContext:
    sub_context_id: str
    task_id: str
    session_revision: int
    context_field: str
    title: str
    question_or_objective: str
    raw_text: str
    canonical_text: str
    source_spans: list[SourceSpan]
    content_hash: str
    cache_key: str
    version: int
    token_count: int
    is_mutable: bool
    reuse_candidate: bool
    semantic_independence_reason: str
    segmentation_confidence: float
    policy_version: str
    created_at: str
    updated_at: str
    supersedes: str | None = None
    invalidated: bool = False
    derived_from: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    token_ids: tuple[Any, ...] = ()
    tokenizer_fingerprint: str = "unknown-tokenizer"
    model_fingerprint: str = "unknown-model"
    recompute_policy_version: str = "batched-v1"
    #: Reusable-content identity. Defaulted so a row written before the
    #: identity split still loads; an empty value reads as "provenance-scoped
    #: identity only", which is what those rows actually had.
    semantic_identity: str = ""

    @property
    def source_revision_identity(self) -> str:
        """Exact logical-source revision owning this sibling block."""

        return str((self.metadata or {}).get("source_revision_identity") or "")

    @property
    def source_order(self) -> int:
        return int((self.metadata or {}).get("source_order", 0))

    def token_accounting(self) -> TokenCount:
        """`token_count` with its unit, read from the provenance already stored.

        The unit was never absent from these rows -- `tokenizer_fingerprint`,
        `token_ids` and the segmentation metadata were all being written. What
        was absent was anyone interpreting them, so the same integer meant model
        tokens on the agent path and regex tokens on the plan-preparation path
        and nothing downstream could tell.

        Conservative for rows that predate the unit metadata: no recorded unit
        means `EstimatedTokenCount`, because an old number is not authoritative
        for being old.
        """

        declared = str((self.metadata or {}).get("token_unit") or "")
        identity = (self.metadata or {}).get("authoritative_tokenizer_identity")
        if declared == TokenUnit.MODEL.value and isinstance(identity, dict):
            return ModelTokenCount(
                value=self.token_count,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
                model_id=str(identity.get("model_id") or self.model_fingerprint),
            )
        return EstimatedTokenCount(
            value=self.token_count,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_spans"] = [span.to_dict() for span in self.source_spans]
        row["token_accounting"] = self.token_accounting().to_dict()
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SemanticSubContext":
        value = dict(row)
        # Derived on read from the stored provenance, never trusted from the
        # artifact: a file claiming model tokens does not make them so.
        value.pop("token_accounting", None)
        value["source_spans"] = [SourceSpan.from_dict(span) for span in row["source_spans"]]
        value["derived_from"] = tuple(row.get("derived_from") or ())
        value["metadata"] = dict(row.get("metadata") or {})
        value["token_ids"] = tuple(row.get("token_ids") or ())
        value["tokenizer_fingerprint"] = str(
            row.get("tokenizer_fingerprint") or "unknown-tokenizer"
        )
        value["model_fingerprint"] = str(
            row.get("model_fingerprint") or "unknown-model"
        )
        value["recompute_policy_version"] = str(
            row.get("recompute_policy_version") or "batched-v1"
        )
        # A row written before the identity split has no content identity, and
        # is not given one here: deriving it on read would invent an equality
        # the row never recorded.
        value["semantic_identity"] = str(row.get("semantic_identity") or "")
        return cls(**value)

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        session_revision: int,
        context_field: str,
        title: str,
        question_or_objective: str,
        raw_text: str,
        source_spans: list[SourceSpan],
        token_count: int,
        token_accounting: TokenCount | None = None,
        is_mutable: bool,
        reuse_candidate: bool,
        semantic_independence_reason: str,
        segmentation_confidence: float,
        policy_version: str = "semantic-v1",
        recompute_policy_version: str = "batched-v1",
        model_fingerprint: str = "unknown-model",
        tokenizer_fingerprint: str = "unknown-tokenizer",
        token_ids: tuple[Any, ...] = (),
        version: int = 1,
        supersedes: str | None = None,
        derived_from: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> "SemanticSubContext":
        # Declaring the unit is how a producer says what it counted with. A
        # producer that says nothing gets ESTIMATED, which is the safe reading
        # and the one every pre-unit caller deserves.
        metadata = dict(metadata or {})
        if token_accounting is not None:
            if token_accounting.value != token_count:
                raise ValueError(
                    "token_accounting must agree with token_count: "
                    f"{token_accounting.value} != {token_count}"
                )
            metadata.setdefault("token_unit", token_accounting.unit.value)
            if isinstance(token_accounting, ModelTokenCount):
                metadata.setdefault(
                    "authoritative_tokenizer_identity",
                    {
                        "model_id": token_accounting.model_id,
                        "fingerprint": token_accounting.tokenizer_fingerprint,
                    },
                )
        token_ids = tuple(token_ids)
        if token_ids and len(token_ids) != token_count:
            raise ValueError(
                "authoritative token IDs must match the recorded token count"
            )
        # Provenance identity: where this occurrence came from. Task-scoped on
        # purpose -- it is the revision anchor for supersede/version history.
        block_id = semantic_block_id(
            task_id=task_id,
            context_field=context_field,
            source_spans=source_spans,
        )
        digest = semantic_content_hash(raw_text)
        # Content identity: what the KV is of. Carries no task, agent or span.
        identity = semantic_identity(
            context_field=context_field,
            raw_text=raw_text,
            token_ids=token_ids,
        )
        cache_key = semantic_cache_key(
            semantic_identity=identity,
            context_field=context_field,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            segmentation_policy_version=policy_version,
            recompute_policy_version=recompute_policy_version,
        )
        now = _now()
        return cls(
            sub_context_id=f"{block_id}:v{version}:{digest[:8]}",
            task_id=task_id,
            session_revision=session_revision,
            context_field=context_field,
            title=title,
            question_or_objective=question_or_objective,
            raw_text=raw_text,
            canonical_text=raw_text,
            source_spans=list(source_spans),
            content_hash=digest,
            cache_key=cache_key,
            version=version,
            token_count=token_count,
            is_mutable=is_mutable,
            reuse_candidate=reuse_candidate,
            semantic_independence_reason=semantic_independence_reason,
            segmentation_confidence=float(segmentation_confidence),
            policy_version=policy_version,
            created_at=now,
            updated_at=now,
            supersedes=supersedes,
            derived_from=derived_from,
            metadata=metadata,
            token_ids=token_ids,
            tokenizer_fingerprint=tokenizer_fingerprint,
            model_fingerprint=model_fingerprint,
            recompute_policy_version=recompute_policy_version,
            semantic_identity=identity,
        )


class SubContextStore(ABC):
    @abstractmethod
    def create(self, value: SemanticSubContext) -> SemanticSubContext: ...

    @abstractmethod
    def update(self, previous_id: str, value: SemanticSubContext) -> SemanticSubContext: ...

    @abstractmethod
    def settle(self, sub_context_id: str) -> SemanticSubContext: ...

    @abstractmethod
    def get(self, sub_context_id: str) -> SemanticSubContext | None: ...

    @abstractmethod
    def list_by_task(self, task_id: str, *, include_invalidated: bool = False) -> list[SemanticSubContext]: ...

    @abstractmethod
    def list_by_field(self, task_id: str, context_field: str) -> list[SemanticSubContext]: ...

    @abstractmethod
    def list_by_revision(self, task_id: str, revision: int) -> list[SemanticSubContext]: ...

    @abstractmethod
    def invalidate(self, sub_context_id: str) -> None: ...

    @abstractmethod
    def supersede(self, previous_id: str, value: SemanticSubContext) -> SemanticSubContext: ...

    @abstractmethod
    def find_by_cache_key(self, cache_key: str) -> SemanticSubContext | None: ...

    @abstractmethod
    def find_by_content_hash(self, task_id: str, digest: str) -> list[SemanticSubContext]: ...

    @abstractmethod
    def find_by_source_id(self, task_id: str, source_id: str) -> list[SemanticSubContext]: ...

    @abstractmethod
    def revision_history(self, sub_context_id: str) -> list[SemanticSubContext]: ...

    @abstractmethod
    def replace_source_revision(
        self,
        task_id: str,
        source_id: str,
        source_revision_identity: str,
        values: Sequence[SemanticSubContext],
    ) -> list[SemanticSubContext]: ...

    def replace_source_revisions(
        self,
        task_id: str,
        revisions: Sequence[tuple[str, str, Sequence[SemanticSubContext]]],
    ) -> list[SemanticSubContext]:
        """Replace several source revisions as one unit: all of them or none.

        One ingest revision can change several sources. Storing them one by one
        lets a failure half way leave some sources at the new revision and the
        rest at the old one. Stores that can commit them together override
        this; the default is only as atomic as `replace_source_revision`.
        """

        stored: list[SemanticSubContext] = []
        for source_id, source_revision_identity, values in revisions:
            stored.extend(
                self.replace_source_revision(
                    task_id, source_id, source_revision_identity, values
                )
            )
        return stored


class SQLiteSubContextStore(SubContextStore):
    """Small durable repository; payload JSON keeps schema migrations conservative."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_subcontexts (
                sub_context_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                session_revision INTEGER NOT NULL,
                context_field TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                version INTEGER NOT NULL,
                invalidated INTEGER NOT NULL,
                supersedes TEXT,
                source_ids TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sc_task ON semantic_subcontexts(task_id)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sc_field ON semantic_subcontexts(task_id, context_field)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sc_revision ON semantic_subcontexts(task_id, session_revision)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sc_hash ON semantic_subcontexts(task_id, content_hash)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sc_cache ON semantic_subcontexts(cache_key)")
        self._connection.commit()

    @staticmethod
    def _payload(value: SemanticSubContext) -> str:
        return json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True)

    def create(self, value: SemanticSubContext) -> SemanticSubContext:
        with self._lock:
            duplicate = self._connection.execute(
                "SELECT payload FROM semantic_subcontexts WHERE task_id=? AND content_hash=? AND context_field=? AND cache_key=? AND invalidated=0 ORDER BY version DESC LIMIT 1",
                (value.task_id, value.content_hash, value.context_field, value.cache_key),
            ).fetchone()
            if duplicate:
                return SemanticSubContext.from_dict(json.loads(duplicate["payload"]))
            self._insert(value)
            return value

    def _insert(self, value: SemanticSubContext, *, commit: bool = True) -> None:
        source_ids = json.dumps(sorted({span.source_id for span in value.source_spans}))
        self._connection.execute(
            "INSERT INTO semantic_subcontexts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                value.sub_context_id,
                value.task_id,
                value.session_revision,
                value.context_field,
                value.content_hash,
                value.cache_key,
                value.version,
                int(value.invalidated),
                value.supersedes,
                source_ids,
                self._payload(value),
            ),
        )
        if commit:
            self._connection.commit()

    def update(self, previous_id: str, value: SemanticSubContext) -> SemanticSubContext:
        with self._lock:
            previous = self.get(previous_id)
            if previous is None:
                raise KeyError(previous_id)
            if value.task_id != previous.task_id or value.context_field != previous.context_field:
                raise ValueError("an update cannot change task or context field")
            self.invalidate(previous_id)
            next_version = max(value.version, previous.version + 1)
            block_id = value.sub_context_id.split(":v", 1)[0]
            replacement = replace(
                value,
                sub_context_id=f"{block_id}:v{next_version}:{value.content_hash[:8]}",
                version=next_version,
                supersedes=previous_id,
                updated_at=_now(),
            )
            self._insert(replacement)
            return replacement

    def settle(self, sub_context_id: str) -> SemanticSubContext:
        """Record that an append-only entry can no longer change.

        This is not an update: the content, content hash, cache key and version
        are untouched, so a settled entry keeps the cache key it was compiled
        under and stays reusable across turns. Only the mutability flags change,
        which is why it rewrites the payload in place rather than superseding
        the record the way :meth:`update` does.
        """

        with self._lock:
            current = self.get(sub_context_id)
            if current is None:
                raise KeyError(sub_context_id)
            if current.invalidated:
                raise ValueError(
                    f"cannot settle an invalidated sub-context: {sub_context_id}"
                )
            if not current.is_mutable and current.reuse_candidate:
                return current
            settled = replace(
                current,
                is_mutable=False,
                reuse_candidate=True,
                updated_at=_now(),
            )
            self._connection.execute(
                "UPDATE semantic_subcontexts SET payload=? WHERE sub_context_id=?",
                (self._payload(settled), sub_context_id),
            )
            self._connection.commit()
            return settled

    def _rows(self, query: str, args: tuple[Any, ...]) -> list[SemanticSubContext]:
        with self._lock:
            rows = self._connection.execute(query, args).fetchall()
        return [SemanticSubContext.from_dict(json.loads(row["payload"])) for row in rows]

    def get(self, sub_context_id: str) -> SemanticSubContext | None:
        rows = self._rows("SELECT payload FROM semantic_subcontexts WHERE sub_context_id=?", (sub_context_id,))
        return rows[0] if rows else None

    def list_by_task(self, task_id: str, *, include_invalidated: bool = False) -> list[SemanticSubContext]:
        where = "task_id=?" if include_invalidated else "task_id=? AND invalidated=0"
        return self._rows(f"SELECT payload FROM semantic_subcontexts WHERE {where} ORDER BY session_revision, version", (task_id,))

    def list_by_field(self, task_id: str, context_field: str) -> list[SemanticSubContext]:
        return self._rows(
            "SELECT payload FROM semantic_subcontexts WHERE task_id=? AND context_field=? ORDER BY version",
            (task_id, context_field),
        )

    def list_by_revision(self, task_id: str, revision: int) -> list[SemanticSubContext]:
        return self._rows(
            "SELECT payload FROM semantic_subcontexts WHERE task_id=? AND session_revision=? ORDER BY context_field, version",
            (task_id, revision),
        )

    def invalidate(self, sub_context_id: str) -> None:
        with self._lock:
            current = self.get(sub_context_id)
            if current is None:
                raise KeyError(sub_context_id)
            invalid = replace(current, invalidated=True, updated_at=_now())
            self._connection.execute(
                "UPDATE semantic_subcontexts SET invalidated=1, payload=? WHERE sub_context_id=?",
                (self._payload(invalid), sub_context_id),
            )
            self._connection.commit()

    def supersede(self, previous_id: str, value: SemanticSubContext) -> SemanticSubContext:
        return self.update(previous_id, value)

    def find_by_cache_key(self, cache_key: str) -> SemanticSubContext | None:
        rows = self._rows(
            "SELECT payload FROM semantic_subcontexts WHERE cache_key=? AND invalidated=0 ORDER BY version DESC LIMIT 1",
            (cache_key,),
        )
        return rows[0] if rows else None

    def find_by_content_hash(self, task_id: str, digest: str) -> list[SemanticSubContext]:
        return self._rows(
            "SELECT payload FROM semantic_subcontexts WHERE task_id=? AND content_hash=? ORDER BY version",
            (task_id, digest),
        )

    def find_by_source_id(self, task_id: str, source_id: str) -> list[SemanticSubContext]:
        return [
            value
            for value in self.list_by_task(task_id, include_invalidated=True)
            if any(span.source_id == source_id for span in value.source_spans)
        ]

    def revision_history(self, sub_context_id: str) -> list[SemanticSubContext]:
        value = self.get(sub_context_id)
        if value is None:
            return []
        history = [value]
        while history[-1].supersedes:
            previous = self.get(history[-1].supersedes or "")
            if previous is None:
                break
            history.append(previous)
        return list(reversed(history))

    @staticmethod
    def _source_position(value: SemanticSubContext, source_id: str) -> tuple[int, int, str]:
        spans = [span for span in value.source_spans if span.source_id == source_id]
        if not spans:
            return (2**63 - 1, 2**63 - 1, value.sub_context_id)
        return (
            min(span.start for span in spans),
            max(span.end for span in spans),
            value.sub_context_id,
        )

    def replace_source_revision(
        self,
        task_id: str,
        source_id: str,
        source_revision_identity: str,
        values: Sequence[SemanticSubContext],
    ) -> list[SemanticSubContext]:
        """Atomically replace one source revision with its complete sibling set.

        A source revision owns a set of blocks.  Blocks inside that set never
        supersede one another; only a later source revision replaces the set.
        """

        siblings = self._checked_siblings(task_id, source_id, values)
        if not siblings:
            return []
        siblings.sort(key=lambda value: self._source_position(value, source_id))
        with self._lock:
            try:
                stored = self._replace_source_revision_uncommitted(
                    task_id, source_id, source_revision_identity, siblings
                )
                self._connection.commit()
                return stored
            except Exception:
                self._connection.rollback()
                raise

    def replace_source_revisions(
        self,
        task_id: str,
        revisions: Sequence[tuple[str, str, Sequence[SemanticSubContext]]],
    ) -> list[SemanticSubContext]:
        """Replace several source revisions in one SQLite transaction."""

        with self._lock:
            try:
                stored: list[SemanticSubContext] = []
                for source_id, source_revision_identity, values in revisions:
                    siblings = self._checked_siblings(task_id, source_id, values)
                    if not siblings:
                        continue
                    siblings.sort(
                        key=lambda value, source=source_id: self._source_position(value, source)
                    )
                    stored.extend(
                        self._replace_source_revision_uncommitted(
                            task_id, source_id, source_revision_identity, siblings
                        )
                    )
                self._connection.commit()
                return stored
            except Exception:
                self._connection.rollback()
                raise

    def _checked_siblings(
        self, task_id: str, source_id: str, values: Sequence[SemanticSubContext]
    ) -> list[SemanticSubContext]:
        siblings = list(values)
        if not siblings:
            return []
        if any(value.task_id != task_id for value in siblings):
            raise ValueError("source sibling set cannot mix tasks")
        if any(
            not any(span.source_id == source_id for span in value.source_spans)
            for value in siblings
        ):
            raise ValueError("every source sibling must carry the owning source_id")
        fields = {value.context_field for value in siblings}
        revisions = {value.session_revision for value in siblings}
        if len(fields) != 1 or len(revisions) != 1:
            raise ValueError("source sibling set cannot mix fields or revisions")
        return siblings

    def _replace_source_revision_uncommitted(
        self,
        task_id: str,
        source_id: str,
        source_revision_identity: str,
        siblings: list[SemanticSubContext],
    ) -> list[SemanticSubContext]:
        """The body of a source-revision replacement; the caller commits or rolls back."""

        active = [
            value
            for value in self.find_by_source_id(task_id, source_id)
            if not value.invalidated
        ]
        same_revision = [
            value
            for value in active
            if value.source_revision_identity == source_revision_identity
        ]
        if same_revision:
            existing = sorted(
                same_revision,
                key=lambda value: self._source_position(value, source_id),
            )
            signatures = lambda rows: [
                (
                    value.content_hash,
                    tuple(
                        (span.start, span.end)
                        for span in value.source_spans
                        if span.source_id == source_id
                    ),
                )
                for value in rows
            ]
            if signatures(existing) == signatures(siblings):
                return existing
            raise ValueError(
                "source revision was already stored with a different sibling set"
            )

        old = sorted(
            active,
            key=lambda value: self._source_position(value, source_id),
        )
        old_ids = tuple(value.sub_context_id for value in old)
        next_version = max((value.version for value in old), default=0) + 1
        for value in old:
            invalid = replace(value, invalidated=True, updated_at=_now())
            self._connection.execute(
                "UPDATE semantic_subcontexts SET invalidated=1, payload=? "
                "WHERE sub_context_id=?",
                (self._payload(invalid), value.sub_context_id),
            )

        stored: list[SemanticSubContext] = []
        for index, value in enumerate(siblings):
            metadata = dict(value.metadata or {})
            metadata["source_revision_identity"] = source_revision_identity
            metadata["source_sibling_index"] = index
            metadata["source_sibling_count"] = len(siblings)
            metadata["superseded_source_sibling_ids"] = list(old_ids)
            block_id = value.sub_context_id.split(":v", 1)[0]
            predecessor = old[index].sub_context_id if index < len(old) else None
            replacement = replace(
                value,
                sub_context_id=(
                    f"{block_id}:v{next_version}:{value.content_hash[:8]}"
                ),
                version=next_version,
                supersedes=predecessor,
                derived_from=tuple(dict.fromkeys((*value.derived_from, *old_ids))),
                metadata=metadata,
                updated_at=_now(),
            )
            self._insert(replacement, commit=False)
            stored.append(replacement)
        return stored

    def close(self) -> None:
        self._connection.close()
