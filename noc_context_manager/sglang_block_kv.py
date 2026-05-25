"""Semantic/canonical block-level KV storage prototype for OpenClaw.

This module models the runtime policy we want SGLang to internalize:
store each semantic sub-context as an independently addressable KV block and
look it up by exact content hash first, then by canonical semantic key when the
reuse policy allows it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticBlockMetadata:
    block_id: str
    context_field: str
    canonical_key: str
    content_hash: str
    token_count: int
    reuse_policy: str = "exact"
    canonical_text: str = ""
    raw_text: str = ""
    tool_name: str = ""
    tool_type: str = ""
    tool_call_id: str = ""
    is_mutating_tool: bool = False
    task_id: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SemanticBlockMetadata":
        raw_text = str(row.get("raw_text") or row.get("sub_context") or row.get("canonical_text") or "")
        canonical_text = str(row.get("canonical_text") or raw_text)
        content_hash = str(row.get("content_hash") or stable_hash(raw_text))
        block_id = str(row.get("sub_context_id") or row.get("id") or content_hash[:16])
        return cls(
            block_id=block_id,
            context_field=str(row.get("context_field", "")),
            canonical_key=str(row.get("canonical_key") or content_hash),
            content_hash=content_hash,
            token_count=int(row.get("token_count") or row.get("prompt_tokens") or len(normalize_text(canonical_text).split()) or 1),
            reuse_policy=str(row.get("reuse_policy") or "exact"),
            canonical_text=canonical_text,
            raw_text=raw_text,
            tool_name=str(row.get("tool_name", "")),
            tool_type=str(row.get("tool_type", "")),
            tool_call_id=str(row.get("tool_call_id", "")),
            is_mutating_tool=as_bool(row.get("is_mutating_tool")),
            task_id=str(row.get("task_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockKVRecord:
    block_id: str
    canonical_key: str
    content_hash: str
    token_count: int
    context_field: str
    reuse_policy: str
    kv_handle: str
    tool_name: str = ""
    tool_type: str = ""
    is_mutating_tool: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockKVLookupResult:
    query_block_id: str
    query_canonical_key: str
    query_content_hash: str
    lookup_status: str
    relation: str
    cache_hit: bool
    reused_tokens: int
    prefill_tokens: int
    matched_block_id: str = ""
    matched_kv_handle: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def allows_canonical_reuse(meta: SemanticBlockMetadata) -> bool:
    if meta.is_mutating_tool:
        return False
    if meta.reuse_policy in {"no_reuse", "no_reuse_mutating_tool"}:
        return False
    return bool(meta.canonical_key)


class SemanticBlockKVStore:
    """In-memory block KV registry keyed by content and canonical semantics."""

    def __init__(self) -> None:
        self.by_content_hash: dict[str, BlockKVRecord] = {}
        self.by_canonical_key: dict[str, BlockKVRecord] = {}
        self.events: list[dict[str, Any]] = []

    def lookup(self, meta: SemanticBlockMetadata) -> BlockKVLookupResult:
        if meta.is_mutating_tool or meta.reuse_policy in {"no_reuse", "no_reuse_mutating_tool"}:
            return self._miss(meta, "policy_no_reuse", "tool or policy forbids reuse")

        exact = self.by_content_hash.get(meta.content_hash)
        if exact is not None:
            return self._hit(meta, exact, "exact_content", "content_hash matched")

        canonical = self.by_canonical_key.get(meta.canonical_key) if allows_canonical_reuse(meta) else None
        if canonical is not None:
            return self._hit(meta, canonical, "canonical_semantic", "canonical_key matched")

        return self._miss(meta, "miss", "no exact or canonical KV block found")

    def store(self, meta: SemanticBlockMetadata, kv_handle: str | None = None) -> BlockKVRecord:
        handle = kv_handle or f"openclaw-kv:{meta.block_id}:{meta.content_hash[:12]}"
        record = BlockKVRecord(
            block_id=meta.block_id,
            canonical_key=meta.canonical_key,
            content_hash=meta.content_hash,
            token_count=meta.token_count,
            context_field=meta.context_field,
            reuse_policy=meta.reuse_policy,
            kv_handle=handle,
            tool_name=meta.tool_name,
            tool_type=meta.tool_type,
            is_mutating_tool=meta.is_mutating_tool,
        )
        self.by_content_hash[meta.content_hash] = record
        if allows_canonical_reuse(meta):
            self.by_canonical_key.setdefault(meta.canonical_key, record)
        self.events.append({"event": "store", **record.to_dict()})
        return record

    def lookup_or_store(self, meta: SemanticBlockMetadata) -> BlockKVLookupResult:
        result = self.lookup(meta)
        self.events.append({"event": "lookup", **result.to_dict()})
        if not result.cache_hit:
            self.store(meta)
        return result

    def _hit(self, meta: SemanticBlockMetadata, record: BlockKVRecord, relation: str, reason: str) -> BlockKVLookupResult:
        return BlockKVLookupResult(
            query_block_id=meta.block_id,
            query_canonical_key=meta.canonical_key,
            query_content_hash=meta.content_hash,
            lookup_status="hit",
            relation=relation,
            cache_hit=True,
            reused_tokens=meta.token_count,
            prefill_tokens=0,
            matched_block_id=record.block_id,
            matched_kv_handle=record.kv_handle,
            reason=reason,
        )

    def _miss(self, meta: SemanticBlockMetadata, relation: str, reason: str) -> BlockKVLookupResult:
        return BlockKVLookupResult(
            query_block_id=meta.block_id,
            query_canonical_key=meta.canonical_key,
            query_content_hash=meta.content_hash,
            lookup_status="miss",
            relation=relation,
            cache_hit=False,
            reused_tokens=0,
            prefill_tokens=meta.token_count,
            reason=reason,
        )

    def records(self) -> list[BlockKVRecord]:
        return list(self.by_content_hash.values())
