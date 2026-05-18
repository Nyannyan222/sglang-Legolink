"""Runtime-level prototype for independent sub-context prefill/cache.

This is a lightweight proof layer. It does not patch SGLang internals yet; it
models the block contract that an SGLang implementation should enforce:

- each sub-context is a separate prefill unit,
- each unit has its own cache key,
- no pairwise attention is allowed between different sub-context blocks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]+|[\u4e00-\u9fff]|[^\s]")


def estimate_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


@dataclass(frozen=True)
class PrefillBlock:
    block_id: str
    cache_key: str
    context_field: str
    functional_role: str
    text: str
    token_count: int
    source_sub_context_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_subcontext_row(cls, row: dict[str, Any]) -> "PrefillBlock":
        plan = row.get("sglang_storage_plan") or {}
        text = row.get("sub_context") or row.get("text") or ""
        cache_key = str(plan.get("cache_key") or row.get("content_hash") or stable_hash(text))
        return cls(
            block_id=str(plan.get("prefill_unit_id") or row.get("id") or cache_key[:16]),
            cache_key=cache_key,
            context_field=str(row.get("context_field") or ""),
            functional_role=str(row.get("functional_role") or ""),
            text=text,
            token_count=len(estimate_tokens(text)),
            source_sub_context_id=str(row.get("id") or ""),
            metadata={
                "task_id": row.get("task_id", ""),
                "title": row.get("title", ""),
                "attention_policy_target": row.get("attention_policy_target", ""),
                "semantic_independence_target": row.get("semantic_independence_target", ""),
                "cache_scope": plan.get("cache_scope", ""),
                "kv_reuse_granularity": plan.get("kv_reuse_granularity", ""),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    block_id: str
    token_count: int
    source_sub_context_id: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrefillResult:
    block_id: str
    cache_key: str
    source_sub_context_id: str
    token_count: int
    cache_hit: bool
    prefill_tokens: int
    reused_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IsolationReport:
    block_count: int
    token_count: int
    intra_block_allowed_edges: int
    cross_block_allowed_edges: int
    cross_block_disallowed_edges: int
    pairwise_attention_ok: bool
    block_pair_matrix: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


class IndependentPrefillCache:
    """In-memory sub-context cache keyed by block content hash."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def prefill(self, block: PrefillBlock) -> PrefillResult:
        entry = self._entries.get(block.cache_key)
        if entry:
            return PrefillResult(
                block_id=block.block_id,
                cache_key=block.cache_key,
                source_sub_context_id=block.source_sub_context_id,
                token_count=block.token_count,
                cache_hit=True,
                prefill_tokens=0,
                reused_tokens=block.token_count,
            )

        self._entries[block.cache_key] = CacheEntry(
            cache_key=block.cache_key,
            block_id=block.block_id,
            token_count=block.token_count,
            source_sub_context_id=block.source_sub_context_id,
            digest=stable_hash(block.text),
        )
        return PrefillResult(
            block_id=block.block_id,
            cache_key=block.cache_key,
            source_sub_context_id=block.source_sub_context_id,
            token_count=block.token_count,
            cache_hit=False,
            prefill_tokens=block.token_count,
            reused_tokens=0,
        )

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())


def block_isolation_report(blocks: Iterable[PrefillBlock]) -> IsolationReport:
    block_list = list(blocks)
    token_count = sum(block.token_count for block in block_list)
    intra_edges = sum(block.token_count * (block.token_count + 1) // 2 for block in block_list)
    cross_disallowed = 0
    matrix: list[dict[str, Any]] = []

    for left_index, left in enumerate(block_list):
        for right_index, right in enumerate(block_list):
            if left_index == right_index:
                allowed = left.token_count * (left.token_count + 1) // 2
                disallowed = 0
                relation = "self"
            else:
                allowed = 0
                disallowed = left.token_count * right.token_count
                relation = "cross_block"
                cross_disallowed += disallowed
            matrix.append(
                {
                    "query_block_id": left.block_id,
                    "key_block_id": right.block_id,
                    "relation": relation,
                    "query_tokens": left.token_count,
                    "key_tokens": right.token_count,
                    "allowed_attention_edges": allowed,
                    "disallowed_attention_edges": disallowed,
                }
            )

    return IsolationReport(
        block_count=len(block_list),
        token_count=token_count,
        intra_block_allowed_edges=intra_edges,
        cross_block_allowed_edges=0,
        cross_block_disallowed_edges=cross_disallowed,
        pairwise_attention_ok=True,
        block_pair_matrix=matrix,
    )


def build_blocks(rows: Iterable[dict[str, Any]]) -> list[PrefillBlock]:
    return [PrefillBlock.from_subcontext_row(row) for row in rows]
