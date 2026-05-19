"""Order-invariant block-level KV cache policy prototype.

This module models the cache behavior that the SGLang runtime patch should
eventually implement: lookup/store by semantic sub-context `cache_key`, not by
prompt prefix position.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BlockKVEntry:
    cache_key: str
    block_id: str
    token_count: int
    first_seen_sequence: str
    first_seen_position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockKVLookup:
    sequence_id: str
    position: int
    block_id: str
    cache_key: str
    token_count: int
    cache_hit: bool
    reused_tokens: int
    prefill_tokens: int
    first_seen_sequence: str
    first_seen_position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrderInvariantBlockKVCache:
    """A minimal semantic block KV cache keyed by `cache_key`."""

    def __init__(self) -> None:
        self._entries: dict[str, BlockKVEntry] = {}

    def lookup_or_store(
        self,
        *,
        sequence_id: str,
        position: int,
        block_id: str,
        cache_key: str,
        token_count: int,
    ) -> BlockKVLookup:
        entry = self._entries.get(cache_key)
        if entry is not None:
            return BlockKVLookup(
                sequence_id=sequence_id,
                position=position,
                block_id=block_id,
                cache_key=cache_key,
                token_count=token_count,
                cache_hit=True,
                reused_tokens=token_count,
                prefill_tokens=0,
                first_seen_sequence=entry.first_seen_sequence,
                first_seen_position=entry.first_seen_position,
            )

        entry = BlockKVEntry(
            cache_key=cache_key,
            block_id=block_id,
            token_count=token_count,
            first_seen_sequence=sequence_id,
            first_seen_position=position,
        )
        self._entries[cache_key] = entry
        return BlockKVLookup(
            sequence_id=sequence_id,
            position=position,
            block_id=block_id,
            cache_key=cache_key,
            token_count=token_count,
            cache_hit=False,
            reused_tokens=0,
            prefill_tokens=token_count,
            first_seen_sequence=sequence_id,
            first_seen_position=position,
        )


def prefix_reuse_tokens(
    previous_sequence: Iterable[dict[str, Any]],
    current_sequence: Iterable[dict[str, Any]],
) -> int:
    """Estimate native prefix-cache reuse at block granularity.

    Reuse stops at the first reordered or changed block, matching the limitation
    we want to overcome for semantic sub-context blocks.
    """

    reused = 0
    for previous, current in zip(previous_sequence, current_sequence):
        if previous["cache_key"] != current["cache_key"]:
            break
        reused += int(current["token_count"])
    return reused


def summarize_lookups(lookups: list[BlockKVLookup]) -> dict[str, Any]:
    token_total = sum(lookup.token_count for lookup in lookups)
    reused_total = sum(lookup.reused_tokens for lookup in lookups)
    prefill_total = sum(lookup.prefill_tokens for lookup in lookups)
    return {
        "blocks": len(lookups),
        "tokens": token_total,
        "cache_hits": sum(1 for lookup in lookups if lookup.cache_hit),
        "reused_tokens": reused_total,
        "prefill_tokens": prefill_total,
        "reuse_ratio": reused_total / token_total if token_total else 0.0,
    }
