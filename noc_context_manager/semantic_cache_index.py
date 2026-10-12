"""Logical RadixTree roots over the one shared physical KV registry.

ARCHITECTURE_STAGE: 6 -- Logical Context Assembly + Reuse Plan (namespace-first lookup)
CANONICAL_ENTRYPOINT: ``scheduler.NOCRequestPlanner.plan`` (through ``cache_namespace.NamespaceResolver``)

"One category has one logical RadixTree namespace/root, but all logical roots share the same physical
SGLang RadixCache and KV pool."

What this module is, and is not:

* It is a **host-side index**: ``namespace -> logical root -> logical entries``. A logical entry is a *name*
  for a compiled semantic block (its own ``cache_key`` for the entry that was compiled, or an *alias* name
  for a block another entry compiled). Each entry references one **runtime cache key**, which is the key the
  patched runtime already registers its block under.
* It is **not** a second cache. There is no second RadixCache, allocator, KV pool, eviction manager or
  reference counter: every number the runtime owns (slots, radix node, lock refs, active link refcount, pin
  count, eviction) stays in ``RadixCache.legolink_blocks`` and nothing here reads or writes it.
* A namespace is deliberately **not** part of the physical identity. Two entries in two roots that carry the
  same ``physical_kv_identity`` reference the *same* runtime block, so a block is compiled once however many
  logical names it has.

Why this can be done without touching the runtime. The runtime registry is keyed by one ``cache_key`` per
physical block and a Link names blocks by that key; presenting a consumer block under the producer's key is
the same shape as a repeated occurrence of one compiled block, which the runtime already supports (its
acquire de-duplicates repeated keys, and one release balances it). Compile also uses the cache key as the
radix ``extra_key``, so each compiled block already lives in its own subtree of the single radix root;
per-namespace roots are therefore a *host* lookup layer, not a tree the runtime would have to grow.

Lifecycle. An alias is derived state. It is valid only while the catalog still holds a live entry for its
runtime key *with the same physical identity*; the runtime is the only authority on residency, so the host
learns of an eviction through the catalog (``mark_stale`` / ``invalidate`` -- fed by the runtime's own
"missing/stale" Link error) and every alias of that key goes stale in the same call. No KV is duplicated to
avoid solving this.

Namespaces are plain strings here so the catalog (which owns an index) can import this module without a
cycle; ``cache_namespace`` pins that these names equal ``PERSISTENT_NAMESPACES``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: The persistent reusable roots, in lookup order. ``mutable_current_turn`` and ``no_cache`` have no root.
PERSISTENT_ROOTS: tuple[str, ...] = (
    "instruction",
    "tool_schema",
    "conversation",
    "tool_output",
    "workspace",
)

KIND_PRIMARY = "primary"
KIND_ALIAS = "alias"
ALIAS_SAME_ROOT = "same_root"
ALIAS_CROSS_ROOT = "cross_root"

_EVICTION_LOG_LIMIT = 512


@dataclass(frozen=True)
class LogicalEntry:
    """One name, in one logical root, for a compiled semantic block."""

    namespace: str
    logical_entry_key: str
    #: The key the runtime registered the block under -- what a Link names.
    runtime_cache_key: str
    physical_kv_identity: str | None
    kind: str = KIND_PRIMARY
    semantic_identity: str | None = None
    physical_token_ids_sha256: str | None = None
    context_field: str | None = None
    token_count: int = 0
    compiled_at: float | None = None

    @property
    def is_alias(self) -> bool:
        return self.kind == KIND_ALIAS

    @property
    def aliasable(self) -> bool:
        """Can another name be presented through this entry? Needs the exact token layout it was compiled
        under and the semantic identity the runtime matches, both recorded at Compile time."""

        return bool(self.physical_token_ids_sha256 and self.semantic_identity and self.physical_kv_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "logical_entry_key": self.logical_entry_key,
            "runtime_cache_key": self.runtime_cache_key,
            "physical_kv_identity": self.physical_kv_identity,
            "kind": self.kind,
            "context_field": self.context_field,
            "token_count": self.token_count,
        }


@dataclass
class _AliasRecord:
    entry: LogicalEntry
    created_at: float
    stale: bool = False
    stale_reason: str | None = None


@dataclass(frozen=True)
class LogicalRoot:
    """A read-only view of one namespace's entries (primary entries, then live aliases)."""

    namespace: str
    entries: tuple[LogicalEntry, ...]

    @property
    def primary_entries(self) -> tuple[LogicalEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.is_alias)

    @property
    def alias_entries(self) -> tuple[LogicalEntry, ...]:
        return tuple(entry for entry in self.entries if entry.is_alias)


class SemanticCacheIndex:
    """``namespace -> logical root -> logical entries -> runtime cache key``, derived from one catalog.

    Primary entries are the catalog's own rows (the host record of what it compiled), so they cannot drift
    from it. Aliases are the only state held here, and they are re-derivable: dropping them changes nothing
    but how many names a block currently has.
    """

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog
        self._aliases: dict[tuple[str, str], _AliasRecord] = {}
        self._guard = threading.RLock()
        self.eviction_log: list[dict[str, Any]] = []
        self.aliases_invalidated = 0
        self.aliases_recorded = 0

    # --- entries ------------------------------------------------------------

    @staticmethod
    def _primary(key: str, meta: Mapping[str, Any]) -> LogicalEntry | None:
        namespace = str(meta.get("namespace") or "")
        if namespace not in PERSISTENT_ROOTS:
            return None
        return LogicalEntry(
            namespace=namespace,
            logical_entry_key=key,
            runtime_cache_key=key,
            physical_kv_identity=meta.get("physical_kv_identity") or None,
            kind=KIND_PRIMARY,
            semantic_identity=meta.get("semantic_identity") or None,
            physical_token_ids_sha256=meta.get("physical_token_ids_sha256") or None,
            context_field=meta.get("context_field") or None,
            token_count=int(meta.get("token_count") or 0),
            compiled_at=meta.get("compiled_at"),
        )

    def _live_alias(self, record: _AliasRecord) -> bool:
        if record.stale:
            return False
        target = self._catalog.metadata(record.entry.runtime_cache_key)
        if target is None or not self._catalog.contains(record.entry.runtime_cache_key):
            return False
        # A key recompiled with different content is a different physical block.
        return (target.get("physical_kv_identity") or None) == record.entry.physical_kv_identity

    def lookup(
        self, namespace: str, logical_entry_key: str, *, any_root: bool = False
    ) -> LogicalEntry | None:
        """The primary entry named ``logical_entry_key`` in ``namespace``'s root, if it is live.

        ``any_root`` accepts an entry of any namespace that holds this key (``GLOBAL``: the namespace never
        stops a key that is already resident).
        """

        if not self._catalog.contains(logical_entry_key):
            return None
        meta = self._catalog.metadata(logical_entry_key) or {}
        entry_namespace = str(meta.get("namespace") or namespace)
        if not any_root and entry_namespace != namespace:
            return None
        built = self._primary(logical_entry_key, {**meta, "namespace": entry_namespace})
        if built is not None:
            return built
        # A live row outside the persistent roots (a legacy / GLOBAL compile of a mutable block): still a
        # resident physical block, just not a root entry.
        return LogicalEntry(
            namespace=entry_namespace,
            logical_entry_key=logical_entry_key,
            runtime_cache_key=logical_entry_key,
            physical_kv_identity=meta.get("physical_kv_identity") or None,
            semantic_identity=meta.get("semantic_identity") or None,
            physical_token_ids_sha256=meta.get("physical_token_ids_sha256") or None,
            context_field=meta.get("context_field") or None,
            token_count=int(meta.get("token_count") or 0),
            compiled_at=meta.get("compiled_at"),
        )

    def lookup_alias(self, namespace: str, logical_entry_key: str) -> LogicalEntry | None:
        with self._guard:
            record = self._aliases.get((namespace, logical_entry_key))
            if record is not None and self._live_alias(record):
                return record.entry
        return None

    def find_by_identity(self, digest: str | None) -> list[LogicalEntry]:
        """Live, aliasable primary entries with this physical identity, in root lookup order.

        Exact identity only: equal authoritative tokens, source bytes, model, tokenizer and policy versions.
        Similarity is never consulted.
        """

        if not digest:
            return []
        found: list[LogicalEntry] = []
        for key, meta in self._catalog.find_by_physical_identity(digest):
            entry = self._primary(key, meta)
            if entry is not None and entry.aliasable:
                found.append(entry)
        order = {name: index for index, name in enumerate(PERSISTENT_ROOTS)}
        found.sort(key=lambda item: order[item.namespace])  # stable: insertion order within a root
        return found

    # --- aliases ------------------------------------------------------------

    def record_alias(
        self,
        *,
        namespace: str,
        logical_entry_key: str,
        target: LogicalEntry,
        context_field: str | None = None,
        token_count: int | None = None,
    ) -> LogicalEntry:
        """Register ``namespace/logical_entry_key`` as another name for ``target``'s physical block."""

        if namespace not in PERSISTENT_ROOTS:
            raise ValueError(f"{namespace!r} has no logical root; only {PERSISTENT_ROOTS} do")
        if namespace == target.namespace and logical_entry_key == target.logical_entry_key:
            return target
        entry = LogicalEntry(
            namespace=namespace,
            logical_entry_key=logical_entry_key,
            runtime_cache_key=target.runtime_cache_key,
            physical_kv_identity=target.physical_kv_identity,
            kind=KIND_ALIAS,
            semantic_identity=target.semantic_identity,
            physical_token_ids_sha256=target.physical_token_ids_sha256,
            context_field=context_field,
            token_count=int(token_count if token_count is not None else target.token_count),
            compiled_at=target.compiled_at,
        )
        with self._guard:
            existing = self._aliases.get((namespace, logical_entry_key))
            if (
                existing is not None
                and not existing.stale
                and existing.entry.runtime_cache_key == entry.runtime_cache_key
                and existing.entry.physical_kv_identity == entry.physical_kv_identity
            ):
                return existing.entry
            self._aliases[(namespace, logical_entry_key)] = _AliasRecord(entry, created_at=time.time())
            self.aliases_recorded += 1
        return entry

    def aliases_of(self, runtime_cache_key: str, *, live_only: bool = True) -> list[LogicalEntry]:
        with self._guard:
            return [
                record.entry
                for record in self._aliases.values()
                if record.entry.runtime_cache_key == runtime_cache_key
                and (not live_only or self._live_alias(record))
            ]

    def logical_refs(self, runtime_cache_key: str) -> int:
        """How many logical names reference this physical block right now (its own, plus live aliases)."""

        primary = 1 if self._catalog.contains(runtime_cache_key) else 0
        return primary + len(self.aliases_of(runtime_cache_key))

    # --- lifecycle ----------------------------------------------------------

    def on_physical_removed(
        self, runtime_cache_key: str, entry: Mapping[str, Any] | None, *, reason: str
    ) -> list[LogicalEntry]:
        """The physical block behind ``runtime_cache_key`` is gone (stale, evicted or invalidated).

        Every alias of it goes stale in the same call. The entry itself is the catalog's row and the
        catalog is what just removed it, so there is nothing else to reconcile.
        """

        invalidated: list[LogicalEntry] = []
        with self._guard:
            for record in self._aliases.values():
                if record.entry.runtime_cache_key == runtime_cache_key and not record.stale:
                    record.stale = True
                    record.stale_reason = reason
                    invalidated.append(record.entry)
            self.aliases_invalidated += len(invalidated)
            meta = dict(entry or {})
            compiled_at = meta.get("compiled_at")
            self.eviction_log.append(
                {
                    "runtime_cache_key": runtime_cache_key,
                    "namespace": str(meta.get("namespace") or "unknown"),
                    "context_field": meta.get("context_field"),
                    "token_count": int(meta.get("token_count") or 0),
                    "residency_age_s": (
                        max(time.time() - float(compiled_at), 0.0) if compiled_at is not None else None
                    ),
                    "reason": reason,
                    "aliases_invalidated": [
                        {"namespace": item.namespace, "logical_entry_key": item.logical_entry_key}
                        for item in invalidated
                    ],
                }
            )
            del self.eviction_log[:-_EVICTION_LOG_LIMIT]
        return invalidated

    def sweep(self) -> int:
        """Mark stale every alias whose physical block is no longer live (a persistent catalog can lose a
        key through another process's append, which fires no hook here). Returns how many it marked."""

        marked = 0
        with self._guard:
            for record in self._aliases.values():
                if not record.stale and not self._live_alias(record):
                    record.stale = True
                    record.stale_reason = "physical_block_not_live"
                    marked += 1
            self.aliases_invalidated += marked
        return marked

    # --- views --------------------------------------------------------------

    def roots(self) -> dict[str, LogicalRoot]:
        self.sweep()
        entries: dict[str, list[LogicalEntry]] = {name: [] for name in PERSISTENT_ROOTS}
        for key, meta in self._catalog.live_entries():
            built = self._primary(key, meta)
            if built is not None:
                entries[built.namespace].append(built)
        with self._guard:
            for record in self._aliases.values():
                if not record.stale:
                    entries[record.entry.namespace].append(record.entry)
        return {name: LogicalRoot(name, tuple(items)) for name, items in entries.items()}

    def snapshot(self) -> dict[str, Any]:
        """Observability: per-root logical entries over the (single) set of physical blocks."""

        roots = self.roots()
        physical = {
            entry.runtime_cache_key
            for root in roots.values()
            for entry in root.entries
        }
        aliases = sum(len(root.alias_entries) for root in roots.values())
        evicted: dict[str, int] = {}
        for row in self.eviction_log:
            evicted[row["namespace"]] = evicted.get(row["namespace"], 0) + 1
        stale_tokens: dict[str, int] = {}
        for row in self.eviction_log:
            stale_tokens[row["namespace"]] = stale_tokens.get(row["namespace"], 0) + row["token_count"]
        return {
            "logical_roots": list(PERSISTENT_ROOTS),
            "logical_root_count": len(PERSISTENT_ROOTS),
            "physical_cache_instances": 1,
            "by_root": {
                name: {
                    "primary_entries": len(root.primary_entries),
                    "alias_entries": len(root.alias_entries),
                    "tokens": sum(entry.token_count for entry in root.primary_entries),
                }
                for name, root in roots.items()
            },
            "physical_blocks": len(physical),
            "logical_entries": sum(len(root.entries) for root in roots.values()),
            "physical_kv_dedup_count": aliases,
            "aliases_recorded": self.aliases_recorded,
            "aliases_invalidated": self.aliases_invalidated,
            "physical_removals_by_namespace": evicted,
            "removed_tokens_by_namespace": stale_tokens,
        }


def semantic_index_of(catalog: Any) -> SemanticCacheIndex:
    """The catalog's own index; a stand-in catalog that has none gets one lazily so a resolver still works."""

    index = getattr(catalog, "semantic_index", None)
    if index is None:
        index = SemanticCacheIndex(catalog)
        try:
            catalog.semantic_index = index
        except AttributeError:
            pass
    return index


__all__ = [
    "ALIAS_CROSS_ROOT",
    "ALIAS_SAME_ROOT",
    "KIND_ALIAS",
    "KIND_PRIMARY",
    "PERSISTENT_ROOTS",
    "LogicalEntry",
    "LogicalRoot",
    "SemanticCacheIndex",
    "semantic_index_of",
]
