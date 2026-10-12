"""Logical cache namespaces over the one shared semantic KV registry.

ARCHITECTURE_STAGE: 6 -- Logical Context Assembly + Reuse Plan (namespace-first lookup)
CANONICAL_ENTRYPOINT: ``scheduler.NOCRequestPlanner.plan`` (it calls ``NamespaceResolver``)

Three different things, kept apart on purpose:

* **functional field** -- what kind of context a block is (``session.CONTEXT_FIELDS``);
* **cache namespace** -- where the *host* looks a block up and how it retains it
  (this module). A pure function of the field, the block's finality and an
  explicit opt-in;
* **physical KV identity** -- whether one compiled KV tensor may stand in for
  another block: exact token identity, model, tokenizer and the policy versions
  that decide how it was compiled (:func:`physical_kv_identity`).

A namespace is **host-side bookkeeping**. There is still one SGLang RadixCache,
one allocator and one KV pool; the runtime keeps owning slots, ranges, eviction,
reference counts, pinning, positions, ``imp_indices``, ``kvlink`` and recompute
positions. Nothing in this module reaches the runtime, and no module here emits
a field the runtime owns.

What the audit found (see ``docs/semantic_cache_namespaces.md``): today's
``cache_key`` already contains ``context_field``, so identical text under two
fields has two cache keys. Left alone, that would be two physical KV copies, so
each namespace is a **logical root** (``semantic_cache_index``) whose entries are
*names* for compiled blocks, and an entry may be a name for a block another root
compiled. Lookup is namespace-first:

1. look in the block's own root; a hit is used;
2. on a miss, ``GLOBAL`` accepts any exact physical-identity match, ``HARD_NAMESPACE``
   stops, and ``LOGICAL_NAMESPACE`` accepts one only for a pair listed in
   ``CacheNamespacePolicy.cross_namespace_pairs``.

Every fallback needs the same :func:`physical_kv_identity` (exact token ids and the
policy versions the KV was compiled under) and the target entry carrying the
physical token hash it was compiled under; similarity is never evidence. A name in
any mode may also resolve to a block compiled under another logical name *in its own
root* (two fields that map to one namespace).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .compile_policy import CompileAction, CompileDecision
from .semantic_cache_index import (
    ALIAS_CROSS_ROOT,
    ALIAS_SAME_ROOT,
    PERSISTENT_ROOTS,
    LogicalEntry,
    semantic_index_of,
)
from .subcontext_store import SemanticSubContext


class CacheNamespace(str, Enum):
    INSTRUCTION = "instruction"
    TOOL_SCHEMA = "tool_schema"
    CONVERSATION = "conversation"
    TOOL_OUTPUT = "tool_output"
    WORKSPACE = "workspace"
    #: Not persisted: the current turn's content changes every turn.
    MUTABLE_CURRENT_TURN = "mutable_current_turn"
    #: Never enters the persistent semantic cache.
    NO_CACHE = "no_cache"


PERSISTENT_NAMESPACES = frozenset(
    {
        CacheNamespace.INSTRUCTION,
        CacheNamespace.TOOL_SCHEMA,
        CacheNamespace.CONVERSATION,
        CacheNamespace.TOOL_OUTPUT,
        CacheNamespace.WORKSPACE,
    }
)
# The index names its roots as strings (it must not import this module); keep the two lists one list.
assert tuple(sorted(PERSISTENT_ROOTS)) == tuple(sorted(item.value for item in PERSISTENT_NAMESPACES))


@dataclass(frozen=True)
class NamespaceProfile:
    """Metadata a later residency/eviction policy can read. Nothing here acts on it yet."""

    namespace: CacheNamespace
    persistent: bool
    expected_reuse: str
    lifecycle: str


NAMESPACE_PROFILES: dict[CacheNamespace, NamespaceProfile] = {
    CacheNamespace.INSTRUCTION: NamespaceProfile(
        CacheNamespace.INSTRUCTION, True, "high", "stable"
    ),
    CacheNamespace.TOOL_SCHEMA: NamespaceProfile(
        CacheNamespace.TOOL_SCHEMA, True, "high", "stable"
    ),
    CacheNamespace.CONVERSATION: NamespaceProfile(
        CacheNamespace.CONVERSATION, True, "medium", "append_heavy_reusable_after_finalized"
    ),
    CacheNamespace.WORKSPACE: NamespaceProfile(
        CacheNamespace.WORKSPACE, True, "medium", "task_or_cross_task_evidence"
    ),
    CacheNamespace.TOOL_OUTPUT: NamespaceProfile(
        CacheNamespace.TOOL_OUTPUT, True, "to_be_measured", "often_request_specific"
    ),
    CacheNamespace.MUTABLE_CURRENT_TURN: NamespaceProfile(
        CacheNamespace.MUTABLE_CURRENT_TURN, False, "none", "current_turn_only"
    ),
    CacheNamespace.NO_CACHE: NamespaceProfile(
        CacheNamespace.NO_CACHE, False, "none", "never_persisted"
    ),
}

#: Metadata key an alias view carries: the owner (root and field) of the physical block that stands in for it.
PHYSICAL_ENTRY_METADATA_KEY = "physical_entry"

#: Why a block was kept out of the persistent cache by namespace policy.
NAMESPACE_NOT_PERSISTENT = "namespace is not persistent"


class NamespaceMode(str, Enum):
    #: Existing behaviour: the one semantic registry, no namespace restriction.
    GLOBAL = "global"
    #: A block may only be resident through an entry of its own namespace.
    HARD_NAMESPACE = "hard_namespace"
    #: Namespace-first lookup, then an optional policy-gated physical alias.
    LOGICAL_NAMESPACE = "logical_namespace"

    @classmethod
    def parse(cls, value: "NamespaceMode | str | None") -> "NamespaceMode":
        if value is None:
            return cls.GLOBAL
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise ValueError(
                f"unknown semantic_cache_namespace_mode {value!r}; expected one of "
                + ", ".join(mode.value for mode in cls)
            ) from None


#: Version of the cross-namespace pair rules below (validation + the vetted set). Recorded on every experiment row.
CROSS_NAMESPACE_PAIR_POLICY_VERSION = "cross-namespace-pairs-q1-v1"

#: The only ordered ``(block_namespace, entry_namespace)`` pairs the experiment runner will configure (Q-1).
#:
#: ``workspace <-> tool_output``: the same bytes can legitimately arrive through two channels -- a file read by a tool
#: (``tool_output``) and the same file surfaced as workspace evidence (``workspace``). Both namespaces are persistent and both
#: are compiled standalone from position 0, so equal ``physical_kv_identity`` means the two KV tensors are the same tensor.
#: Nothing else is vetted: ``instruction``/``tool_schema``/``conversation`` carry no second channel for the same bytes in any
#: workload this repository has, and a pair is only worth naming when a second channel exists. A pair is a statement about
#: where *identical* blocks can appear; it never widens what counts as identical.
VETTED_CROSS_NAMESPACE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {("workspace", "tool_output"), ("tool_output", "workspace")}
)


def parse_cross_namespace_pairs(
    spec: Iterable[Any] | None,
    *,
    require_vetted: bool = False,
) -> tuple[tuple["CacheNamespace", "CacheNamespace"], ...]:
    """Pairs from config text: ``"a<->b"`` (both orders), ``"a->b"`` (block ``a`` may use an entry of ``b``) or a 2-sequence.

    Every namespace must be persistent and the two must differ; a non-persistent namespace (``mutable_current_turn``,
    ``no_cache``) can never hold or receive an alias. With ``require_vetted`` a pair outside
    :data:`VETTED_CROSS_NAMESPACE_PAIRS` is refused, so a config typo cannot widen reuse.
    """

    pairs: list[tuple[CacheNamespace, CacheNamespace]] = []
    for item in spec or ():
        if isinstance(item, str):
            text = item.strip()
            if "<->" in text:
                left, right = (part.strip() for part in text.split("<->", 1))
                ordered = [(left, right), (right, left)]
            elif "->" in text:
                left, right = (part.strip() for part in text.split("->", 1))
                ordered = [(left, right)]
            else:
                raise ValueError(f"cross-namespace pair {item!r} must look like 'a<->b' or 'a->b'")
        else:
            left, right = item
            ordered = [(left, right)]
        for left, right in ordered:
            pair = (CacheNamespace(left), CacheNamespace(right))
            if pair[0] is pair[1]:
                raise ValueError(f"cross-namespace pair {item!r} names one namespace twice")
            if pair not in pairs:
                pairs.append(pair)
    for first, second in pairs:
        for namespace in (first, second):
            if namespace not in PERSISTENT_NAMESPACES:
                raise ValueError(
                    f"cross-namespace pair {first.value}->{second.value} names {namespace.value!r}, which is not "
                    "persistent; a non-persistent namespace never aliases"
                )
        if require_vetted and (first.value, second.value) not in VETTED_CROSS_NAMESPACE_PAIRS:
            raise ValueError(
                f"cross-namespace pair {first.value}->{second.value} is not vetted; vetted pairs: "
                + ", ".join(sorted(f"{a}->{b}" for a, b in VETTED_CROSS_NAMESPACE_PAIRS))
            )
    return tuple(pairs)


class CacheNamespacePolicy:
    """The single place field -> namespace is decided.

    ``cross_namespace_pairs`` is the whole of the cross-namespace fallback
    policy: ordered ``(block_namespace, entry_namespace)`` pairs for which a
    block may be served from an entry compiled under another namespace. Empty by
    default, which is the conservative setting. Pairs are validated on
    construction (:func:`parse_cross_namespace_pairs`): persistent namespaces
    only, never a namespace with itself.
    """

    version = "cache-namespace-policy-v1"

    _FIELD_NAMESPACES: Mapping[str, CacheNamespace] = {
        "user_prompt": CacheNamespace.MUTABLE_CURRENT_TURN,
        "task_instruction": CacheNamespace.INSTRUCTION,
        "tool_definitions": CacheNamespace.TOOL_SCHEMA,
        "tool_history_output": CacheNamespace.TOOL_OUTPUT,
        "workspace_evidence": CacheNamespace.WORKSPACE,
    }

    def __init__(
        self,
        *,
        cross_namespace_pairs: Iterable[tuple[CacheNamespace, CacheNamespace]] = (),
        require_vetted_pairs: bool = False,
    ) -> None:
        self.cross_namespace_pairs = frozenset(
            parse_cross_namespace_pairs(cross_namespace_pairs, require_vetted=require_vetted_pairs)
        )

    def namespace_for(
        self,
        context_field: str,
        *,
        finalized: bool = True,
        explicitly_cacheable: bool = False,
    ) -> CacheNamespace:
        """``finalized`` is "this block can no longer change"; ``explicitly_cacheable`` is the
        opt-in for ``task_metadata`` (model-visible, stable and reusable)."""

        if context_field == "conversation_history":
            return (
                CacheNamespace.CONVERSATION
                if finalized
                else CacheNamespace.MUTABLE_CURRENT_TURN
            )
        if context_field == "task_metadata":
            return (
                CacheNamespace.INSTRUCTION
                if explicitly_cacheable and finalized
                else CacheNamespace.NO_CACHE
            )
        try:
            return self._FIELD_NAMESPACES[context_field]
        except KeyError:
            raise ValueError(f"unknown functional field {context_field!r}") from None

    def namespace_of(self, value: SemanticSubContext) -> CacheNamespace:
        metadata = value.metadata or {}
        explicit = all(
            metadata.get(key) is True
            for key in ("model_visible", "stable", "reusable")
        )
        return self.namespace_for(
            value.context_field,
            finalized=not value.is_mutable,
            explicitly_cacheable=explicit,
        )

    @staticmethod
    def profile(namespace: CacheNamespace) -> NamespaceProfile:
        return NAMESPACE_PROFILES[CacheNamespace(namespace)]

    def is_persistent(self, namespace: CacheNamespace) -> bool:
        return CacheNamespace(namespace) in PERSISTENT_NAMESPACES

    def permits_fallback(self, block: CacheNamespace, entry: CacheNamespace) -> bool:
        return (CacheNamespace(block), CacheNamespace(entry)) in self.cross_namespace_pairs


# --- physical KV identity -----------------------------------------------------


PHYSICAL_KV_IDENTITY_VERSION = "physical-kv-identity-v1"


@dataclass(frozen=True)
class PhysicalKVIdentity:
    digest: str


def physical_kv_identity(value: SemanticSubContext) -> PhysicalKVIdentity | None:
    """What decides whether one compiled block's KV may serve another block.

    Exact authoritative token ids, the exact source bytes (the Compile request
    tokenizes the block's own text plus the stitch separator), the model and
    tokenizer, and the segmentation / recompute policy versions the runtime keys
    its registry on. It deliberately **excludes** the functional field, the
    namespace, the task and the session: none of them enters the compiled KV.

    Returns ``None`` when the block has no recorded token ids. A text-only
    identity proves equal bytes, not equal tokens, so it never licenses sharing.
    The runtime-side fingerprints (kv dtype, rope, attention backend, tensor
    parallel size, adapter) are constant for one server and checked by the
    runtime's own registry key; they are not, and cannot be, host knowledge.
    """

    if not value.token_ids:
        return None
    payload = {
        "version": PHYSICAL_KV_IDENTITY_VERSION,
        "token_ids": list(value.token_ids),
        "content_hash": value.content_hash,
        "model_fingerprint": value.model_fingerprint,
        "tokenizer_fingerprint": value.tokenizer_fingerprint,
        "segmentation_policy_version": value.policy_version,
        "recompute_policy_version": value.recompute_policy_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return PhysicalKVIdentity(hashlib.sha256(encoded.encode("utf-8")).hexdigest())


# --- lookup ---------------------------------------------------------------


@dataclass(frozen=True)
class NamespaceLookupResult:
    sub_context_id: str
    context_field: str
    namespace: str
    mode: str
    token_count: int
    #: The block is served from an already-compiled entry (its own, or an alias).
    resident: bool
    same_namespace_hit: bool = False
    fallback_attempted: bool = False
    cross_namespace_hit: bool = False
    #: Cache key the Link will name. Differs from the block's own only for an alias.
    physical_cache_key: str | None = None
    physical_semantic_identity: str | None = None
    #: Tokens of a compatible entry that existed in another namespace and was not used.
    missed_reuse_tokens: int = 0
    #: The block could be served through another name at all: immutable, a reuse candidate, not invalidated. A mutable
    #: block is never linked, so a foreign twin of it is not a reuse opportunity (``LOGICAL_NAMESPACE`` refuses it outright).
    alias_eligible: bool = True
    #: An exact-identity entry of ANOTHER root exists for an eligible, not-already-resident block -- whatever the mode did
    #: with it. Computed in every mode so Hard and Logical are measured against the same denominator.
    cross_namespace_opportunity: bool = False
    #: The cross-namespace pair ``block->entry`` that was permitted (``LOGICAL_NAMESPACE``), else ``None``.
    cross_namespace_pair: str | None = None
    not_persistent: bool = False
    lookup_latency_s: float = 0.0
    entry_age_s: float | None = None
    #: The logical root this block's name lives in; ``None`` for a namespace with no root.
    logical_root: str | None = None
    #: The block's own name inside that root (its own cache key).
    logical_entry_key: str | None = None
    #: Namespace-independent identity of the KV this block needs (``physical_kv_identity``).
    physical_kv_identity: str | None = None
    #: ``same_root`` / ``cross_root`` when the block is served through another name's physical block.
    alias_kind: str | None = None
    #: This block's name was already recorded as an alias of the block it is served from.
    alias_recorded: bool = False
    #: Logical names (primary + live aliases, counting this one) that reference the serving physical block.
    physical_alias_count: int = 0
    #: The root and field of the entry that owns the serving physical block (an alias is another name for it).
    physical_namespace: str | None = None
    physical_context_field: str | None = None

    @property
    def physical_kv_reused(self) -> bool:
        """Served from a physical block that already exists -- no Compile is needed for it."""

        return self.resident

    @property
    def second_compile_avoided(self) -> bool:
        """An alias: the block's own name was absent, yet an existing physical block serves it."""

        return self.alias_kind is not None

    @property
    def cross_namespace_second_compile_avoided(self) -> bool:
        return self.alias_kind == ALIAS_CROSS_ROOT

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_context_id": self.sub_context_id,
            "context_field": self.context_field,
            "namespace": self.namespace,
            "cache_namespace": self.namespace,
            "mode": self.mode,
            "token_count": self.token_count,
            "resident": self.resident,
            "namespace_lookup_hit": self.same_namespace_hit,
            "namespace_lookup_miss": not self.resident and not self.not_persistent,
            "logical_root": self.logical_root,
            "logical_entry_key": self.logical_entry_key,
            "logical_lookup_hit": self.same_namespace_hit or self.alias_recorded,
            "logical_lookup_miss": not (self.same_namespace_hit or self.alias_recorded)
            and not self.not_persistent,
            "cross_namespace_fallback_attempted": self.fallback_attempted,
            "cross_namespace_attempt": self.fallback_attempted,
            "cross_namespace_hit": self.cross_namespace_hit,
            "alias_kind": self.alias_kind,
            "alias_recorded": self.alias_recorded,
            "physical_alias_count": self.physical_alias_count,
            "physical_namespace": self.physical_namespace,
            "physical_context_field": self.physical_context_field,
            "physical_kv_identity": self.physical_kv_identity,
            "physical_kv_reused": self.physical_kv_reused,
            "second_compile_avoided": self.second_compile_avoided,
            "cross_namespace_second_compile_avoided": self.cross_namespace_second_compile_avoided,
            "physical_cache_key": self.physical_cache_key,
            "cross_namespace_missed_reuse_tokens": self.missed_reuse_tokens,
            "cross_namespace_opportunity": self.cross_namespace_opportunity,
            "cross_namespace_opportunity_tokens": self.token_count if self.cross_namespace_opportunity else 0,
            "cross_namespace_pair": self.cross_namespace_pair,
            "alias_eligible": self.alias_eligible,
            "not_persistent": self.not_persistent,
            "lookup_latency_s": self.lookup_latency_s,
            "entry_age_s": self.entry_age_s,
            "physical_block_residency_age_s": self.entry_age_s,
        }


class NamespaceResolver:
    """Namespace-first lookup over the logical roots, per the selected mode.

    The resolver reads the catalog's logical index (``semantic_cache_index``) and rewrites a block's
    *presentation* (an alias view carrying the serving entry's cache key and semantic identity) -- the same
    shape as a repeated occurrence of one compiled block, which the runtime already supports. It never decides
    compile economics, layout or repair, and it never touches runtime-owned state.

    Lookup order for one block:

    1. its own root: the entry named by the block's own cache key (``GLOBAL``: an entry of any root that
       holds the key);
    2. another name for the same physical KV *inside its own root* (two fields that share a namespace);
    3. on a miss, an exact-physical-identity entry of another root, if the mode permits it:
       ``GLOBAL`` always (``global_physical_dedup``), ``LOGICAL_NAMESPACE`` for a pair named in the policy,
       ``HARD_NAMESPACE`` never.
    """

    def __init__(
        self,
        mode: NamespaceMode | str = NamespaceMode.GLOBAL,
        policy: CacheNamespacePolicy | None = None,
        *,
        hard_persistence_constraint: bool = True,
        global_physical_dedup: bool = True,
    ) -> None:
        self.mode = NamespaceMode.parse(mode)
        self.policy = policy or CacheNamespacePolicy()
        #: Outside GLOBAL, a block in a non-persistent namespace may not be compiled by *any* later stage,
        #: including the layout's prefix closure. ``False`` restores the pre-constraint behaviour (the
        #: closure may promote it) and exists only so the before/after diagnosis can be reproduced.
        self.hard_persistence_constraint = bool(hard_persistence_constraint)
        #: ``GLOBAL`` lets an exact physical-identity match in any root serve a block (the namespace never
        #: blocks exact-compatible reuse). ``False`` restores the field-partitioned registry the mode had
        #: before logical roots, so that behaviour can still be reproduced.
        self.global_physical_dedup = bool(global_physical_dedup)

    @property
    def enforces_persistence(self) -> bool:
        return self.mode is not NamespaceMode.GLOBAL

    def describe(self) -> dict[str, Any]:
        return {
            "semantic_cache_namespace_mode": self.mode.value,
            "cache_namespace_policy_version": self.policy.version,
            "cross_namespace_pair_policy_version": CROSS_NAMESPACE_PAIR_POLICY_VERSION,
            "cross_namespace_pairs": sorted(
                f"{a.value}->{b.value}" for a, b in self.policy.cross_namespace_pairs
            ),
            "global_physical_dedup": self.global_physical_dedup,
            "logical_roots": list(PERSISTENT_ROOTS),
            "physical_cache_instances": 1,
        }

    def _cross_root_permitted(self, namespace: CacheNamespace, entry_namespace: str) -> bool:
        if self.mode is NamespaceMode.GLOBAL:
            return self.global_physical_dedup
        if self.mode is NamespaceMode.LOGICAL_NAMESPACE:
            return self.policy.permits_fallback(namespace, CacheNamespace(entry_namespace))
        return False

    def resolve(self, value: SemanticSubContext, catalog: Any) -> NamespaceLookupResult:
        started = time.perf_counter()
        namespace = self.policy.namespace_of(value)
        persistent = self.policy.is_persistent(namespace)
        identity = physical_kv_identity(value)
        digest = None if identity is None else identity.digest
        base = dict(
            sub_context_id=value.sub_context_id,
            context_field=value.context_field,
            namespace=namespace.value,
            mode=self.mode.value,
            token_count=int(value.token_count),
            logical_root=namespace.value if persistent else None,
            logical_entry_key=value.cache_key if persistent else None,
            physical_kv_identity=digest,
        )
        if self.enforces_persistence and not persistent:
            return NamespaceLookupResult(
                resident=False,
                not_persistent=True,
                lookup_latency_s=time.perf_counter() - started,
                **base,
            )
        index = semantic_index_of(catalog)

        # Stage 1 -- the block's own logical root.
        entry = index.lookup(
            namespace.value, value.cache_key, any_root=self.mode is NamespaceMode.GLOBAL
        )
        own_hit = entry is not None
        same_namespace_hit = bool(own_hit and entry.namespace == namespace.value)
        age = None
        if own_hit and entry.compiled_at is not None:
            age = max(time.time() - float(entry.compiled_at), 0.0)

        alias: LogicalEntry | None = None
        alias_kind: str | None = None
        pair_text: str | None = None
        fallback_attempted = False
        alias_eligible = True
        opportunity = False
        compatible: list[LogicalEntry] = []
        if not own_hit:
            compatible = [
                candidate
                for candidate in index.find_by_identity(digest)
                if candidate.logical_entry_key != value.cache_key
            ]
            # Stage 1b -- another name for the same physical KV in the block's own root.
            for candidate in compatible:
                if candidate.namespace == namespace.value:
                    alias, alias_kind = candidate, ALIAS_SAME_ROOT
                    break
            # Stage 2 -- the mode's cross-root policy.
            if alias is None:
                foreign = [c for c in compatible if c.namespace != namespace.value]
                alias_eligible = (
                    not value.is_mutable and not value.invalidated and bool(value.reuse_candidate)
                )
                opportunity = bool(foreign) and alias_eligible
                # LOGICAL_NAMESPACE (the Q-1 policy) never aliases a block that cannot be linked anyway. GLOBAL keeps its
                # historical behaviour untouched; HARD never crosses.
                if self.mode is NamespaceMode.LOGICAL_NAMESPACE and not alias_eligible:
                    foreign = []
                permitted = [
                    c for c in foreign if self._cross_root_permitted(namespace, c.namespace)
                ]
                if foreign and (
                    (self.mode is NamespaceMode.GLOBAL and self.global_physical_dedup)
                    or (
                        self.mode is NamespaceMode.LOGICAL_NAMESPACE
                        and bool(self.policy.cross_namespace_pairs)
                    )
                ):
                    fallback_attempted = True
                if permitted:
                    alias, alias_kind = permitted[0], ALIAS_CROSS_ROOT
                    pair_text = f"{namespace.value}->{alias.namespace}"
        if alias is not None and alias.compiled_at is not None:
            age = max(time.time() - float(alias.compiled_at), 0.0)

        serving_key = (
            alias.runtime_cache_key if alias is not None else (value.cache_key if own_hit else None)
        )
        recorded = alias is not None and any(
            item.namespace == namespace.value and item.logical_entry_key == value.cache_key
            for item in index.aliases_of(alias.runtime_cache_key)
        )
        refs = 0
        if serving_key is not None:
            refs = index.logical_refs(serving_key) + (1 if alias is not None and not recorded else 0)
        missed = int(value.token_count) if compatible and alias is None and not own_hit else 0
        return NamespaceLookupResult(
            resident=bool(own_hit or alias is not None),
            same_namespace_hit=same_namespace_hit,
            fallback_attempted=fallback_attempted,
            cross_namespace_hit=alias_kind == ALIAS_CROSS_ROOT,
            physical_cache_key=serving_key,
            physical_semantic_identity=(
                str(alias.semantic_identity) if alias is not None else None
            ),
            missed_reuse_tokens=missed,
            lookup_latency_s=time.perf_counter() - started,
            entry_age_s=age,
            alias_kind=alias_kind,
            alias_recorded=recorded,
            physical_alias_count=refs,
            physical_namespace=alias.namespace if alias is not None else None,
            physical_context_field=alias.context_field if alias is not None else None,
            alias_eligible=alias_eligible,
            cross_namespace_opportunity=opportunity,
            cross_namespace_pair=pair_text,
            **base,
        )

    def apply(
        self,
        ordered: Sequence[SemanticSubContext],
        catalog: Any,
    ) -> tuple[tuple[SemanticSubContext, ...], tuple[NamespaceLookupResult, ...]]:
        """Resolve every block; present an aliased one under the physical block that serves it."""

        views: list[SemanticSubContext] = []
        results: list[NamespaceLookupResult] = []
        for value in ordered:
            result = self.resolve(value, catalog)
            results.append(result)
            if result.alias_kind is not None:
                # The view stands for the owner's physical block. It remembers whose block that is, so a
                # recovery Compile of this view registers the block as the owner's, not as this block's.
                views.append(
                    replace(
                        value,
                        cache_key=str(result.physical_cache_key),
                        semantic_identity=str(result.physical_semantic_identity),
                        metadata={
                            **(value.metadata or {}),
                            PHYSICAL_ENTRY_METADATA_KEY: {
                                "namespace": result.physical_namespace,
                                "context_field": result.physical_context_field,
                            },
                        },
                    )
                )
            else:
                views.append(value)
        return tuple(views), tuple(results)

    def record_aliases(
        self,
        results: Sequence[NamespaceLookupResult],
        catalog: Any,
        *,
        linked_block_ids: Iterable[str] | None = None,
    ) -> int:
        """Register the names a completed Link used, so each root shows what it references.

        Called after a Link succeeded (the planner stays side-effect free). Only blocks that were actually
        served through another name are recorded. Returns how many aliases were newly recorded.
        """

        index = semantic_index_of(catalog)
        linked = None if linked_block_ids is None else set(linked_block_ids)
        recorded = 0
        for result in results:
            if result.alias_kind is None or result.logical_root is None:
                continue
            if linked is not None and result.sub_context_id not in linked:
                continue
            target = index.lookup(result.namespace, str(result.physical_cache_key), any_root=True)
            if target is None:
                continue
            before = index.aliases_recorded
            index.record_alias(
                namespace=result.logical_root,
                logical_entry_key=str(result.logical_entry_key),
                target=target,
                context_field=result.context_field,
                token_count=result.token_count,
            )
            recorded += int(index.aliases_recorded > before)
        return recorded

    def layout_permissions(
        self, results: Sequence[NamespaceLookupResult]
    ) -> tuple[bool, ...] | None:
        """Per block: may the layout stage compile it? ``None`` means no namespace restriction.

        This is what makes non-persistence a hard constraint. ``LinkLayoutFeasibility`` promotes an
        immutable reusable block that precedes a compiled one (prefix closure), unless it is told the
        block may not be linked at all, in which case it demotes the *later* compiled blocks instead.
        A block in a non-persistent namespace is told exactly that.
        """

        if not (self.enforces_persistence and self.hard_persistence_constraint):
            return None
        return tuple(not result.not_persistent for result in results)

    def enforce_persistence(
        self,
        results: Sequence[NamespaceLookupResult],
        decisions: Sequence[CompileDecision],
    ) -> tuple[CompileDecision, ...]:
        """Keep a block out of the persistent cache when its namespace is not persistent.

        A no-op in ``GLOBAL`` mode, which preserves the existing behaviour. Only a
        ``COMPILE`` decision is touched; the reason is recorded verbatim and the
        policy's provenance is kept.
        """

        if not self.enforces_persistence:
            return tuple(decisions)
        adjusted: list[CompileDecision] = []
        for result, decision in zip(results, decisions, strict=True):
            if result.not_persistent and decision.action is CompileAction.COMPILE:
                decision = replace(
                    decision,
                    action=CompileAction.KEEP_MUTABLE,
                    reason=NAMESPACE_NOT_PERSISTENT,
                    estimated_saved_prefill_tokens=0,
                )
            adjusted.append(decision)
        return tuple(adjusted)


# --- per-request summary ---------------------------------------------------


def summarize_namespace_request(
    results: Sequence[NamespaceLookupResult],
    *,
    linked_block_ids: Iterable[str],
    link_cache_key_by_block: Mapping[str, str],
    repair_tokens_by_cache_key: Mapping[str, int] | None = None,
    recovered_cache_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Aggregate one request's planning-time lookups into the counters the experiment compares.

    ``linked_block_ids`` are the blocks the plan sends down the Link path;
    ``link_cache_key_by_block`` is the key each is linked under (an alias key for
    a cross-namespace hit). A linked block that was resident when planned counts
    as reused, one that was not counts as compiled. Tokens are host-accounted
    block tokens, not runtime measurements.

    ``recovered_cache_keys`` are the keys the runtime reported as no longer resident and the host compiled again
    (residency recovery). The host planned them from its catalog, which records intent and not residency, so the plan
    counted them as reused; they were not. Their tokens are counted as compiled, never as reused, and are also named
    in ``residency_recovered_*`` so a recovered eviction stays visible. Lookup hit/miss counters are left as planned:
    they are what the host believed.
    """

    linked = set(linked_block_ids)
    recovered = {str(key) for key in (recovered_cache_keys or ())}
    repair = dict(repair_tokens_by_cache_key or {})
    counters = (
        "namespace_lookup_hit",
        "namespace_lookup_miss",
        "cross_namespace_fallback_attempted",
        "cross_namespace_hit",
        "cross_namespace_missed_reuse_tokens",
        "cross_namespace_opportunity",
        "cross_namespace_opportunity_tokens",
        "reused_token_count",
        "compiled_token_count",
        "repair_token_count",
        "logical_lookup_hit",
        "logical_lookup_miss",
        "cross_namespace_attempt",
        "physical_kv_reused",
        "physical_kv_dedup_count",
        "cross_namespace_second_compile_avoided",
        "alias_count",
        "residency_recovered_blocks",
        "residency_recovered_token_count",
    )
    per: dict[str, dict[str, Any]] = {}
    for result in results:
        row = per.setdefault(
            result.namespace,
            {"blocks": 0, "functional_fields": [], **{key: 0 for key in counters}},
        )
        row["blocks"] += 1
        if result.context_field not in row["functional_fields"]:
            row["functional_fields"].append(result.context_field)
        row["namespace_lookup_hit"] += int(result.resident)
        row["namespace_lookup_miss"] += int(not result.resident and not result.not_persistent)
        row["cross_namespace_fallback_attempted"] += int(result.fallback_attempted)
        row["cross_namespace_hit"] += int(result.cross_namespace_hit)
        row["cross_namespace_missed_reuse_tokens"] += result.missed_reuse_tokens
        row["cross_namespace_opportunity"] += int(result.cross_namespace_opportunity)
        row["cross_namespace_opportunity_tokens"] += result.token_count if result.cross_namespace_opportunity else 0
        row["logical_lookup_hit"] += int(result.resident)
        row["logical_lookup_miss"] += int(not result.resident and not result.not_persistent)
        row["cross_namespace_attempt"] += int(result.fallback_attempted)
        if result.sub_context_id in linked:
            link_key = str(link_cache_key_by_block.get(result.sub_context_id))
            was_recovered = link_key in recovered
            if was_recovered:
                row["residency_recovered_blocks"] += 1
                row["residency_recovered_token_count"] += result.token_count
                # compiled again after the runtime said it was gone: real Compile work, whatever the plan believed
                row["compiled_token_count"] += result.token_count
                key = link_cache_key_by_block.get(result.sub_context_id)
                row["repair_token_count"] += min(int(repair.get(str(key), 0)), result.token_count)
            elif result.resident:
                row["physical_kv_reused"] += 1
                row["physical_kv_dedup_count"] += int(result.second_compile_avoided)
                row["cross_namespace_second_compile_avoided"] += int(
                    result.cross_namespace_second_compile_avoided
                )
                row["alias_count"] += int(result.alias_kind is not None)
                row["reused_token_count"] += result.token_count
                key = link_cache_key_by_block.get(result.sub_context_id)
                row["repair_token_count"] += min(int(repair.get(str(key), 0)), result.token_count)
            else:
                row["compiled_token_count"] += result.token_count
    totals: dict[str, Any] = {key: sum(row[key] for row in per.values()) for key in counters}
    totals["namespace_lookup_latency_s"] = sum(row.lookup_latency_s for row in results)
    ages = [row.entry_age_s for row in results if row.entry_age_s is not None]
    totals["physical_block_residency_age_s_max"] = max(ages) if ages else None
    return {"by_namespace": per, "totals": totals, "blocks": len(results)}


__all__ = [
    "CROSS_NAMESPACE_PAIR_POLICY_VERSION",
    "VETTED_CROSS_NAMESPACE_PAIRS",
    "parse_cross_namespace_pairs",
    "PHYSICAL_ENTRY_METADATA_KEY",
    "NAMESPACE_NOT_PERSISTENT",
    "NAMESPACE_PROFILES",
    "PERSISTENT_NAMESPACES",
    "PHYSICAL_KV_IDENTITY_VERSION",
    "CacheNamespace",
    "CacheNamespacePolicy",
    "NamespaceLookupResult",
    "NamespaceMode",
    "NamespaceProfile",
    "NamespaceResolver",
    "PhysicalKVIdentity",
    "physical_kv_identity",
    "summarize_namespace_request",
]
