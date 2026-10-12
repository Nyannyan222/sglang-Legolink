from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import hashlib
import heapq
import contextlib
import logging
import os
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.mem_cache.utils import split_node_hash_value

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _token_content_hash(token_ids: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    for token_id in token_ids:
        hasher.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return hasher.hexdigest()


def _emit_legolink_trace(event: str, **details: Any) -> None:
    try:
        from sglang.srt.pic_runtime import _emit_pic_trace

        _emit_pic_trace(event, **details)
    except (ImportError, AttributeError):
        return


@contextlib.contextmanager
def _legolink_profile_range(name: str):
    nvtx_enabled = _env_flag("SGLANG_PIC_NVTX_ENABLED", "0") and torch.cuda.is_available()
    if not nvtx_enabled and not _env_flag(
        "SGLANG_PIC_PROFILE_RANGES_ENABLED", "0"
    ):
        yield
        return
    region_names = {
        "LinkPlanBuild": "LegoLink::link_prepare",
        "CompatibilityValidation": "LegoLink::lookup",
        "Lookup": "LegoLink::lookup",
        "Acquire": "LegoLink::acquire",
        "Release": "LegoLink::release",
    }
    region_name = region_names.get(name, f"LegoLink::{name}")
    with torch.profiler.record_function(region_name):
        if nvtx_enabled:
            torch.cuda.nvtx.range_push(region_name)
        try:
            yield
        finally:
            if nvtx_enabled:
                torch.cuda.nvtx.range_pop()


class LegoLinkDuplicateRegistrationError(ValueError):
    """Two different reusable contents claim one cache key.

    Raised only when the *content* disagrees -- a different token hash, or
    fingerprints that make the resident KV unusable for the new claim. A caller
    is expected to abandon its registration and keep the resident entry, never
    to overwrite it. Still a ValueError, so existing handlers keep working.
    """


@dataclass(frozen=True)
class LegoLinkBlockKey:
    """Compatibility-qualified logical key for one reusable semantic block."""

    logical_cache_key: str
    token_hash: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    kv_dtype: str
    rope_fingerprint: str
    attention_backend_fingerprint: str
    adapter_id: str
    tp_world_size: int
    segmentation_policy_version: str
    legolink_policy_version: str

    @classmethod
    def build(
        cls,
        cache_key: str,
        token_ids: tuple[int, ...],
        compatibility: Optional[dict[str, Any]] = None,
    ) -> "LegoLinkBlockKey":
        values = compatibility or {}
        return cls(
            logical_cache_key=str(cache_key),
            token_hash=str(values.get("token_hash") or _token_content_hash(token_ids)),
            model_fingerprint=str(values.get("model_fingerprint") or "runtime-local"),
            tokenizer_fingerprint=str(
                values.get("tokenizer_fingerprint") or "runtime-local"
            ),
            kv_dtype=str(values.get("kv_dtype") or "runtime-local"),
            rope_fingerprint=str(values.get("rope_fingerprint") or "runtime-local"),
            attention_backend_fingerprint=str(
                values.get("attention_backend_fingerprint") or "runtime-local"
            ),
            adapter_id=str(values.get("adapter_id") or "none"),
            tp_world_size=int(values.get("tp_world_size") or 1),
            segmentation_policy_version=str(
                values.get("segmentation_policy_version") or "field-first-v1"
            ),
            legolink_policy_version=str(
                values.get("legolink_policy_version")
                or values.get("policy_version")
                or "legolink-v1"
            ),
        )

    @property
    def policy_version(self) -> str:
        """Compatibility alias for the initial prototype field."""

        return self.legolink_policy_version


@dataclass
class LegoLinkKVSlotRange:
    """An exact logical token range backed by rank-local physical KV slots."""

    slots: Optional[torch.Tensor]
    token_length: int
    page_size: int
    aligned_token_count: int
    tail_token_count: int

    @classmethod
    def build(
        cls,
        slots: Optional[torch.Tensor],
        token_length: int,
        page_size: int,
    ) -> "LegoLinkKVSlotRange":
        normalized = None if slots is None else slots.long().clone()
        if normalized is not None and len(normalized) != token_length:
            raise ValueError(
                "LegoLink KV slot count must equal the exact logical token length."
            )
        page_size = max(int(page_size), 1)
        aligned = token_length // page_size * page_size
        return cls(
            slots=normalized,
            token_length=int(token_length),
            page_size=page_size,
            aligned_token_count=aligned,
            tail_token_count=int(token_length) - aligned,
        )


@dataclass(frozen=True)
class LegoLinkLinkPlan:
    requested_cache_keys: tuple[str, ...]
    destination_offsets: tuple[int, ...]
    request_id: str = ""
    semantic_block_ids: tuple[str, ...] = ()
    recompute_tokens_per_block: tuple[int, ...] = ()
    block_token_ranges: tuple[tuple[int, int], ...] = ()
    destination_global_positions: tuple[tuple[int, ...], ...] = ()
    rope_position_deltas: tuple[int, ...] = ()
    kvlink: tuple[int, ...] = ()
    imp_indices: tuple[int, ...] = ()
    mutable_suffix_range: tuple[int, int] = (0, 0)
    fallback_cache_keys: tuple[str, ...] = ()

    @classmethod
    def from_legacy(
        cls,
        cache_keys: list[str],
        destination_offsets: Optional[list[int]] = None,
        semantic_block_ids: Optional[list[str]] = None,
        link_plan: Optional[dict[str, Any]] = None,
    ) -> "LegoLinkLinkPlan":
        ranges = (link_plan or {}).get("block_ranges") or (link_plan or {}).get(
            "attention_ranges"
        ) or []
        if destination_offsets is None:
            destination_offsets = [
                int(ranges[index].get("start_token", 0)) if index < len(ranges) else 0
                for index in range(len(cache_keys))
            ]
        if semantic_block_ids is not None:
            block_ids = semantic_block_ids
        elif ranges:
            block_ids = [
                str(ranges[index].get("block_id", cache_key))
                if index < len(ranges)
                else str(cache_key)
                for index, cache_key in enumerate(cache_keys)
            ]
        else:
            block_ids = []
        recompute = (link_plan or {}).get("recompute_tokens_per_block") or []
        token_ranges = tuple(
            (
                int(item.get("start_token", 0)),
                int(item.get("end_token", 0)),
            )
            for item in ranges
        )
        positions = tuple(
            tuple(range(start, end)) for start, end in token_ranges
        )
        mutable_start = int((link_plan or {}).get("mutable_start_token") or 0)
        total_tokens = int((link_plan or {}).get("total_token_count") or mutable_start)
        return cls(
            requested_cache_keys=tuple(str(item) for item in cache_keys),
            destination_offsets=tuple(int(item) for item in destination_offsets),
            request_id=str((link_plan or {}).get("request_id") or ""),
            semantic_block_ids=tuple(str(item) for item in block_ids),
            recompute_tokens_per_block=tuple(int(item) for item in recompute),
            block_token_ranges=token_ranges,
            destination_global_positions=positions,
            rope_position_deltas=tuple(int(offset) for offset in destination_offsets),
            kvlink=tuple(
                int(item) for item in ((link_plan or {}).get("kvlink") or ())
            ),
            imp_indices=tuple(
                int(item)
                for item in ((link_plan or {}).get("imp_indices") or ())
            ),
            mutable_suffix_range=(mutable_start, total_tokens),
        )


@dataclass
class LegoLinkBlockHandle:
    """RadixCache-owned logical handle for immutable compiled KV."""

    cache_key: str
    block_key: LegoLinkBlockKey
    block_id: str
    sub_context_id: str
    semantic_block_id: str
    context_field: str
    token_range: tuple[int, int]
    token_ids: tuple[int, ...]
    slot_range: LegoLinkKVSlotRange
    compile_local_positions: tuple[int, ...]
    snapshot: Any = None
    radix_node: Any = None
    ownership_lock_params: Any = None
    pin_count: int = 0
    active_link_refcount: int = 0
    reuse_count: int = 0
    reused_tokens: int = 0
    recomputed_tokens: int = 0
    evicted_tokens: int = 0
    last_access_time: float = field(default_factory=time.monotonic)
    generation: int = 0
    allocator_generation: int = 0
    estimated_saved_recompute_cost: float = 0.0
    kv_residency_cost: float = 0.0
    valid: bool = True
    stale_reason: Optional[str] = None

    @property
    def token_count(self) -> int:
        return max(int(self.token_range[1]) - int(self.token_range[0]), 0)

    @property
    def token_length(self) -> int:
        return self.token_count

    @property
    def kv_slots(self) -> Optional[torch.Tensor]:
        return self.slot_range.slots

    @kv_slots.setter
    def kv_slots(self, value: Optional[torch.Tensor]) -> None:
        self.slot_range = LegoLinkKVSlotRange.build(
            value, self.token_count, self.slot_range.page_size
        )

    @property
    def page_size(self) -> int:
        return self.slot_range.page_size

    @property
    def aligned_token_count(self) -> int:
        return self.slot_range.aligned_token_count

    @property
    def tail_token_count(self) -> int:
        return self.slot_range.tail_token_count

    @property
    def refcount(self) -> int:
        return self.active_link_refcount

    @refcount.setter
    def refcount(self, value: int) -> None:
        self.active_link_refcount = int(value)

    @property
    def evicted(self) -> bool:
        return (not self.valid) or (
            self.radix_node is not None and getattr(self.radix_node, "evicted", False)
        )

    @property
    def compatibility_fingerprint(self) -> LegoLinkBlockKey:
        return self.block_key

    @property
    def cache_generation(self) -> int:
        return self.generation

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.stale_reason = str(reason)


# Compatibility name for callers written against the initial prototype.
LegoLinkBlockMetadata = LegoLinkBlockHandle


@dataclass
class LegoLinkBlockLookupResult:
    requested_cache_keys: list[str] = field(default_factory=list)
    matched_handles: list[LegoLinkBlockHandle] = field(default_factory=list)
    matched_blocks: dict[str, LegoLinkBlockHandle] = field(default_factory=dict)
    kv_slot_ranges: dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    ordered_slot_ranges: list[LegoLinkKVSlotRange] = field(default_factory=list)
    global_offsets: dict[str, int] = field(default_factory=dict)
    destination_offsets: list[int] = field(default_factory=list)
    compile_local_positions: dict[str, tuple[int, ...]] = field(default_factory=dict)
    destination_global_positions: list[tuple[int, ...]] = field(default_factory=list)
    missing_blocks: list[str] = field(default_factory=list)
    stale_blocks: list[str] = field(default_factory=list)
    incompatible_blocks: list[str] = field(default_factory=list)
    candidate_reused_tokens: int = 0
    missing_tokens: int = 0
    lookup_elapsed_s: float = 0.0
    timing_breakdown_ns: dict[str, int] = field(default_factory=dict)

    @property
    def reused_tokens(self) -> int:
        """Compatibility alias; lookup reuse is only a candidate estimate."""

        return self.candidate_reused_tokens


@dataclass
class LegoLinkExecutionResult:
    actual_reused_tokens: int = 0
    actual_recomputed_tokens: int = 0
    copied_tokens: int = 0
    remapped_tokens: int = 0
    writeback_tokens: int = 0
    fallback_tokens: int = 0
    linked_blocks: list[str] = field(default_factory=list)
    failed_blocks: list[str] = field(default_factory=list)
    execution_failure_reason: Optional[str] = None
    per_block_accounting: dict[str, dict[str, int]] = field(default_factory=dict)


LegoLinkLookupResult = LegoLinkBlockLookupResult


class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram")

    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence (raw ints in both modes)
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        if self.is_bigram:
            n = len(self.token_ids)
            return n - 1 if n > 0 else 0
        return len(self.token_ids)

    def __iter__(self) -> Iterator:
        if self.is_bigram:
            t = self.token_ids
            for i in range(len(t) - 1):
                yield (t[i], t[i + 1])
        else:
            yield from self.token_ids

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        # Normalize int -> 1-element slice so the rest handles one shape.
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigrams [start, stop) span raw tokens [start, stop + 1);
            # empty slice -> empty raw tokens (not a dangling boundary token).
            raw = self.token_ids[start : stop + 1] if stop > start else []
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> "RadixKey":
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple["RadixKey", Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: "RadixKey") -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )

    def match(self, other: "RadixKey", page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids

        if self.is_bigram:
            # Walk raw tokens; L matching tokens imply L-1 matching bigrams.
            i = 0
            for a, b in zip(t0, t1):
                if a != b:
                    break
                i += 1
            matched = max(0, min(i - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        if page_size == 1:
            i = 0
            for a, b in zip(t0, t1):
                if a != b:
                    break
                i += 1
            return i

        min_len = min(len(self), len(other))
        i = 0
        while i < min_len:
            if t0[i : i + page_size] != t1[i : i + page_size]:
                break
            i += page_size
        return i

    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str] = None) -> str:
        """SHA256 for logical units [start, end); bigram mode feeds overlapping (t_i, t_{i+1}) byte pairs."""
        hasher = hashlib.sha256()
        if prior_hash:
            hasher.update(bytes.fromhex(prior_hash))
        t = self.token_ids
        if self.is_bigram:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
                hasher.update(t[j + 1].to_bytes(4, byteorder="little", signed=False))
        else:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
        return hasher.hexdigest()


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


class RadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy: EvictionStrategy = SLRUStrategy()

        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru'."
            )

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        previous_handles = list(getattr(self, "legolink_blocks", {}).values())
        for handle in previous_handles:
            if (
                handle.ownership_lock_params is not None
                and handle.radix_node is not None
                and not handle.radix_node.evicted
            ):
                self.dec_lock_ref(handle.radix_node, handle.ownership_lock_params)
                handle.ownership_lock_params = None
            handle.invalidate("radix_cache_reset")
        self.legolink_generation = int(getattr(self, "legolink_generation", 0)) + 1
        self.legolink_enabled = _env_flag(
            "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "1"
        )
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self.legolink_blocks: OrderedDict[str, LegoLinkBlockMetadata] = OrderedDict()
        self.legolink_blocks_by_node: dict[int, set[str]] = defaultdict(set)
        self.legolink_evicted_tokens_by_key: dict[str, int] = {}
        self.legolink_stats = {
            "registered_blocks": 0,
            "active_blocks": 0,
            "linked_blocks": 0,
            "candidate_reused_tokens": 0,
            "reused_tokens": 0,
            "recomputed_tokens": 0,
            "fallback_tokens": 0,
            "evicted_tokens": 0,
            "evicted_blocks": 0,
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "lookup_hits": 0,
            "lookup_misses": 0,
            "incompatible_lookups": 0,
            "pinned_blocks": 0,
            "active_link_refs": 0,
            "stale_lookup_attempts": 0,
            "duplicate_register_hits": 0,
        }
        self._record_all_cleared_event()

    def register_legolink_block(
        self,
        *,
        cache_key: str,
        block_id: Optional[str] = None,
        sub_context_id: Optional[str] = None,
        semantic_block_id: Optional[str] = None,
        context_field: Optional[str] = None,
        token_ids: Optional[tuple[int, ...]] = None,
        token_range: Optional[tuple[int, int]] = None,
        kv_slots: Optional[torch.Tensor] = None,
        snapshot: Any = None,
        radix_node: Optional[TreeNode] = None,
        recomputed_tokens: int = 0,
        pinned: bool = False,
        compatibility: Optional[dict[str, Any]] = None,
        allocator_generation: Optional[int] = None,
        ownership_lock_params: Any = None,
    ) -> LegoLinkBlockMetadata:
        """Register an independently compiled LegoLink KV block.

        This is intentionally separate from ``insert``/``match_prefix``:
        contiguous prefix KV still lives in the radix tree, while semantic
        LegoLink blocks are looked up by cache key and carry the radix node that
        owns/pins their underlying KV slots.
        """

        if not self.legolink_enabled:
            raise RuntimeError("RadixCache LegoLink integration is disabled.")
        cache_key = str(cache_key)
        if token_ids is None:
            token_ids = tuple(int(token_id) for token_id in getattr(snapshot, "token_ids", ()))
        else:
            token_ids = tuple(int(token_id) for token_id in token_ids)
        if token_range is None:
            token_count = len(token_ids)
            token_range = (0, token_count)
        token_range = (int(token_range[0]), int(token_range[1]))
        token_count = max(token_range[1] - token_range[0], 0)
        if token_count != len(token_ids):
            raise ValueError("LegoLink token range must match the exact token IDs.")
        block_key = LegoLinkBlockKey.build(cache_key, token_ids, compatibility)
        slot_range = LegoLinkKVSlotRange.build(kv_slots, token_count, self.page_size)
        semantic_block_id = str(semantic_block_id or cache_key)
        block_id = str(block_id or semantic_block_id)
        sub_context_id = str(sub_context_id or semantic_block_id)
        context_field = str(context_field or "unknown")
        current_allocator_generation = int(
            getattr(
                self.token_to_kv_pool_allocator,
                "generation",
                getattr(self.token_to_kv_pool_allocator, "_generation", 0),
            )
            or 0
        )
        if allocator_generation is None:
            allocator_generation = current_allocator_generation
        allocator_generation = int(allocator_generation)

        old = self.legolink_blocks.get(cache_key)
        if old is not None and old.valid and old.generation == self.legolink_generation:
            # Content decides; labels and slots do not. `block_key` already
            # carries the cache key, the token hash and every compatibility
            # fingerprint, so two registrations that agree on it are two
            # computations of the same reusable KV -- which is what happens now
            # that a group key can be claimed by a compile or by a promotion.
            # They may legitimately differ in provenance labels and in which
            # slots hold the result; neither makes the resident entry wrong, so
            # it stays authoritative and the caller keeps what it brought.
            same_content = (
                old.block_key == block_key
                and old.token_range == token_range
                and old.allocator_generation == allocator_generation
            )
            if not same_content:
                raise LegoLinkDuplicateRegistrationError(
                    f"LegoLink incompatible duplicate registration for cache key "
                    f"{cache_key}: the resident block describes different content."
                )
            differing_labels = sorted(
                name
                for name, resident, offered in (
                    ("block_id", old.block_id, block_id),
                    ("sub_context_id", old.sub_context_id, sub_context_id),
                    ("semantic_block_id", old.semantic_block_id, semantic_block_id),
                    ("context_field", old.context_field, context_field),
                )
                if resident != offered
            )
            offered_slots_adopted = bool(
                old.kv_slots is None
                or slot_range.slots is None
                or torch.equal(old.kv_slots.cpu(), slot_range.slots.cpu())
            )
            if ownership_lock_params is not None and radix_node is not None:
                self.dec_lock_ref(radix_node, ownership_lock_params)
            if old.snapshot is None and snapshot is not None:
                old.snapshot = snapshot
            if pinned:
                old.pin_count += 1
            old.last_access_time = time.monotonic()
            self.legolink_stats["duplicate_register_hits"] += 1
            self.legolink_blocks.move_to_end(cache_key)
            _emit_legolink_trace(
                "radix_legolink_duplicate_register_hit",
                phase="compile",
                cache_key=cache_key,
                block_id=old.block_id,
                sub_context_id=old.sub_context_id,
                semantic_block_id=old.semantic_block_id,
                context_field=old.context_field,
                token_count=old.token_count,
                generation=old.generation,
                allocator_generation=old.allocator_generation,
                # So a caller can tell whether the slots it offered became the
                # resident ones or are still its own to account for.
                offered_slots_adopted=offered_slots_adopted,
                differing_labels=differing_labels,
            )
            return old

        acquired_ownership = False
        if radix_node is not None and ownership_lock_params is None:
            lock_result = self.inc_lock_ref(radix_node)
            ownership_lock_params = lock_result.to_dec_params()
            acquired_ownership = True
        try:
            metadata = LegoLinkBlockMetadata(
                cache_key=cache_key,
                block_key=block_key,
                block_id=block_id,
                sub_context_id=sub_context_id,
                semantic_block_id=semantic_block_id,
                context_field=context_field,
                token_ids=token_ids,
                token_range=token_range,
                slot_range=slot_range,
                compile_local_positions=tuple(range(token_count)),
                snapshot=snapshot,
                radix_node=radix_node,
                ownership_lock_params=ownership_lock_params,
                pin_count=1 if pinned else 0,
                recomputed_tokens=int(recomputed_tokens),
                generation=self.legolink_generation,
                allocator_generation=allocator_generation,
                estimated_saved_recompute_cost=float(token_count),
                kv_residency_cost=float(token_count),
            )
            self.legolink_blocks[cache_key] = metadata
            self.legolink_evicted_tokens_by_key.pop(cache_key, None)
            self.legolink_blocks.move_to_end(cache_key)
            if radix_node is not None:
                self.legolink_blocks_by_node[id(radix_node)].add(cache_key)
        except Exception:
            if acquired_ownership and radix_node is not None:
                self.dec_lock_ref(radix_node, ownership_lock_params)
            raise
        self.legolink_stats["registered_blocks"] = len(self.legolink_blocks)
        self.legolink_stats["active_blocks"] = len(self.legolink_blocks)
        _emit_legolink_trace(
            "radix_legolink_block_registered",
            phase="compile",
            cache_key=cache_key,
            block_id=metadata.block_id,
            sub_context_id=metadata.sub_context_id,
            semantic_block_id=metadata.semantic_block_id,
            context_field=metadata.context_field,
            token_count=metadata.token_count,
            generation=metadata.generation,
            allocator_generation=metadata.allocator_generation,
            page_size=metadata.page_size,
            aligned_token_count=metadata.aligned_token_count,
            tail_token_count=metadata.tail_token_count,
        )
        return metadata

    def has_legolink_block(self, cache_key: str) -> bool:
        metadata = self.legolink_blocks.get(str(cache_key))
        return bool(
            self.legolink_enabled
            and metadata is not None
            and metadata.valid
            and metadata.generation == self.legolink_generation
            and not metadata.evicted
        )

    def get_legolink_block(
        self, cache_key: str
    ) -> Optional[LegoLinkBlockMetadata]:
        return (
            self.legolink_blocks.get(str(cache_key))
            if self.has_legolink_block(cache_key)
            else None
        )

    def get_legolink_block_tokens(self, cache_key: str) -> Optional[tuple[int, ...]]:
        metadata = self.get_legolink_block(cache_key)
        if metadata is None:
            return None
        return metadata.token_ids

    def get_legolink_block_snapshot(self, cache_key: str) -> Any:
        metadata = self.get_legolink_block(cache_key)
        return None if metadata is None else metadata.snapshot

    def legolink_memory_bytes(self) -> tuple[int, int]:
        registry_bytes = 0
        represented_bytes = 0
        for metadata in self.legolink_blocks.values():
            snapshot = metadata.snapshot
            if snapshot is None:
                if metadata.kv_slots is not None:
                    registry_bytes += metadata.kv_slots.numel() * metadata.kv_slots.element_size()
                continue
            registry_bytes += int(getattr(snapshot, "owned_bytes", 0))
            represented_bytes += int(getattr(snapshot, "kv_bytes", 0))
        return registry_bytes, represented_bytes

    def rebind_legolink_block(
        self,
        cache_key: str,
        kv_slots: torch.Tensor,
        *,
        radix_node: Optional[TreeNode] = None,
    ) -> LegoLinkBlockMetadata:
        metadata = self.legolink_blocks.get(str(cache_key))
        if metadata is None:
            raise KeyError(f"Unknown LegoLink block: {cache_key}")
        metadata.kv_slots = kv_slots
        if radix_node is not None:
            if metadata.radix_node is not None:
                self.legolink_blocks_by_node.get(
                    id(metadata.radix_node), set()
                ).discard(metadata.cache_key)
            metadata.radix_node = radix_node
            self.legolink_blocks_by_node[id(radix_node)].add(metadata.cache_key)
        metadata.last_access_time = time.monotonic()
        self.legolink_blocks.move_to_end(metadata.cache_key)
        return metadata

    def lookup_legolink_blocks(
        self,
        cache_keys: list[str],
        destination_offsets: Optional[list[int]] = None,
        *,
        semantic_block_ids: Optional[list[str]] = None,
        link_plan: Optional[dict[str, Any]] = None,
        pin: bool = False,
        compatibility: Optional[dict[str, Any]] = None,
    ) -> LegoLinkLookupResult:
        """Lookup multiple semantic KV blocks by cache key.

        Returns matched metadata, KV slot ranges, global offsets derived from
        the link plan, and token-level reuse accounting. Missing or radix-evicted
        blocks are reported explicitly so callers can fall back or fail cleanly.
        """

        lookup_started = time.perf_counter()
        lookup_started_ns = time.perf_counter_ns()
        plan_started_ns = time.perf_counter_ns()
        with _legolink_profile_range("LinkPlanBuild"):
            plan = LegoLinkLinkPlan.from_legacy(
                cache_keys,
                destination_offsets,
                semantic_block_ids,
                link_plan,
            )
        plan_ns = time.perf_counter_ns() - plan_started_ns
        result = LegoLinkLookupResult(
            requested_cache_keys=list(plan.requested_cache_keys),
            destination_offsets=list(plan.destination_offsets),
        )
        self.legolink_stats["lookups"] += 1
        _emit_legolink_trace(
            "radix_legolink_lookup_start",
            phase="link",
            cache_keys=list(plan.requested_cache_keys),
            destination_offsets=list(plan.destination_offsets),
            generation=self.legolink_generation,
        )

        compatibility_validation_ns = 0
        classification_ns = 0
        for index, raw_cache_key in enumerate(plan.requested_cache_keys):
            cache_key = str(raw_cache_key)
            metadata = self.legolink_blocks.get(cache_key)
            global_offset = plan.destination_offsets[index]
            result.global_offsets[cache_key] = global_offset
            planned_range = (
                plan.block_token_ranges[index]
                if index < len(plan.block_token_ranges)
                else (global_offset, global_offset)
            )
            planned_tokens = max(planned_range[1] - planned_range[0], 0)

            if metadata is None:
                classification_started_ns = time.perf_counter_ns()
                result.missing_blocks.append(cache_key)
                self.legolink_stats["misses"] += 1
                self.legolink_stats["lookup_misses"] += 1
                result.missing_tokens += planned_tokens
                _emit_legolink_trace(
                    "radix_legolink_block_miss",
                    phase="link",
                    cache_key=cache_key,
                    reason="missing",
                    generation=self.legolink_generation,
                )
                classification_ns += (
                    time.perf_counter_ns() - classification_started_ns
                )
                continue

            if not metadata.valid or metadata.generation != self.legolink_generation:
                classification_started_ns = time.perf_counter_ns()
                result.stale_blocks.append(cache_key)
                self.legolink_stats["misses"] += 1
                self.legolink_stats["lookup_misses"] += 1
                self.legolink_stats["stale_lookup_attempts"] += 1
                _emit_legolink_trace(
                    "radix_legolink_block_stale",
                    phase="link",
                    cache_key=cache_key,
                    reason="stale_generation",
                    handle_generation=metadata.generation,
                    owner_generation=self.legolink_generation,
                )
                classification_ns += (
                    time.perf_counter_ns() - classification_started_ns
                )
                continue

            if metadata.evicted:
                classification_started_ns = time.perf_counter_ns()
                result.missing_blocks.append(cache_key)
                self.legolink_stats["misses"] += 1
                self.legolink_stats["lookup_misses"] += 1
                result.missing_tokens += metadata.token_count
                self._mark_legolink_block_evicted(metadata)
                classification_ns += (
                    time.perf_counter_ns() - classification_started_ns
                )
                continue

            current_allocator_generation = int(
                getattr(
                    self.token_to_kv_pool_allocator,
                    "generation",
                    getattr(self.token_to_kv_pool_allocator, "_generation", 0),
                )
                or 0
            )
            if metadata.allocator_generation != current_allocator_generation:
                classification_started_ns = time.perf_counter_ns()
                result.stale_blocks.append(cache_key)
                self.legolink_stats["misses"] += 1
                self.legolink_stats["lookup_misses"] += 1
                self.legolink_stats["stale_lookup_attempts"] += 1
                result.missing_tokens += metadata.token_count
                _emit_legolink_trace(
                    "radix_legolink_block_stale",
                    phase="link",
                    cache_key=cache_key,
                    reason="allocator_generation_mismatch",
                    handle_allocator_generation=metadata.allocator_generation,
                    owner_allocator_generation=current_allocator_generation,
                )
                classification_ns += (
                    time.perf_counter_ns() - classification_started_ns
                )
                continue

            compatibility_started_ns = time.perf_counter_ns()
            with _legolink_profile_range("CompatibilityValidation"):
                expected_id = (
                    plan.semantic_block_ids[index]
                    if index < len(plan.semantic_block_ids)
                    else metadata.semantic_block_id
                )
                semantic_id_compatible = not expected_id or metadata.semantic_block_id in (
                    expected_id,
                    cache_key,
                )
            compatibility_validation_ns += (
                time.perf_counter_ns() - compatibility_started_ns
            )
            if not semantic_id_compatible:
                classification_started_ns = time.perf_counter_ns()
                result.incompatible_blocks.append(cache_key)
                self.legolink_stats["misses"] += 1
                self.legolink_stats["lookup_misses"] += 1
                self.legolink_stats["incompatible_lookups"] += 1
                result.missing_tokens += metadata.token_count
                _emit_legolink_trace(
                    "radix_legolink_block_incompatible",
                    phase="link",
                    cache_key=cache_key,
                    reason="semantic_block_id_mismatch",
                    expected_semantic_block_id=expected_id,
                    actual_semantic_block_id=metadata.semantic_block_id,
                )
                classification_ns += (
                    time.perf_counter_ns() - classification_started_ns
                )
                continue

            if compatibility is not None:
                compatibility_started_ns = time.perf_counter_ns()
                with _legolink_profile_range("CompatibilityValidation"):
                    expected_key = LegoLinkBlockKey.build(
                        cache_key, metadata.token_ids, compatibility
                    )
                compatibility_validation_ns += (
                    time.perf_counter_ns() - compatibility_started_ns
                )
                if metadata.block_key != expected_key:
                    classification_started_ns = time.perf_counter_ns()
                    result.incompatible_blocks.append(cache_key)
                    self.legolink_stats["misses"] += 1
                    self.legolink_stats["lookup_misses"] += 1
                    self.legolink_stats["incompatible_lookups"] += 1
                    result.missing_tokens += metadata.token_count
                    _emit_legolink_trace(
                        "radix_legolink_block_incompatible",
                        phase="link",
                        cache_key=cache_key,
                        semantic_block_id=metadata.semantic_block_id,
                        reason="compatibility_fingerprint_mismatch",
                    )
                    classification_ns += (
                        time.perf_counter_ns() - classification_started_ns
                    )
                    continue

            metadata.last_access_time = time.monotonic()

            result.matched_handles.append(metadata)
            result.matched_blocks[cache_key] = metadata
            result.kv_slot_ranges[cache_key] = metadata.kv_slots
            result.ordered_slot_ranges.append(metadata.slot_range)
            result.compile_local_positions[cache_key] = metadata.compile_local_positions
            result.destination_global_positions.append(
                tuple(global_offset + position for position in metadata.compile_local_positions)
            )
            result.candidate_reused_tokens += metadata.token_count
            self.legolink_stats["hits"] += 1
            self.legolink_stats["lookup_hits"] += 1
            self.legolink_stats["candidate_reused_tokens"] += metadata.token_count
            self.legolink_blocks.move_to_end(cache_key)
            _emit_legolink_trace(
                "radix_legolink_block_hit",
                phase="link",
                cache_key=cache_key,
                semantic_block_id=metadata.semantic_block_id,
                sub_context_id=metadata.sub_context_id,
                context_field=metadata.context_field,
                token_count=metadata.token_count,
                candidate_reused_tokens=metadata.token_count,
                generation=metadata.generation,
                allocator_generation=metadata.allocator_generation,
                global_offset=global_offset,
            )

        classification_counts = {
            "matched": len(result.matched_handles),
            "missing": len(result.missing_blocks),
            "stale": len(result.stale_blocks),
            "incompatible": len(result.incompatible_blocks),
        }
        if pin and result.matched_handles:
            self.pin_legolink_blocks(result.matched_handles)
        result.lookup_elapsed_s = time.perf_counter() - lookup_started
        result.timing_breakdown_ns = {
            "radix_link_plan_materialization": int(plan_ns),
            "compatibility_validation": int(compatibility_validation_ns),
            "missing_stale_incompatible_classification": int(classification_ns),
            "radixcache_lookup_total": int(time.perf_counter_ns() - lookup_started_ns),
        }
        _emit_legolink_trace(
            "radix_legolink_lookup_complete",
            phase="link",
            cache_keys=list(plan.requested_cache_keys),
            classification=classification_counts,
            timing_breakdown_ns=result.timing_breakdown_ns,
            timing_breakdown_ms={
                key: value / 1_000_000.0
                for key, value in result.timing_breakdown_ns.items()
            },
        )
        return result

    def acquire_legolink_blocks(
        self,
        blocks: list[Union[str, LegoLinkBlockHandle]],
        *,
        pin: bool = False,
    ) -> list[LegoLinkBlockHandle]:
        """Atomically acquire active-LinkPlan references for valid handles."""

        handles: list[LegoLinkBlockHandle] = []
        with _legolink_profile_range("Acquire"):
            seen: set[int] = set()
            for block in blocks:
                metadata = (
                    block
                    if isinstance(block, LegoLinkBlockHandle)
                    else self.legolink_blocks.get(str(block))
                )
                if metadata is None or not self.has_legolink_block(
                    metadata.cache_key
                ):
                    raise RuntimeError(
                        f"Cannot acquire stale LegoLink block: {block}"
                    )
                if id(metadata) in seen:
                    continue
                seen.add(id(metadata))
                handles.append(metadata)

            for metadata in handles:
                metadata.active_link_refcount += 1
                if pin:
                    metadata.pin_count += 1
        self._refresh_legolink_lifetime_stats()
        if handles:
            _emit_legolink_trace(
                "radix_legolink_blocks_acquired",
                phase="link",
                cache_keys=[handle.cache_key for handle in handles],
                generations=[handle.generation for handle in handles],
                pin=pin,
            )
        return handles

    def record_legolink_execution(
        self,
        handles: list[LegoLinkBlockHandle],
        result: LegoLinkExecutionResult,
    ) -> None:
        """Record post-execution accounting, separate from availability lookup."""

        by_key = {handle.cache_key: handle for handle in handles}
        linked = [by_key[key] for key in result.linked_blocks if key in by_key]
        if linked:
            for handle in linked:
                accounting = result.per_block_accounting.get(handle.cache_key, {})
                per_block_reused = int(accounting.get("actual_reused_tokens", 0))
                per_block_recomputed = int(
                    accounting.get("actual_recomputed_tokens", 0)
                )
                handle.reuse_count += 1
                handle.reused_tokens += per_block_reused
                handle.recomputed_tokens += per_block_recomputed
                handle.last_access_time = time.monotonic()
        self.legolink_stats["linked_blocks"] += len(linked)
        self.legolink_stats["reused_tokens"] += int(result.actual_reused_tokens)
        self.legolink_stats["recomputed_tokens"] += int(
            result.actual_recomputed_tokens
        )
        self.legolink_stats["fallback_tokens"] += int(result.fallback_tokens)

    def record_legolink_fallback(self, fallback_tokens: int) -> None:
        self.legolink_stats["fallback_tokens"] += max(int(fallback_tokens), 0)

    def _refresh_legolink_lifetime_stats(self) -> None:
        self.legolink_stats["active_blocks"] = sum(
            int(handle.valid) for handle in self.legolink_blocks.values()
        )
        self.legolink_stats["pinned_blocks"] = sum(
            int(handle.pin_count > 0) for handle in self.legolink_blocks.values()
        )
        self.legolink_stats["active_link_refs"] = sum(
            handle.active_link_refcount for handle in self.legolink_blocks.values()
        )

    def release_legolink_blocks(
        self,
        blocks: list[Union[str, LegoLinkBlockHandle]],
        *,
        unpin: bool = False,
    ) -> None:
        released: list[str] = []
        with _legolink_profile_range("Release"):
            for block in blocks:
                metadata = (
                    block
                    if isinstance(block, LegoLinkBlockHandle)
                    else self.legolink_blocks.get(str(block))
                )
                if metadata is None:
                    continue
                if metadata.active_link_refcount <= 0:
                    raise RuntimeError(
                        f"LegoLink block {metadata.cache_key} released without acquire."
                    )
                metadata.active_link_refcount -= 1
                if unpin:
                    if metadata.pin_count <= 0:
                        raise RuntimeError(
                            f"LegoLink block {metadata.cache_key} unpinned without pin."
                        )
                    metadata.pin_count -= 1
                released.append(metadata.cache_key)
        self._refresh_legolink_lifetime_stats()
        if released:
            _emit_legolink_trace(
                "radix_legolink_blocks_released",
                phase="link",
                cache_keys=released,
                generation=self.legolink_generation,
            )

    def unref_legolink_blocks(self, cache_keys: list[str], *, unpin: bool = False):
        self.release_legolink_blocks(cache_keys, unpin=unpin)

    def pin_legolink_blocks(
        self, blocks: list[Union[str, LegoLinkBlockHandle]]
    ) -> None:
        pinned: list[str] = []
        seen: set[int] = set()
        for block in blocks:
            metadata = (
                block
                if isinstance(block, LegoLinkBlockHandle)
                else self.legolink_blocks.get(str(block))
            )
            if metadata is not None and id(metadata) in seen:
                continue
            if metadata is not None:
                seen.add(id(metadata))
            if (
                metadata is not None
                and metadata.valid
                and metadata.generation == self.legolink_generation
            ):
                metadata.pin_count += 1
                pinned.append(metadata.cache_key)
        self._refresh_legolink_lifetime_stats()
        if pinned:
            _emit_legolink_trace(
                "radix_legolink_blocks_pinned",
                phase="link",
                cache_keys=pinned,
                generation=self.legolink_generation,
            )

    def unpin_legolink_blocks(self, cache_keys: list[str]):
        for raw_cache_key in cache_keys:
            metadata = self.legolink_blocks.get(str(raw_cache_key))
            if metadata is not None:
                metadata.pin_count = max(metadata.pin_count - 1, 0)
        self._refresh_legolink_lifetime_stats()

    def unregister_legolink_block(
        self,
        cache_key: str,
        *,
        reason: str = "unregistered",
        force: bool = False,
    ) -> Optional[LegoLinkBlockMetadata]:
        metadata = self.legolink_blocks.get(str(cache_key))
        if metadata is None:
            return None
        if not force and (metadata.pin_count > 0 or metadata.refcount > 0):
            return None
        metadata = self.legolink_blocks.pop(str(cache_key))
        if metadata.radix_node is not None:
            self.legolink_blocks_by_node.get(id(metadata.radix_node), set()).discard(
                metadata.cache_key
            )
        metadata.invalidate(reason)
        _emit_legolink_trace(
            "radix_legolink_generation_invalidated",
            phase="cache_management",
            cache_key=metadata.cache_key,
            semantic_block_id=metadata.semantic_block_id,
            generation=metadata.generation,
            reason=reason,
        )
        if (
            metadata.ownership_lock_params is not None
            and metadata.radix_node is not None
            and not metadata.radix_node.evicted
        ):
            self.dec_lock_ref(metadata.radix_node, metadata.ownership_lock_params)
            metadata.ownership_lock_params = None
        self._record_legolink_eviction(metadata)
        self.legolink_stats["registered_blocks"] = len(self.legolink_blocks)
        self._refresh_legolink_lifetime_stats()
        return metadata

    def remove_legolink_block(self, cache_key: str) -> Optional[LegoLinkBlockMetadata]:
        return self.unregister_legolink_block(cache_key, reason="evicted")

    def clear_legolink_blocks(self):
        for cache_key in list(self.legolink_blocks):
            self.unregister_legolink_block(
                cache_key, reason="registry_cleared", force=True
            )
        self.legolink_blocks_by_node.clear()
        self.legolink_evicted_tokens_by_key.clear()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0).
            ``last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        if self.disable or len(key) == 0:
            return self._empty_match_result

        key = key.page_aligned(self.page_size)

        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        # Remove req slot release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        self._release_legolink_eviction_candidates(num_tokens)
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if self._node_has_active_legolink_refs(x):
                continue
            self._on_legolink_node_evicted(x)
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def capacity_evictable_size(self) -> int:
        return self.evictable_size_ + self._legolink_capacity_evictable_size()

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = key.child_key(self.page_size)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _delete_leaf(self, node):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _node_has_active_legolink_refs(self, node: TreeNode) -> bool:
        return any(
            metadata is not None
            and metadata.valid
            and (metadata.pin_count > 0 or metadata.refcount > 0)
            for cache_key in self.legolink_blocks_by_node.get(id(node), set())
            for metadata in [self.legolink_blocks.get(cache_key)]
        )

    def _legolink_capacity_evictable_size(self) -> int:
        """Count ownership locks that capacity eviction is allowed to release.

        Scheduler admission consults ``capacity_evictable_size()`` before
        allocation calls ``evict()``. Without this accounting, independently
        compiled LegoLink blocks can fill the KV pool while their releasable
        ownership locks make the scheduler believe that zero tokens are
        evictable. The regular ``evictable_size()`` remains disjoint from
        ``protected_size()`` for pool invariant accounting.
        """

        releasable_locks_by_node: dict[int, tuple[TreeNode, int]] = {}
        for metadata in self.legolink_blocks.values():
            node = metadata.radix_node
            if (
                not metadata.valid
                or metadata.pin_count > 0
                or metadata.refcount > 0
                or metadata.ownership_lock_params is None
                or node is None
            ):
                continue
            # One ownership lock contributes one lock_ref along the complete
            # path to the root. Count those contributions so shared prefix
            # nodes are capacity-evictable only when every lock on the node is
            # owned by a currently releasable LegoLink block.
            while node != self.root_node:
                node_id = id(node)
                existing = releasable_locks_by_node.get(node_id)
                releasable_locks_by_node[node_id] = (
                    node,
                    (existing[1] if existing is not None else 0) + 1,
                )
                node = node.parent

        return sum(
            len(node.key)
            for node, releasable_locks in releasable_locks_by_node.values()
            if node.lock_ref > 0 and node.lock_ref == releasable_locks
        )

    def legolink_eviction_score(
        self,
        metadata: LegoLinkBlockMetadata,
        *,
        expected_reuse_benefit: Optional[float] = None,
        recompute_cost: Optional[float] = None,
        residency_cost: Optional[float] = None,
    ) -> float:
        """Lower scores are evicted first; inputs remain explicit for tuning."""

        age = max(time.monotonic() - metadata.last_access_time, 0.0)
        expected_reuse_benefit = (
            float(expected_reuse_benefit)
            if expected_reuse_benefit is not None
            else float(metadata.reuse_count + 1)
        )
        recompute_cost = (
            float(recompute_cost)
            if recompute_cost is not None
            else float(max(metadata.token_count, 1))
        )
        residency_cost = (
            float(residency_cost)
            if residency_cost is not None
            else float(max(metadata.aligned_token_count, 1))
        )
        return (expected_reuse_benefit * recompute_cost) / (
            residency_cost * (age + 1.0)
        )

    def _release_legolink_eviction_candidates(self, requested_tokens: int) -> int:
        if requested_tokens <= self.evictable_size_:
            return 0
        candidates = [
            metadata
            for metadata in self.legolink_blocks.values()
            if metadata.valid
            and metadata.pin_count == 0
            and metadata.refcount == 0
            and metadata.ownership_lock_params is not None
        ]
        candidates.sort(key=self.legolink_eviction_score)
        released_tokens = 0
        for metadata in candidates:
            if requested_tokens <= self.evictable_size_:
                break
            token_count = metadata.token_count
            removed = self.unregister_legolink_block(
                metadata.cache_key, reason="capacity_eviction"
            )
            if removed is not None:
                released_tokens += token_count
        return released_tokens

    def _on_legolink_node_evicted(self, node: TreeNode):
        cache_keys = list(self.legolink_blocks_by_node.pop(id(node), set()))
        for cache_key in cache_keys:
            metadata = self.legolink_blocks.pop(cache_key, None)
            if metadata is None:
                continue
            metadata.invalidate("radix_node_evicted")
            metadata.ownership_lock_params = None
            self._record_legolink_eviction(metadata)
        self.legolink_stats["registered_blocks"] = len(self.legolink_blocks)

    def _mark_legolink_block_evicted(
        self, metadata: LegoLinkBlockMetadata
    ) -> int:
        removed = self.legolink_blocks.pop(metadata.cache_key, None)
        if removed is None:
            return 0
        if removed.radix_node is not None:
            self.legolink_blocks_by_node.get(id(removed.radix_node), set()).discard(
                removed.cache_key
            )
        removed.invalidate("radix_node_evicted")
        removed.ownership_lock_params = None
        evicted = self._record_legolink_eviction(removed)
        self.legolink_stats["registered_blocks"] = len(self.legolink_blocks)
        self._refresh_legolink_lifetime_stats()
        return evicted

    def _record_legolink_eviction(self, metadata: LegoLinkBlockMetadata) -> int:
        token_count = metadata.token_count
        if token_count <= 0:
            return 0
        previous = self.legolink_evicted_tokens_by_key.get(metadata.cache_key, 0)
        if previous:
            return 0
        metadata.evicted_tokens += token_count
        self.legolink_evicted_tokens_by_key[metadata.cache_key] = token_count
        self.legolink_stats["evicted_tokens"] += token_count
        self.legolink_stats["evicted_blocks"] += 1
        _emit_legolink_trace(
            "radix_legolink_block_evicted",
            phase="cache_management",
            cache_key=metadata.cache_key,
            block_id=metadata.block_id,
            sub_context_id=metadata.sub_context_id,
            semantic_block_id=metadata.semantic_block_id,
            context_field=metadata.context_field,
            token_count=token_count,
            evicted_tokens=token_count,
            generation=metadata.generation,
            reason=metadata.stale_reason,
        )
        return token_count

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
