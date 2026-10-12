import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from tests.pic.support import SRT_ROOT  # noqa: E402

MEM_CACHE = SRT_ROOT / "mem_cache"


def install_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for package_name in (
    "sglang",
    "sglang.srt",
    "sglang.srt.mem_cache",
    "sglang.srt.observability",
    "sglang.srt.disaggregation",
):
    install_module(package_name, types.ModuleType(package_name))

allocator = types.ModuleType("sglang.srt.mem_cache.allocator")
allocator.BaseTokenToKVPoolAllocator = object
install_module("sglang.srt.mem_cache.allocator", allocator)

memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")
memory_pool.ReqToTokenPool = object
install_module("sglang.srt.mem_cache.memory_pool", memory_pool)

metrics = types.ModuleType("sglang.srt.observability.metrics_collector")
metrics.RadixCacheMetricsCollector = object
install_module("sglang.srt.observability.metrics_collector", metrics)

kv_events = types.ModuleType("sglang.srt.disaggregation.kv_events")
for class_name in ("AllBlocksCleared", "BlockRemoved", "BlockStored"):
    setattr(kv_events, class_name, type(class_name, (), {"__init__": lambda self, **_: None}))
kv_events.StorageMedium = types.SimpleNamespace(GPU="GPU", CPU="CPU")
install_module("sglang.srt.disaggregation.kv_events", kv_events)

utils = types.ModuleType("sglang.srt.mem_cache.utils")
utils.split_node_hash_value = lambda hash_value, split_len, page_size: (None, None)
utils.compute_node_hash_values = lambda node, page_size: []
utils.hash_str_to_int64 = lambda value: 0
install_module("sglang.srt.mem_cache.utils", utils)

load_module(
    "sglang.srt.mem_cache.cache_init_params",
    MEM_CACHE / "cache_init_params.py",
)
load_module("sglang.srt.mem_cache.base_prefix_cache", MEM_CACHE / "base_prefix_cache.py")
load_module("sglang.srt.mem_cache.events", MEM_CACHE / "events.py")
load_module("sglang.srt.mem_cache.evict_policy", MEM_CACHE / "evict_policy.py")
radix_cache = load_module("sglang.srt.mem_cache.radix_cache", MEM_CACHE / "radix_cache.py")
RadixCache = radix_cache.RadixCache
TreeNode = radix_cache.TreeNode
LegoLinkBlockKey = radix_cache.LegoLinkBlockKey
LegoLinkBlockHandle = radix_cache.LegoLinkBlockHandle


class MockAllocator:
    device = "cpu"

    def __init__(self):
        self.freed = []
        self.generation = 0

    def free(self, slots):
        self.freed.append(tuple(int(item) for item in slots.tolist()))


class RadixCacheLegoLinkTest(unittest.TestCase):
    def test_registration_builds_compatibility_key_and_page_tail_metadata(self):
        tree = RadixCache.create_simulated(page_size=4)
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2, 3, 4, 5, 6),
            token_range=(0, 6),
            kv_slots=torch.tensor([10, 11, 12, 13, 14, 15]),
            compatibility={
                "model_fingerprint": "model-r1",
                "tokenizer_fingerprint": "tokenizer-r1",
                "kv_dtype": "bfloat16",
                "rope_fingerprint": "rope-r1",
                "adapter_id": "none",
                "tp_world_size": 1,
                "policy_version": "legolink-v1",
            },
        )

        self.assertIsInstance(handle, LegoLinkBlockHandle)
        self.assertIsInstance(handle.block_key, LegoLinkBlockKey)
        self.assertEqual(handle.page_size, 4)
        self.assertEqual(handle.aligned_token_count, 4)
        self.assertEqual(handle.tail_token_count, 2)
        self.assertEqual(handle.generation, tree.legolink_generation)
        self.assertEqual(handle.block_id, "block-a")
        self.assertEqual(handle.sub_context_id, "block-a")
        self.assertEqual(handle.context_field, "unknown")
        self.assertEqual(handle.block_key.segmentation_policy_version, "field-first-v1")
        self.assertEqual(handle.block_key.attention_backend_fingerprint, "runtime-local")
        self.assertTrue(handle.valid)
        self.assertTrue(tree.has_legolink_block("cache-a"))

    def test_duplicate_registration_reuses_compatible_handle_and_rejects_conflict(self):
        tree = RadixCache.create_simulated()
        first = tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2, 3),
            kv_slots=torch.tensor([7, 8, 9]),
        )
        duplicate = tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2, 3),
            kv_slots=torch.tensor([7, 8, 9]),
        )

        self.assertIs(duplicate, first)
        self.assertEqual(tree.legolink_stats["duplicate_register_hits"], 1)
        with self.assertRaisesRegex(ValueError, "incompatible duplicate"):
            tree.register_legolink_block(
                cache_key="cache-a",
                semantic_block_id="block-a",
                token_ids=(1, 2, 4),
                kv_slots=torch.tensor([7, 8, 10]),
            )

    def test_ordered_lookup_preserves_duplicate_request_keys_and_offsets(self):
        tree = RadixCache.create_simulated()
        for suffix in ("a", "b"):
            tree.register_legolink_block(
                cache_key=f"cache-{suffix}",
                semantic_block_id=f"block-{suffix}",
                token_ids=(1, 2),
                kv_slots=torch.tensor([1, 2]),
            )

        result = tree.lookup_legolink_blocks(
            ["cache-b", "cache-a", "cache-b"],
            [0, 2, 4],
            semantic_block_ids=["block-b", "block-a", "block-b"],
            pin=False,
        )

        self.assertEqual(
            [handle.cache_key for handle in result.matched_handles],
            ["cache-b", "cache-a", "cache-b"],
        )
        self.assertEqual(result.destination_offsets, [0, 2, 4])
        self.assertEqual(result.reused_tokens, 6)
        acquired = tree.acquire_legolink_blocks(result.matched_handles, pin=True)
        tree.release_legolink_blocks(acquired, unpin=True)
        self.assertTrue(all(handle.refcount == 0 for handle in result.matched_handles))

    def test_reset_invalidates_old_handle_and_advances_generation(self):
        tree = RadixCache.create_simulated()
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2),
            kv_slots=torch.tensor([1, 2]),
        )
        generation = tree.legolink_generation

        tree.reset()

        self.assertEqual(tree.legolink_generation, generation + 1)
        self.assertFalse(handle.valid)
        self.assertEqual(handle.stale_reason, "radix_cache_reset")
        result = tree.lookup_legolink_blocks(["cache-a"])
        self.assertEqual(result.missing_blocks, ["cache-a"])

    def test_stale_generation_and_semantic_mismatch_are_reported_separately(self):
        tree = RadixCache.create_simulated()
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2),
            kv_slots=torch.tensor([1, 2]),
        )
        handle.generation -= 1

        stale = tree.lookup_legolink_blocks(["cache-a"])
        self.assertEqual(stale.stale_blocks, ["cache-a"])
        self.assertEqual(tree.legolink_stats["stale_lookup_attempts"], 1)

        handle.generation = tree.legolink_generation
        handle.valid = True
        incompatible = tree.lookup_legolink_blocks(
            ["cache-a"], semantic_block_ids=["different-block"]
        )
        self.assertEqual(incompatible.incompatible_blocks, ["cache-a"])

    def test_feature_disabled_keeps_prefix_cache_and_disables_secondary_registry(self):
        with mock.patch.dict(os.environ, {"SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "0"}):
            tree = RadixCache.create_simulated()
        tree.insert(
            radix_cache.InsertParams(
                key=radix_cache.RadixKey([1, 2, 3]),
                value=torch.tensor([11, 12, 13]),
            )
        )
        prefix = tree.match_prefix(
            radix_cache.MatchPrefixParams(key=radix_cache.RadixKey([1, 2, 9]))
        )

        self.assertEqual(prefix.device_indices.tolist(), [11, 12])
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            tree.register_legolink_block(cache_key="cache-a", token_ids=(1, 2))

    def test_concurrent_readers_hold_independent_active_references(self):
        tree = RadixCache.create_simulated()
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2, 3),
            kv_slots=torch.tensor([7, 8, 9]),
        )

        first = tree.lookup_legolink_blocks(["cache-a"])
        second = tree.lookup_legolink_blocks(["cache-a"])
        first_handles = tree.acquire_legolink_blocks(first.matched_handles, pin=True)
        second_handles = tree.acquire_legolink_blocks(second.matched_handles, pin=True)
        self.assertEqual(handle.refcount, 2)
        self.assertEqual(handle.pin_count, 2)

        tree.release_legolink_blocks(first_handles, unpin=True)
        self.assertEqual(handle.refcount, 1)
        tree.release_legolink_blocks(second_handles, unpin=True)
        self.assertEqual(handle.refcount, 0)
        self.assertEqual(handle.pin_count, 0)

    def test_node_split_keeps_exact_slot_handle_valid(self):
        tree = RadixCache.create_simulated()
        tree.insert(
            radix_cache.InsertParams(
                key=radix_cache.RadixKey([1, 2, 3, 4]),
                value=torch.tensor([11, 12, 13, 14]),
            )
        )
        original = tree.match_prefix(
            radix_cache.MatchPrefixParams(key=radix_cache.RadixKey([1, 2, 3, 4]))
        )
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2, 3, 4),
            kv_slots=original.device_indices,
            radix_node=original.last_device_node,
        )

        tree.match_prefix(
            radix_cache.MatchPrefixParams(key=radix_cache.RadixKey([1, 2, 9]))
        )
        result = tree.lookup_legolink_blocks(["cache-a"])

        self.assertIs(result.matched_handles[0], handle)
        self.assertEqual(result.kv_slot_ranges["cache-a"].tolist(), [11, 12, 13, 14])

    def test_eviction_skips_pinned_block_then_frees_once_after_release(self):
        allocator = MockAllocator()
        tree = RadixCache.create_simulated(mock_allocator=allocator)
        tree.insert(
            radix_cache.InsertParams(
                key=radix_cache.RadixKey([1, 2, 3]),
                value=torch.tensor([11, 12, 13]),
            )
        )
        match = tree.match_prefix(
            radix_cache.MatchPrefixParams(key=radix_cache.RadixKey([1, 2, 3]))
        )
        lock = tree.inc_lock_ref(match.last_device_node)
        tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2, 3),
            kv_slots=match.device_indices,
            radix_node=match.last_device_node,
            ownership_lock_params=lock.to_dec_params(),
            pinned=True,
        )

        blocked = tree.evict(radix_cache.EvictParams(num_tokens=3))
        self.assertEqual(blocked.num_tokens_evicted, 0)
        self.assertEqual(allocator.freed, [])

        tree.unpin_legolink_blocks(["cache-a"])
        evicted = tree.evict(radix_cache.EvictParams(num_tokens=3))
        self.assertEqual(evicted.num_tokens_evicted, 3)
        self.assertEqual(allocator.freed, [(11, 12, 13)])

    def test_capacity_evictable_size_includes_releasable_legolink_ownership(self):
        allocator = MockAllocator()
        tree = RadixCache.create_simulated(mock_allocator=allocator)
        tree.insert(
            radix_cache.InsertParams(
                key=radix_cache.RadixKey([1, 2, 3]),
                value=torch.tensor([11, 12, 13]),
            )
        )
        match = tree.match_prefix(
            radix_cache.MatchPrefixParams(key=radix_cache.RadixKey([1, 2, 3]))
        )
        tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2, 3),
            kv_slots=match.device_indices,
            radix_node=match.last_device_node,
            pinned=False,
        )

        self.assertEqual(tree.evictable_size_, 0)
        self.assertEqual(tree.evictable_size(), 0)
        self.assertEqual(tree.capacity_evictable_size(), 3)

        acquired = tree.acquire_legolink_blocks(["cache-a"], pin=True)
        self.assertEqual(tree.capacity_evictable_size(), 0)
        tree.release_legolink_blocks(acquired, unpin=True)
        self.assertEqual(tree.capacity_evictable_size(), 3)

        evicted = tree.evict(radix_cache.EvictParams(num_tokens=3))
        self.assertEqual(evicted.num_tokens_evicted, 3)
        self.assertFalse(tree.has_legolink_block("cache-a"))

        repeated = tree.evict(radix_cache.EvictParams(num_tokens=3))
        self.assertEqual(repeated.num_tokens_evicted, 0)
        self.assertEqual(allocator.freed, [(11, 12, 13)])

    def test_register_and_lookup_legolink_blocks_accounts_reuse(self):
        tree = RadixCache.create_simulated()
        snapshot = object()
        slots = torch.tensor([7, 8, 9, 10], dtype=torch.long)

        tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2, 3, 4),
            token_range=(0, 4),
            kv_slots=slots,
            snapshot=snapshot,
            recomputed_tokens=0,
            pinned=True,
        )

        result = tree.lookup_legolink_blocks(
            ["cache-a"],
            semantic_block_ids=["block-a"],
            link_plan={
                "block_ranges": [
                    {
                        "block_id": "block-a",
                        "start_token": 12,
                        "end_token": 16,
                        "token_count": 4,
                    }
                ],
                "recompute_tokens_per_block": [1],
            },
            pin=False,
        )

        self.assertEqual(result.missing_blocks, [])
        self.assertIs(result.matched_blocks["cache-a"].snapshot, snapshot)
        self.assertEqual(result.global_offsets["cache-a"], 12)
        self.assertEqual(result.candidate_reused_tokens, 4)
        self.assertGreaterEqual(
            result.timing_breakdown_ns["radixcache_lookup_total"], 0
        )
        self.assertGreaterEqual(
            result.timing_breakdown_ns["compatibility_validation"], 0
        )
        self.assertGreaterEqual(
            result.timing_breakdown_ns[
                "missing_stale_incompatible_classification"
            ],
            0,
        )
        self.assertFalse(hasattr(result, "recomputed_tokens"))
        self.assertEqual(tree.legolink_stats["hits"], 1)
        acquired = tree.acquire_legolink_blocks(result.matched_handles, pin=True)
        tree.record_legolink_execution(
            acquired,
            radix_cache.LegoLinkExecutionResult(
                actual_reused_tokens=3,
                actual_recomputed_tokens=1,
                copied_tokens=4,
                remapped_tokens=0,
                writeback_tokens=4,
                linked_blocks=["cache-a"],
                per_block_accounting={
                    "cache-a": {
                        "actual_reused_tokens": 3,
                        "actual_recomputed_tokens": 1,
                    }
                },
            ),
        )
        self.assertEqual(result.matched_blocks["cache-a"].pin_count, 2)
        self.assertEqual(result.matched_blocks["cache-a"].refcount, 1)
        self.assertEqual(tree.get_legolink_block_tokens("cache-a"), (1, 2, 3, 4))
        torch.testing.assert_close(result.kv_slot_ranges["cache-a"], slots)

        tree.release_legolink_blocks(acquired, unpin=True)
        self.assertEqual(result.matched_blocks["cache-a"].pin_count, 1)
        self.assertEqual(result.matched_blocks["cache-a"].refcount, 0)
        self.assertEqual(tree.legolink_stats["reused_tokens"], 3)
        self.assertEqual(tree.legolink_stats["recomputed_tokens"], 1)
        self.assertEqual(result.matched_blocks["cache-a"].reused_tokens, 3)
        self.assertEqual(result.matched_blocks["cache-a"].recomputed_tokens, 1)

    def test_lookup_reports_missing_and_evicted_blocks(self):
        tree = RadixCache.create_simulated()
        node = TreeNode()
        node.key = tree.root_node.key
        node.value = torch.tensor([1, 2], dtype=torch.long)
        node.parent = tree.root_node

        tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2),
            token_range=(0, 2),
            kv_slots=torch.tensor([1, 2], dtype=torch.long),
            radix_node=node,
        )
        tree._on_legolink_node_evicted(node)

        result = tree.lookup_legolink_blocks(
            ["cache-a", "cache-b"],
            semantic_block_ids=["block-a", "block-b"],
            link_plan={
                "block_ranges": [
                    {"block_id": "block-a", "start_token": 0, "end_token": 2},
                    {"block_id": "block-b", "start_token": 2, "end_token": 5},
                ]
            },
        )

        self.assertEqual(result.missing_blocks, ["cache-a", "cache-b"])
        self.assertEqual(result.missing_tokens, 5)
        self.assertEqual(tree.legolink_stats["misses"], 2)
        self.assertEqual(tree.legolink_stats["evicted_tokens"], 2)

        result = tree.lookup_legolink_blocks(
            ["cache-a"],
            semantic_block_ids=["block-a"],
            link_plan={
                "block_ranges": [
                    {"block_id": "block-a", "start_token": 0, "end_token": 2}
                ]
            },
        )
        self.assertEqual(result.missing_blocks, ["cache-a"])
        self.assertEqual(tree.legolink_stats["evicted_tokens"], 2)

    def test_remove_legolink_block_respects_pin_refcount_and_counts_eviction_once(self):
        tree = RadixCache.create_simulated()
        tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2, 3, 4),
            token_range=(0, 4),
            kv_slots=torch.tensor([1, 2, 3, 4], dtype=torch.long),
            pinned=True,
        )

        self.assertIsNone(tree.remove_legolink_block("cache-a"))
        tree.unpin_legolink_blocks(["cache-a"])

        result = tree.lookup_legolink_blocks(["cache-a"])
        acquired = tree.acquire_legolink_blocks(result.matched_handles, pin=True)
        self.assertEqual(result.matched_blocks["cache-a"].refcount, 1)
        self.assertIsNone(tree.remove_legolink_block("cache-a"))

        tree.release_legolink_blocks(acquired, unpin=True)
        removed = tree.remove_legolink_block("cache-a")
        self.assertIsNotNone(removed)
        self.assertEqual(tree.legolink_stats["evicted_tokens"], 4)

        self.assertIsNone(tree.remove_legolink_block("cache-a"))
        self.assertEqual(tree.legolink_stats["evicted_tokens"], 4)

    def test_registration_atomically_acquires_and_releases_prefix_ownership(self):
        tree = RadixCache.create_simulated()
        tree.insert(
            radix_cache.InsertParams(
                key=radix_cache.RadixKey([1, 2, 3]),
                value=torch.tensor([11, 12, 13]),
            )
        )
        match = tree.match_prefix(
            radix_cache.MatchPrefixParams(key=radix_cache.RadixKey([1, 2, 3]))
        )
        node = match.last_device_node
        initial_lock_ref = node.lock_ref

        tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2, 3),
            kv_slots=match.device_indices,
            radix_node=node,
        )

        self.assertEqual(node.lock_ref, initial_lock_ref + 1)
        tree.unregister_legolink_block("cache-a")
        self.assertEqual(node.lock_ref, initial_lock_ref)

    def test_lookup_rejects_allocator_generation_and_compatibility_mismatch(self):
        allocator = MockAllocator()
        tree = RadixCache.create_simulated(mock_allocator=allocator)
        tree.register_legolink_block(
            cache_key="cache-a",
            semantic_block_id="block-a",
            token_ids=(1, 2),
            kv_slots=torch.tensor([7, 8]),
            compatibility={"attention_backend_fingerprint": "flashinfer-v1"},
        )

        incompatible = tree.lookup_legolink_blocks(
            ["cache-a"],
            semantic_block_ids=["block-a"],
            compatibility={"attention_backend_fingerprint": "triton-v2"},
        )
        self.assertEqual(incompatible.incompatible_blocks, ["cache-a"])

        allocator.generation = 1
        stale = tree.lookup_legolink_blocks(["cache-a"])
        self.assertEqual(stale.stale_blocks, ["cache-a"])

    def test_lookup_does_not_mutate_execution_accounting(self):
        tree = RadixCache.create_simulated()
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2, 3),
            kv_slots=torch.tensor([7, 8, 9]),
        )

        result = tree.lookup_legolink_blocks(
            ["cache-a"],
            link_plan={"recompute_tokens_per_block": [2]},
        )

        self.assertEqual(result.candidate_reused_tokens, 3)
        self.assertEqual(handle.reuse_count, 0)
        self.assertEqual(handle.reused_tokens, 0)
        self.assertEqual(handle.recomputed_tokens, 0)

    def test_release_without_acquire_fails_loudly(self):
        tree = RadixCache.create_simulated()
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1,),
            kv_slots=torch.tensor([7]),
        )

        with self.assertRaisesRegex(RuntimeError, "released without acquire"):
            tree.release_legolink_blocks([handle])
        self.assertEqual(handle.active_link_refcount, 0)

    def test_link_plan_keeps_request_positions_out_of_block_handle(self):
        tree = RadixCache.create_simulated()
        handle = tree.register_legolink_block(
            cache_key="cache-a",
            token_ids=(1, 2),
            kv_slots=torch.tensor([7, 8]),
        )
        plan = radix_cache.LegoLinkLinkPlan.from_legacy(
            ["cache-a"],
            link_plan={
                "request_id": "request-1",
                "block_ranges": [
                    {"block_id": "cache-a", "start_token": 30, "end_token": 32}
                ],
                "mutable_start_token": 32,
                "total_token_count": 35,
                "imp_indices": [30, 31, 32, 33, 34],
            },
        )

        self.assertEqual(plan.request_id, "request-1")
        self.assertEqual(plan.destination_offsets, (30,))
        self.assertEqual(plan.mutable_suffix_range, (32, 35))
        self.assertFalse(hasattr(handle, "global_offset"))


if __name__ == "__main__":
    unittest.main()
