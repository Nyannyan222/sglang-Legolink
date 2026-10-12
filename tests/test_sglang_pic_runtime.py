import importlib.util
import json
import os
import sys
import unittest
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from tests.pic.support import SGLANG_PYTHON  # noqa: E402
PIC_RUNTIME_PATH = SGLANG_PYTHON / "sglang" / "srt" / "pic_runtime.py"
SPARSE_REPAIR_PATH = (
    SGLANG_PYTHON / "sglang" / "srt" / "flashinfer_sparse_repair.py"
)
SPARSE_COMPUTE_PATH = (
    SGLANG_PYTHON / "sglang" / "srt" / "sparse_repair_compute.py"
)
TEST_TMP = PROJECT_ROOT / ".tmp" / "test-temp"
TEST_TMP.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("sglang_pic_runtime_test_module", PIC_RUNTIME_PATH)
pic_runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pic_runtime
assert spec.loader is not None
spec.loader.exec_module(pic_runtime)

compute_spec = importlib.util.spec_from_file_location(
    "sglang.srt.sparse_repair_compute", SPARSE_COMPUTE_PATH
)
sparse_compute = importlib.util.module_from_spec(compute_spec)
sys.modules[compute_spec.name] = sparse_compute
assert compute_spec.loader is not None
compute_spec.loader.exec_module(sparse_compute)

sparse_spec = importlib.util.spec_from_file_location(
    "sglang.srt.flashinfer_sparse_repair", SPARSE_REPAIR_PATH
)
sparse_repair = importlib.util.module_from_spec(sparse_spec)
sys.modules[sparse_spec.name] = sparse_repair
assert sparse_spec.loader is not None
sparse_spec.loader.exec_module(sparse_repair)


def sampling_params(
    *,
    cache_key=None,
    link_plan=None,
    storage_mode=None,
    compile_metadata=None,
):
    custom_params = {}
    if cache_key is not None:
        custom_params["sub_context_metadata"] = {
            "cache_key": cache_key,
        }
        custom_params["sub_context_metadata"].update(
            compile_metadata or {}
        )
        if storage_mode is not None:
            custom_params["sub_context_metadata"][
                "pic_storage_mode"
            ] = storage_mode
    if link_plan is not None:
        custom_params["pic_link"] = link_plan
    return SimpleNamespace(custom_params=custom_params)


class FakeRotary:
    def __init__(self, max_position=32, rotary_dim=4):
        self.rotary_dim = rotary_dim
        self.is_neox_style = True
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def _ensure_cos_sin_cache_length(self, needed_max_pos):
        if needed_max_pos >= len(self.cos_sin_cache):
            raise AssertionError("fixture cache is too short")


class FakeKVPool:
    def __init__(self, size=16):
        self.dtype = torch.float32
        self.store_dtype = torch.float32
        self.key = torch.zeros(size, 1, 4)
        self.value = torch.zeros(size, 1, 4)

    def get_key_buffer(self, layer_id):
        assert layer_id == 0
        return self.key

    def get_value_buffer(self, layer_id):
        assert layer_id == 0
        return self.value


def apply_rope(x, rotary, positions):
    cos, sin = rotary.cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class FakeForwardMode(IntEnum):
    EXTEND = 1
    DECODE = 2

    def is_extend(self):
        return self == FakeForwardMode.EXTEND


@dataclass
class FakeForwardBatch:
    forward_mode: FakeForwardMode
    input_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens_sum: int
    req_pool_indices: torch.Tensor | None = None
    req_to_token_pool: object | None = None
    global_forward_mode: FakeForwardMode | None = None
    batch_size: int = 1
    seq_lens_cpu: torch.Tensor | None = None
    orig_seq_lens: torch.Tensor | None = None
    extend_num_tokens: int | None = None
    extend_seq_lens: torch.Tensor | None = None
    extend_prefix_lens: torch.Tensor | None = None
    extend_start_loc: torch.Tensor | None = None
    extend_prefix_lens_cpu: list[int] | None = None
    extend_seq_lens_cpu: list[int] | None = None
    extend_logprob_start_lens_cpu: list[int] | None = None
    return_logprob: bool = False
    input_embeds: torch.Tensor | None = None
    replace_embeds: torch.Tensor | None = None
    replace_positions: torch.Tensor | None = None
    out_cache_loc_swa: torch.Tensor | None = None
    spec_info: object | None = None
    global_num_tokens_cpu: list[int] | None = None
    global_num_tokens_gpu: torch.Tensor | None = None
    global_num_tokens_for_logprob_cpu: list[int] | None = None
    global_num_tokens_for_logprob_gpu: torch.Tensor | None = None
    num_token_non_padded: torch.Tensor | None = None
    num_token_non_padded_cpu: int | None = None
    is_extend_in_batch: bool = False
    all_extend_in_batch: bool = False
    pic_request_metadata: list[dict] | None = None
    full_input_ids: list[list[int]] | None = None


class SGLangPICRuntimeTest(unittest.TestCase):
    def test_triton_fused_kv_link_mode_is_explicit(self):
        cases = {
            "0": "disabled",
            "false": "disabled",
            "1": "required",
            "true": "required",
            "auto": "auto",
        }
        for raw_mode, expected in cases.items():
            with self.subTest(raw_mode=raw_mode):
                with mock.patch.dict(
                    pic_runtime.os.environ,
                    {"SGLANG_PIC_TRITON_FUSED_KV_LINK": raw_mode},
                ):
                    self.assertEqual(
                        pic_runtime.get_triton_fused_kv_link_mode(),
                        expected,
                    )

        with mock.patch.dict(
            pic_runtime.os.environ,
            {"SGLANG_PIC_TRITON_FUSED_KV_LINK": "sometimes"},
        ):
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError, "must be 0, 1, or auto"
            ):
                pic_runtime.get_triton_fused_kv_link_mode()

    def test_tp_rank_manifest_validation_accepts_rank_local_slots(self):
        logical = {
            "cache_keys": ["a", "b"],
            "block_token_lengths": [3, 4],
            "destination_global_offsets": [0, 3],
            "imp_indices": [0, 3, 6],
            "local_lookup_success": True,
        }
        other_rank = dict(logical, local_slots=[101, 102])

        pic_runtime._validate_tp_rank_manifests([logical, other_rank])

    def test_tp_rank_manifest_validation_rejects_logical_mismatch_and_miss(self):
        baseline = {
            "cache_keys": ["a"],
            "block_token_lengths": [3],
            "destination_global_offsets": [0],
            "imp_indices": [0, 2],
            "local_lookup_success": True,
        }
        with self.assertRaisesRegex(pic_runtime.PICRuntimeError, "cache_keys"):
            pic_runtime._validate_tp_rank_manifests(
                [baseline, dict(baseline, cache_keys=["b"])]
            )
        with self.assertRaisesRegex(pic_runtime.PICRuntimeError, "rank 1"):
            pic_runtime._validate_tp_rank_manifests(
                [baseline, dict(baseline, local_lookup_success=False)]
            )

    def test_compile_storage_mode_is_request_configurable(self):
        self.assertEqual(
            pic_runtime.get_pic_compile_storage_mode(
                sampling_params(cache_key="a", storage_mode="clone")
            ),
            "clone",
        )
        self.assertEqual(
            pic_runtime.get_pic_compile_storage_mode(
                sampling_params(cache_key="a", storage_mode="reference")
            ),
            "reference",
        )
        with self.assertRaises(pic_runtime.PICRuntimeError):
            pic_runtime.get_pic_compile_storage_mode(
                sampling_params(cache_key="a", storage_mode="invalid")
            )

    def test_link_interception_only_applies_to_extend_phase(self):
        metadata = [{"link_plan": {"cache_keys": ["a"]}}]
        extend_batch = SimpleNamespace(
            forward_mode=FakeForwardMode.EXTEND,
            pic_request_metadata=metadata,
        )
        decode_batch = SimpleNamespace(
            forward_mode=FakeForwardMode.DECODE,
            pic_request_metadata=metadata,
        )

        self.assertTrue(pic_runtime.PICRuntime.has_link(extend_batch))
        self.assertFalse(pic_runtime.PICRuntime.has_link(decode_batch))

    def test_legacy_mode_can_fall_back_to_native_prefill(self):
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime._legolink_cache = SimpleNamespace(
            has_legolink_block=lambda cache_key: cache_key != "missing"
        )
        batch = SimpleNamespace(
            forward_mode=FakeForwardMode.EXTEND,
            pic_request_metadata=[
                {"link_plan": {"cache_keys": ["present", "missing"]}}
            ],
        )

        with mock.patch.dict(
            pic_runtime.os.environ,
            {
                "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "1",
                "SGLANG_PIC_LEGOLINK_ONLY": "0",
            },
        ):
            self.assertFalse(runtime.can_run_link(batch))

    def test_legolink_only_mode_fails_closed_on_missing_block(self):
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime._legolink_cache = SimpleNamespace(
            get_legolink_block=lambda cache_key: None
        )
        batch = SimpleNamespace(
            forward_mode=FakeForwardMode.EXTEND,
            pic_request_metadata=[
                {"link_plan": {"cache_keys": ["missing"], "block_ranges": []}}
            ],
        )

        with mock.patch.dict(
            pic_runtime.os.environ,
            {
                "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "1",
                "SGLANG_PIC_LEGOLINK_ONLY": "1",
            },
        ):
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError, "cannot continue"
            ):
                runtime.can_run_link(batch)

    def test_legolink_only_mode_fails_closed_without_radixcache_registry(self):
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime._legolink_cache = None
        batch = SimpleNamespace(
            forward_mode=FakeForwardMode.EXTEND,
            pic_request_metadata=[
                {"link_plan": {"cache_keys": ["required-block"]}}
            ],
        )

        with mock.patch.dict(
            pic_runtime.os.environ,
            {
                "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "1",
                "SGLANG_PIC_LEGOLINK_ONLY": "1",
            },
        ):
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError, "registry is unavailable"
            ):
                runtime.can_run_link(batch)

    def test_scheduler_reserves_sources_before_model_runner(self):
        pinned = []
        handle = SimpleNamespace(cache_key="compiled")
        lookup_result = SimpleNamespace(
            matched_handles=[handle],
            missing_blocks=[],
            stale_blocks=[],
            incompatible_blocks=[],
        )
        tree_cache = SimpleNamespace(
            lookup_legolink_blocks=mock.Mock(return_value=lookup_result),
            pin_legolink_blocks=lambda handles: pinned.extend(handles),
        )
        registry = pic_runtime.PICRequestTokenRegistry()
        registry._tree_cache = tree_cache
        link_plan = {
            "request_id": "reserved-request",
            "cache_keys": ["compiled"],
            "block_ranges": [
                {
                    "block_id": "compiled",
                    "start_token": 0,
                    "end_token": 2,
                }
            ],
        }

        with mock.patch.dict(
            pic_runtime.os.environ,
            {
                "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "1",
                "SGLANG_PIC_LEGOLINK_ONLY": "1",
            },
        ):
            reserved = registry.reserve_link_sources(
                sampling_params(link_plan=link_plan)
            )

        self.assertEqual(reserved, ["compiled"])
        self.assertEqual(pinned, [handle])
        self.assertEqual(
            link_plan["_scheduler_reserved_cache_keys"], ["compiled"]
        )
        tree_cache.lookup_legolink_blocks.assert_called_once()

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime._legolink_cache = None
        batch = SimpleNamespace(
            forward_mode=FakeForwardMode.EXTEND,
            pic_request_metadata=[{"link_plan": link_plan}],
        )
        self.assertTrue(runtime.can_run_link(batch))

    def test_scheduler_source_preflight_fails_before_pinning(self):
        tree_cache = SimpleNamespace(
            lookup_legolink_blocks=mock.Mock(
                return_value=SimpleNamespace(
                    matched_handles=[],
                    missing_blocks=["missing"],
                    stale_blocks=[],
                    incompatible_blocks=[],
                )
            ),
            pin_legolink_blocks=mock.Mock(),
        )
        registry = pic_runtime.PICRequestTokenRegistry()
        registry._tree_cache = tree_cache
        params = sampling_params(
            link_plan={
                "request_id": "missing-request",
                "cache_keys": ["missing"],
                "block_ranges": [],
            }
        )

        with mock.patch.dict(
            pic_runtime.os.environ,
            {
                "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "1",
                "SGLANG_PIC_LEGOLINK_ONLY": "1",
            },
        ):
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError, "preflight failed"
            ):
                registry.reserve_link_sources(params)

        tree_cache.pin_legolink_blocks.assert_not_called()

    def test_scheduler_capacity_preflight_unpins_before_failing(self):
        handle = SimpleNamespace(cache_key="compiled")
        lookup_result = SimpleNamespace(
            matched_handles=[handle],
            missing_blocks=[],
            stale_blocks=[],
            incompatible_blocks=[],
        )
        unpinned = []
        tree_cache = SimpleNamespace(
            lookup_legolink_blocks=mock.Mock(return_value=lookup_result),
            pin_legolink_blocks=mock.Mock(),
            unpin_legolink_blocks=lambda keys: unpinned.extend(keys),
            token_to_kv_pool_allocator=SimpleNamespace(
                available_size=lambda: 100
            ),
            capacity_evictable_size=lambda: 50,
        )
        registry = pic_runtime.PICRequestTokenRegistry()
        registry._tree_cache = tree_cache
        link_plan = {
            "request_id": "capacity-request",
            "cache_keys": ["compiled"],
            "block_ranges": [
                {
                    "block_id": "compiled",
                    "start_token": 0,
                    "end_token": 200,
                }
            ],
            "total_token_count": 200,
        }

        with mock.patch.dict(
            pic_runtime.os.environ,
            {
                "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "1",
                "SGLANG_PIC_LEGOLINK_ONLY": "1",
            },
        ):
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError,
                "request-private destination KV",
            ):
                registry.reserve_link_sources(
                    sampling_params(link_plan=link_plan)
                )

        tree_cache.pin_legolink_blocks.assert_called_once_with([handle])
        self.assertEqual(unpinned, ["compiled"])
        self.assertNotIn("_scheduler_reserved_cache_keys", link_plan)

    def test_early_run_link_error_releases_scheduler_source_pins(self):
        released = []
        tree_cache = SimpleNamespace(
            unpin_legolink_blocks=lambda cache_keys: released.extend(
                cache_keys
            )
        )

        class FailingRuntime:
            def __init__(self):
                self._legolink_cache = tree_cache

            @pic_runtime._guard_legolink_release
            def run_link(self, forward_batch):
                raise pic_runtime.PICRuntimeError("injected failure")

        batch = SimpleNamespace(
            pic_request_metadata=[
                {
                    "link_plan": {
                        "cache_keys": ["compiled"],
                        "_scheduler_reserved_cache_keys": ["compiled"],
                    }
                }
            ]
        )
        with self.assertRaisesRegex(
            pic_runtime.PICRuntimeError, "injected failure"
        ):
            FailingRuntime().run_link(batch)

        self.assertEqual(released, ["compiled"])

    def test_legolink_only_mode_rejects_plain_generation_request(self):
        registry = pic_runtime.PICRequestTokenRegistry()

        with mock.patch.dict(
            pic_runtime.os.environ,
            {"SGLANG_PIC_LEGOLINK_ONLY": "1"},
        ):
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError, "LegoLink-only"
            ):
                registry.prepare_request([10, 11], sampling_params())

    def test_legolink_only_mode_rejects_legacy_picruntime_owner(self):
        registry = pic_runtime.PICRequestTokenRegistry()

        with mock.patch.dict(
            pic_runtime.os.environ,
            {
                "SGLANG_PIC_LEGOLINK_ONLY": "1",
                "SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS": "0",
            },
        ):
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError, "requires RadixCache"
            ):
                registry.prepare_request(
                    [10, 11], sampling_params(cache_key="compiled-block")
                )

    def test_legolink_only_mode_accepts_compile_and_link_requests(self):
        registry = pic_runtime.PICRequestTokenRegistry()

        with mock.patch.dict(
            pic_runtime.os.environ,
            {"SGLANG_PIC_LEGOLINK_ONLY": "1"},
        ):
            registry.prepare_request([10, 11], sampling_params(cache_key="a"))
            linked = registry.prepare_request(
                [20],
                sampling_params(
                    link_plan={"cache_keys": ["a"], "recompute_initial_tokens": 0}
                ),
            )

        self.assertEqual(linked, [10, 11, 20])

    def test_aborted_request_drops_pic_metadata_before_model_runner(self):
        params = sampling_params(
            cache_key="compiled-block",
            link_plan={"cache_keys": ["compiled-block"]},
        )
        active = SimpleNamespace(
            sampling_params=params,
            to_finish=None,
            finished_reason=None,
        )
        aborted = SimpleNamespace(
            sampling_params=params,
            to_finish=object(),
            finished_reason=None,
        )

        self.assertIsNotNone(
            pic_runtime.get_pic_request_metadata(active)["link_plan"]
        )
        self.assertEqual(
            pic_runtime.get_pic_request_metadata(aborted),
            {
                "compile_cache_key": None,
                "compile_storage_mode": None,
                "link_plan": None,
            },
        )

    def test_overlength_pic_link_is_rejected_without_auto_truncation(self):
        params = sampling_params(link_plan={"cache_keys": ["a"]})

        self.assertIsNone(
            pic_runtime.validate_pic_link_input_length([1, 2], params, 4)
        )
        error = pic_runtime.validate_pic_link_input_length(
            [1, 2, 3, 4], params, 4
        )
        self.assertIn("cannot be auto-truncated", error)
        self.assertIsNone(
            pic_runtime.validate_pic_link_input_length(
                [1, 2, 3, 4], sampling_params(), 4
            )
        )

    def test_contiguous_ranges_preserve_sparse_execution_order(self):
        self.assertEqual(
            pic_runtime._contiguous_ranges([3, 4, 7, 8, 9, 15]),
            [(3, 5), (7, 10), (15, 16)],
        )
        with self.assertRaises(pic_runtime.PICRuntimeError):
            pic_runtime._contiguous_ranges([3, 3])

    def test_scheduler_registry_recomputes_offsets_from_real_token_ids(self):
        registry = pic_runtime.PICRequestTokenRegistry()
        registry.prepare_request([10, 11, 12], sampling_params(cache_key="a"))
        registry.prepare_request([20, 21], sampling_params(cache_key="b"))
        link_plan = {
            "cache_keys": ["a", "b"],
            "block_ids": ["block-a", "block-b"],
            "recompute_initial_tokens": 1,
        }

        linked = registry.prepare_request(
            [30, 31], sampling_params(link_plan=link_plan)
        )

        self.assertEqual(linked, [10, 11, 12, 20, 21, 30, 31])
        self.assertEqual(link_plan["mutable_start_token"], 5)
        self.assertEqual(link_plan["imp_indices"], [3, 5, 6])
        self.assertEqual(link_plan["kvlink"], [3, 5, 6])
        self.assertEqual(link_plan["legolink_k"], 1)
        self.assertEqual(
            link_plan["block_ranges"],
            [
                {
                    "block_id": "block-a",
                    "cache_key": "a",
                    "start_token": 0,
                    "end_token": 3,
                    "token_count": 3,
                },
                {
                    "block_id": "block-b",
                    "cache_key": "b",
                    "start_token": 3,
                    "end_token": 5,
                    "token_count": 2,
                },
            ],
        )

    def test_scheduler_normalized_layout_matches_one_chat_template(self):
        registry = pic_runtime.PICRequestTokenRegistry()
        registry.prepare_request([10, 11, 12], sampling_params(cache_key="a"))
        registry.prepare_request([20, 21], sampling_params(cache_key="b"))
        link_input = [101, 30, 102]
        expected = [101, 10, 11, 12, 20, 21, 30, 102]
        link_plan = {
            "cache_keys": ["a", "b"],
            "block_ids": ["block-a", "block-b"],
            "recompute_initial_tokens": 1,
            "physical_token_layout_version": (
                pic_runtime.PHYSICAL_TOKEN_LAYOUT_VERSION
            ),
            "physical_input_ids": link_input,
            "physical_token_count": len(link_input),
            "physical_token_ids_sha256": (
                pic_runtime.physical_token_ids_sha256(link_input)
            ),
            "physical_prefix_token_count": 1,
            "physical_block_segments": [
                {
                    "cache_key": "a",
                    "physical_token_count": 3,
                    "physical_token_ids_sha256": (
                        pic_runtime.physical_token_ids_sha256([10, 11, 12])
                    ),
                },
                {
                    "cache_key": "b",
                    "physical_token_count": 2,
                    "physical_token_ids_sha256": (
                        pic_runtime.physical_token_ids_sha256([20, 21])
                    ),
                },
            ],
            "expected_complete_token_count": len(expected),
            "expected_complete_token_ids_sha256": (
                pic_runtime.physical_token_ids_sha256(expected)
            ),
        }

        linked = registry.prepare_request(
            link_input,
            sampling_params(link_plan=link_plan),
        )

        self.assertEqual(linked, expected)
        self.assertEqual(link_plan["physical_prefix_token_count"], 1)
        self.assertEqual(link_plan["block_ranges"][0]["start_token"], 1)
        self.assertEqual(link_plan["block_ranges"][1]["start_token"], 4)
        self.assertEqual(link_plan["mutable_start_token"], 6)
        self.assertEqual(link_plan["imp_indices"], [0, 4, 6, 7])
        self.assertEqual(
            link_plan["token_layout_source"],
            "single_chat_template_physical_override",
        )

    def test_scheduler_normalized_layout_fails_closed_on_token_hash_mismatch(
        self,
    ):
        registry = pic_runtime.PICRequestTokenRegistry()
        registry.prepare_request([10, 11], sampling_params(cache_key="a"))
        link_input = [101, 30, 102]
        link_plan = {
            "cache_keys": ["a"],
            "physical_token_layout_version": (
                pic_runtime.PHYSICAL_TOKEN_LAYOUT_VERSION
            ),
            "physical_input_ids": link_input,
            "physical_token_count": len(link_input),
            "physical_token_ids_sha256": (
                pic_runtime.physical_token_ids_sha256(link_input)
            ),
            "physical_prefix_token_count": 1,
            "physical_block_segments": [
                {
                    "cache_key": "a",
                    "physical_token_count": 2,
                    "physical_token_ids_sha256": "wrong",
                }
            ],
            "expected_complete_token_count": 5,
            "expected_complete_token_ids_sha256": "wrong",
        }

        with self.assertRaisesRegex(
            pic_runtime.PICRuntimeError,
            "physical block token hash mismatch",
        ):
            registry.prepare_request(
                link_input,
                sampling_params(link_plan=link_plan),
            )

    def test_compile_physical_token_override_is_validated(self):
        token_ids = [10, 11, 12]
        metadata = {
            "physical_token_layout_version": (
                pic_runtime.PHYSICAL_TOKEN_LAYOUT_VERSION
            ),
            "physical_input_ids": token_ids,
            "physical_token_count": len(token_ids),
            "physical_token_ids_sha256": (
                pic_runtime.physical_token_ids_sha256(token_ids)
            ),
        }
        registry = pic_runtime.PICRequestTokenRegistry()

        self.assertEqual(
            registry.prepare_request(
                token_ids,
                sampling_params(
                    cache_key="a",
                    compile_metadata=metadata,
                ),
            ),
            token_ids,
        )
        with self.assertRaisesRegex(
            pic_runtime.PICRuntimeError,
            "differ from the validated",
        ):
            registry.prepare_request(
                [10, 99, 12],
                sampling_params(
                    cache_key="b",
                    compile_metadata=metadata,
                ),
            )

    def test_scheduler_link_layout_prefers_radixcache_block_tokens(self):
        class FakeTreeCache:
            def get_legolink_block_tokens(self, cache_key):
                return {
                    "a": (10, 11, 12),
                    "b": (20, 21),
                }.get(cache_key)

        registry = pic_runtime.PICRequestTokenRegistry()
        registry._tree_cache = FakeTreeCache()
        link_plan = {
            "cache_keys": ["a", "b"],
            "recompute_initial_tokens": 1,
        }

        linked = registry.prepare_request(
            [30, 31], sampling_params(link_plan=link_plan)
        )

        self.assertEqual(linked, [10, 11, 12, 20, 21, 30, 31])
        self.assertEqual(link_plan["mutable_start_token"], 5)
        self.assertEqual(link_plan["imp_indices"], [3, 5, 6])

    def test_scheduler_can_recompute_initial_tokens_from_first_block(self):
        registry = pic_runtime.PICRequestTokenRegistry()
        registry.prepare_request([10, 11, 12], sampling_params(cache_key="a"))
        registry.prepare_request([20, 21], sampling_params(cache_key="b"))
        link_plan = {
            "cache_keys": ["a", "b"],
            "recompute_initial_tokens": 1,
            "recompute_first_block": True,
        }

        registry.prepare_request([30, 31], sampling_params(link_plan=link_plan))

        self.assertEqual(link_plan["imp_indices"], [0, 3, 5, 6])
        self.assertTrue(link_plan["recompute_first_block"])

    def test_scheduler_supports_per_block_recompute_counts(self):
        registry = pic_runtime.PICRequestTokenRegistry()
        registry.prepare_request([10, 11, 12], sampling_params(cache_key="a"))
        registry.prepare_request([20, 21, 22], sampling_params(cache_key="b"))
        link_plan = {
            "cache_keys": ["a", "b"],
            "recompute_initial_tokens": 1,
            "recompute_tokens_per_block": [2, 1],
        }

        registry.prepare_request([30], sampling_params(link_plan=link_plan))

        self.assertEqual(link_plan["imp_indices"], [0, 1, 3, 6])
        self.assertEqual(link_plan["recompute_tokens_per_block"], [2, 1])

    def test_scheduler_trace_records_cache_keys_offsets_and_kvlink(self):
        trace_path = TEST_TMP / "pic-trace-scheduler.jsonl"
        trace_path.write_text("", encoding="utf-8")
        previous = os.environ.get("SGLANG_PIC_TRACE_PATH")
        os.environ["SGLANG_PIC_TRACE_PATH"] = str(trace_path)
        try:
            registry = pic_runtime.PICRequestTokenRegistry()
            registry.prepare_request([10, 11], sampling_params(cache_key="a"))
            registry.prepare_request([20, 21], sampling_params(cache_key="b"))
            link_plan = {
                "request_id": "trace-request",
                "task_id": "trace-task",
                "cache_keys": ["a", "b"],
                "block_ids": ["block-a", "block-b"],
                "recompute_initial_tokens": 1,
            }
            registry.prepare_request([30], sampling_params(link_plan=link_plan))
        finally:
            if previous is None:
                os.environ.pop("SGLANG_PIC_TRACE_PATH", None)
            else:
                os.environ["SGLANG_PIC_TRACE_PATH"] = previous
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        link_event = next(
            event for event in events if event["event"] == "scheduler_link_prepared"
        )
        self.assertEqual(link_event["cache_keys"], ["a", "b"])
        self.assertEqual(link_event["block_ranges"][1]["start_token"], 2)
        self.assertEqual(link_event["kvlink"]["values"], [2, 4])

    def test_rope_remap_matches_direct_global_rotation(self):
        rotary = FakeRotary()
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime._rotary_by_layer = {0: rotary}
        local_positions = torch.tensor([0, 1, 2], dtype=torch.long)
        global_positions = torch.tensor([5, 6, 7], dtype=torch.long)
        raw_key = torch.randn(3, 2, 4)
        local_key = apply_rope(raw_key, rotary, local_positions)

        remapped = runtime._remap_key_positions(
            0, local_key, local_positions, global_positions
        )

        expected = apply_rope(raw_key, rotary, global_positions)
        torch.testing.assert_close(remapped, expected, rtol=1e-5, atol=1e-5)

    def test_kv_retrieval_remaps_keys_and_copies_values_to_global_slots(self):
        rotary = FakeRotary()
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = SimpleNamespace(start_layer=0, end_layer=1)
        runtime._rotary_by_layer = {0: rotary}
        runtime._blocks = pic_runtime.OrderedDict()

        raw_a = torch.randn(2, 1, 4)
        raw_b = torch.randn(2, 1, 4)
        local_positions = torch.tensor([0, 1], dtype=torch.long)
        values_a = torch.randn(2, 1, 4)
        values_b = torch.randn(2, 1, 4)
        runtime._blocks["a"] = pic_runtime.PICBlockSnapshot(
            cache_key="a",
            token_ids=(10, 11),
            keys={0: apply_rope(raw_a, rotary, local_positions)},
            values={0: values_a},
        )
        runtime._blocks["b"] = pic_runtime.PICBlockSnapshot(
            cache_key="b",
            token_ids=(20, 21),
            keys={0: apply_rope(raw_b, rotary, local_positions)},
            values={0: values_b},
        )
        pool = FakeKVPool()
        destination_slots = torch.tensor([4, 5, 6, 7], dtype=torch.long)
        link_plan = {
            "cache_keys": ["a", "b"],
            "block_ranges": [
                {"start_token": 0, "end_token": 2},
                {"start_token": 2, "end_token": 4},
            ],
        }

        with mock.patch.dict(
            pic_runtime.os.environ,
            {"SGLANG_PIC_LEGOLINK_ONLY": "0"},
        ):
            runtime._copy_linked_kv(pool, destination_slots, link_plan)

        expected_keys = torch.cat(
            (
                apply_rope(raw_a, rotary, torch.tensor([0, 1])),
                apply_rope(raw_b, rotary, torch.tensor([2, 3])),
            )
        )
        torch.testing.assert_close(pool.key[destination_slots], expected_keys)
        torch.testing.assert_close(
            pool.value[destination_slots], torch.cat((values_a, values_b))
        )

    def test_multi_block_kv_link_batches_layer_buffer_operations(self):
        rotary = FakeRotary()
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = SimpleNamespace(start_layer=0, end_layer=1)
        runtime._rotary_by_layer = {0: rotary}
        runtime._blocks = pic_runtime.OrderedDict()
        local_positions = torch.tensor([0, 1], dtype=torch.long)
        for index, cache_key in enumerate(("a", "b", "c")):
            raw_key = torch.randn(2, 1, 4)
            runtime._blocks[cache_key] = pic_runtime.PICBlockSnapshot(
                cache_key=cache_key,
                token_ids=(index * 2, index * 2 + 1),
                keys={0: apply_rope(raw_key, rotary, local_positions)},
                values={0: torch.randn(2, 1, 4)},
            )
        pool = FakeKVPool()
        destination_slots = torch.tensor([4, 5, 6, 7, 8, 9], dtype=torch.long)
        link_plan = {
            "cache_keys": ["a", "b", "c"],
            "block_ranges": [
                {"start_token": 0, "end_token": 2},
                {"start_token": 2, "end_token": 4},
                {"start_token": 4, "end_token": 6},
            ],
        }

        with (
            mock.patch.dict(
                pic_runtime.os.environ,
                {
                    "SGLANG_PIC_LEGOLINK_ONLY": "0",
                    "SGLANG_PIC_BATCH_BLOCK_KV_LINK": "1",
                },
            ),
            mock.patch.object(
                pool, "get_key_buffer", wraps=pool.get_key_buffer
            ) as get_key_buffer,
            mock.patch.object(
                pool, "get_value_buffer", wraps=pool.get_value_buffer
            ) as get_value_buffer,
            mock.patch.object(
                runtime,
                "_remap_key_positions",
                wraps=runtime._remap_key_positions,
            ) as remap,
        ):
            runtime._copy_linked_kv(pool, destination_slots, link_plan)

        self.assertEqual(get_key_buffer.call_count, 1)
        self.assertEqual(get_value_buffer.call_count, 1)
        self.assertEqual(remap.call_count, 1)

    def test_kv_retrieval_uses_radixcache_legolink_lookup_when_attached(self):
        class FakeLookupResult:
            def __init__(self, snapshot):
                self.matched_blocks = {
                    "a": SimpleNamespace(snapshot=snapshot),
                }
                self.kv_slot_ranges = {"a": None}
                self.missing_blocks = []

        class FakeLegoLinkCache:
            def __init__(self, snapshot):
                self.snapshot = snapshot
                self.calls = []

            def lookup_legolink_blocks(self, cache_keys, **kwargs):
                self.calls.append((cache_keys, kwargs))
                return FakeLookupResult(self.snapshot)

        rotary = FakeRotary()
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = SimpleNamespace(start_layer=0, end_layer=1)
        runtime._rotary_by_layer = {0: rotary}
        runtime._blocks = pic_runtime.OrderedDict()

        raw = torch.randn(2, 1, 4)
        values = torch.randn(2, 1, 4)
        snapshot = pic_runtime.PICBlockSnapshot(
            cache_key="a",
            token_ids=(10, 11),
            keys={0: apply_rope(raw, rotary, torch.tensor([0, 1]))},
            values={0: values},
        )
        source_keys_before = snapshot.keys[0].clone()
        source_values_before = snapshot.values[0].clone()
        cache = FakeLegoLinkCache(snapshot)
        runtime.set_legolink_cache(cache)

        pool = FakeKVPool()
        destination_slots = torch.tensor([4, 5], dtype=torch.long)
        runtime._copy_linked_kv(
            pool,
            destination_slots,
            {
                "cache_keys": ["a"],
                "segmentation_policy_version": "segment-v2",
                "legolink_policy_version": "layout-v2",
                "block_ranges": [
                    {"block_id": "semantic-a", "start_token": 0, "end_token": 2}
                ],
            },
        )

        self.assertEqual(cache.calls[0][0], ["a"])
        self.assertEqual(cache.calls[0][1]["semantic_block_ids"], ["semantic-a"])
        compatibility = cache.calls[0][1]["compatibility"]
        self.assertEqual(
            compatibility["segmentation_policy_version"],
            "segment-v2",
        )
        self.assertEqual(
            compatibility["legolink_policy_version"],
            "layout-v2",
        )
        torch.testing.assert_close(pool.value[destination_slots], values)
        torch.testing.assert_close(snapshot.keys[0], source_keys_before)
        torch.testing.assert_close(snapshot.values[0], source_values_before)

    def test_radixcache_lookup_references_are_released_after_link_error(self):
        handle = object()
        snapshot = pic_runtime.PICBlockSnapshot(
            cache_key="a",
            token_ids=(10,),
            keys={0: torch.zeros(1, 1, 4)},
            values={0: torch.zeros(1, 1, 4)},
        )

        class FakeLegoLinkCache:
            def __init__(self):
                self.released = []

            def lookup_legolink_blocks(self, cache_keys, **kwargs):
                return SimpleNamespace(
                    matched_blocks={"a": SimpleNamespace(snapshot=snapshot)},
                    matched_handles=[handle],
                    kv_slot_ranges={"a": None},
                    missing_blocks=[],
                    stale_blocks=[],
                    incompatible_blocks=[],
                    evicted_tokens=0,
                )

            def release_legolink_blocks(self, handles, *, unpin=False):
                self.released.append((list(handles), unpin))

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = SimpleNamespace(start_layer=0, end_layer=1)
        runtime._rotary_by_layer = {0: FakeRotary()}
        runtime._blocks = pic_runtime.OrderedDict()
        cache = FakeLegoLinkCache()
        runtime.set_legolink_cache(cache)

        guarded_copy = pic_runtime._guard_legolink_release(
            pic_runtime.PICRuntime._copy_linked_kv
        )
        with self.assertRaisesRegex(pic_runtime.PICRuntimeError, "token count mismatch"):
            guarded_copy(
                runtime,
                FakeKVPool(),
                torch.tensor([4, 5]),
                {
                    "cache_keys": ["a"],
                    "block_ranges": [
                        {"block_id": "a", "start_token": 0, "end_token": 2}
                    ],
                },
            )

        self.assertEqual(cache.released, [([handle], True)])

    def test_source_handles_release_before_native_decode_handoff(self):
        events = []
        source = {"compiled_kv": [1, 2, 3]}
        destination = {"request_private_kv": []}
        handle = object()

        class FakeCache:
            def release_legolink_blocks(self, handles, *, unpin=False):
                events.append(("release", list(handles), unpin))
                # Simulate immediate source eviction after the active reference
                # is released. Decode must only depend on destination KV.
                source["compiled_kv"].clear()
                events.append(("source_evicted", []))

        runtime = SimpleNamespace(_pending_legolink_release=None)
        cache = FakeCache()

        def link(self):
            destination["request_private_kv"] = list(source["compiled_kv"])
            events.append(("writeback", list(destination["request_private_kv"])))
            self._pending_legolink_release = (cache, [handle], ["key"])
            return destination

        guarded_link = pic_runtime._guard_legolink_release(link)
        decode_input = guarded_link(runtime)
        events.append(("native_decode", list(decode_input["request_private_kv"])))

        self.assertEqual(events[0][0], "writeback")
        self.assertEqual(events[1], ("release", [handle], True))
        self.assertEqual(events[2], ("source_evicted", []))
        self.assertEqual(events[3], ("native_decode", [1, 2, 3]))
        self.assertEqual(source["compiled_kv"], [])

    def test_kv_trace_records_global_offset_rope_and_writeback(self):
        rotary = FakeRotary()
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = SimpleNamespace(start_layer=0, end_layer=1)
        runtime._rotary_by_layer = {0: rotary}
        runtime._blocks = pic_runtime.OrderedDict()
        runtime._active_trace_context = {
            "request_id": "trace-request",
            "task_id": "trace-task",
            "link_trace_id": "trace-link",
        }
        local_positions = torch.tensor([0, 1], dtype=torch.long)
        for cache_key in ("a", "b"):
            raw_key = torch.randn(2, 1, 4)
            runtime._blocks[cache_key] = pic_runtime.PICBlockSnapshot(
                cache_key=cache_key,
                token_ids=(10, 11),
                keys={0: apply_rope(raw_key, rotary, local_positions)},
                values={0: torch.randn(2, 1, 4)},
            )
        pool = FakeKVPool()
        destination_slots = torch.tensor([4, 5, 6, 7], dtype=torch.long)

        trace_path = TEST_TMP / "pic-trace-kv.jsonl"
        trace_path.write_text("", encoding="utf-8")
        previous = os.environ.get("SGLANG_PIC_TRACE_PATH")
        os.environ["SGLANG_PIC_TRACE_PATH"] = str(trace_path)
        try:
            with mock.patch.dict(
                pic_runtime.os.environ,
                {"SGLANG_PIC_LEGOLINK_ONLY": "0"},
            ):
                runtime._copy_linked_kv(
                    pool,
                    destination_slots,
                    {
                        "cache_keys": ["a", "b"],
                        "block_ranges": [
                            {
                                "block_id": "block-a",
                                "start_token": 0,
                                "end_token": 2,
                            },
                            {
                                "block_id": "block-b",
                                "start_token": 2,
                                "end_token": 4,
                            },
                        ],
                    },
                )
        finally:
            if previous is None:
                os.environ.pop("SGLANG_PIC_TRACE_PATH", None)
            else:
                os.environ["SGLANG_PIC_TRACE_PATH"] = previous
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        linked = [event for event in events if event["event"] == "kv_block_linked"]
        remapped = [
            event
            for event in events
            if event["event"] == "rope_global_position_remap"
        ]
        self.assertEqual(len(linked), 2)
        self.assertEqual(len(remapped), 1)
        self.assertEqual(remapped[0]["layers_remapped"], 1)
        self.assertEqual(remapped[0]["execution"], "layerwise_batched_blocks")
        self.assertEqual(linked[1]["cache_key"], "b")
        self.assertEqual(linked[1]["global_offset"], 2)
        self.assertTrue(linked[1]["rope_remapped"])
        self.assertEqual(linked[1]["destination_slots"]["values"], [6, 7])
        self.assertEqual(linked[1]["kv_writeback"], "index_copy")

    def test_reference_registry_reuses_source_slots_without_cloning_kv(self):
        rotary = FakeRotary()
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = SimpleNamespace(start_layer=0, end_layer=1)
        runtime._rotary_by_layer = {0: rotary}
        runtime._blocks = pic_runtime.OrderedDict()
        pool = FakeKVPool(size=20)

        source_slots = torch.tensor([2, 3], dtype=torch.long)
        destination_slots = torch.tensor([10, 11], dtype=torch.long)
        raw_key = torch.randn(2, 1, 4)
        local_positions = torch.tensor([0, 1], dtype=torch.long)
        source_values = torch.randn(2, 1, 4)
        pool.key[source_slots] = apply_rope(raw_key, rotary, local_positions)
        pool.value[source_slots] = source_values
        runtime._blocks["a"] = pic_runtime.PICBlockSnapshot(
            cache_key="a",
            token_ids=(10, 11),
            storage_mode="reference",
            source_slots=source_slots.clone(),
            reference_ready=True,
            kv_bytes=pool.key[source_slots].numel()
            * pool.key.element_size()
            + pool.value[source_slots].numel() * pool.value.element_size(),
        )

        with mock.patch.dict(
            pic_runtime.os.environ,
            {"SGLANG_PIC_LEGOLINK_ONLY": "0"},
        ):
            runtime._copy_linked_kv(
                pool,
                destination_slots,
                {
                    "cache_keys": ["a"],
                    "block_ranges": [{"start_token": 0, "end_token": 2}],
                },
            )

        torch.testing.assert_close(pool.key[destination_slots], pool.key[source_slots])
        torch.testing.assert_close(
            pool.value[destination_slots], source_values
        )
        registry_bytes, represented_bytes = runtime.registry_memory_bytes()
        self.assertEqual(registry_bytes, source_slots.numel() * source_slots.element_size())
        self.assertGreater(represented_bytes, registry_bytes)

    def test_reference_slots_must_be_rebound_before_link(self):
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime._blocks = pic_runtime.OrderedDict(
            {
                "a": pic_runtime.PICBlockSnapshot(
                    cache_key="a",
                    token_ids=(10, 11),
                    storage_mode="reference",
                    source_slots=torch.tensor([7, 8]),
                )
            }
        )

        runtime.rebind_reference_slots("a", torch.tensor([2, 3]))

        snapshot = runtime._blocks["a"]
        self.assertTrue(snapshot.reference_ready)
        self.assertEqual(snapshot.source_slots.tolist(), [2, 3])

    def test_clear_registry_drops_runtime_references(self):
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime._blocks = pic_runtime.OrderedDict(
            {
                "a": pic_runtime.PICBlockSnapshot(
                    cache_key="a",
                    token_ids=(10,),
                    storage_mode="reference",
                    source_slots=torch.tensor([2]),
                    reference_ready=True,
                )
            }
        )

        runtime.clear_registry()

        self.assertEqual(len(runtime._blocks), 0)

    def test_extend_batch_represents_contiguous_sparse_span(self):
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([12]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=12,
        )

        batch = pic_runtime.PICRuntime._make_extend_batch(
            original,
            token_ids=[20, 21, 22],
            start=5,
            end=8,
            destination_slots=torch.tensor([50, 51, 52]),
        )

        self.assertEqual(batch.forward_mode, FakeForwardMode.EXTEND)
        self.assertEqual(batch.input_ids.tolist(), [20, 21, 22])
        self.assertEqual(batch.positions.tolist(), [5, 6, 7])
        self.assertEqual(batch.seq_lens.tolist(), [8])
        self.assertEqual(batch.extend_prefix_lens.tolist(), [5])
        self.assertEqual(batch.extend_seq_lens.tolist(), [3])
        self.assertEqual(batch.extend_start_loc.tolist(), [0])
        self.assertEqual(batch.out_cache_loc.tolist(), [50, 51, 52])
        self.assertEqual(batch.num_token_non_padded_cpu, 3)

    def test_batched_recompute_uses_one_extend_forward_per_sparse_span(self):
        calls = []

        class FakeModelRunner:
            def forward_extend(self, batch):
                calls.append(
                    {
                        "input_ids": batch.input_ids.tolist(),
                        "positions": batch.positions.tolist(),
                        "prefix": batch.extend_prefix_lens.tolist(),
                    }
                )
                return f"output-{len(calls)}", False

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = FakeModelRunner()
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([12]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=12,
        )

        output, forward_count = runtime._run_batched_recompute(
            original,
            list(range(20)),
            torch.arange(100, 120),
            [3, 4, 7, 8, 9, 15],
        )

        self.assertEqual(output, "output-3")
        self.assertEqual(forward_count, 3)
        self.assertEqual(
            calls,
            [
                {"input_ids": [3, 4], "positions": [3, 4], "prefix": [3]},
                {
                    "input_ids": [7, 8, 9],
                    "positions": [7, 8, 9],
                    "prefix": [7],
                },
                {"input_ids": [15], "positions": [15], "prefix": [15]},
            ],
        )

    def test_batched_recompute_attaches_unified_view_without_changing_execution(self):
        captured = []

        class FakeModelRunner:
            def forward_extend(self, batch):
                captured.append(batch.pic_sparse_repair_metadata)
                return "output", False

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = FakeModelRunner()
        runtime._active_trace_context = {"request_id": "request-a"}
        runtime._active_sparse_repair_block_count = 3
        runtime._active_sparse_repair_block_size = 128
        req_to_token = torch.arange(200, 240, dtype=torch.long).reshape(1, -1)
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([12]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=12,
        )
        original.req_pool_indices = torch.tensor([0])
        original.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND": "native",
                "SGLANG_PIC_SPARSE_REPAIR_METADATA_VIEW_ENABLED": "1",
            },
            clear=False,
        ):
            output, forward_count = runtime._run_batched_recompute(
                original,
                list(range(20)),
                torch.arange(200, 220),
                [3, 4, 7, 8],
            )

        self.assertEqual(output, "output")
        self.assertEqual(forward_count, 2)
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["unified_metadata_view"]["repair_start"], 3)
        self.assertEqual(captured[0]["unified_metadata_view"]["ordered_slot_count"], 5)
        self.assertEqual(captured[0]["batch_plan_group_size"], 2)
        self.assertEqual(
            captured[0]["batch_plan_execution_mode"],
            "analysis_only_sequential_forward_extend",
        )

    def test_native_row_batching_groups_middle_rows_and_keeps_final_row_single(self):
        calls = []

        class FakeModelRunner:
            def forward_extend(self, batch):
                calls.append(
                    {
                        "batch_size": batch.batch_size,
                        "input_ids": batch.input_ids.tolist(),
                        "positions": batch.positions.tolist(),
                        "prefixes": batch.extend_prefix_lens.tolist(),
                        "req_pool_indices": batch.req_pool_indices.tolist(),
                        "metadata": batch.pic_sparse_repair_metadata,
                    }
                )
                return f"output-{len(calls)}", False

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = FakeModelRunner()
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([20]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=20,
            req_pool_indices=torch.tensor([4]),
        )

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND": "native",
                "SGLANG_PIC_SPARSE_REPAIR_ROW_BATCHING": "1",
            },
            clear=False,
        ):
            output, forward_count = runtime._run_batched_recompute(
                original,
                list(range(20)),
                torch.arange(100, 120),
                [3, 4, 7, 8, 15],
            )

        self.assertEqual(output, "output-2")
        self.assertEqual(forward_count, 2)
        self.assertEqual(calls[0]["batch_size"], 2)
        self.assertEqual(calls[0]["input_ids"], [3, 4, 7, 8])
        self.assertEqual(calls[0]["positions"], [3, 4, 7, 8])
        self.assertEqual(calls[0]["prefixes"], [3, 7])
        self.assertEqual(calls[0]["req_pool_indices"], [4, 4])
        self.assertEqual(calls[0]["metadata"]["repair_tokens"], 4)
        self.assertEqual(calls[0]["metadata"]["repair_row_count"], 2)
        self.assertEqual(calls[1]["batch_size"], 1)
        self.assertEqual(calls[1]["input_ids"], [15])

    def test_row_batching_rejects_flashinfer_without_experiment_flag(self):
        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=8,
        )

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND": "flashinfer",
                "SGLANG_PIC_SPARSE_REPAIR_ROW_BATCHING": "1",
            },
            clear=False,
        ), self.assertRaisesRegex(pic_runtime.PICRuntimeError, "requires native"):
            runtime._run_batched_recompute(
                original,
                list(range(8)),
                torch.arange(100, 108),
                [3, 4, 7],
            )

    def test_row_batching_allows_forced_flashinfer_with_experiment_flag(self):
        calls = []

        class FakeModelRunner:
            def forward_extend(self, batch):
                calls.append(batch.pic_sparse_repair_metadata)
                return "output", False

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = FakeModelRunner()
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=8,
            req_pool_indices=torch.tensor([4]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(40).view(5, 8)
            ),
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND": "flashinfer",
                    "SGLANG_PIC_SPARSE_REPAIR_ROW_BATCHING": "1",
                    "SGLANG_PIC_ALLOW_FLASHINFER_ROW_BATCHING_EXPERIMENT": "1",
                },
                clear=False,
            ),
            mock.patch.object(
                sparse_repair, "_flashinfer_available", return_value=(True, None)
            ),
        ):
            output, forward_count = runtime._run_batched_recompute(
                original,
                list(range(8)),
                torch.arange(100, 108),
                [1, 2, 4, 5, 7],
            )

        self.assertEqual(output, "output")
        self.assertEqual(forward_count, 2)
        self.assertTrue(calls[0]["flashinfer_row_batching_experiment"])
        self.assertEqual(
            calls[0]["batch_plan_execution_mode"],
            "experimental_flashinfer_per_row_grouped_forward_extend",
        )

    def test_flashinfer_paged_full_row_batching_returns_final_request_logits(self):
        calls = []

        class FakeModelRunner:
            def forward_extend(self, batch):
                calls.append(batch)
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        next_token_logits=torch.tensor([[1.0], [2.0], [3.0]])
                    )
                ), False

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = FakeModelRunner()
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([20]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=20,
            req_pool_indices=torch.tensor([0]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(20).view(1, 20)
            ),
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND": "flashinfer",
                    "SGLANG_PIC_SPARSE_REPAIR_ROW_BATCHING": "1",
                    "SGLANG_PIC_SPARSE_REPAIR_ROWWISE_FFN": "1",
                    "SGLANG_PIC_SPARSE_REPAIR_ROWWISE_OUTPUT_PROJECTION": "1",
                    "SGLANG_PIC_ALLOW_FLASHINFER_ROW_BATCHING_EXPERIMENT": "1",
                    "SGLANG_PIC_FLASHINFER_GROUPED_ATTENTION_MODE": "paged_batched",
                    "SGLANG_PIC_FLASHINFER_INCLUDE_FINAL_ROW_IN_BATCH": "1",
                },
                clear=False,
            ),
            mock.patch.object(
                sparse_repair, "_flashinfer_available", return_value=(True, None)
            ),
        ):
            output, forward_count = runtime._run_batched_recompute(
                original,
                list(range(20)),
                torch.arange(100, 120),
                [3, 4, 7, 8, 15],
            )

        self.assertEqual(forward_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].batch_size, 3)
        self.assertEqual(
            calls[0].pic_sparse_repair_metadata["repair_ranges"],
            [
                {"start": 3, "end": 5},
                {"start": 7, "end": 9},
                {"start": 15, "end": 16},
            ],
        )
        self.assertEqual(
            output.logits_output.next_token_logits.tolist(), [[3.0]]
        )

    def test_request_row_batching_override_is_fail_closed_and_explicit(self):
        batch = SimpleNamespace(
            pic_request_metadata=[
                {
                    "link_plan": {
                        "diagnostic_sparse_repair_row_batching": True,
                        "diagnostic_sparse_repair_rowwise_ffn": True,
                        "diagnostic_sparse_repair_rowwise_output_projection": True,
                    }
                }
            ]
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                "SGLANG_PIC_ALLOW_REQUEST_ROW_BATCHING_OVERRIDE", None
            )
            with self.assertRaisesRegex(
                pic_runtime.PICRuntimeError, "override is disabled"
            ):
                pic_runtime._sparse_repair_row_batching_options(batch)

        with mock.patch.dict(
            os.environ,
            {"SGLANG_PIC_ALLOW_REQUEST_ROW_BATCHING_OVERRIDE": "1"},
            clear=False,
        ):
            self.assertEqual(
                pic_runtime._sparse_repair_row_batching_options(batch),
                (True, True, True, "request_diagnostic_override"),
            )

    def test_full_causal_baseline_recomputes_the_entire_sequence_once(self):
        calls = []

        class FakeModelRunner:
            def forward_extend(self, batch):
                calls.append(
                    {
                        "input_ids": batch.input_ids.tolist(),
                        "positions": batch.positions.tolist(),
                        "prefix": batch.extend_prefix_lens.tolist(),
                    }
                )
                return "full-output", False

        runtime = pic_runtime.PICRuntime.__new__(pic_runtime.PICRuntime)
        runtime.model_runner = FakeModelRunner()
        original = FakeForwardBatch(
            forward_mode=FakeForwardMode.EXTEND,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([4]),
            out_cache_loc=torch.tensor([99]),
            seq_lens_sum=4,
        )

        output, forward_count = runtime._run_full_causal(
            original,
            [10, 11, 12, 13],
            torch.tensor([40, 41, 42, 43]),
        )

        self.assertEqual(output, "full-output")
        self.assertEqual(forward_count, 1)
        self.assertEqual(
            calls,
            [
                {
                    "input_ids": [10, 11, 12, 13],
                    "positions": [0, 1, 2, 3],
                    "prefix": [0],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
