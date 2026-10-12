from __future__ import annotations

import pytest
import torch

from .reference.attention_reference import gqa_attention, logical_causal_mask
from .support import fixed_generator, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_sparse_metadata_reconstructs_reference_gqa_attention():
    module = load_srt_module("flashinfer_sparse_repair", "flashinfer_sparse_repair.py")
    generator = fixed_generator()
    query = torch.randn(2, 4, 8, generator=generator)
    logical_key = torch.randn(4, 2, 8, generator=generator)
    logical_value = torch.randn(4, 2, 8, generator=generator)
    physical_slots = torch.tensor([11, 2, 19, 7], dtype=torch.int32)
    key_pool = torch.zeros(24, 2, 8)
    value_pool = torch.zeros(24, 2, 8)
    key_pool[physical_slots.long()] = logical_key
    value_pool[physical_slots.long()] = logical_value
    _, indices, mask_data = module.build_explicit_causal_masked_pattern(
        physical_slots, repair_start=2, repair_end=4, pool_size=24
    )

    expected = gqa_attention(
        query,
        logical_key,
        logical_value,
        logical_causal_mask(torch.tensor([2, 3]), torch.arange(4)),
    )
    actual = gqa_attention(
        query,
        key_pool[indices.long()],
        value_pool[indices.long()],
        mask_data[:, :, 0].T,
    )

    torch.testing.assert_close(actual, expected)


def test_gqa_head_mapping_is_query_head_divided_by_group_size():
    query = torch.zeros(1, 4, 2)
    key = torch.zeros(1, 2, 2)
    value = torch.tensor([[[10.0, 11.0], [20.0, 21.0]]])
    output = gqa_attention(query, key, value, torch.ones(1, 1, dtype=torch.bool))

    assert output[0, :, 0].tolist() == [10.0, 10.0, 20.0, 20.0]


@pytest.mark.cuda
@pytest.mark.flashinfer
@pytest.mark.integration
def test_flashinfer_sparse_output_matches_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        import flashinfer
    except ImportError:
        pytest.skip("FlashInfer is not installed")
    sparse = getattr(flashinfer, "sparse", None)
    if sparse is None or not hasattr(sparse, "BlockSparseAttentionWrapper"):
        pytest.skip("FlashInfer block-sparse API is unavailable")

    torch.manual_seed(20260824)
    query = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = sparse.BlockSparseAttentionWrapper(workspace, backend="fa2")
    indptr = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    indices = torch.arange(4, device="cuda", dtype=torch.int32)
    wrapper.plan(
        indptr,
        indices,
        M=2,
        N=4,
        R=2,
        C=1,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )

    actual = wrapper.run(query, key, value).cpu()
    expected = gqa_attention(
        query.cpu(),
        key.cpu(),
        value.cpu(),
        logical_causal_mask(torch.tensor([2, 3]), torch.arange(4)),
    )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    assert torch.isfinite(actual).all()
