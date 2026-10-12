"""Optional Triton fusion for LegoLink KV gather, RoPE remap, and scatter.

Boundary tokens are repaired by the model's sparse ``forward_extend`` pass.
The fused kernel can recognize those tokens and avoid copying stale source KV
that the repair pass will immediately overwrite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only development hosts.
    triton = None
    tl = None


@dataclass(frozen=True)
class LegoLinkFusionEligibility:
    supported: bool
    reason: str


@dataclass(frozen=True)
class LegoLinkLaunchConfig:
    block_size: int
    num_warps: int
    num_stages: int
    source: str


def resolve_fused_kv_launch_config(
    *,
    work_items: int,
    device: torch.device | None = None,
    capability: tuple[int, int] | None = None,
    block_size: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> LegoLinkLaunchConfig:
    """Select a guarded per-architecture launch configuration.

    Only the RTX 5070 SM120 production shape has completed a local launch
    sweep. Other architectures retain the pre-tuning configuration until
    their own sweep is available. Environment variables provide an explicit
    experiment override without changing production defaults.
    """

    if capability is None:
        if not torch.cuda.is_available():
            capability = (0, 0)
        else:
            capability = torch.cuda.get_device_capability(device)
    major, _minor = (int(value) for value in capability)
    if major >= 12 and int(work_items) == 384:
        defaults = (128, 2, 2)
        source = "sm120_qwen_kv_tuned"
    else:
        defaults = (256, 4, 2)
        source = "conservative_cross_gpu"

    env_values = (
        os.getenv("SGLANG_PIC_TRITON_BLOCK_SIZE"),
        os.getenv("SGLANG_PIC_TRITON_NUM_WARPS"),
        os.getenv("SGLANG_PIC_TRITON_NUM_STAGES"),
    )
    explicit_values = (block_size, num_warps, num_stages)
    resolved = []
    overridden = False
    for explicit, environment, default in zip(
        explicit_values, env_values, defaults, strict=True
    ):
        if explicit is not None:
            resolved.append(int(explicit))
            overridden = True
        elif environment not in (None, ""):
            resolved.append(int(environment))
            overridden = True
        else:
            resolved.append(default)
    if overridden:
        source = "explicit_or_environment_override"
    resolved_block_size, resolved_num_warps, resolved_num_stages = resolved
    if (
        resolved_block_size <= 0
        or resolved_block_size > 1024
        or resolved_block_size & (resolved_block_size - 1)
    ):
        raise ValueError("block_size must be a power of two between 1 and 1024")
    if resolved_num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if not 1 <= resolved_num_stages <= 5:
        raise ValueError("num_stages must be between 1 and 5")
    return LegoLinkLaunchConfig(*resolved, source)


def triton_legolink_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


def estimate_kv_traffic_bytes(
    *,
    token_count: int,
    key_row_bytes: int,
    value_row_bytes: int,
    remapped_token_count: int,
    repair_token_count: int = 0,
) -> dict[str, int | float]:
    """Return a lower-bound traffic model for Torch and fused pipelines."""

    token_count = max(int(token_count), 0)
    remapped_token_count = max(min(int(remapped_token_count), token_count), 0)
    repair_token_count = max(min(int(repair_token_count), token_count), 0)
    copied_token_count = token_count - repair_token_count
    key_bytes = token_count * max(int(key_row_bytes), 0)
    value_bytes = token_count * max(int(value_row_bytes), 0)
    remapped_key_bytes = remapped_token_count * max(int(key_row_bytes), 0)
    copied_key_bytes = copied_token_count * max(int(key_row_bytes), 0)
    copied_value_bytes = copied_token_count * max(int(value_row_bytes), 0)

    # Torch materializes gathered K/V, materializes remapped K, and then reads
    # those temporaries again for index_copy_. This is a conservative lower
    # bound that excludes position-map and trigonometric-cache traffic.
    torch_read_bytes = 2 * (key_bytes + value_bytes) + remapped_key_bytes
    torch_write_bytes = 2 * (key_bytes + value_bytes) + remapped_key_bytes
    torch_bytes = torch_read_bytes + torch_write_bytes
    # Triton reads source K/V and writes destination K/V directly. Repair
    # positions are intentionally untouched because sparse forward_extend
    # writes their final KV values.
    fused_read_bytes = copied_key_bytes + copied_value_bytes
    fused_write_bytes = copied_key_bytes + copied_value_bytes
    fused_bytes = fused_read_bytes + fused_write_bytes
    reduction = (
        1.0 - (fused_bytes / torch_bytes)
        if torch_bytes
        else 0.0
    )
    return {
        "torch_lower_bound_bytes": torch_bytes,
        "triton_lower_bound_bytes": fused_bytes,
        "torch_estimated_hbm_read_bytes": torch_read_bytes,
        "torch_estimated_hbm_write_bytes": torch_write_bytes,
        "triton_estimated_hbm_read_bytes": fused_read_bytes,
        "triton_estimated_hbm_write_bytes": fused_write_bytes,
        "torch_estimated_hbm_traffic_bytes": torch_bytes,
        "triton_estimated_hbm_traffic_bytes": fused_bytes,
        "estimated_reduction_ratio": reduction,
        "copied_token_count": copied_token_count,
        "repair_token_count": repair_token_count,
    }


def check_fused_kv_link_eligibility(
    *,
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    local_positions: torch.Tensor,
    global_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    repair_mask: torch.Tensor | None = None,
) -> LegoLinkFusionEligibility:
    tensors = (
        key_buffer,
        value_buffer,
        source_slots,
        destination_slots,
        local_positions,
        global_positions,
        cos_sin_cache,
    )
    if triton is None or tl is None:
        return LegoLinkFusionEligibility(False, "triton_not_installed")
    if not all(tensor.is_cuda for tensor in tensors):
        return LegoLinkFusionEligibility(False, "cuda_tensors_required")
    if key_buffer.ndim != 3 or value_buffer.ndim != 3:
        return LegoLinkFusionEligibility(False, "mha_3d_kv_buffers_required")
    if key_buffer.shape[:2] != value_buffer.shape[:2]:
        return LegoLinkFusionEligibility(False, "kv_head_layout_mismatch")
    if key_buffer.dtype != value_buffer.dtype:
        return LegoLinkFusionEligibility(False, "kv_dtype_mismatch")
    if key_buffer.dtype not in (torch.float16, torch.bfloat16):
        return LegoLinkFusionEligibility(False, "fp16_or_bf16_required")
    if not key_buffer.is_contiguous() or not value_buffer.is_contiguous():
        return LegoLinkFusionEligibility(False, "contiguous_kv_buffers_required")
    key_head_dim = int(key_buffer.shape[-1])
    value_head_dim = int(value_buffer.shape[-1])
    if (
        key_head_dim <= 0
        or value_head_dim <= 0
        or key_head_dim > 256
        or value_head_dim > 256
    ):
        return LegoLinkFusionEligibility(False, "head_dimension_out_of_range")
    token_count = int(source_slots.numel())
    if token_count == 0:
        return LegoLinkFusionEligibility(False, "empty_link")
    if any(
        int(tensor.numel()) != token_count
        for tensor in (
            destination_slots,
            local_positions,
            global_positions,
        )
    ):
        return LegoLinkFusionEligibility(False, "token_map_length_mismatch")
    if repair_mask is not None:
        if not repair_mask.is_cuda:
            return LegoLinkFusionEligibility(False, "cuda_repair_mask_required")
        if int(repair_mask.numel()) != token_count:
            return LegoLinkFusionEligibility(False, "repair_mask_length_mismatch")
        if repair_mask.dtype not in (torch.bool, torch.uint8):
            return LegoLinkFusionEligibility(False, "boolean_repair_mask_required")
        if not repair_mask.is_contiguous():
            return LegoLinkFusionEligibility(False, "contiguous_repair_mask_required")
    if any(
        tensor.dtype not in (torch.int32, torch.int64)
        for tensor in (
            source_slots,
            destination_slots,
            local_positions,
            global_positions,
        )
    ):
        return LegoLinkFusionEligibility(False, "integer_token_maps_required")
    rotary_dim = int(rotary_dim)
    if rotary_dim <= 0 or rotary_dim % 2:
        return LegoLinkFusionEligibility(False, "even_rotary_dim_required")
    if rotary_dim > int(key_buffer.shape[-1]):
        return LegoLinkFusionEligibility(False, "rotary_dim_exceeds_head_dim")
    if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) < rotary_dim:
        return LegoLinkFusionEligibility(False, "unsupported_cos_sin_cache")
    if cos_sin_cache.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return LegoLinkFusionEligibility(False, "unsupported_cos_sin_dtype")
    return LegoLinkFusionEligibility(True, "supported")


if triton is not None:

    @triton.jit
    def _mul_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            asm="mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _add_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            asm="add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            asm="sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _round_to_bf16_f32(x):
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b16 rounded;
                cvt.rn.bf16.f32 rounded, $1;
                cvt.f32.bf16 $0, rounded;
            }
            """,
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fused_kv_gather_rope_scatter_kernel(
        key_ptr,
        value_ptr,
        source_slots_ptr,
        destination_slots_ptr,
        local_positions_ptr,
        global_positions_ptr,
        repair_mask_ptr,
        cos_sin_ptr,
        key_stride_slot: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_slot: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_dim: tl.constexpr,
        cache_stride_position: tl.constexpr,
        cache_stride_dim: tl.constexpr,
        prefix_token_count,
        NUM_HEADS: tl.constexpr,
        KEY_HEAD_DIM: tl.constexpr,
        VALUE_HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        IS_NEOX_STYLE: tl.constexpr,
        KEY_IS_BF16: tl.constexpr,
        CACHE_IS_FP32: tl.constexpr,
        HAS_REPAIR_MASK: tl.constexpr,
        WORK_ITEMS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_index = tl.program_id(0)
        work_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        work_mask = work_offsets < WORK_ITEMS

        source_slot = tl.load(source_slots_ptr + token_index)
        destination_slot = tl.load(destination_slots_ptr + token_index)
        local_position = tl.load(local_positions_ptr + token_index)
        global_position = tl.load(global_positions_ptr + token_index)
        if HAS_REPAIR_MASK:
            is_repair_token = (
                tl.load(repair_mask_ptr + token_index).to(tl.int1)
            )
        else:
            is_repair_token = False
        copy_token = ~is_repair_token
        remap = token_index >= prefix_token_count

        rotary_half = ROTARY_DIM // 2
        rotary_work_items = NUM_HEADS * rotary_half
        rotary_mask = (
            work_mask & (work_offsets < rotary_work_items) & copy_token
        )
        rotary_head = work_offsets // rotary_half
        rotary_pair = work_offsets - rotary_head * rotary_half

        if IS_NEOX_STYLE:
            first_dim = rotary_pair
            second_dim = rotary_pair + rotary_half
        else:
            first_dim = rotary_pair * 2
            second_dim = first_dim + 1

        source_key_base = (
            source_slot * key_stride_slot
            + rotary_head * key_stride_head
        )
        destination_key_base = (
            destination_slot * key_stride_slot
            + rotary_head * key_stride_head
        )
        first_value = tl.load(
            key_ptr + source_key_base + first_dim * key_stride_dim,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        second_value = tl.load(
            key_ptr + source_key_base + second_dim * key_stride_dim,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)

        local_cache_base = local_position * cache_stride_position
        global_cache_base = global_position * cache_stride_position
        local_cos = tl.load(
            cos_sin_ptr
            + local_cache_base
            + rotary_pair * cache_stride_dim,
            mask=rotary_mask,
            other=1.0,
        ).to(tl.float32)
        local_sin = tl.load(
            cos_sin_ptr
            + local_cache_base
            + (rotary_pair + rotary_half) * cache_stride_dim,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        global_cos = tl.load(
            cos_sin_ptr
            + global_cache_base
            + rotary_pair * cache_stride_dim,
            mask=rotary_mask,
            other=1.0,
        ).to(tl.float32)
        global_sin = tl.load(
            cos_sin_ptr
            + global_cache_base
            + (rotary_pair + rotary_half) * cache_stride_dim,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        # Torch materializes the two FP32 products before the add/sub in
        # PICRuntime._remap_key_positions. Keep those rounding boundaries and
        # prohibit FMA contraction; at long RoPE positions an FMA can cross a
        # BF16 rounding midpoint and change a linked K element by one ULP.
        delta_cos = _add_rn_f32(
            _mul_rn_f32(global_cos, local_cos),
            _mul_rn_f32(global_sin, local_sin),
        )
        delta_sin = _sub_rn_f32(
            _mul_rn_f32(global_sin, local_cos),
            _mul_rn_f32(global_cos, local_sin),
        )
        if KEY_IS_BF16:
            if CACHE_IS_FP32:
                # PyTorch promotes BF16 K multiplied by an FP32 RoPE delta to
                # FP32 and rounds only when the remapped K is stored.
                rotated_first = _sub_rn_f32(
                    _mul_rn_f32(first_value, delta_cos),
                    _mul_rn_f32(second_value, delta_sin),
                )
                rotated_second = _add_rn_f32(
                    _mul_rn_f32(second_value, delta_cos),
                    _mul_rn_f32(first_value, delta_sin),
                )
            else:
                first_bf16 = _round_to_bf16_f32(first_value)
                second_bf16 = _round_to_bf16_f32(second_value)
                # Eager Torch materializes each BF16 trigonometric product
                # before the BF16 add/sub, so preserve both rounding points.
                delta_cos_bf16 = _round_to_bf16_f32(
                    _add_rn_f32(
                        _round_to_bf16_f32(
                            _mul_rn_f32(global_cos, local_cos)
                        ),
                        _round_to_bf16_f32(
                            _mul_rn_f32(global_sin, local_sin)
                        ),
                    )
                )
                delta_sin_bf16 = _round_to_bf16_f32(
                    _sub_rn_f32(
                        _round_to_bf16_f32(
                            _mul_rn_f32(global_sin, local_cos)
                        ),
                        _round_to_bf16_f32(
                            _mul_rn_f32(global_cos, local_sin)
                        ),
                    )
                )
                first_cos = _round_to_bf16_f32(first_bf16 * delta_cos_bf16)
                second_sin = _round_to_bf16_f32(second_bf16 * delta_sin_bf16)
                second_cos = _round_to_bf16_f32(second_bf16 * delta_cos_bf16)
                first_sin = _round_to_bf16_f32(first_bf16 * delta_sin_bf16)
                rotated_first = _round_to_bf16_f32(first_cos - second_sin)
                rotated_second = _round_to_bf16_f32(second_cos + first_sin)
        else:
            rotated_first = first_value * delta_cos - second_value * delta_sin
            rotated_second = second_value * delta_cos + first_value * delta_sin
        output_first = tl.where(remap, rotated_first, first_value)
        output_second = tl.where(remap, rotated_second, second_value)
        tl.store(
            key_ptr
            + destination_key_base
            + first_dim * key_stride_dim,
            output_first,
            mask=rotary_mask,
        )
        tl.store(
            key_ptr
            + destination_key_base
            + second_dim * key_stride_dim,
            output_second,
            mask=rotary_mask,
        )

        key_pass_dim = KEY_HEAD_DIM - ROTARY_DIM
        if key_pass_dim > 0:
            key_pass_start = rotary_work_items
            key_pass_items = NUM_HEADS * key_pass_dim
            key_pass_index = work_offsets - key_pass_start
            key_pass_mask = (
                work_mask
                & (key_pass_index >= 0)
                & (key_pass_index < key_pass_items)
                & copy_token
            )
            key_pass_head = key_pass_index // key_pass_dim
            key_pass_offset = key_pass_index - key_pass_head * key_pass_dim
            key_pass_value = tl.load(
                key_ptr
                + source_slot * key_stride_slot
                + key_pass_head * key_stride_head
                + (ROTARY_DIM + key_pass_offset) * key_stride_dim,
                mask=key_pass_mask,
                other=0.0,
            )
            tl.store(
                key_ptr
                + destination_slot * key_stride_slot
                + key_pass_head * key_stride_head
                + (ROTARY_DIM + key_pass_offset) * key_stride_dim,
                key_pass_value,
                mask=key_pass_mask,
            )
        else:
            key_pass_start = rotary_work_items
            key_pass_items = 0

        value_start = key_pass_start + key_pass_items
        value_items = NUM_HEADS * VALUE_HEAD_DIM
        value_index = work_offsets - value_start
        value_mask = (
            work_mask
            & (value_index >= 0)
            & (value_index < value_items)
            & copy_token
        )
        value_head = value_index // VALUE_HEAD_DIM
        value_offset = value_index - value_head * VALUE_HEAD_DIM
        value = tl.load(
            value_ptr
            + source_slot * value_stride_slot
            + value_head * value_stride_head
            + value_offset * value_stride_dim,
            mask=value_mask,
            other=0.0,
        )
        tl.store(
            value_ptr
            + destination_slot * value_stride_slot
            + value_head * value_stride_head
            + value_offset * value_stride_dim,
            value,
            mask=value_mask,
        )


def fused_kv_link_inplace(
    *,
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    local_positions: torch.Tensor,
    global_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    prefix_token_count: int,
    rotary_dim: int,
    is_neox_style: bool,
    repair_mask: torch.Tensor | None = None,
    block_size: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> None:
    eligibility = check_fused_kv_link_eligibility(
        key_buffer=key_buffer,
        value_buffer=value_buffer,
        source_slots=source_slots,
        destination_slots=destination_slots,
        local_positions=local_positions,
        global_positions=global_positions,
        cos_sin_cache=cos_sin_cache,
        rotary_dim=rotary_dim,
        repair_mask=repair_mask,
    )
    if not eligibility.supported:
        raise RuntimeError(
            "Triton LegoLink fusion is unavailable: "
            f"{eligibility.reason}"
        )
    if not 0 <= int(prefix_token_count) <= int(source_slots.numel()):
        raise ValueError("prefix_token_count is outside the linked token range")
    num_heads = int(key_buffer.shape[1])
    key_head_dim = int(key_buffer.shape[2])
    value_head_dim = int(value_buffer.shape[2])
    rotary_dim = int(rotary_dim)
    work_items = num_heads * (
        rotary_dim // 2
        + key_head_dim
        - rotary_dim
        + value_head_dim
    )
    launch_config = resolve_fused_kv_launch_config(
        work_items=work_items,
        device=key_buffer.device,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    block_size = launch_config.block_size
    num_warps = launch_config.num_warps
    num_stages = launch_config.num_stages
    if block_size <= 0 or block_size > 1024 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two between 1 and 1024")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if not 1 <= num_stages <= 5:
        raise ValueError("num_stages must be between 1 and 5")
    grid = (
        int(source_slots.numel()),
        triton.cdiv(work_items, block_size),
    )
    _fused_kv_gather_rope_scatter_kernel[grid](
        key_buffer,
        value_buffer,
        source_slots,
        destination_slots,
        local_positions,
        global_positions,
        repair_mask if repair_mask is not None else source_slots,
        cos_sin_cache,
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        value_buffer.stride(0),
        value_buffer.stride(1),
        value_buffer.stride(2),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        int(prefix_token_count),
        NUM_HEADS=num_heads,
        KEY_HEAD_DIM=key_head_dim,
        VALUE_HEAD_DIM=value_head_dim,
        ROTARY_DIM=rotary_dim,
        IS_NEOX_STYLE=bool(is_neox_style),
        KEY_IS_BF16=key_buffer.dtype == torch.bfloat16,
        CACHE_IS_FP32=cos_sin_cache.dtype == torch.float32,
        HAS_REPAIR_MASK=repair_mask is not None,
        WORK_ITEMS=work_items,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def fusion_metadata(**kwargs: Any) -> dict[str, Any]:
    """Return bounded metadata suitable for runtime trace records."""

    boundary_repair_aware = bool(kwargs.get("boundary_repair_aware", False))
    return {
        "implementation": (
            "triton_fused_gather_rope_scatter_boundary_repair_aware"
            if boundary_repair_aware
            else "triton_fused_gather_rope_scatter"
        ),
        "kernel_launches_per_layer": 1,
        **kwargs,
    }
