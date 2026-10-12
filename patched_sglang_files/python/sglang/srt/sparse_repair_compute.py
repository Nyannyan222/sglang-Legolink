"""Shared metadata and batching analysis for LegoLink sparse repair.

This module deliberately does not execute Transformer layers.  It provides the
read-only request view needed by the native and FlashInfer attention paths and
an analysis-only batching plan.  Actual layer-outer batching remains gated on
repaired-KV and hidden-state correctness tests.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class SparseRepairComputeError(RuntimeError):
    pass


def _sparse_repair_row_slices(
    forward_batch: Any,
    token_count: int,
    *,
    override_key: str,
    environment_key: str,
    label: str,
    trace_prefix: str,
) -> tuple[slice, ...]:
    metadata = getattr(forward_batch, "pic_sparse_repair_metadata", None)
    if not isinstance(metadata, dict):
        return ()
    override = metadata.get(override_key)
    if override is not None and not isinstance(override, bool):
        raise SparseRepairComputeError(f"row-wise {label} override must be boolean")
    enabled = (
        override
        if override is not None
        else os.getenv(environment_key, "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not enabled:
        return ()
    unit_size = int(metadata.get("batch_execution_unit_size", 1))
    if unit_size <= 1:
        return ()
    raw_ranges = metadata.get("repair_ranges")
    if not isinstance(raw_ranges, list) or len(raw_ranges) != unit_size:
        raise SparseRepairComputeError(
            f"row-wise {label} requires one repair range per grouped execution row"
        )
    lengths: list[int] = []
    for item in raw_ranges:
        if not isinstance(item, dict):
            raise SparseRepairComputeError(
                f"row-wise {label} repair ranges must use start/end mappings"
            )
        start = int(item.get("start", -1))
        end = int(item.get("end", -1))
        if start < 0 or end <= start:
            raise SparseRepairComputeError(
                f"row-wise {label} received an invalid range"
            )
        lengths.append(end - start)
    if sum(lengths) != int(token_count):
        raise SparseRepairComputeError(
            f"row-wise {label} ranges do not match the flattened token count"
        )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    metadata[f"{trace_prefix}_observed"] = True
    metadata[f"{trace_prefix}_row_count"] = len(lengths)
    metadata[f"{trace_prefix}_lengths"] = list(lengths)
    return tuple(
        slice(offsets[index], offsets[index + 1])
        for index in range(len(lengths))
    )


def sparse_repair_rowwise_ffn_slices(
    forward_batch: Any,
    token_count: int,
) -> tuple[slice, ...]:
    """Return flattened slices for shape-preserving grouped FFN calls."""

    return _sparse_repair_row_slices(
        forward_batch,
        token_count,
        override_key="rowwise_ffn_enabled",
        environment_key="SGLANG_PIC_SPARSE_REPAIR_ROWWISE_FFN",
        label="FFN",
        trace_prefix="rowwise_ffn",
    )


def sparse_repair_rowwise_output_projection_slices(
    forward_batch: Any,
    token_count: int,
) -> tuple[slice, ...]:
    """Return original repair-row slices for grouped attention output projection."""

    return _sparse_repair_row_slices(
        forward_batch,
        token_count,
        override_key="rowwise_output_projection_enabled",
        environment_key="SGLANG_PIC_SPARSE_REPAIR_ROWWISE_OUTPUT_PROJECTION",
        label="output projection",
        trace_prefix="rowwise_output_projection",
    )


def _tensor_outputs(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


class SparseRepairStateCapture:
    """Opt-in diagnostic capture for row-batching equivalence gates.

    Captures layer outputs while repair forwards execute, then snapshots repaired
    K/V and final logits in logical repair-token order. The output is intentionally
    unsuitable for performance measurements because it synchronizes GPU tensors
    to CPU.
    """

    def __init__(self, model_runner: Any, output_path: Path) -> None:
        self.model_runner = model_runner
        self.output_path = output_path
        self.layer_outputs: dict[int, list[dict[str, torch.Tensor]]] = {}
        self.component_outputs: dict[
            int, dict[str, list[dict[str, torch.Tensor]]]
        ] = {}
        self.handles: list[Any] = []

    @classmethod
    def create_if_enabled(
        cls, model_runner: Any, output_dir: str | None
    ) -> SparseRepairStateCapture | None:
        if not output_dir:
            return None
        output_path = Path(output_dir) / "sparse_repair_state.pt"
        if output_path.exists():
            return None
        capture = cls(model_runner, output_path)
        capture.register()
        return capture

    def _model_layers(self) -> Any:
        model = self.model_runner.model
        for attribute in ("model", "language_model", "transformer"):
            candidate = getattr(model, attribute, None)
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate.layers
        if hasattr(model, "layers"):
            return model.layers
        raise SparseRepairComputeError("model does not expose Transformer layers")

    def register(self) -> None:
        layers = self._model_layers()
        start_layer = int(self.model_runner.start_layer)
        end_layer = int(self.model_runner.end_layer)
        component_layer_ids = self._component_layer_ids(start_layer, end_layer)
        for layer_id in range(start_layer, end_layer):
            layer = layers[layer_id]

            def hook(
                _module: Any, _inputs: Any, output: Any, *, index: int = layer_id
            ) -> None:
                tensors = _tensor_outputs(output)
                if not tensors:
                    raise SparseRepairComputeError(
                        f"layer {index} produced no tensor output"
                    )
                names = ("hidden", "residual")
                self.layer_outputs.setdefault(index, []).append(
                    {
                        names[position]
                        if position < len(names)
                        else f"tensor_{position}": tensor.detach().cpu().clone()
                        for position, tensor in enumerate(tensors)
                    }
                )

            self.handles.append(layer.register_forward_hook(hook))
            if layer_id in component_layer_ids:
                self._register_component_hooks(layer_id, layer)

    @staticmethod
    def _component_layer_ids(start_layer: int, end_layer: int) -> set[int]:
        raw = os.getenv(
            "SGLANG_PIC_SPARSE_REPAIR_CAPTURE_COMPONENT_LAYERS", ""
        ).strip()
        if not raw:
            return set()
        try:
            layer_ids = {int(value.strip()) for value in raw.split(",")}
        except ValueError as exc:
            raise SparseRepairComputeError(
                "component capture layers must be comma-separated integers"
            ) from exc
        if not layer_ids or any(
            layer_id < start_layer or layer_id >= end_layer
            for layer_id in layer_ids
        ):
            raise SparseRepairComputeError(
                "component capture layer is outside the local model stage"
            )
        return layer_ids

    def _register_component_hooks(self, layer_id: int, layer: Any) -> None:
        self_attn = getattr(layer, "self_attn", None)
        layer_active = {"value": False}

        def activate_layer(_module: Any, _inputs: Any) -> None:
            layer_active["value"] = True

        def deactivate_layer(_module: Any, _inputs: Any, _output: Any) -> None:
            layer_active["value"] = False

        self.handles.append(layer.register_forward_pre_hook(activate_layer))
        specs = (
            ("pre_attention_norm", getattr(layer, "input_layernorm", None), ("hidden", "residual")),
            ("qkv_projection", getattr(self_attn, "qkv_proj", None), ("qkv",)),
            ("rope", getattr(self_attn, "rotary_emb", None), ("q", "k")),
            ("attention", getattr(self_attn, "attn", None), ("attention_output",)),
            ("output_projection", getattr(self_attn, "o_proj", None), ("projected_output",)),
            ("post_attention_norm", getattr(layer, "post_attention_layernorm", None), ("hidden", "residual")),
            ("mlp", getattr(layer, "mlp", None), ("mlp_output",)),
        )
        missing = [name for name, module, _ in specs if module is None]
        if missing:
            raise SparseRepairComputeError(
                "component capture requires Qwen2-compatible modules: "
                + ", ".join(missing)
            )

        for component_name, module, names in specs:

            def component_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                index: int = layer_id,
                label: str = component_name,
                tensor_names: tuple[str, ...] = names,
            ) -> None:
                if not layer_active["value"]:
                    return
                tensors = _tensor_outputs(output)
                if not tensors:
                    raise SparseRepairComputeError(
                        f"layer {index} component {label} produced no tensor output"
                    )
                self.component_outputs.setdefault(index, {}).setdefault(
                    label, []
                ).append(
                    {
                        tensor_names[position]
                        if position < len(tensor_names)
                        else f"tensor_{position}": tensor.detach().cpu().clone()
                        for position, tensor in enumerate(tensors)
                    }
                )

            self.handles.append(module.register_forward_hook(component_hook))
        self.handles.append(layer.register_forward_hook(deactivate_layer))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(
        self,
        *,
        forward_batch: Any,
        destination_slots: torch.Tensor,
        repair_positions: list[int],
        execution_units: tuple[tuple[int, ...], ...],
        output: Any,
    ) -> Path:
        pool = forward_batch.token_to_kv_pool
        logical_indices = torch.tensor(
            repair_positions, device=destination_slots.device, dtype=torch.long
        )
        repair_slots = destination_slots.index_select(0, logical_indices).long()
        repaired_kv = {}
        for layer_id in range(
            int(self.model_runner.start_layer), int(self.model_runner.end_layer)
        ):
            repaired_kv[layer_id] = {
                "key": pool.get_key_buffer(layer_id)
                .index_select(0, repair_slots)
                .detach()
                .cpu()
                .clone(),
                "value": pool.get_value_buffer(layer_id)
                .index_select(0, repair_slots)
                .detach()
                .cpu()
                .clone(),
            }
        logits_output = getattr(output, "logits_output", output)
        logits = getattr(logits_output, "next_token_logits", None)
        final_hidden = getattr(logits_output, "hidden_states", None)
        payload = {
            "schema_version": 2,
            "diagnostic_only": True,
            "repair_positions": torch.tensor(repair_positions, dtype=torch.long),
            "execution_units": [list(unit) for unit in execution_units],
            "layer_outputs": self.layer_outputs,
            "component_outputs": self.component_outputs,
            "repaired_kv": repaired_kv,
            "logits": logits.detach().cpu().clone() if logits is not None else None,
            "final_hidden": (
                final_hidden.detach().cpu().clone()
                if final_hidden is not None
                else None
            ),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.output_path)
        return self.output_path


@dataclass(frozen=True)
class UnifiedSparseRepairMetadataView:
    """Immutable identity for one contiguous repair row.

    ``destination_slots`` is a detached contiguous tensor pointing at the
    request-private TokenToKVPool slots.  Callers must treat it as read-only.
    The byte identity and hash are captured once so attention layers do not
    repeat a device-to-host slot lookup merely to rebuild cache keys.
    """

    request_id: str
    repair_start: int
    repair_end: int
    repair_positions: tuple[int, ...]
    kv_tokens: int
    repair_tokens: int
    block_count: int
    block_size: int
    destination_slots: torch.Tensor
    ordered_slot_bytes: bytes
    ordered_slot_hash: str
    device: str

    def flashinfer_metadata_key(self, pool_size: int) -> tuple[Any, ...]:
        if pool_size <= 0:
            raise SparseRepairComputeError("pool_size must be positive")
        return (
            self.device,
            int(pool_size),
            self.repair_start,
            self.repair_end,
            self.ordered_slot_bytes,
        )

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "repair_start": self.repair_start,
            "repair_end": self.repair_end,
            "repair_tokens": self.repair_tokens,
            "kv_tokens": self.kv_tokens,
            "block_count": self.block_count,
            "block_size": self.block_size,
            "ordered_slot_hash": self.ordered_slot_hash,
            "ordered_slot_count": int(self.destination_slots.numel()),
            "device": self.device,
            "read_only": True,
        }


@dataclass(frozen=True)
class SparseRepairBatchGroup:
    group_id: int
    row_indices: tuple[int, ...]
    repair_lengths: tuple[int, ...]
    max_repair_tokens: int
    real_tokens: int
    padded_tokens: int
    padding_ratio: float
    requires_layer_outer_execution: bool

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "row_indices": list(self.row_indices),
            "repair_lengths": list(self.repair_lengths),
            "max_repair_tokens": self.max_repair_tokens,
            "real_tokens": self.real_tokens,
            "padded_tokens": self.padded_tokens,
            "padding_ratio": self.padding_ratio,
            "requires_layer_outer_execution": self.requires_layer_outer_execution,
        }


@dataclass(frozen=True)
class SparseRepairBatchPlan:
    groups: tuple[SparseRepairBatchGroup, ...]
    row_to_group: tuple[int, ...]
    execution_mode: str = "analysis_only_sequential_forward_extend"

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "execution_mode": self.execution_mode,
            "group_count": len(self.groups),
            "row_count": len(self.row_to_group),
            "groups": [group.trace_metadata() for group in self.groups],
        }


def build_unified_sparse_repair_metadata_view(
    forward_batch: Any,
    *,
    start: int,
    end: int,
    block_count: int,
    block_size: int,
    request_id: str | None = None,
) -> UnifiedSparseRepairMetadataView:
    """Capture the actual logical-to-physical mapping used by one repair row."""

    if start < 0 or end <= start:
        raise SparseRepairComputeError("invalid sparse repair range")
    if block_count < 0 or block_size < 0:
        raise SparseRepairComputeError(
            "block_count and block_size must be non-negative"
        )
    if int(getattr(forward_batch, "batch_size", 1)) != 1:
        raise SparseRepairComputeError(
            "unified metadata currently requires batch_size=1"
        )

    try:
        req_pool_idx = int(forward_batch.req_pool_indices[0].item())
        slot_table = forward_batch.req_to_token_pool.req_to_token[req_pool_idx]
    except (AttributeError, IndexError, TypeError) as exc:
        raise SparseRepairComputeError(
            "forward_batch is missing request-private TokenToKVPool mapping"
        ) from exc
    if int(slot_table.numel()) < end:
        raise SparseRepairComputeError(
            "destination slot mapping is shorter than repair_end"
        )

    slots = slot_table[:end].detach().to(dtype=torch.int32).contiguous()
    slot_bytes = slots.cpu().numpy().tobytes()
    slot_hash = hashlib.sha256(slot_bytes).hexdigest()
    return UnifiedSparseRepairMetadataView(
        request_id=str(request_id or "unknown"),
        repair_start=int(start),
        repair_end=int(end),
        repair_positions=tuple(range(start, end)),
        kv_tokens=int(end),
        repair_tokens=int(end - start),
        block_count=int(block_count),
        block_size=int(block_size),
        destination_slots=slots,
        ordered_slot_bytes=slot_bytes,
        ordered_slot_hash=slot_hash,
        device=str(slots.device),
    )


def plan_sparse_repair_batches(
    ranges: Iterable[tuple[int, int]],
    *,
    max_padding_ratio: float = 0.0,
) -> SparseRepairBatchPlan:
    """Group compatible repair rows without changing their execution.

    Rows are bucketed by repair length by default.  A positive padding budget
    can merge adjacent length buckets only when the resulting padded fraction
    stays within the caller's explicit limit.  The returned plan remains
    analysis-only because row-outer ``forward_extend`` cannot preserve all
    same-layer repaired-KV dependencies when executed as one ordinary batch.
    """

    rows = tuple((int(start), int(end)) for start, end in ranges)
    if not 0.0 <= max_padding_ratio < 1.0:
        raise SparseRepairComputeError("max_padding_ratio must be in [0, 1)")
    previous_end = -1
    for start, end in rows:
        if start < 0 or end <= start:
            raise SparseRepairComputeError("invalid sparse repair range")
        if start < previous_end:
            raise SparseRepairComputeError("repair ranges must be ordered and disjoint")
        previous_end = end

    buckets: list[list[int]] = []
    for row_index, (start, end) in enumerate(rows):
        length = end - start
        append_to_previous = False
        if buckets:
            members = buckets[-1]
            lengths = [rows[index][1] - rows[index][0] for index in members]
            candidate_lengths = [*lengths, length]
            padded = max(candidate_lengths) * len(candidate_lengths)
            padding_ratio = (padded - sum(candidate_lengths)) / padded
            append_to_previous = padding_ratio <= max_padding_ratio
        if append_to_previous:
            buckets[-1].append(row_index)
        else:
            buckets.append([row_index])

    groups = []
    row_to_group = [-1] * len(rows)
    for group_id, members in enumerate(buckets):
        lengths = tuple(rows[index][1] - rows[index][0] for index in members)
        maximum = max(lengths)
        real_tokens = sum(lengths)
        padded_tokens = maximum * len(lengths)
        for row_index in members:
            row_to_group[row_index] = group_id
        groups.append(
            SparseRepairBatchGroup(
                group_id=group_id,
                row_indices=tuple(members),
                repair_lengths=lengths,
                max_repair_tokens=maximum,
                real_tokens=real_tokens,
                padded_tokens=padded_tokens,
                padding_ratio=(padded_tokens - real_tokens) / padded_tokens,
                requires_layer_outer_execution=len(members) > 1,
            )
        )
    return SparseRepairBatchPlan(groups=tuple(groups), row_to_group=tuple(row_to_group))


def build_sparse_repair_execution_units(
    plan: SparseRepairBatchPlan,
) -> tuple[tuple[int, ...], ...]:
    """Create ordered native execution units while keeping the final row single.

    The last row owns the logits returned to the original one-request scheduler
    batch.  Keeping it separate avoids changing the downstream output shape.
    """

    if not plan.row_to_group:
        return ()
    final_row = len(plan.row_to_group) - 1
    units: list[tuple[int, ...]] = []
    for group in plan.groups:
        members = list(group.row_indices)
        if members and members[-1] == final_row:
            prefix = tuple(members[:-1])
            if len(prefix) > 1:
                units.append(prefix)
            else:
                units.extend((row_index,) for row_index in prefix)
            units.append((final_row,))
        elif len(members) > 1:
            units.append(tuple(members))
        else:
            units.extend((row_index,) for row_index in members)
    return tuple(sorted(units, key=lambda unit: unit[0]))
