"""Content-aware sparse boundary repair window selection.

The runtime already accepts a per-block repair window: ``_prepare_link`` reads
``pic_link.recompute_tokens_per_block`` and falls back to a single ``legolink_k``
only when it is absent on a debug path. The live adapter produces this list;
formal execution additionally requires the attributed decision and quality profile.

A scalar K can overstate work on short blocks unless counts are clamped. Different
preceding contexts motivate measuring different windows; position alone does not
prove how much repair is sufficient.

Why blocks differ
-----------------

``NOCSGLangAdapter._standalone_compile_segment`` tokenizes a block on its own,
so a compiled block's KV is computed at local positions ``0..n-1`` with no
preceding context at all. At link time the block sits at a global offset behind
the chat prefix and every earlier block. Two things changed:

* position -- repaired exactly by the RoPE global-position remap, no recompute
  needed;
* visible preceding context -- the block's leading tokens attended to nothing at
  compile time and must now attend to everything before them.

Preceding context is a policy feature, not a proven bound on required repair.
Quality still needs model/workload/framing-specific measurement.

Safety
------

Reducing a window below a K with recorded exact-output evidence is a correctness
risk: the K screen found K=96 and K=88 passing 100-request gates while K=64
failed output equivalence, and ``test_k88_is_not_unconditionally_promotable``
exists precisely because one passing shape does not generalise. The historical
gate below prevents unmeasured reductions relative to the caller's baseline; it
does not certify that baseline. ``validated_k`` is a legacy parameter name.
Formal execution additionally requires an exact compatible RepairQualityProfile
through execution_governance; absent evidence never certifies any K.

ARCHITECTURE_STAGE: 7 -- Repair Policy / K Budget
CANONICAL_ENTRYPOINT: RepairWindowPlanner (.plan -> RepairWindowDecision,
    .execution_decision -> the attributed RepairDecision)
DELEGATED_HELPERS: group_repair_profile.py. The policy chooses window *sizes*;
    physical positions are the runtime's (Stage 12). It is asked for its windows
    from ``pic_link_contract.LogicalPICRequestFramer.frame`` (Stage 8) because the
    exact token layout is what defines the blocks it sizes.
NOT_IN_CURRENT_BASELINE: legolink_repair_policy.py (the EPIC first-block repair
    rule) lives only on branch fix/legolink-policy-accounting (e76cc81). It is
    deliberately not merged by the stage-entrypoint refactor.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPAIR_WINDOW_POLICY_VERSION = "repair-window-v1"


class RepairWindowError(ValueError):
    """A repair window plan violated a safety invariant."""


@dataclass(frozen=True)
class RepairWindowBlock:
    """One linked block, described by everything known before any GPU work."""

    block_index: int
    cache_key: str
    sub_context_id: str
    context_field: str
    token_count: int
    global_start: int

    @property
    def preceding_token_count(self) -> int:
        """Preceding context this block gained relative to compile time.

        A compiled block is prefilled standalone from local position 0, so every
        token before ``global_start`` is context its leading tokens never
        attended to.
        """

        return self.global_start

    def to_metadata(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "cache_key": self.cache_key,
            "sub_context_id": self.sub_context_id,
            "context_field": self.context_field,
            "token_count": self.token_count,
            "global_start": self.global_start,
            "preceding_token_count": self.preceding_token_count,
        }


@dataclass(frozen=True)
class RepairWindowDecision:
    """Per-block repair windows plus why each one was chosen."""

    windows: tuple[int, ...]
    policy: str
    validated_k: int
    reasons: tuple[str, ...]
    evidence_id: str | None = None

    @property
    def total_repair_tokens(self) -> int:
        return sum(self.windows)

    def uniform_repair_tokens(self, blocks: Sequence[RepairWindowBlock]) -> int:
        """Tokens a uniform validated K would have planned, for comparison."""

        return sum(min(self.validated_k, block.token_count) for block in blocks)

    def to_metadata(
        self, blocks: Sequence[RepairWindowBlock] | None = None
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "repair_window_policy": self.policy,
            "repair_window_policy_version": REPAIR_WINDOW_POLICY_VERSION,
            "repair_window_validated_k": self.validated_k,
            "repair_window_evidence_id": self.evidence_id,
            "repair_window_reasons": list(self.reasons),
            "repair_window_total_tokens": self.total_repair_tokens,
        }
        if blocks is not None:
            uniform = self.uniform_repair_tokens(blocks)
            metadata["repair_window_uniform_total_tokens"] = uniform
            metadata["repair_window_saved_tokens"] = uniform - self.total_repair_tokens
        return metadata


class RepairWindowEvidence:
    """Shapes with recorded exact-output evidence at a reduced window.

    An entry authorises one ``(task_id, context_field, token_count bucket)`` to
    run at ``min_validated_window``. Evidence is produced by a K screen that
    recorded exact generated output, never inferred from latency.
    """

    def __init__(
        self,
        entries: Iterable[Mapping[str, Any]] = (),
        *,
        evidence_id: str | None = None,
    ) -> None:
        self.evidence_id = evidence_id
        self._entries: dict[tuple[str, str, int], int] = {}
        for entry in entries:
            key = (
                str(entry["task_id"]),
                str(entry["context_field"]),
                int(entry["token_count_bucket"]),
            )
            window = int(entry["min_validated_window"])
            if window < 0:
                raise RepairWindowError(
                    "min_validated_window must be non-negative"
                )
            if not bool(entry.get("exact_output", False)):
                # Latency evidence never authorises a reduction.
                continue
            previous = self._entries.get(key)
            self._entries[key] = (
                window if previous is None else max(previous, window)
            )

    @staticmethod
    def token_count_bucket(token_count: int) -> int:
        """Power-of-two bucket, so evidence generalises across near shapes."""

        if token_count <= 1:
            return 1
        return 1 << (token_count - 1).bit_length()

    def minimum_window(
        self, task_id: str, block: RepairWindowBlock
    ) -> int | None:
        return self._entries.get(
            (
                str(task_id),
                block.context_field,
                self.token_count_bucket(block.token_count),
            )
        )

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_json(cls, path: str | Path) -> "RepairWindowEvidence":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload.get("entries", ()),
            evidence_id=payload.get("evidence_id"),
        )

    @classmethod
    def empty(cls) -> "RepairWindowEvidence":
        return cls(())


class RepairWindowPolicy(ABC):
    """Choose a repair window per linked block."""

    name = "abstract"

    @abstractmethod
    def _propose(
        self,
        blocks: Sequence[RepairWindowBlock],
        *,
        validated_k: int,
        task_id: str,
    ) -> list[tuple[int, str]]:
        """Return one ``(window, reason)`` per block before safety clamping."""

    def plan(
        self,
        blocks: Sequence[RepairWindowBlock],
        *,
        validated_k: int,
        task_id: str = "",
        evidence: RepairWindowEvidence | None = None,
    ) -> RepairWindowDecision:
        if validated_k < 0:
            raise RepairWindowError("validated_k must be non-negative")
        for expected_index, block in enumerate(blocks):
            if block.block_index != expected_index:
                raise RepairWindowError(
                    "repair window blocks must be in link order"
                )
            if block.token_count <= 0:
                raise RepairWindowError(
                    f"block {block.cache_key} has no tokens"
                )
        evidence = evidence or RepairWindowEvidence.empty()
        proposed = self._propose(
            blocks, validated_k=validated_k, task_id=task_id
        )
        if len(proposed) != len(blocks):
            raise RepairWindowError(
                "policy returned the wrong number of repair windows"
            )
        windows: list[int] = []
        reasons: list[str] = []
        for block, (window, reason) in zip(blocks, proposed):
            window = int(window)
            if window < 0:
                raise RepairWindowError("repair window must be non-negative")
            ceiling = min(validated_k, block.token_count)
            if window < ceiling:
                permitted = evidence.minimum_window(task_id, block)
                if permitted is None:
                    # Nothing recorded for this shape. Fall back to the
                    # validated K rather than guess.
                    window = ceiling
                    reason = (
                        f"{reason}; raised to validated K, "
                        "no exact-output evidence for this shape"
                    )
                elif window < permitted:
                    # Evidence proves this shape exact at `permitted`, so take
                    # that rather than discarding the evidence entirely.
                    window = min(permitted, ceiling)
                    reason = (
                        f"{reason}; raised to the lowest window with "
                        f"exact-output evidence ({permitted})"
                    )
            # A block can never repair more tokens than it has.
            if window > block.token_count:
                window = block.token_count
                reason = f"{reason}; clamped to block length"
            windows.append(window)
            reasons.append(reason)
        return RepairWindowDecision(
            windows=tuple(windows),
            policy=self.name,
            validated_k=validated_k,
            reasons=tuple(reasons),
            evidence_id=evidence.evidence_id,
        )


class UniformRepairWindow(RepairWindowPolicy):
    """Every block repairs the validated K. The current production behaviour."""

    name = "uniform"

    def _propose(self, blocks, *, validated_k, task_id):
        return [(validated_k, "uniform validated K") for _ in blocks]


class BlockLengthCappedRepairWindow(RepairWindowPolicy):
    """Cap each window at the block's own length.

    Output-identical to :class:`UniformRepairWindow`: ``_prepare_link`` already
    clamps recompute positions with ``min(start + window, end)``. What changes is
    the plan and every count derived from it, which stopped over-reporting repair
    tokens for blocks shorter than K.
    """

    name = "block_length_capped"

    def _propose(self, blocks, *, validated_k, task_id):
        return [
            (
                min(validated_k, block.token_count),
                f"min(K={validated_k}, block tokens={block.token_count})",
            )
            for block in blocks
        ]


class ContextGainRepairWindow(RepairWindowPolicy):
    """Scale the window with the preceding context a block actually gained.

    A block's leading tokens attended to nothing at compile time. At link time
    they must attend to everything before them, so the amount of repair needed is
    bounded by how much preceding context appeared -- which is exactly
    ``global_start``. The first block behind a 30-token chat prefix gained far
    less than the sixth block behind several thousand tokens.

    The window grows logarithmically in that gain and saturates at the validated
    K, because attention to distant context contributes progressively less to a
    leading token's KV. ``saturation_tokens`` is the gain at which the window
    reaches K and must be supplied explicitly -- there is no defensible default
    before calibration.

    Every reduction below the validated K still passes the evidence gate, so this
    policy is inert until a K screen has recorded exact-output evidence.
    """

    name = "context_gain"

    def __init__(self, *, saturation_tokens: int, floor_tokens: int = 0) -> None:
        if saturation_tokens <= 0:
            raise RepairWindowError("saturation_tokens must be positive")
        if floor_tokens < 0:
            raise RepairWindowError("floor_tokens must be non-negative")
        self.saturation_tokens = int(saturation_tokens)
        self.floor_tokens = int(floor_tokens)

    def _propose(self, blocks, *, validated_k, task_id):
        proposals: list[tuple[int, str]] = []
        span = max(validated_k - self.floor_tokens, 0)
        saturation = math.log2(1.0 + self.saturation_tokens)
        for block in blocks:
            gain = block.preceding_token_count
            fraction = (
                1.0 if saturation <= 0 else min(1.0, math.log2(1.0 + gain) / saturation)
            )
            window = self.floor_tokens + math.ceil(span * fraction)
            window = min(window, validated_k, block.token_count)
            proposals.append(
                (
                    window,
                    (
                        f"context gain={gain} tokens, "
                        f"saturation={self.saturation_tokens}, "
                        f"fraction={fraction:.3f}"
                    ),
                )
            )
        return proposals


@dataclass
class RepairWindowPlanner:
    """Bind a policy and its evidence to one task."""

    policy: RepairWindowPolicy = field(default_factory=UniformRepairWindow)
    evidence: RepairWindowEvidence = field(default_factory=RepairWindowEvidence.empty)

    def execution_decision(self, decision, *, framing_layout_version,
                           workload_class, quality_profile=None):
        """Attribute the actual plan; historical validated_k is not quality proof."""
        from .execution_governance import RepairDecision

        quality = quality_profile or {}
        return RepairDecision(
            selected_k=decision.validated_k,
            windows=decision.windows,
            policy="quality_guarded" if quality else "fixed_unvalidated_debug",
            policy_version="repair-decision-v1",
            framing_layout_version=framing_layout_version,
            quality_profile_id=quality.get("profile_id"),
            evidence_id=quality.get("evidence_id"),
            reason=f"{decision.policy}: " + "; ".join(decision.reasons),
            workload_class=workload_class,
        )

    def plan(
        self,
        blocks: Sequence[RepairWindowBlock],
        *,
        validated_k: int,
        task_id: str = "",
    ) -> RepairWindowDecision:
        return self.policy.plan(
            blocks,
            validated_k=validated_k,
            task_id=task_id,
            evidence=self.evidence,
        )


def build_repair_window_blocks(
    layout: Any,
    linked_contexts: Sequence[Any],
) -> tuple[RepairWindowBlock, ...]:
    """Describe each linked block from the exact physical token layout.

    ``layout`` is a :class:`~noc_context_manager.physical_framing.NormalizedChatTokenLayout`.
    Global offsets are taken from the real segment token counts, so they match
    the ``block_ranges`` the scheduler rebuilds rather than approximating them.
    """

    fields = {
        getattr(value, "cache_key"): getattr(value, "context_field", "unknown")
        for value in linked_contexts
    }
    identifiers = {
        getattr(value, "cache_key"): getattr(value, "sub_context_id", "")
        for value in linked_contexts
    }
    blocks: list[RepairWindowBlock] = []
    cursor = len(layout.chat_prefix_token_ids)
    # Only the blocks this Link reuses get a window; fresh tokens between them
    # (token-stable framing) move every later block's global start.
    segments = getattr(layout, "link_segments", layout.block_segments)
    gaps = getattr(layout, "interblock_fresh_token_ids", ())
    for index, segment in enumerate(segments):
        if index > 0 and index - 1 < len(gaps):
            cursor += len(gaps[index - 1])
        token_count = len(segment.token_ids)
        blocks.append(
            RepairWindowBlock(
                block_index=index,
                cache_key=segment.cache_key,
                sub_context_id=identifiers.get(segment.cache_key, ""),
                context_field=fields.get(segment.cache_key, "unknown"),
                token_count=token_count,
                global_start=cursor,
            )
        )
        cursor += token_count
    return tuple(blocks)
