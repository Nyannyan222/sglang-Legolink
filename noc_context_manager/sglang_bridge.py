"""Bridge objects for mapping sub-context blocks into SGLang runtime inputs.

The objects here are intentionally runtime-facing but SGLang-version-neutral.
They define the metadata that a future SGLang scheduler / KV-cache patch should
consume without depending on one exact upstream source layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .runtime_isolation import IsolationReport, PrefillBlock, block_isolation_report


@dataclass(frozen=True)
class AttentionRange:
    block_id: str
    cache_key: str
    start_token: int
    end_token: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SGLangPrefillRequest:
    request_id: str
    task_id: str
    model: str
    block_id: str
    cache_key: str
    context_field: str
    functional_role: str
    text: str
    token_count: int
    extra_key: str
    openclaw_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_chat_completion_payload(self, max_tokens: int = 1) -> dict[str, Any]:
        """Return a payload shape that can travel through SGLang's OpenAI API.

        Stock SGLang will ignore OpenClaw metadata unless patched, but keeping
        the metadata in the request body gives the runtime patch a stable place
        to read from.
        """

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": self.text}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "extra_key": self.extra_key,
            "custom_labels": {
                "openclaw_prefill_unit_id": self.block_id,
                "openclaw_cache_key": self.cache_key,
                "openclaw_context_field": self.context_field,
            },
            "openclaw_sub_context": self.openclaw_metadata,
        }


@dataclass(frozen=True)
class SGLangBlockRuntimePlan:
    request_id: str
    task_id: str
    model: str
    prefill_requests: list[SGLangPrefillRequest]
    attention_ranges: list[AttentionRange]
    isolation_report: IsolationReport
    scheduler_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "model": self.model,
            "prefill_requests": [request.to_dict() for request in self.prefill_requests],
            "attention_ranges": [attention_range.to_dict() for attention_range in self.attention_ranges],
            "isolation_report": self.isolation_report.to_dict(),
            "scheduler_contract": self.scheduler_contract,
        }

    def chat_completion_payloads(self, max_tokens: int = 1) -> list[dict[str, Any]]:
        return [
            request.to_chat_completion_payload(max_tokens=max_tokens)
            for request in self.prefill_requests
        ]


def build_attention_ranges(blocks: Iterable[PrefillBlock]) -> list[AttentionRange]:
    ranges: list[AttentionRange] = []
    offset = 0
    for block in blocks:
        end = offset + block.token_count
        ranges.append(
            AttentionRange(
                block_id=block.block_id,
                cache_key=block.cache_key,
                start_token=offset,
                end_token=end,
                token_count=block.token_count,
            )
        )
        offset = end
    return ranges


def scheduler_contract() -> dict[str, Any]:
    return {
        "metadata_fields": {
            "prefill_unit_id": "Stable runtime id for one semantic sub-context block.",
            "cache_key": "Content-derived key used to store/reuse the block KV cache.",
            "extra_key": "SGLang request namespace key; should be set to cache_key.",
            "attention_policy": "block_isolated_no_pairwise_attention",
        },
        "scheduler_requirements": [
            "Parse openclaw_sub_context metadata into each SGLang request object.",
            "Treat every prefill_unit_id as an independent prefill/cache unit.",
            "Use cache_key/extra_key to lookup and store block-local KV entries.",
            "Never concatenate independent sub-context blocks before prefill.",
            "When composing selected blocks, apply a block-diagonal attention mask.",
            "Expose per-block cached/prefill token counters for evaluation.",
        ],
        "kv_cache_requirements": [
            "KV entries are keyed by cache_key, not by prompt position alone.",
            "Reused KV blocks must keep block-local positions or an explicit offset map.",
            "A cache hit for block A must not require block A to appear as a prefix.",
        ],
        "attention_mask_contract": {
            "same_block": "causal_attention_allowed",
            "different_block": "attention_disallowed",
            "validation_metric": "cross_block_allowed_edges == 0",
        },
    }


def build_sglang_block_runtime_plan(
    blocks: Iterable[PrefillBlock],
    *,
    request_id: str,
    task_id: str,
    model: str,
) -> SGLangBlockRuntimePlan:
    block_list = list(blocks)
    prefill_requests: list[SGLangPrefillRequest] = []
    for block in block_list:
        prefill_requests.append(
            SGLangPrefillRequest(
                request_id=request_id,
                task_id=task_id,
                model=model,
                block_id=block.block_id,
                cache_key=block.cache_key,
                context_field=block.context_field,
                functional_role=block.functional_role,
                text=block.text,
                token_count=block.token_count,
                extra_key=block.cache_key,
                openclaw_metadata={
                    "prefill_unit_id": block.block_id,
                    "cache_key": block.cache_key,
                    "source_sub_context_id": block.source_sub_context_id,
                    "context_field": block.context_field,
                    "functional_role": block.functional_role,
                    "token_count": block.token_count,
                    "attention_policy": "block_isolated_no_pairwise_attention",
                    "cache_scope": "sub_context_block",
                    "kv_reuse_granularity": "sub_context",
                    "task_id": block.metadata.get("task_id", task_id),
                },
            )
        )

    return SGLangBlockRuntimePlan(
        request_id=request_id,
        task_id=task_id,
        model=model,
        prefill_requests=prefill_requests,
        attention_ranges=build_attention_ranges(block_list),
        isolation_report=block_isolation_report(block_list),
        scheduler_contract=scheduler_contract(),
    )


def validate_sglang_block_runtime_plan(plan: SGLangBlockRuntimePlan) -> list[str]:
    errors: list[str] = []
    seen_block_ids: set[str] = set()
    for request in plan.prefill_requests:
        if not request.block_id:
            errors.append("missing block_id")
        if request.block_id in seen_block_ids:
            errors.append(f"duplicate block_id: {request.block_id}")
        seen_block_ids.add(request.block_id)
        if request.extra_key != request.cache_key:
            errors.append(f"extra_key does not match cache_key for {request.block_id}")
        if request.openclaw_metadata.get("attention_policy") != "block_isolated_no_pairwise_attention":
            errors.append(f"missing block isolation policy for {request.block_id}")
    if plan.isolation_report.cross_block_allowed_edges != 0:
        errors.append("cross_block_allowed_edges is not zero")
    if not plan.isolation_report.pairwise_attention_ok:
        errors.append("pairwise attention check failed")
    return errors
