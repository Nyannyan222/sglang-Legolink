"""Bridge objects for mapping sub-context blocks into SGLang PIC inputs.

The objects here are intentionally runtime-facing but SGLang-version-neutral.
They separate EPIC-style independent compilation from sparse causal linking and
decode, without depending on one exact upstream SGLang source layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .pic_link_contract import (
    PICLinkRequest,
    PICLinkRequestAudit,
    build_pic_link_payload,
    build_pic_link_request,
)
from .runtime_isolation import (
    IsolationReport,
    PrefillBlock,
    block_isolation_report,
)
from .tokenizer import estimate_tokens


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
class LegoLinkRecomputeRange:
    block_id: str
    source: str
    start_token: int
    end_token: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SGLangPICLinkRequest:
    request_id: str
    task_id: str
    model: str
    block_ids: list[str]
    cache_keys: list[str]
    attention_ranges: list[AttentionRange]
    mutable_text: str
    mutable_start_token: int
    mutable_token_count: int
    total_token_count: int
    recompute_initial_tokens: int
    recompute_ranges: list[LegoLinkRecomputeRange]
    recompute_positions: list[int]
    #: Whether the opening tokens of the first block are repaired too. The EPIC
    #: demo says no -- the first chunk has no preceding boundary -- and the live
    #: adapter says yes, because a chat prefix precedes it. Until this field
    #: existed the bridge could only say no, so the plan-file benchmarks and the
    #: agent were running different repair policies through the same runtime.
    recompute_first_block: bool = False
    execution_mode: str = "legacy_debug"
    framing_spec: dict[str, Any] | None = None
    repair_decision: dict[str, Any] | None = None
    execution_profile: dict[str, Any] | None = None
    token_layout: dict[str, Any] | None = None
    compile_attention_policy: str = "isolated_causal"
    link_attention_policy: str = "legolink_sparse_causal"
    decode_attention_policy: str = "full_over_linked_kv"
    position_policy: str = "block_local_compile_global_link"

    @property
    def kvlink(self) -> list[int]:
        """EPIC demo compatibility alias for LegoLink recompute positions."""

        return self.recompute_positions

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "model": self.model,
            "block_ids": self.block_ids,
            "cache_keys": self.cache_keys,
            "attention_ranges": [
                attention_range.to_dict() for attention_range in self.attention_ranges
            ],
            "mutable_text": self.mutable_text,
            "mutable_start_token": self.mutable_start_token,
            "mutable_token_count": self.mutable_token_count,
            "total_token_count": self.total_token_count,
            "recompute_initial_tokens": self.recompute_initial_tokens,
            "recompute_ranges": [
                recompute_range.to_dict() for recompute_range in self.recompute_ranges
            ],
            "recompute_positions": self.recompute_positions,
            "imp_indices": self.recompute_positions,
            "kvlink": self.kvlink,
            "legolink_k": self.recompute_initial_tokens,
            "compile_attention_policy": self.compile_attention_policy,
            "link_attention_policy": self.link_attention_policy,
            "decode_attention_policy": self.decode_attention_policy,
            "position_policy": self.position_policy,
        }

    def to_contract(self) -> PICLinkRequest:
        """The logical request this plan expresses.

        Everything else on this dataclass -- attention ranges, recompute
        positions, the mutable boundary -- is a host-side *estimate* of what
        the runtime will decide, computed from a preliminary tokenizer for the
        plan report. It used to be sent, and the runtime overwrote all of it.
        """

        return build_pic_link_request(
            execution_mode=self.execution_mode,
            framing_spec=self.framing_spec,
            repair_decision=self.repair_decision,
            execution_profile=self.execution_profile,
            token_layout=self.token_layout,
            recompute_tokens_per_block=(self.repair_decision["windows"] if self.repair_decision else None),
            request_id=self.request_id,
            task_id=self.task_id,
            cache_keys=self.cache_keys,
            block_ids=self.block_ids,
            recompute_initial_tokens=self.recompute_initial_tokens,
            recompute_first_block=self.recompute_first_block,
            recompute_first_block_tokens=(
                self.repair_decision["windows"][0] if self.repair_decision else
                self.recompute_initial_tokens if self.recompute_first_block else 0
            ),
        )

    def to_audit(self) -> PICLinkRequestAudit:
        """The estimate, kept where a report can read it without pretending it ran."""

        return PICLinkRequestAudit(
            request_id=self.request_id,
            task_id=self.task_id,
            selected_order=tuple(self.cache_keys),
            estimated_layout={
                "attention_ranges": [
                    attention_range.to_dict()
                    for attention_range in self.attention_ranges
                ],
                "mutable_start_token": self.mutable_start_token,
                "mutable_token_count": self.mutable_token_count,
                "total_token_count": self.total_token_count,
                "recompute_positions": list(self.recompute_positions),
                "recompute_ranges": [
                    recompute_range.to_dict()
                    for recompute_range in self.recompute_ranges
                ],
                "position_policy": self.position_policy,
                "compile_attention_policy": self.compile_attention_policy,
                "link_attention_policy": self.link_attention_policy,
                "decode_attention_policy": self.decode_attention_policy,
            },
        )

    def to_runtime_payload(self, max_tokens: int = 1) -> dict[str, Any]:
        """One chat-completions request carrying the canonical Link contract."""

        payload = build_pic_link_payload(
            self.to_contract(),
            model=self.model,
            mutable_text=self.mutable_text,
            generation={"temperature": 0, "max_tokens": max_tokens},
        )
        # The estimate travels beside the request rather than inside it, so a
        # plan file still carries everything a report needs.
        payload["pic_link_audit"] = self.to_audit().to_dict()
        return payload


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
    sub_context_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_chat_completion_payload(self, max_tokens: int = 1) -> dict[str, Any]:
        """Return a payload shape that can travel through SGLang's OpenAI API.

        Stock SGLang will ignore SubContext metadata unless patched, but keeping
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
                "prefill_unit_id": self.block_id,
                "cache_key": self.cache_key,
                "context_field": self.context_field,
            },
            "sub_context": self.sub_context_metadata,
        }


@dataclass(frozen=True)
class SGLangBlockRuntimePlan:
    request_id: str
    task_id: str
    model: str
    prefill_requests: list[SGLangPrefillRequest]
    attention_ranges: list[AttentionRange]
    link_request: SGLangPICLinkRequest
    isolation_report: IsolationReport
    scheduler_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "model": self.model,
            "prefill_requests": [request.to_dict() for request in self.prefill_requests],
            "attention_ranges": [attention_range.to_dict() for attention_range in self.attention_ranges],
            "link_request": self.link_request.to_dict(),
            "isolation_report": self.isolation_report.to_dict(),
            "scheduler_contract": self.scheduler_contract,
        }

    def chat_completion_payloads(self, max_tokens: int = 1) -> list[dict[str, Any]]:
        return [
            request.to_chat_completion_payload(max_tokens=max_tokens)
            for request in self.prefill_requests
        ]

    def pic_link_payload(self, max_tokens: int = 1) -> dict[str, Any]:
        return self.link_request.to_runtime_payload(max_tokens=max_tokens)


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


def build_sglang_pic_link_plan(
    blocks: Iterable[PrefillBlock],
    *,
    request_id: str,
    task_id: str,
    model: str,
    recompute_initial_tokens: int = 16,
    recompute_first_block: bool = False,
    mutable_text: str = "",
) -> SGLangPICLinkRequest:
    if recompute_initial_tokens < 0:
        raise ValueError("recompute_initial_tokens must be non-negative")

    block_list = list(blocks)
    attention_ranges = build_attention_ranges(block_list)
    mutable_start_token = sum(block.token_count for block in block_list)
    # An estimate, and one the runtime does not trust either: PICRuntime
    # overwrites link_plan["mutable_token_count"] with len(mutable_input_ids)
    # once it has tokenized the request itself. This value sizes the host-side
    # plan and is recorded as an estimate in the artifact.
    mutable_token_count = estimate_tokens(mutable_text).value
    recompute_ranges: list[LegoLinkRecomputeRange] = []

    # The DerekHJH/epic demo exposes this as cache_fuse_metadata["kvlink"]:
    # recompute the first k tokens of each later chunk, then recompute the
    # mutable query. By default the first chunk has no preceding chunk boundary
    # to repair; a caller that puts a chat prefix in front of it asks for one.
    repaired = attention_ranges if recompute_first_block else attention_ranges[1:]
    for attention_range in repaired:
        token_count = min(recompute_initial_tokens, attention_range.token_count)
        if token_count:
            recompute_ranges.append(
                LegoLinkRecomputeRange(
                    block_id=attention_range.block_id,
                    source="chunk_initial",
                    start_token=attention_range.start_token,
                    end_token=attention_range.start_token + token_count,
                    token_count=token_count,
                )
            )

    if mutable_token_count:
        recompute_ranges.append(
            LegoLinkRecomputeRange(
                block_id="__mutable_query__",
                source="mutable_query",
                start_token=mutable_start_token,
                end_token=mutable_start_token + mutable_token_count,
                token_count=mutable_token_count,
            )
        )

    recompute_positions = [
        position
        for recompute_range in recompute_ranges
        for position in range(recompute_range.start_token, recompute_range.end_token)
    ]
    return SGLangPICLinkRequest(
        request_id=request_id,
        task_id=task_id,
        model=model,
        block_ids=[block.block_id for block in block_list],
        cache_keys=[block.cache_key for block in block_list],
        attention_ranges=attention_ranges,
        mutable_text=mutable_text,
        mutable_start_token=mutable_start_token,
        mutable_token_count=mutable_token_count,
        total_token_count=mutable_start_token + mutable_token_count,
        recompute_initial_tokens=recompute_initial_tokens,
        recompute_ranges=recompute_ranges,
        recompute_positions=recompute_positions,
        recompute_first_block=recompute_first_block,
    )


def scheduler_contract() -> dict[str, Any]:
    return {
        "metadata_fields": {
            "prefill_unit_id": "Stable runtime id for one semantic sub-context block.",
            "cache_key": "Content-derived key used to store/reuse the block KV cache.",
            "extra_key": "SGLang request namespace key; should be set to cache_key.",
            "attention_policy": "Legacy compile-only value: block_isolated_no_pairwise_attention.",
            "compile_attention_policy": "isolated_causal",
            "link_attention_policy": "legolink_sparse_causal",
            "imp_indices": "Global token positions recomputed during sparse causal linking.",
            "kvlink": "EPIC demo alias for the same LegoLink recompute positions as imp_indices.",
            "legolink_k": "Number of initial tokens recomputed for each later immutable block.",
        },
        "runtime_phases": [
            "Compile: prefill each semantic block independently with block-local causal positions.",
            "Link: retrieve selected KV blocks, assign global offsets, and recompute imp_indices with sparse causal attention over preceding linked KV.",
            "Decode: generate with causal attention over the complete linked KV context.",
        ],
        "scheduler_requirements": [
            "Parse sub_context and pic_link metadata.",
            "Treat every prefill_unit_id as an independent compile/cache unit.",
            "Use cache_key/extra_key to lookup and store block-local KV entries.",
            "Preserve the selected block order and its explicit global offset map.",
            "Recompute only imp_indices during link; do not permanently use a block-diagonal mask.",
            "Expose compile hits, link recompute tokens, TTFT, throughput, and peak memory.",
        ],
        "kv_cache_requirements": [
            "KV entries are keyed by cache_key, not by prompt position alone.",
            "Compile stores block-local positions; link applies the explicit global offset map.",
            "A cache hit for block A must not require block A to appear as a prefix.",
            "The physical cache identity must also bind model, tokenizer, adapter, and token ids.",
        ],
        "attention_mask_contract": {
            "compile": "causal attention is confined to the independently compiled block",
            "link": "each recomputed query position may attend global key positions <= query position",
            "decode": "new tokens may attend the full linked KV context",
            "compile_validation_metric": "cross_block_allowed_edges == 0",
            "link_validation_metric": "kvlink/imp_indices are sorted, unique, and in global range",
        },
    }


def build_sglang_block_runtime_plan(
    blocks: Iterable[PrefillBlock],
    *,
    request_id: str,
    task_id: str,
    model: str,
    recompute_initial_tokens: int = 16,
    recompute_first_block: bool = False,
    mutable_text: str = "",
) -> SGLangBlockRuntimePlan:
    block_list = list(blocks)
    attention_ranges = build_attention_ranges(block_list)
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
                sub_context_metadata={
                    "prefill_unit_id": block.block_id,
                    "cache_key": block.cache_key,
                    "source_sub_context_id": block.source_sub_context_id,
                    "context_field": block.context_field,
                    "functional_role": block.functional_role,
                    "token_count": block.token_count,
                    "attention_policy": "block_isolated_no_pairwise_attention",
                    "compile_attention_policy": "isolated_causal",
                    "link_attention_policy": "legolink_sparse_causal",
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
        attention_ranges=attention_ranges,
        link_request=build_sglang_pic_link_plan(
            block_list,
            request_id=request_id,
            task_id=task_id,
            model=model,
            recompute_initial_tokens=recompute_initial_tokens,
            recompute_first_block=recompute_first_block,
            mutable_text=mutable_text,
        ),
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
        if request.sub_context_metadata.get("attention_policy") != "block_isolated_no_pairwise_attention":
            errors.append(f"missing block isolation policy for {request.block_id}")
        if request.sub_context_metadata.get("compile_attention_policy") != "isolated_causal":
            errors.append(f"missing compile attention policy for {request.block_id}")
    if plan.isolation_report.cross_block_allowed_edges != 0:
        errors.append("cross_block_allowed_edges is not zero")
    if not plan.isolation_report.pairwise_attention_ok:
        errors.append("pairwise attention check failed")

    link_request = plan.link_request
    if link_request.block_ids != [request.block_id for request in plan.prefill_requests]:
        errors.append("link block order does not match prefill request order")
    if link_request.cache_keys != [request.cache_key for request in plan.prefill_requests]:
        errors.append("link cache key order does not match prefill request order")
    expected_mutable_start = sum(request.token_count for request in plan.prefill_requests)
    if link_request.mutable_start_token != expected_mutable_start:
        errors.append("mutable_start_token does not follow immutable blocks")
    if link_request.total_token_count != (
        link_request.mutable_start_token + link_request.mutable_token_count
    ):
        errors.append("total_token_count is inconsistent")
    if link_request.recompute_positions != sorted(set(link_request.recompute_positions)):
        errors.append("kvlink/imp_indices are not sorted and unique")
    if any(
        position < 0 or position >= link_request.total_token_count
        for position in link_request.recompute_positions
    ):
        errors.append("kvlink/imp_indices contain an out-of-range position")
    expected_positions = [
        position
        for attention_range in link_request.attention_ranges[1:]
        for position in range(
            attention_range.start_token,
            attention_range.start_token
            + min(link_request.recompute_initial_tokens, attention_range.token_count),
        )
    ]
    expected_positions.extend(
        range(link_request.mutable_start_token, link_request.total_token_count)
    )
    if link_request.recompute_positions != expected_positions:
        errors.append("kvlink/imp_indices do not match the LegoLink selection rule")
    return errors
