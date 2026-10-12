"""Unified NOC agent JSONL metrics with explicit N/A semantics."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NOCAgentMetric:
    task_id: str
    agent_id: str
    request_id: str
    session_revision: int
    loop_stage: str
    objective: str
    variant: str
    operation_mode: str
    measurement_kind: str = "formal_latency"

    full_session_tokens: int | None = None
    candidate_context_tokens: int | None = None
    selected_context_tokens: int | None = None
    mutable_tokens: int | None = None
    context_reduction_tokens: int | None = None
    context_reduction_ratio: float | None = None
    selected_sub_context_count: int | None = None
    excluded_sub_context_count: int | None = None

    segmentation_model: str | None = None
    segmentation_latency_ms: float | None = None
    segment_count: int | None = None
    validation_status: str | None = None
    repair_count: int | None = None
    segmentation_fallback_used: bool | None = None

    selection_model: str | None = None
    selection_latency_ms: float | None = None
    token_budget: int | None = None
    selected_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)

    compile_decisions: list[dict[str, Any]] = field(default_factory=list)
    compile_requests: int | None = None
    compiled_blocks: int | None = None
    compiled_tokens: int | None = None
    compile_latency_ms: float | None = None
    duplicate_compile_avoided: int | None = None

    requested_cache_keys: list[str] = field(default_factory=list)
    cache_hits: int | None = None
    cache_misses: int | None = None
    stale_blocks: int | None = None
    incompatible_blocks: int | None = None
    reused_tokens: int | None = None
    recomputed_tokens: int | None = None
    cache_reuse_ratio: float | None = None
    recompute_ratio: float | None = None
    eviction_count: int | None = None
    fallback_reason: str | None = None

    lookup_latency_ms: float | None = None
    link_plan_latency_ms: float | None = None
    link_latency_ms: float | None = None
    native_prefill_latency_ms: float | None = None
    ttft_ms: float | None = None
    decode_latency_ms: float | None = None
    itl_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    queue_wait_ms: float | None = None
    request_latency_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_per_second: float | None = None

    agent_steps: int | None = None
    planner_calls: int | None = None
    replanner_calls: int | None = None
    llm_calls: int | None = None
    tool_calls: int | None = None
    successful_tool_calls: int | None = None
    failed_tool_calls: int | None = None
    repeated_action_count: int | None = None
    invalid_tool_call_count: int | None = None
    tool_execution_time_ms: float | None = None
    planning_latency_ms: float | None = None
    replanning_latency_ms: float | None = None
    total_llm_latency_ms: float | None = None
    context_management_latency_ms: float | None = None
    e2e_latency_ms: float | None = None
    task_success: bool | None = None
    verifier_pass: bool | None = None
    verifier_score: float | None = None
    timeout: bool | None = None
    oom: bool | None = None
    error: str | None = None

    unique_semantic_blocks: int | None = None
    average_reuse_count_per_block: float | None = None
    superseded_blocks: int | None = None
    invalidated_blocks: int | None = None
    evicted_blocks: int | None = None
    evicted_tokens: int | None = None
    fallback_count: int | None = None
    recovery_success_count: int | None = None
    backend_input_tokens: int | None = None
    backend_output_tokens: int | None = None
    local_model_input_tokens: int | None = None
    local_model_output_tokens: int | None = None
    concurrent_tasks: int | None = None
    warmup: bool = False

    def finalize_ratios(self) -> None:
        if self.full_session_tokens is not None and self.selected_context_tokens is not None:
            self.context_reduction_tokens = self.full_session_tokens - self.selected_context_tokens
            self.context_reduction_ratio = (
                self.context_reduction_tokens / self.full_session_tokens
                if self.full_session_tokens
                else 0.0
            )
        if self.reused_tokens is not None and self.recomputed_tokens is not None:
            total = self.reused_tokens + self.recomputed_tokens
            self.cache_reuse_ratio = self.reused_tokens / total if total else 0.0
            self.recompute_ratio = self.recomputed_tokens / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        self.finalize_ratios()
        return asdict(self)


class NOCAgentMetricsRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, metric: NOCAgentMetric | dict[str, Any]) -> None:
        row = metric.to_dict() if isinstance(metric, NOCAgentMetric) else dict(metric)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    @staticmethod
    def original_sglang_metric(**identity: Any) -> NOCAgentMetric:
        """LegoLink-only fields intentionally remain None (JSON null / report N/A)."""

        return NOCAgentMetric(**identity)
