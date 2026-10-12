"""Two-or-more-round Planner -> direct Tool -> Re-planner integration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .agents import AgentRouter
from .compile_policy import CompileAction
from .manager import IncrementalContextManager
from .metrics import NOCAgentMetric, NOCAgentMetricsRecorder
from .profiling import HostTimingRecorder, profile_range
from .session import EventType
from .sglang_adapter import NOCSGLangAdapter


@dataclass(frozen=True)
class ToolAction:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ActionParseResult:
    action: ToolAction | None
    status: str
    error: str | None = None


@dataclass(frozen=True)
class AgentRoundResult:
    round_index: int
    loop_stage: str
    llm_output: str
    tool_action: ToolAction | None
    tool_result: str | None
    adapter_route: str
    manifest_path: str
    parse_status: str = "unknown"
    termination_reason: str | None = None


class DirectToolExecutor:
    """Tools execute directly; only planner/re-planner calls use an LLM."""

    def __init__(self, tools: dict[str, Callable[..., Any]]) -> None:
        self.tools = dict(tools)

    def execute(self, action: ToolAction) -> str:
        if action.tool_name not in self.tools:
            raise KeyError(f"tool not allowed: {action.tool_name}")
        result = self.tools[action.tool_name](**action.arguments)
        return (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )


def parse_tool_action(output: str) -> ActionParseResult:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        return ActionParseResult(None, "invalid_json", str(error))
    if not isinstance(value, dict):
        return ActionParseResult(None, "invalid_arguments", "top-level JSON is not an object")
    tool = value.get("tool") or value.get("tool_name")
    if not tool:
        if "arguments" in value:
            return ActionParseResult(None, "missing_tool_name", "tool/tool_name is missing")
        return ActionParseResult(None, "no_tool_requested", None)
    arguments = value.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ActionParseResult(None, "invalid_arguments", "arguments is not an object")
    return ActionParseResult(ToolAction(str(tool), dict(arguments)), "valid_tool_action")


def default_action_parser(output: str) -> ToolAction | None:
    return parse_tool_action(output).action


class NOCPlannerToolLoop:
    def __init__(
        self,
        *,
        context_manager: IncrementalContextManager,
        router: AgentRouter,
        adapter: NOCSGLangAdapter,
        tool_executor: DirectToolExecutor,
        artifacts_dir: str | Path,
        metrics: NOCAgentMetricsRecorder | None = None,
        action_parser: Callable[[str], ToolAction | None] = default_action_parser,
    ) -> None:
        self.context_manager = context_manager
        self.router = router
        self.adapter = adapter
        self.tool_executor = tool_executor
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
        self.action_parser = action_parser

    def _parse_action(self, output: str) -> ActionParseResult:
        if self.action_parser is default_action_parser:
            return parse_tool_action(output)
        parsed = self.action_parser(output)
        if isinstance(parsed, ActionParseResult):
            return parsed
        if isinstance(parsed, ToolAction):
            return ActionParseResult(parsed, "valid_tool_action")
        diagnostic = parse_tool_action(output)
        return ActionParseResult(None, diagnostic.status, diagnostic.error)

    @staticmethod
    def _termination_for_parse(result: ActionParseResult, output: str) -> str | None:
        if result.action is not None:
            return None
        if result.status == "no_tool_requested":
            lowered = output.lower()
            if any(marker in lowered for marker in ('"final"', '"task_complete"', "final answer")):
                return "explicit_final"
            return "completed_no_action"
        if result.status == "invalid_json":
            stripped = output.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                return "parser_error"
            return "completed_no_action"
        if result.status in {"missing_tool_name", "invalid_arguments"}:
            return "invalid_action"
        return "runtime_error"

    def _write_round(
        self,
        index: int,
        payload: dict[str, Any],
    ) -> Path:
        path = self.artifacts_dir / f"round_{index:02d}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def run(
        self,
        *,
        objective: str,
        initial_events: list[tuple[EventType | str, str, str, dict[str, Any]]],
        max_rounds: int = 2,
    ) -> list[AgentRoundResult]:
        latest_ingest = self.context_manager.ingest(initial_events)
        results: list[AgentRoundResult] = []
        plan_step = objective
        for index in range(1, max_rounds + 1):
            stage = "planner" if index == 1 else "replanner"
            timing = HostTimingRecorder(
                "noc_agent_step",
                metadata={
                    "task_id": self.context_manager.task_id,
                    "phase": stage,
                },
            )
            with profile_range(
                "NOC::agent_step",
                metadata={"task_id": self.context_manager.task_id, "phase": stage},
            ):
                agent = self.router.route(
                    task_id=self.context_manager.task_id,
                    loop_stage=stage,
                    objective=objective,
                )
                selection_started = time.perf_counter()
                with profile_range(
                    "NOC::context_selection",
                    metadata={"task_id": self.context_manager.task_id, "phase": stage},
                ), timing.span("context_selection"):
                    manifest, candidates = self.context_manager.select(
                        agent_id=agent.agent_id,
                        loop_stage=stage,
                        objective=objective,
                        plan_step=plan_step,
                        available_tools=agent.tool_permissions,
                        token_budget=agent.token_budget,
                        mutable_text=f"Current {stage} objective: {plan_step}",
                    )
                selection_latency = time.perf_counter() - selection_started
            timing.metadata.update(
                {
                    "request_id": manifest.request_id,
                    "variant": "selected_context_radixcache_legolink",
                }
            )
            agent.selected_context_manifest = manifest
            with profile_range(
                "NOC::request_planning", metadata=timing.metadata
            ), timing.span("request_planning"):
                execution_plan = self.adapter.planner.plan(
                    manifest,
                    candidates,
                    expected_future_reuse=max_rounds - index + 1,
                )
            with profile_range(
                "NOC::context_stitching", metadata=timing.metadata
            ), timing.span("context_stitching"):
                stitched = self.adapter.stitcher.stitch(
                    execution_plan,
                    generation_parameters=self.adapter.generation,
                )
            with profile_range(
                "NOC::compile_policy", metadata=timing.metadata
            ), timing.span("compile_policy"):
                execution_plan, compile_requests, compile_latency = (
                    self.adapter.compile_plan(
                        execution_plan,
                        stitched=stitched,
                    )
                )
            with profile_range(
                "NOC::context_stitching", metadata=timing.metadata
            ), timing.span("context_stitching"):
                stitched = self.adapter.stitcher.stitch(
                    execution_plan,
                    generation_parameters=self.adapter.generation,
                )
            decisions = execution_plan.decisions
            started = time.perf_counter()
            request_region = (
                "NOC::planner_request" if index == 1 else "NOC::replanner_request"
            )
            with profile_range(request_region, metadata=timing.metadata), profile_range(
                "NOC::sglang_adapter", metadata=timing.metadata
            ), timing.span("llm_request"):
                response = self.adapter.link(
                    manifest,
                    candidates,
                    execution_plan=execution_plan,
                    stitched=stitched,
                )
            request_latency = time.perf_counter() - started
            parse_result = self._parse_action(response.content)
            action = parse_result.action
            termination_reason = self._termination_for_parse(
                parse_result, response.content
            )
            tool_result: str | None = None
            tool_latency = 0.0
            if action is not None:
                tool_started = time.perf_counter()
                try:
                    with profile_range(
                        "NOC::tool_execution", metadata=timing.metadata
                    ), timing.span("tool_execution"):
                        tool_result = self.tool_executor.execute(action)
                except Exception as error:
                    tool_latency = time.perf_counter() - tool_started
                    tool_result = f"{type(error).__name__}: {error}"
                    termination_reason = "tool_error"
                    raise
                tool_latency = time.perf_counter() - tool_started
                next_ingest = self.context_manager.ingest(
                    [
                        (
                            EventType.TOOL_CALL,
                            f"tool-call-{index}",
                            json.dumps(asdict(action), ensure_ascii=False),
                            {
                                "tool_call_id": f"tool-call-{index}",
                                "tool_name": action.tool_name,
                            },
                        ),
                        (
                            EventType.TOOL_OUTPUT,
                            f"tool-output-{index}",
                            tool_result,
                            {
                                "tool_call_id": f"tool-call-{index}",
                                "tool_name": action.tool_name,
                            },
                        ),
                    ]
                )
                plan_step = (
                    f"Use the latest {action.tool_name} result to continue {objective}"
                )
            else:
                plan_step = response.content
            artifact = {
                "round": index,
                "session_revision": self.context_manager.collector.revision,
                "loop_stage": stage,
                "candidate_blocks": [value.to_dict() for value in candidates],
                "selected_blocks": list(manifest.selected_sub_context_ids),
                "excluded_blocks": manifest.excluded_reason,
                "compile_actions": [
                    {
                        "sub_context_id": value.sub_context_id,
                        "action": value.action.value,
                        "reason": value.reason,
                    }
                    for value in decisions
                ],
                "request_execution_plan": execution_plan.to_dict(),
                "stitch_equivalence": stitched.equivalence.to_dict(),
                "link_request": response.request_payload,
                "mutable_suffix": manifest.mutable_text,
                "fallback_reason": response.fallback_reason,
                "llm_output": response.content,
                "parse_status": parse_result.status,
                "parse_error": parse_result.error,
                "tool_action": asdict(action) if action else None,
                "tool_result": tool_result,
                "request_latency": request_latency,
                "request_latency_ms": request_latency * 1000,
                "tool_latency": tool_latency,
                "tool_latency_ms": tool_latency * 1000,
                "adapter_route": response.route,
                "termination_candidate": termination_reason,
                "manifest": manifest.to_dict(),
            }
            path = self._write_round(index, artifact)
            if self.metrics:
                usage = response.response.get("usage") or {}
                compiled = [
                    value
                    for value in decisions
                    if value.action == CompileAction.COMPILE
                ]
                compiled_tokens = sum(
                    next(
                        item.token_count
                        for item in candidates
                        if item.sub_context_id == value.sub_context_id
                    )
                    for value in compiled
                )
                reused_tokens = usage.get("reused_tokens")
                recomputed_tokens = usage.get("recomputed_tokens") or usage.get(
                    "recompute_tokens"
                )
                metric = NOCAgentMetric(
                    task_id=manifest.task_id,
                    agent_id=manifest.agent_id,
                    request_id=manifest.request_id,
                    session_revision=manifest.session_revision,
                    loop_stage=stage,
                    objective=objective,
                    variant=response.route,
                    operation_mode=self.adapter.mode.value,
                    full_session_tokens=manifest.full_session_tokens,
                    candidate_context_tokens=manifest.candidate_context_tokens,
                    selected_context_tokens=manifest.selected_context_tokens,
                    selected_sub_context_count=len(manifest.selected_sub_context_ids),
                    excluded_sub_context_count=len(manifest.excluded_reason),
                    mutable_tokens=sum(
                        item.token_count
                        for item in candidates
                        if item.sub_context_id in manifest.selected_sub_context_ids
                        and item.is_mutable
                    ),
                    segmentation_model=self.context_manager.segment_pipeline.segmenter.model_name,
                    segmentation_latency_ms=latest_ingest.segmentation_latency_s * 1000,
                    segment_count=len(latest_ingest.segmentation.sub_contexts),
                    validation_status=latest_ingest.segmentation.report.status.value,
                    repair_count=latest_ingest.segmentation.report.repair_count,
                    segmentation_fallback_used=latest_ingest.segmentation.report.fallback_used,
                    selection_model=type(self.context_manager.selector).__name__,
                    selection_latency_ms=selection_latency * 1000,
                    token_budget=agent.token_budget,
                    selected_ids=list(manifest.selected_sub_context_ids),
                    excluded_ids=list(manifest.excluded_reason),
                    compile_decisions=[
                        {
                            "id": value.sub_context_id,
                            "action": value.action.value,
                            "reason": value.reason,
                        }
                        for value in decisions
                    ],
                    compile_requests=compile_requests,
                    compiled_blocks=len(compiled),
                    compiled_tokens=compiled_tokens,
                    compile_latency_ms=compile_latency * 1000,
                    duplicate_compile_avoided=response.duplicate_compile_avoided,
                    requested_cache_keys=list(manifest.selected_cache_keys),
                    cache_hits=max(0, len(compiled) - compile_requests),
                    cache_misses=compile_requests,
                    reused_tokens=int(reused_tokens)
                    if reused_tokens is not None
                    else None,
                    recomputed_tokens=int(recomputed_tokens)
                    if recomputed_tokens is not None
                    else None,
                    fallback_reason=response.fallback_reason,
                    link_latency_ms=(
                        response.request_latency_s * 1000
                        if response.route == "selected_context_radixcache_legolink"
                        else None
                    ),
                    native_prefill_latency_ms=(
                        response.request_latency_s * 1000
                        if response.route == "selected_context_original_sglang"
                        else None
                    ),
                    request_latency_ms=request_latency * 1000,
                    prompt_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    agent_steps=1,
                    llm_calls=1,
                    tool_calls=int(action is not None),
                    tool_execution_time_ms=tool_latency * 1000,
                    e2e_latency_ms=(request_latency + tool_latency) * 1000,
                )
                self.metrics.record(metric)
            results.append(
                AgentRoundResult(
                    round_index=index,
                    loop_stage=stage,
                    llm_output=response.content,
                    tool_action=action,
                    tool_result=tool_result,
                    adapter_route=response.route,
                    manifest_path=str(path),
                    parse_status=parse_result.status,
                    termination_reason=termination_reason,
                )
            )
            if action is not None:
                latest_ingest = next_ingest
            elif termination_reason is not None:
                timing.emit()
                break
            timing.emit()
        return results
