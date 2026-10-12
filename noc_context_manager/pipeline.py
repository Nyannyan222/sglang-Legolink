"""End-to-end facade for NOC context preparation and request execution.

This module orchestrates Stages 4, 6 and 9; it owns none of their semantics.
Stage 4 is entered through ``ContextSelectionStage`` (context_selection.py),
Stage 6 through ``NOCRequestPlanner.plan`` (scheduler.py, reached as
``adapter.planner``) and Stage 9 through ``NOCSGLangAdapter``
(sglang_adapter.py). Do not add planning, compile-decision or payload logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .compile_policy import CompileDecision
from .context_selection import AgentContextManifest, ContextSelectionStage
from .manager import IncrementalContextManager, IngestResult
from .profiling import HostTimingRecorder, profile_range
from .scheduler import NOCRequestExecutionPlan
from .session import EventType
from .sglang_adapter import AdapterResult, NOCSGLangAdapter
from .stitch import StitchedContext
from .subcontext_store import SemanticSubContext


class PipelineRoute(str, Enum):
    FULL_CONTEXT_ORIGINAL = "full_context_original_sglang"
    SELECTED_CONTEXT_ORIGINAL = "selected_context_original_sglang"
    SELECTED_CONTEXT_LEGOLINK = "selected_context_radixcache_legolink"


@dataclass(frozen=True)
class PipelineExecution:
    route: PipelineRoute
    adapter_result: AdapterResult
    ingest_result: IngestResult | None = None
    manifest: AgentContextManifest | None = None
    contexts: tuple[SemanticSubContext, ...] = ()
    execution_plan: NOCRequestExecutionPlan | None = None
    stitched: StitchedContext | None = None
    compile_decisions: tuple[CompileDecision, ...] = ()
    compile_requests: int = 0
    compile_latency_s: float = 0.0

    @property
    def schedule(self) -> NOCRequestExecutionPlan | None:
        """Deprecated compatibility view of :attr:`execution_plan`."""

        return self.execution_plan


class NOCRequestPipeline:
    """Coordinate the modules while keeping each module independently testable."""

    def __init__(
        self,
        *,
        context_manager: IncrementalContextManager,
        adapter: NOCSGLangAdapter,
    ) -> None:
        self.context_manager = context_manager
        self.adapter = adapter

    def execute(
        self,
        events: list[tuple[EventType | str, str, str, dict[str, Any]]],
        *,
        route: PipelineRoute | str,
        agent_id: str,
        loop_stage: str,
        objective: str,
        plan_step: str,
        available_tools: tuple[str, ...],
        token_budget: int,
        mutable_text: str = "",
        expected_future_reuse: int = 1,
        memory_pressure: float = 0.0,
    ) -> PipelineExecution:
        route = PipelineRoute(route)
        timing = HostTimingRecorder(
            "noc_request_pipeline",
            metadata={
                "task_id": self.context_manager.task_id,
                "variant": route.value,
                "phase": loop_stage,
            },
        )
        status = "ok"
        error_message = None
        try:
            if route == PipelineRoute.FULL_CONTEXT_ORIGINAL:
                self.context_manager.collector.add_events(events)
                full_text = "\n\n".join(
                    event.raw_text for event in self.context_manager.collector.events()
                )
                with profile_range(
                    "NOC::sglang_adapter",
                    metadata={"variant": route.value, "phase": loop_stage},
                ), timing.span("sglang_adapter"):
                    result = self.adapter.native_full(full_text)
                return PipelineExecution(route=route, adapter_result=result)

            with profile_range(
                "NOC::context_selection",
                metadata={"variant": route.value, "phase": loop_stage},
            ):
                selection_stage = ContextSelectionStage(self.context_manager)
                ingest_result = selection_stage.ingest(events)
                with timing.span("context_selection"):
                    manifest, context_list = selection_stage.select(
                        agent_id=agent_id,
                        loop_stage=loop_stage,
                        objective=objective,
                        plan_step=plan_step,
                        available_tools=available_tools,
                        token_budget=token_budget,
                        mutable_text=mutable_text,
                    )
            metadata = {
                "request_id": manifest.request_id,
                "task_id": manifest.task_id,
                "variant": route.value,
                "phase": loop_stage,
            }
            timing.metadata.update(metadata)
            contexts = tuple(context_list)
            with profile_range("NOC::request_planning", metadata=metadata), timing.span(
                "request_planning"
            ):
                execution_plan = self.adapter.planner.plan(
                    manifest,
                    context_list,
                    expected_future_reuse=expected_future_reuse,
                    memory_pressure=memory_pressure,
                )
            with profile_range("NOC::context_stitching", metadata=metadata), timing.span(
                "context_stitching"
            ):
                stitched = self.adapter.stitcher.stitch(
                    execution_plan,
                    generation_parameters=self.adapter.generation,
                )
            if route == PipelineRoute.SELECTED_CONTEXT_ORIGINAL:
                with profile_range("NOC::sglang_adapter", metadata=metadata), timing.span(
                    "sglang_adapter"
                ):
                    result = self.adapter.native_selected(
                        manifest,
                        context_list,
                        plan=execution_plan,
                        stitched=stitched,
                    )
                return PipelineExecution(
                    route=route,
                    adapter_result=result,
                    ingest_result=ingest_result,
                    manifest=manifest,
                    contexts=contexts,
                    execution_plan=execution_plan,
                    stitched=stitched,
                    compile_decisions=execution_plan.decisions,
                )

            with profile_range("NOC::compile_policy", metadata=metadata), timing.span(
                "compile_policy"
            ):
                execution_plan, compile_requests, compile_latency = (
                    self.adapter.compile_plan(execution_plan, stitched=stitched)
                )
            with profile_range("NOC::context_stitching", metadata=metadata), timing.span(
                "context_stitching"
            ):
                stitched = self.adapter.stitcher.stitch(
                    execution_plan,
                    generation_parameters=self.adapter.generation,
                )
            with profile_range("NOC::sglang_adapter", metadata=metadata), timing.span(
                "sglang_adapter"
            ):
                result = self.adapter.link(
                    manifest,
                    context_list,
                    execution_plan=execution_plan,
                    stitched=stitched,
                )
            return PipelineExecution(
                route=route,
                adapter_result=result,
                ingest_result=ingest_result,
                manifest=manifest,
                contexts=contexts,
                execution_plan=execution_plan,
                stitched=stitched,
                compile_decisions=execution_plan.decisions,
                compile_requests=compile_requests,
                compile_latency_s=compile_latency,
            )
        except Exception as error:
            status = "error"
            error_message = f"{type(error).__name__}: {error}"
            raise
        finally:
            timing.emit(status=status, error=error_message)
