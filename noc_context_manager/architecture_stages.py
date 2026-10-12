"""The 13-stage architecture, mapped one-to-one onto canonical source files.

This module is *data*. It imports nothing from the pipeline and is read by no
runtime path; ``tests/test_architecture_stage_map.py`` checks every entry against
the source (the file exists, the symbol is defined there, the stage markers are
present, every recorded call really appears in its caller), and
``docs/architecture_stage_map.md`` renders it. If the code moves, the test fails
here rather than the diagram drifting silently.

Two conventions:

* ``canonical_file`` is relative to the repository root for host stages and to
  ``external/sglang/python/sglang/srt/`` for runtime stages (``side == "runtime"``).
* ``CALL_EDGES`` records calls that exist, not calls the diagram implies. The
  diagram's left-to-right order is a *data* order; where the code nests a stage
  inside another (Stage 6 calls Stage 5, Stage 8 calls Stage 7, Stage 9 calls
  Stage 8) the edge says so.
"""

from __future__ import annotations

from dataclasses import dataclass

RUNTIME_ROOT = "external/sglang/python/sglang/srt"

HOST = "host"
RUNTIME = "runtime"


@dataclass(frozen=True)
class ArchitectureStage:
    step: int
    name: str
    side: str
    canonical_file: str
    canonical_symbol: str
    helpers: tuple[str, ...] = ()
    note: str = ""


STAGES: tuple[ArchitectureStage, ...] = (
    ArchitectureStage(
        1, "User / Agent Layer", HOST,
        "terminal_bench_integration/noc_sglang_agent.py",
        "NOCSGLangLLM.call",
        (
            "terminal_bench_integration/sglang_pic_agent.py",
            "terminal_bench_integration/noc_openai_proxy.py",
            "terminal_bench_integration/proxy_sessions.py",
            "noc_context_manager/agent_loop.py",
            "noc_context_manager/agents.py",
            "noc_context_manager/real_ingress.py",
        ),
        "The OpenAI proxy constructs NOCSGLangLLM and calls the same method. real_ingress.py holds the "
        "deterministic Step 1 adapters (OpenAI messages / tools / tool calls / tool results, Terminus turns).",
    ),
    ArchitectureStage(
        2, "Session Events + Field Extraction", HOST,
        "noc_context_manager/session.py",
        "SessionFieldPipeline",
        ("noc_context_manager/tokenizer.py", "noc_context_manager/field_classification.py"),
        "Composes SessionEventCollector and FieldFirstExtractor; field_classification.py "
        "holds the deterministic / optional-Jev classifiers the extractor consults.",
    ),
    ArchitectureStage(
        3, "Semantic Segmentation", HOST,
        "noc_context_manager/semantic_segmentation.py",
        "SemanticSegmentationPipeline",
        (
            "noc_context_manager/segmentation_validation.py",
            "noc_context_manager/atomic_units.py",
            "noc_context_manager/segmentation_prompts.py",
            "noc_context_manager/current_llm_segmenter.py",
            "noc_context_manager/current_llm_guardrails.py",
            "terminal_bench_integration/frozen_segmentation.py",
        ),
        "Defined in segmentation_validation.py; exposed here by a lazy "
        "module __getattr__ (the validator imports this module).",
    ),
    ArchitectureStage(
        4, "Context Selection + SubContextStore", HOST,
        "noc_context_manager/context_selection.py",
        "ContextSelectionStage",
        (
            "noc_context_manager/manager.py",
            "noc_context_manager/subcontext_store.py",
            "noc_context_manager/reuse_eligibility.py",
        ),
    ),
    ArchitectureStage(
        5, "Compile Policy + Link Feasibility", HOST,
        "noc_context_manager/compile_policy.py",
        "CompileAndLinkFeasibilityPlanner",
        (
            "noc_context_manager/link_layout.py",
            "noc_context_manager/compile_cost.py",
        ),
    ),
    ArchitectureStage(
        6, "Logical Context Assembly + Reuse Plan", HOST,
        "noc_context_manager/scheduler.py",
        "NOCRequestPlanner.plan",
        (
            "noc_context_manager/stitch.py",
            "noc_context_manager/reuse_eligibility.py",
            "noc_context_manager/reuse_cost_model.py",
            "noc_context_manager/cache_namespace.py",
            "noc_context_manager/semantic_cache_index.py",
        ),
        "Namespace-first lookup (cache_namespace.NamespaceResolver) runs at the top of plan(), over the logical "
        "roots in semantic_cache_index (one root per category, one shared physical cache).",
    ),
    ArchitectureStage(
        7, "Repair Policy / K Budget", HOST,
        "noc_context_manager/repair_window.py",
        "RepairWindowPlanner",
        ("noc_context_manager/group_repair_profile.py",),
        "NOT_IN_CURRENT_BASELINE: legolink_repair_policy.py "
        "(branch fix/legolink-policy-accounting, e76cc81).",
    ),
    ArchitectureStage(
        8, "PIC Contract + Logical Request Framing", HOST,
        "noc_context_manager/pic_link_contract.py",
        "LogicalPICRequestFramer.frame",
        (
            "noc_context_manager/physical_framing.py",
            "noc_context_manager/token_accounting.py",
            "noc_context_manager/authoritative_tokens.py",
        ),
    ),
    ArchitectureStage(
        9, "NOCSGLang Adapter", HOST,
        "noc_context_manager/sglang_adapter.py",
        "NOCSGLangAdapter.link",
        (
            "noc_context_manager/pipeline.py",
            "noc_context_manager/chat_transport.py",
            "noc_context_manager/sglang_bridge.py",
        ),
        "pipeline.py orchestrates; sglang_bridge.py is the legacy plan-file "
        "bridge, not this path.",
    ),
    ArchitectureStage(
        10, "SGLang Scheduler / ModelRunner", RUNTIME,
        "managers/scheduler.py",
        "Scheduler.handle_generate_request",
        ("model_executor/model_runner.py", "entrypoints/openai/serving_chat.py"),
        "Admission validates the pic_link request; ModelRunner dispatch is a "
        "dependency, not a second implementation.",
    ),
    ArchitectureStage(
        11, "RadixCache + Semantic Block Registry", RUNTIME,
        "mem_cache/radix_cache.py",
        "RadixCache.lookup_legolink_blocks",
        (),
        "11a native RadixCache (RadixCache.match_prefix) and 11b the LegoLink "
        "block registry (register/lookup/acquire/release_legolink_block[s]) "
        "share one class; 11b is the modified part.",
    ),
    ArchitectureStage(
        12, "PIC / LegoLink Runtime", RUNTIME,
        "pic_runtime.py",
        "PICRuntime.run_link",
        (
            "legolink_triton.py",
            "flashinfer_sparse_repair.py",
            "sparse_repair_compute.py",
        ),
        "PICRequestTokenRegistry._prepare_link builds the ResolvedLinkPlan.",
    ),
    ArchitectureStage(
        13, "Native Decode", RUNTIME,
        "model_executor/model_runner.py",
        "ModelRunner.forward",
        ("managers/tp_worker.py",),
        "Unmodified upstream control flow after run_link returns: "
        "ModelRunner.forward -> TpModelWorker.forward_batch_generation -> "
        "ModelRunner.sample, then ModelRunner._forward_raw -> forward_decode.",
    ),
)


@dataclass(frozen=True)
class CallEdge:
    """``caller`` really calls something named ``callee`` in its body."""

    from_step: int
    to_step: int
    side: str
    caller_file: str
    caller: str
    callee: str
    note: str = ""


CALL_EDGES: tuple[CallEdge, ...] = (
    # 1 -> 2 : Stage 1 hands the turn to the pipeline, whose Stage 4 ingest
    # calls into Stage 2. Stage 1 never calls Stage 2 directly.
    CallEdge(1, 2, HOST, "terminal_bench_integration/noc_sglang_agent.py",
             "NOCSGLangLLM.call", "execute"),
    CallEdge(1, 2, HOST, "noc_context_manager/pipeline.py",
             "NOCRequestPipeline.execute", "ingest"),
    CallEdge(1, 2, HOST, "noc_context_manager/context_selection.py",
             "ContextSelectionStage.ingest", "ingest"),
    CallEdge(1, 2, HOST, "noc_context_manager/manager.py",
             "IncrementalContextManager.ingest", "collect"),
    # 2 -> 3
    CallEdge(2, 3, HOST, "noc_context_manager/manager.py",
             "IncrementalContextManager._ingest_accepted", "extract"),
    CallEdge(2, 3, HOST, "noc_context_manager/manager.py",
             "IncrementalContextManager._ingest_accepted", "run"),
    # 3 -> 4 : validated blocks are stored, then the same stage selects.
    CallEdge(3, 4, HOST, "noc_context_manager/manager.py",
             "IncrementalContextManager._ingest_accepted", "replace_source_revisions"),
    CallEdge(3, 4, HOST, "noc_context_manager/pipeline.py",
             "NOCRequestPipeline.execute", "select"),
    CallEdge(3, 4, HOST, "noc_context_manager/manager.py",
             "IncrementalContextManager.select", "select_for_request"),
    # 4 -> 5 : the pipeline asks Stage 6 to plan; Stage 6 calls Stage 5.
    CallEdge(4, 5, HOST, "noc_context_manager/pipeline.py",
             "NOCRequestPipeline.execute", "plan"),
    # 5 -> 6 : nested -- Stage 6 is the caller.
    CallEdge(5, 6, HOST, "noc_context_manager/scheduler.py",
             "NOCRequestPlanner.plan", "provisional_decisions",
             "Stage 6 calls Stage 5"),
    CallEdge(5, 6, HOST, "noc_context_manager/scheduler.py",
             "NOCRequestPlanner._assemble", "apply_feasibility",
             "Stage 6 calls Stage 5 after Gate 1 (_gate)"),
    CallEdge(5, 6, HOST, "noc_context_manager/pipeline.py",
             "NOCRequestPipeline.execute", "stitch",
             "the plan is then stitched (ContextStitcher)"),
    # 6 -> 7 : no direct call. Stage 7 is asked at Link time from Stage 8.
    # 7 -> 8 : nested -- Stage 8 is the caller.
    CallEdge(7, 8, HOST, "noc_context_manager/pic_link_contract.py",
             "LogicalPICRequestFramer.frame", "plan",
             "Stage 8 calls Stage 7 (RepairWindowPlanner.plan)"),
    CallEdge(7, 8, HOST, "noc_context_manager/pic_link_contract.py",
             "LogicalPICRequestFramer.frame", "build_pic_link_request"),
    # 8 -> 9 : nested -- Stage 9 is the caller.
    CallEdge(8, 9, HOST, "noc_context_manager/sglang_adapter.py",
             "NOCSGLangAdapter._link_payload", "frame",
             "Stage 9 calls Stage 8"),
    CallEdge(8, 9, HOST, "noc_context_manager/sglang_adapter.py",
             "NOCSGLangAdapter.link", "_link_payload"),
    CallEdge(8, 9, HOST, "noc_context_manager/pipeline.py",
             "NOCRequestPipeline.execute", "link"),
    # 9 -> 10 : an HTTP request to /v1/chat/completions with `pic_link`.
    CallEdge(9, 10, HOST, "noc_context_manager/sglang_adapter.py",
             "NOCSGLangAdapter.link", "patched_transport",
             "JSONHTTPTransport.__call__ -> chat_transport.post_chat"),
    CallEdge(9, 10, RUNTIME, "entrypoints/openai/serving_chat.py",
             "OpenAIServingChat._convert_to_internal_request", "getattr",
             "pic_link is copied into sampling_params.custom_params"),
    # 10 -> 12 -> 11 : admission and link-source reservation go through the
    # PIC registry, which looks the blocks up in RadixCache (Stage 11b).
    CallEdge(10, 11, RUNTIME, "managers/scheduler.py",
             "Scheduler.handle_generate_request", "prepare_request"),
    CallEdge(10, 11, RUNTIME, "managers/scheduler.py",
             "Scheduler._get_new_batch_prefill_raw", "reserve_link_sources"),
    CallEdge(10, 11, RUNTIME, "pic_runtime.py",
             "PICRequestTokenRegistry.reserve_link_sources", "lookup_legolink_blocks",
             "Stage 12's registry calls Stage 11b"),
    # 11 -> 12 : ModelRunner dispatches the batch into PICRuntime.
    CallEdge(11, 12, RUNTIME, "model_executor/model_runner.py",
             "ModelRunner.forward", "run_link"),
    CallEdge(11, 12, RUNTIME, "pic_runtime.py",
             "PICRuntime._copy_linked_kv", "acquire_legolink_blocks",
             "Stage 12 calls Stage 11b"),
    # 12 -> 13 : run_link returns logits; control is upstream SGLang's.
    CallEdge(12, 13, RUNTIME, "model_executor/model_runner.py",
             "ModelRunner.forward", "ModelRunnerOutput",
             "return ModelRunnerOutput(logits_output=run_link(...))"),
    CallEdge(12, 13, RUNTIME, "managers/tp_worker.py",
             "TpModelWorker.forward_batch_generation", "sample"),
    CallEdge(12, 13, RUNTIME, "model_executor/model_runner.py",
             "ModelRunner._forward_raw", "forward_decode"),
)


def stage(step: int) -> ArchitectureStage:
    for value in STAGES:
        if value.step == step:
            return value
    raise KeyError(step)
