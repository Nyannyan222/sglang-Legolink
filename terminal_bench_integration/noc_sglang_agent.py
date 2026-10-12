"""Official Terminal-Bench agent using incremental NOC context management.

ARCHITECTURE_STAGE: 1 -- User / Agent Layer
CANONICAL_ENTRYPOINT: NOCSGLangLLM.call (builds the Stage 2-4 collaborators in
    ``set_task_context`` and hands each turn to ``NOCRequestPipeline.execute``);
    ``NOCSGLangAgent`` / ``NOCSGLangTerminus2Agent`` are the Terminal-Bench
    agent classes that install it.
DELEGATED_HELPERS: sglang_pic_agent.py (base agent/LLM classes),
    noc_openai_proxy.py (an OpenAI-compatible HTTP ingress that constructs
    ``NOCSGLangLLM`` and calls the same ``call``; it has no pipeline of its own),
    proxy_sessions.py, noc_context_manager/agent_loop.py, agents.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from terminal_bench.llms.base_llm import OutputLengthExceededError

from noc_context_manager.cache_namespace import (
    CROSS_NAMESPACE_PAIR_POLICY_VERSION,
    CacheNamespacePolicy,
    NamespaceMode,
    parse_cross_namespace_pairs,
)
from noc_context_manager.compile_policy import (
    CacheCatalog,
    CompileAllImmutablePolicy,
    CompilePolicy,
    CompilePolicyMode,
    build_compile_policy,
)
from noc_context_manager.compile_cost import (
    CompileCostProfile,
    DeploymentIdentity,
    PlannedTurnsReuseExpectation,
)
from noc_context_manager.context_selection import (
    HybridSemanticContextSelector,
    LLMMetadataSemanticContextSelector,
    RuleBasedSemanticContextSelector,
)
from noc_context_manager.manager import (
    IncrementalContextManager,
    TokenAccountingMode,
)
from noc_context_manager.pipeline import NOCRequestPipeline, PipelineRoute
from noc_context_manager.semantic_segmentation import (
    DeterministicFieldSegmenter,
    CurrentLLMSegmenter,
    SemanticSegmentationPipeline,
    SemanticSegmentationValidator,
)
from noc_context_manager.persistent_catalog import PersistentCacheCatalog
from noc_context_manager.real_ingress import (
    PromptPart,
    PromptRegistry,
    SourceLedger,
    openai_request_events,
    terminus_turn_events,
)
from noc_context_manager.segmentation_prompts import resolve_prompt_variant
from noc_context_manager.session import EventType, SessionEventCollector
from noc_context_manager.sglang_adapter import (
    JSONHTTPTransport,
    NOCSGLangAdapter,
    OperationMode,
)
from noc_context_manager.subcontext_store import SQLiteSubContextStore
from noc_context_manager.tokenizer import (
    RegexTokenizer,
    TextTokenizer,
    load_text_tokenizer,
)

from .sglang_pic_agent import (
    SGLangPICAgent,
    SGLangPICLLM,
    SGLangPICTerminus2Agent,
    _message_value,
)


def _as_bool(value: Any, name: str) -> bool:
    """Parse a flag that may arrive as a string from the Terminal-Bench agent-kwarg surface.

    `bool("False")` is True, which is exactly the way an experiment switch silently stops switching, so
    strings are parsed rather than truthiness-tested and anything unrecognised is refused.
    """

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _coverage_summary(records: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if records is None:
        return None
    statuses: dict[str, int] = {}
    for row in records:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "sources": len(records),
        "statuses": statuses,
        "uncovered": [
            {key: row.get(key) for key in ("source_id", "session_revision", "context_field", "status")}
            for row in records
            if row["status"] not in ("covered", "empty", "excluded_by_policy")
        ],
    }


def _output_token_ids(response: dict[str, Any]) -> list[int] | None:
    """Generated token ids, when the serving response exposes them; None when it does not."""

    choice = (response.get("choices") or [{}])[0]
    for key in ("token_ids", "output_ids"):
        if isinstance(choice.get(key), list):
            return [int(value) for value in choice[key]]
    for meta in (choice.get("meta_info"), response.get("meta_info")):
        if isinstance(meta, dict) and isinstance(meta.get("output_ids"), list):
            return [int(value) for value in meta["output_ids"]]
    return None


def _order_key_ties(contexts: list[Any], selected_ids: Any) -> int:
    """How many selected blocks share (session_revision, created_at) with another."""

    selected = set(selected_ids)
    keys: dict[tuple[Any, Any], int] = {}
    for value in contexts:
        if value.sub_context_id in selected:
            key = (value.session_revision, value.created_at)
            keys[key] = keys.get(key, 0) + 1
    return sum(count for count in keys.values() if count > 1)


class NOCSGLangLLM(SGLangPICLLM):
    """Drop-in Terminus LLM that routes every turn through the NOC pipeline."""

    #: ``call`` accepts ``openai_messages`` / ``tools`` / ``functions`` (the proxy checks this before sending them).
    structured_ingress = True

    def __init__(
        self,
        *,
        variant: str = "selected_context_radixcache_legolink",
        original_api_base: str | None = None,
        operation_mode: str = "strict_experimental",
        segmentation_mode: str = "deterministic",
        segmentation_api_base: str | None = None,
        segmentation_model: str | None = None,
        semantic_classification_enabled: bool | str = True,
        require_llm_segmentation: bool | str = False,
        segmentation_prompt_variant: str | None = None,
        shared_cache_catalog: CacheCatalog | None = None,
        shared_catalog_path: str | None = None,
        semantic_cache_namespace_mode: str = "global",
        global_physical_dedup: bool | str = True,
        cross_namespace_pairs: Sequence[str] | str | None = None,
        repetition_penalty: float | str | None = None,
        record_logprobs: bool | str = False,
        preserve_full_source_content: bool = False,
        selection_mode: str = "rule",
        compile_policy_mode: str = "legacy_fixed_64",
        compile_cost_profile: str | None = None,
        deployment_hardware_id: str | None = None,
        compile_reuse_probability: float = 0.5,
        context_token_budget: int = 24_000,
        artifacts_root: str = "benchmark_results/noc_agent_sessions",
        tokenizer_mode: str = "sglang",
        tokenizer_path: str | None = None,
        tokenizer_revision: str | None = None,
        tokenizer_local_files_only: bool = True,
        tokenizer_trust_remote_code: bool = True,
        tokenizer_api_base: str | None = None,
        residency_recovery: bool = False,
        residency_recovery_attempts: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if variant not in {
            "full_context_original_sglang",
            "selected_context_original_sglang",
            "selected_context_radixcache_legolink",
        }:
            raise ValueError(f"unknown NOC variant: {variant}")
        self.variant = variant
        self.operation_mode = OperationMode(operation_mode)
        # Experimental, default off: recompile a source block the runtime
        # reports as no longer resident and resend, instead of failing the turn
        # (results/cache_residency_correctness/).
        self.residency_recovery = bool(residency_recovery)
        self.residency_recovery_attempts = int(residency_recovery_attempts)
        self.context_token_budget = context_token_budget
        self.artifacts_root = Path(artifacts_root)
        self.original_api_base = original_api_base or kwargs["api_base"]
        self._tokenizer_mode = tokenizer_mode
        self._tokenizer: TextTokenizer = load_text_tokenizer(
            self.model_name,
            mode=tokenizer_mode,
            tokenizer_path=tokenizer_path,
            revision=tokenizer_revision,
            local_files_only=tokenizer_local_files_only,
            trust_remote_code=tokenizer_trust_remote_code,
            # `tokenizer_mode="sglang"` tokenizes over HTTP. By default that is the original endpoint, as
            # always; a performance run points it at an auxiliary server so tokenization never loads the
            # server being measured. The SGLang tokenizer's fingerprint does not contain the URL, so cache
            # keys are unchanged.
            api_base=tokenizer_api_base or self.original_api_base,
            api_key=kwargs.get("api_key", ""),
            timeout=kwargs.get("timeout", 1200),
        )
        # Only the two generation transports stream. The segmenter and the
        # selector are helper calls whose first token nobody waits on, and
        # streaming them would add SSE framing to work that is not being timed.
        self._patched_transport = JSONHTTPTransport(
            kwargs["api_base"],
            api_key=kwargs.get("api_key", ""),
            timeout=kwargs.get("timeout", 1200),
            stream=self.stream_metrics,
        )
        self._original_transport = JSONHTTPTransport(
            self.original_api_base,
            api_key=kwargs.get("api_key", ""),
            timeout=kwargs.get("timeout", 1200),
            stream=self.stream_metrics,
        )
        # Recorded on every metrics row as route provenance; read by nothing else.
        self._segmentation_mode = segmentation_mode
        self._requested_segmentation_mode = "llm" if segmentation_mode == "current_llm" else segmentation_mode
        self._require_llm_segmentation = _as_bool(require_llm_segmentation, "require_llm_segmentation")
        # Stage 2 labels segments Stage 1 already fixed, and reaches no cache key, selector, planner or
        # prompt. A formal serving experiment whose treatment is segmentation granularity turns it off so
        # its latency and its failure modes never enter the measurement.
        self._semantic_classification_enabled = _as_bool(
            semantic_classification_enabled, "semantic_classification_enabled"
        )
        # Stage-1 instruction variant for the segmentation-prompt ablation. `None` is P0, the production
        # prompt, byte for byte; only an LLM segmenter reads it.
        self._segmentation_prompt_variant = resolve_prompt_variant(segmentation_prompt_variant)
        # Host bookkeeping of which immutable exact-safe blocks have a standalone Compile. `None` keeps the
        # historical per-session catalog. A caller that runs several agents against one patched runtime
        # may pass one catalog to all of them, so agent B links the block agent A compiled instead of
        # compiling it again. Only compiled block keys are shared: each session keeps its own store,
        # manifest, mutable suffix and history.
        self._shared_cache_catalog = shared_cache_catalog
        if self._shared_cache_catalog is None and shared_catalog_path:
            # Live multi-agent runs: `tb run` builds a new agent per trial, so the catalog of compiled
            # immutable blocks is kept in a file every session shares. Default off (no path).
            self._shared_cache_catalog = PersistentCacheCatalog(shared_catalog_path)
        self._selection_mode = selection_mode
        self._compile_policy_mode = compile_policy_mode
        # Explicit and logged: global | hard_namespace | logical_namespace. Validated
        # here so a typo fails at construction, not in the middle of a run.
        # Sent on every request when set. An omitted sampling parameter is not the same request on the Link
        # and Native paths (they inherit different server defaults), so a comparison run states it.
        self._repetition_penalty = None if repetition_penalty in (None, "") else float(repetition_penalty)
        # Opt-in: ask the server for per-token logprobs and keep them in the turn artifact, so a run can be
        # compared with a Native reference token by token and not only by text.
        self._record_logprobs = _as_bool(record_logprobs, "record_logprobs")
        self._semantic_cache_namespace_mode = NamespaceMode.parse(
            semantic_cache_namespace_mode
        ).value
        # Arm switch of the cache-policy comparison: False restores the field-partitioned registry lookup.
        self._global_physical_dedup = _as_bool(global_physical_dedup, "global_physical_dedup")
        # Q-1: the cross-namespace pairs of the logical_namespace arm. Only vetted pairs are accepted, and a pair given to any
        # other mode is an error rather than a silent no-op (a no-op pair is how two arms became one treatment, F-04).
        if isinstance(cross_namespace_pairs, str):
            cross_namespace_pairs = [part for part in cross_namespace_pairs.replace(";", ",").split(",") if part.strip()]
        self._cross_namespace_pairs = parse_cross_namespace_pairs(cross_namespace_pairs, require_vetted=True)
        if self._cross_namespace_pairs and self._semantic_cache_namespace_mode != NamespaceMode.LOGICAL_NAMESPACE.value:
            raise ValueError(
                f"cross_namespace_pairs only act in logical_namespace mode, not {self._semantic_cache_namespace_mode!r}"
            )
        self._preserve_full_source_content = bool(preserve_full_source_content)
        self._loop_stage: str | None = None
        explicit_segmentation_api_base = (
            segmentation_api_base or os.getenv("NOC_SEGMENTATION_API_BASE")
        )
        if segmentation_mode in {"llm", "current_llm"} and not explicit_segmentation_api_base:
            raise ValueError(
                "LLM segmentation requires an explicit segmentation_api_base "
                "or NOC_SEGMENTATION_API_BASE; generation endpoints are never inherited"
            )
        self._segmenter = self._make_segmenter(
            segmentation_mode,
            explicit_segmentation_api_base or self.original_api_base,
            segmentation_model or os.getenv("NOC_SEGMENTATION_MODEL", self.model_name),
        )
        rule_selector = RuleBasedSemanticContextSelector()
        if selection_mode in {"llm", "hybrid"}:
            selector_transport = JSONHTTPTransport(
                explicit_segmentation_api_base or self.original_api_base,
                api_key=kwargs.get("api_key", ""),
                timeout=kwargs.get("timeout", 1200),
            )
            llm_selector = LLMMetadataSemanticContextSelector(
                selector_transport,
                segmentation_model or self.model_name,
            )
            self._selector = (
                HybridSemanticContextSelector(llm_selector, rule_selector)
                if selection_mode == "hybrid"
                else llm_selector
            )
        elif selection_mode == "rule":
            self._selector = rule_selector
        else:
            raise ValueError("selection_mode must be rule, llm, or hybrid")
        if compile_policy_mode == "compile_all_immutable":
            self._compile_policy = CompileAllImmutablePolicy()
        elif compile_policy_mode == "legacy_fixed_64":
            self._compile_policy = CompilePolicy()
        elif compile_policy_mode == "cost_benefit":
            if not compile_cost_profile:
                raise ValueError(
                    "cost_benefit requires compile_cost_profile; use "
                    "legacy_fixed_64 for the historical fixed-threshold policy"
                )
            if not deployment_hardware_id:
                raise ValueError(
                    "cost_benefit requires deployment_hardware_id so a profile "
                    "from different hardware cannot be used silently"
                )
            profile = CompileCostProfile.from_json(compile_cost_profile)
            deployment = DeploymentIdentity(
                hardware_id=deployment_hardware_id,
                model_id=self.model_name,
            )
            profile.check_compatibility(deployment)
            self._compile_policy = build_compile_policy(
                CompilePolicyMode.COST_BENEFIT,
                reuse_expectation=PlannedTurnsReuseExpectation(
                    probability=float(compile_reuse_probability)
                ),
                profile=profile,
                deployment=deployment,
            )
        else:
            raise ValueError(
                "compile_policy_mode must be legacy_fixed_64, cost_benefit, "
                "or compile_all_immutable"
            )
        self._manager: IncrementalContextManager | None = None
        self._adapter: NOCSGLangAdapter | None = None
        self._pipeline: NOCRequestPipeline | None = None

    def _make_segmenter(self, mode: str, api_base: str, model: str):
        if mode in {"llm", "current_llm"}:
            if not api_base:
                raise ValueError("LLM segmentation requires segmentation_api_base")
            return CurrentLLMSegmenter(
                api_base=api_base,
                model=model,
                api_key=self.api_key,
                timeout=self.timeout,
                cache_dir=self.artifacts_root / "segmentation_cache",
                semantic_classification_enabled=self._semantic_classification_enabled,
                prompt_variant=self._segmentation_prompt_variant,
            )
        if mode != "deterministic":
            raise ValueError("segmentation_mode must be deterministic, llm, or current_llm")
        return DeterministicFieldSegmenter()

    def set_task_context(self, task_id: str, metrics_path: Path | None) -> None:
        super().set_task_context(task_id, metrics_path)
        task_root = self.artifacts_root / task_id / self.session_id
        task_root.mkdir(parents=True, exist_ok=True)
        self._task_root = task_root
        # Step 1 state of this session: how declared prompts were split (to settle them when they return
        # as history), the agent's pending declaration, and the per-call event trace.
        self._prompt_registry = PromptRegistry()
        self._declared_parts: tuple[PromptPart, ...] | None = None
        self._last_ingress: dict[str, Any] = {}
        self._session_events_path = task_root / "session_events.jsonl"
        self._manager = IncrementalContextManager(
            task_id=task_id,
            collector=SessionEventCollector(task_id),
            segment_pipeline=SemanticSegmentationPipeline(
                self._segmenter,
                validator=SemanticSegmentationValidator(
                    tokenizer=self._tokenizer,
                ),
                preliminary_tokenizer=RegexTokenizer(),
                report_dir=task_root / "validation",
            ),
            store=SQLiteSubContextStore(task_root / "subcontexts.sqlite3"),
            selector=self._selector,
            preliminary_tokenizer=RegexTokenizer(),
            authoritative_tokenizer=self._tokenizer,
            model_fingerprint=f"model:{self.model_name}",
            preserve_full_source_content=self._preserve_full_source_content,
        )
        if self._require_llm_segmentation or os.getenv("NOC_AGENT_REQUIRE_LLM_SEGMENTATION") == "1":
            from .segmentation_evidence import install_experiment_gate
            install_experiment_gate(self._manager.segment_pipeline, self._requested_segmentation_mode,
                                    task_root / "validation")
        # Intent, not availability. `tokenizer_mode="regex"` is a deliberate
        # estimate-only configuration used by fixtures; every other mode loads
        # the serving model's tokenizer and the adapter is held to it.
        self._token_accounting_mode = (
            TokenAccountingMode.ESTIMATE_ONLY
            if self._tokenizer_mode == "regex"
            else TokenAccountingMode.AUTHORITATIVE
        )
        self._adapter = NOCSGLangAdapter(
            model=self.model_name,
            token_accounting_mode=self._token_accounting_mode,
            patched_transport=self._patched_transport,
            original_transport=self._original_transport,
            mode=self.operation_mode,
            compile_policy=self._compile_policy,
            cache_catalog=self._shared_cache_catalog,
            recompute_initial_tokens=self.legolink_k,
            generation={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **(
                    {"repetition_penalty": self._repetition_penalty}
                    if self._repetition_penalty is not None
                    else {}
                ),
                **({"logprobs": True, "top_logprobs": 1} if self._record_logprobs else {}),
            },
            tokenizer=self._tokenizer,
            residency_recovery=self.residency_recovery,
            residency_recovery_attempts=self.residency_recovery_attempts,
            semantic_cache_namespace_mode=self._semantic_cache_namespace_mode,
            cache_namespace_policy=CacheNamespacePolicy(
                cross_namespace_pairs=self._cross_namespace_pairs, require_vetted_pairs=True
            ),
            global_physical_dedup=self._global_physical_dedup,
        )
        self._pipeline = NOCRequestPipeline(
            context_manager=self._manager,
            adapter=self._adapter,
        )

    def declare_prompt_structure(self, parts: list[PromptPart] | tuple[PromptPart, ...] | None) -> None:
        """The agent that assembled the *next* prompt says how (agent directive / tool output parts).

        Step 1 re-verifies the declaration byte for byte against the prompt it receives and discards it if
        it does not re-assemble that prompt exactly. Spent by the first call that succeeds.
        """

        self._declared_parts = tuple(parts) if parts else None

    def _turn_events(
        self,
        prompt: str,
        history: list[Any],
        *,
        openai_messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        functions: list[dict[str, Any]] | None = None,
    ) -> list[tuple[EventType, str, str, dict[str, Any]]]:
        """Step 1 for one call: canonical events from the structure the interface actually provides.

        ``openai_messages`` / ``tools`` (an OpenAI-compatible request, via the proxy) carry roles, tool calls,
        tool results and tool schemas. A bare Terminus ``call(prompt, message_history)`` carries only
        user/assistant text, plus the agent's re-verified declaration of how it built ``prompt``.
        """

        assert self._manager is not None
        ledger = SourceLedger.from_events(self._manager.collector.events())
        declared = self._declared_parts
        # The first prompt of a session is the one that finds the session empty -- not "turn == 1", which a
        # retried first call (its events rolled back) would no longer be.
        first_turn = not ledger.latest
        if openai_messages:
            events = openai_request_events(
                openai_messages, tools=tools, functions=functions, ledger=ledger, first_turn=first_turn
            )
            self._last_ingress = {"kind": "openai_structured", "tools": len(tools or []) + len(functions or [])}
        else:
            events, structure = terminus_turn_events(
                prompt,
                history,
                ledger=ledger,
                registry=self._prompt_registry,
                declared=declared,
                first_turn=first_turn,
                message_value=_message_value,
            )
            self._last_ingress = {"kind": "terminus_call", **structure}
        return events

    def _ingress_row(self, execution: Any) -> dict[str, Any]:
        """What Step 1 produced this call, and append the accepted events to ``session_events.jsonl``."""

        ingest = getattr(execution, "ingest_result", None)
        accepted = list(getattr(ingest, "events", ()) or ())
        counts: dict[str, int] = {}
        for event in accepted:
            counts[event.event_type.value] = counts.get(event.event_type.value, 0) + 1
        path = getattr(self, "_session_events_path", None)
        if path is not None and accepted:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                for event in accepted:
                    handle.write(
                        json.dumps({"turn": self.turn, **event.to_dict()}, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
        return {**self._last_ingress, "accepted_events": len(accepted), "event_types": counts}

    def _last_generation_transport(self) -> Any:
        """The transport that served this turn's generation (see below)."""

        return (
            self._patched_transport
            if self.variant.endswith("legolink")
            else self._original_transport
        )

    def _last_generation_ttft(self) -> float | None:
        """TTFT of the request this turn just made, or None if it did not stream.

        The route decides which transport served it, so read the one that
        matches rather than whichever happens to be non-empty -- a stale value
        from the other arm would be indistinguishable from a real one.
        """

        transport = (
            self._patched_transport
            if self.variant.endswith("legolink")
            else self._original_transport
        )
        return getattr(transport, "last_ttft_s", None)

    def call(
        self,
        prompt: str,
        message_history: list[Any] | None = None,
        response_format: Any = None,
        logging_path: Path | None = None,
        openai_messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        functions: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        if self._manager is None or self._adapter is None or self._pipeline is None:
            self.set_task_context(self.task_id, self.metrics_path)
        assert (
            self._manager is not None
            and self._adapter is not None
            and self._pipeline is not None
        )
        self.turn += 1
        turn_events = self._turn_events(
            prompt,
            message_history or [],
            openai_messages=openai_messages,
            tools=tools,
            functions=functions,
        )
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        self._adapter.generation["max_tokens"] = max_tokens
        if response_format is not None:
            schema = (
                response_format
                if isinstance(response_format, dict)
                else response_format.model_json_schema()
            )
            self._adapter.generation["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(response_format, "__name__", "response"),
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            self._adapter.generation.pop("response_format", None)
        stage = "planner" if self.turn == 1 else "replanner"
        self._loop_stage = stage
        if self._manager is not None:
            self._manager.last_manifest = None
        try:
            execution = self._pipeline.execute(
                turn_events,
                route=PipelineRoute(self.variant),
                agent_id=stage,
                loop_stage=stage,
                objective=prompt,
                plan_step=prompt,
                available_tools=("terminal",),
                token_budget=self.context_token_budget,
                mutable_text=(
                    f"Current {stage}: produce the next terminal-agent action "
                    "for the selected objective."
                ),
                expected_future_reuse=max(1, self.turn),
            )
        except Exception as error:
            # The call raised inside host planning or assembly, before any
            # debug.json existed: persist what it was building, then re-raise.
            # Recording only -- the exception propagates unchanged.
            self._write_error_fixture(error, turn_events, logging_path)
            raise
        # The turn was served: its declaration is spent. (A failed call keeps it, so a retry sees the same structure.)
        self._declared_parts = None
        result = execution.adapter_result
        manifest = execution.manifest
        contexts = list(execution.contexts)
        decisions = execution.compile_decisions
        # `execution` counts the Compiles the pipeline issued before the Link; `result` counts the ones the adapter issued
        # itself while linking, which are the residency-recovery Compiles (a strict plan's blocks are all compiled before the
        # Link). Without the second term a recovered eviction's Compile work vanished from the count and the latency components.
        compile_count = execution.compile_requests + result.compile_requests
        compile_latency = execution.compile_latency_s + result.compile_latency_s
        usage = result.response.get("usage") or {}
        linked_tokens = None
        reused_tokens = None
        recompute_tokens = None
        # A served request never depends on the server having reported usage. Unknown stays unknown:
        # substituting zero would invent a measurement and would make repair tokens exceed the total they
        # belong to, which is how a metrics row used to abort a turn.
        reported_prompt_tokens = usage.get("prompt_tokens")
        reported_prompt_tokens = (
            int(reported_prompt_tokens)
            if isinstance(reported_prompt_tokens, (int, float))
            and not isinstance(reported_prompt_tokens, bool)
            else None
        )
        if self.variant.endswith("legolink") and reported_prompt_tokens is not None:
            linked_tokens = reported_prompt_tokens
            by_id = {value.sub_context_id: value for value in contexts}
            block_token_counts = []
            for decision in decisions:
                if decision.action.value != "compile":
                    continue
                value = by_id[decision.sub_context_id]
                metadata = self._adapter.cache_catalog.metadata(value.cache_key) or {}
                block_token_counts.append(
                    int(metadata.get("token_count") or value.token_count)
                )
            immutable_tokens = sum(block_token_counts)
            mutable_tokens = max(linked_tokens - immutable_tokens, 0)
            recompute_tokens = min(
                linked_tokens,
                mutable_tokens
                + sum(
                    min(self.legolink_k, token_count)
                    for token_count in block_token_counts
                ),
            )
            reused_tokens = max(linked_tokens - recompute_tokens, 0)
        # Gate 3 accounting for the formal S0/S1 x N/K matrix. Recorded, never
        # read back by any decision; unavailable quantities say so. Note
        # `recompute_tokens` above already includes the repair windows.
        reuse_accounting = None
        if execution.execution_plan is not None:
            wire = (result.request_payload or {}).get("pic_link") or {}
            linked_route = self.variant.endswith("legolink") and bool(wire)
            link_keys = [str(key) for key in wire.get("cache_keys") or ()]
            windows = wire.get("recompute_tokens_per_block") or [
                self._adapter.recompute_initial_tokens
            ] * len(link_keys)
            reuse_accounting = execution.execution_plan.reuse_accounting(
                linked_route=linked_route,
                repair_tokens_by_cache_key=(
                    dict(zip(link_keys, windows)) if linked_route else None
                ),
                linked_cache_keys=link_keys if linked_route else None,
                prompt_tokens=reported_prompt_tokens,
                recomputed_tokens=recompute_tokens,
                compile_latency_s=compile_latency if linked_route else None,
                request_latency_s=result.request_latency_s,
                ttft_s=self._last_generation_ttft(),
                route=result.route,
                recovered_cache_keys=result.residency_recovered_cache_keys,
            ).to_dict()
        self._write_metrics(
            {
                "task_id": self.task_id,
                "session_id": self.session_id,
                "turn": self.turn,
                "variant": self.variant,
                "operation_mode": self.operation_mode.value,
                "tokenizer_fingerprint": self._tokenizer.fingerprint,
                # Recorded per row, not just per run. Whether the budget binds
                # is what decides if selection ever drops a block, and every
                # recorded session so far had it at a value that never did --
                # 863 of 885 rows have full_session_tokens equal to
                # selected_context_tokens. A trace has to carry the number that
                # explains its own shape.
                "context_token_budget": self.context_token_budget,
                "request_id": manifest.request_id
                if manifest
                else f"full::{self.session_id}::{self.turn}",
                "full_session_tokens": manifest.full_session_tokens
                if manifest
                else usage.get("prompt_tokens"),
                "selected_context_tokens": manifest.selected_context_tokens
                if manifest
                else None,
                "context_reduction_ratio": manifest.context_reduction_ratio
                if manifest
                else None,
                "selected_sub_context_ids": list(manifest.selected_sub_context_ids)
                if manifest
                else None,
                "selected_cache_keys": list(manifest.selected_cache_keys)
                if manifest
                else None,
                "execution_mode": result.request_payload.get("pic_link", {}).get("execution_mode", "legacy_debug"),
                "framing_spec": result.request_payload.get("pic_link", {}).get("framing_spec"),
                "repair_decision": result.request_payload.get("pic_link", {}).get("repair_decision"),
                # What the planner did with each selected block. Without this
                # the row said what was selected and the trace said what was
                # linked, and a block could quietly stop being reusable in
                # between with nothing recording that it had.
                "block_planning": [
                    record.to_dict()
                    for record in execution.execution_plan.planning_records
                ]
                if execution.execution_plan is not None
                else None,
                # Empty on all 891 reusable selected blocks recorded so far.
                # Non-empty means a block selection offered for reuse stopped
                # being reusable and no recognised policy said why. Recorded
                # rather than raised, so it cannot abort a measurement mid-run.
                "planning_invariant_violations": [
                    record.to_dict()
                    for record in (
                        execution.execution_plan.planning_invariant_violations()
                    )
                ]
                if execution.execution_plan is not None
                else None,
                "newly_compiled_blocks": compile_count
                if self.variant.endswith("legolink")
                else None,
                "compile_latency_s": compile_latency
                if self.variant.endswith("legolink")
                else None,
                "link_latency_s": result.request_latency_s
                if self.variant.endswith("legolink")
                else None,
                # The metric the objective names. None when the route did not
                # stream, which is not the same as a request that streamed and
                # measured zero.
                "ttft_s": self._last_generation_ttft(),
                # Diagnostics for joining this row to server-side timing.
                "sglang_rid": getattr(
                    self._last_generation_transport(), "last_response_id", None
                ),
                **(
                    getattr(self._last_generation_transport(), "last_timing", None)
                    or {}
                ),
                "native_request_latency_s": result.request_latency_s
                if not self.variant.endswith("legolink")
                else None,
                "linked_tokens": linked_tokens,
                "reused_tokens": reused_tokens,
                "recompute_tokens": recompute_tokens,
                "context_cache_ratio": (
                    reused_tokens / linked_tokens
                    if linked_tokens and reused_tokens is not None
                    else None
                ),
                "prompt_tokens": reported_prompt_tokens,
                # Served from the server's own prefix cache, when it reports it (--enable-cache-report);
                # None when it does not, which is not the same as zero.
                "cached_prompt_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                if isinstance(usage.get("prompt_tokens_details"), dict) else None,
                # Cache-report evidence, raw as received (experiment_harness/cache_evidence.py grades it). The server omits prompt_tokens_details when
                # nothing was cached, so `cached_prompt_tokens` None can mean a miss, a disabled report or a missing usage chunk; these fields keep
                # the three apart. Nothing here is a guess: absent stays absent.
                "usage_present": bool(usage) and reported_prompt_tokens is not None and isinstance(usage.get("completion_tokens"), (int, float)),
                "usage_has_prompt_tokens_details_key": "prompt_tokens_details" in usage,
                "usage_prompt_tokens_details": dict(usage["prompt_tokens_details"]) if isinstance(usage.get("prompt_tokens_details"), dict) else None,
                "usage_keys": sorted(str(key) for key in usage),
                "usage_source": "stream_final_usage_chunk" if self.stream_metrics else "response_body",
                # T5: host Stage-1 segmentation wall of this turn's ingest.
                "segmentation_latency_s": getattr(execution.ingest_result, "segmentation_latency_s", None),
                "completion_tokens": usage.get("completion_tokens"),
                "fallback_reason": result.fallback_reason,
                # --- route provenance -------------------------------------
                # Which method actually served this request. Every field is
                # read back from what the turn already computed; nothing here
                # is consulted by segmentation, planning, grouping, repair,
                # scheduling, cache lookup or decode
                # (results/canonical_agent_path/route_provenance_schema.md).
                "agent_loop_stage": getattr(self, "_loop_stage", None),
                "trial_id": (
                    Path(self.metrics_path).parents[1].name
                    if self.metrics_path is not None
                    else None
                ),
                "segmentation_mode": self._segmentation_mode,
                "semantic_classification_enabled": self._semantic_classification_enabled,
                "segmentation_prompt_variant": self._segmentation_prompt_variant.value,
                "shared_cache_catalog": self._shared_cache_catalog is not None,
                "segmentation_used": type(self._segmenter).__name__,
                "segmentation_output_encoding": getattr(self._segmenter, "formal_output_encoding", None),
                "segmentation_guardrail_version": getattr(self._segmenter, "guardrail_version", None),
                "segmentation_evidence": getattr(self._manager.segment_pipeline, "segmentation_evidence", None),
                "source_reconstruction": self._manager.source_reconstruction_evidence,
                # Per request: the active-source coverage `select` checked, and
                # any sibling-atomicity adjustment it made. Recorded, never read.
                "source_coverage": _coverage_summary(self._manager.source_coverage_evidence),
                "source_atomicity_adjustments": list(self._manager.source_atomicity_adjustments),
                "selection_mode": self._selection_mode,
                "compile_policy_mode": self._compile_policy_mode,
                "subcontext_count": len(manifest.selected_sub_context_ids) if manifest else 0,
                "candidate_subcontext_count": len(contexts),
                # Selected blocks whose presentation-order key (session revision,
                # millisecond creation time) ties with another block's. A tie is
                # broken by sub_context_id rather than by ingestion, so a nonzero
                # count marks a request whose context order was not guaranteed
                # chronological (results/agent_effectiveness/arm_definition_audit.md).
                "selected_blocks_sharing_creation_ms": (
                    _order_key_ties(contexts, manifest.selected_sub_context_ids)
                    if manifest
                    else None
                ),
                "semantic_identities": [
                    value.semantic_identity or value.sub_context_id for value in contexts
                ] if self.variant.endswith("legolink") else None,
                # One Link block per semantic sub-context: the live path does
                # no execution grouping, so there is no P3 or group decision to
                # record and no group cache to hit.
                "grouping_mode": "P0_one_link_block_per_semantic_subcontext",
                "p3_decision": None,
                "quality_profile_decision": (
                    result.request_payload.get("pic_link", {}).get("repair_decision")
                    if self.variant.endswith("legolink") else None
                ),
                "repair_policy": "uniform_k" if self.variant.endswith("legolink") else None,
                "repair_initial_tokens": self.legolink_k
                if self.variant.endswith("legolink") else None,
                "repair_tokens_per_block": (
                    result.request_payload.get("pic_link", {}).get(
                        "recompute_tokens_per_block")
                    if self.variant.endswith("legolink") else None
                ),
                "link_count": 1 if (
                    self.variant.endswith("legolink")
                    and result.request_payload.get("pic_link")
                ) else 0,
                "compile_count": compile_count,
                "resident_hit_count": (
                    len(execution.execution_plan.cached_cache_keys)
                    if execution.execution_plan is not None else None
                ),
                "reconstruction_count": 0,
                "promotion_count": 0,
                "group_cache_hit_count": 0,
                "runtime_route": result.route,
                # Token-stable framing evidence from the Link actually sent:
                # framing path, blocks carried fresh at a non-composing
                # boundary, reused interior tokens. Recorded, never read.
                "token_stable_framing": (
                    ((result.request_payload or {}).get("pic_link_audit") or {}).get(
                        "token_stable_framing"
                    )
                    if self.variant.endswith("legolink")
                    else None
                ),
                "error_classification": None,
                # Eviction evidence: how many times this turn found a planned
                # source block gone, and which keys were recompiled for it.
                "residency_recovery_attempts": result.residency_recovery_attempts,
                "residency_recovered_cache_keys": list(
                    result.residency_recovered_cache_keys
                ),
                "residency_recovery_compiles": result.residency_recovery_compiles,
                # Explicit, never silent: what the runtime reported gone, and what the host did about it. `outcome` is
                # reuse_hit (nothing was missing) or recovered_after_source_missing (the named keys were compiled again,
                # their tokens counted as compiled, and the Link retried). An unrecovered miss never reaches this row:
                # the request raises and the experiment runner records it as an error.
                "residency_recovery": {
                    "enabled": self._adapter.residency_recovery,
                    "max_attempts": self._adapter.residency_recovery_attempts,
                    "attempts": result.residency_recovery_attempts,
                    "recovered_cache_keys": list(result.residency_recovered_cache_keys),
                    "recovered_blocks": len(result.residency_recovered_cache_keys),
                    "compiles": result.residency_recovery_compiles,
                    # Single-flight (residency_recovery.py): a key a PEER recompiled is waited on (or already settled), never compiled
                    # again by this request, so it is not in recovered_cache_keys and adds no Compile work to this row.
                    "waited_cache_keys": list(result.residency_waited_cache_keys),
                    "owner_count": result.residency_recovery_owner_count,
                    "waiter_count": result.residency_recovery_waiter_count,
                    "settled_count": result.residency_recovery_settled_count,
                    "wait_time_s": result.residency_recovery_wait_time_s,
                    "duplicate_compile_count": result.residency_duplicate_recovery_compile_count,
                    "outcome": (
                        "recovered_after_source_missing" if result.residency_recovered_cache_keys
                        else "recovered_by_peer_after_source_missing" if result.residency_waited_cache_keys
                        else "reuse_hit"
                    ),
                },
                # Gates 1-3 per request: eligibility counts, boundary structure,
                # repair/recompute decomposition, observe-only profitability.
                "reuse_eligibility_summary": (
                    execution.execution_plan.eligibility_summary()
                    if execution.execution_plan is not None
                    else None
                ),
                "reuse_accounting": reuse_accounting,
                # Namespace-aware lookup (cache_namespace). The mode is the experiment arm;
                # the summary is per-request counters by namespace. Recorded, never read.
                "semantic_cache_namespace_mode": self._semantic_cache_namespace_mode,
                "cross_namespace_pairs": sorted(f"{a.value}->{b.value}" for a, b in self._cross_namespace_pairs),
                "cross_namespace_pair_policy_version": CROSS_NAMESPACE_PAIR_POLICY_VERSION,
                "ingress": self._ingress_row(execution),
                "namespace_metrics": (
                    execution.execution_plan.namespace_summary(
                        repair_tokens_by_cache_key=(
                            dict(zip(link_keys, windows)) if reuse_accounting is not None and linked_route else None
                        ),
                        recovered_cache_keys=result.residency_recovered_cache_keys,
                    )
                    if execution.execution_plan is not None
                    else None
                ),
                "namespace_active_blocks": self._adapter.cache_catalog.active_by_namespace(),
                "namespace_removals": dict(self._adapter.cache_catalog.removals_by_namespace),
                # Logical roots over the one shared physical cache: entries, aliases, dedup, evictions.
                "namespace_logical_index": self._adapter.cache_catalog.semantic_index.snapshot(),
            },
            None,
        )
        native_payload = self._adapter._payload(result.selected_content)
        if hasattr(self._tokenizer, "tokenize_messages"):
            logical_prompt_token_ids = self._tokenizer.tokenize_messages(
                native_payload.get("messages", [])
            )
        else:
            logical_prompt_token_ids = self._tokenizer.tokenize(
                result.selected_content
            )
        artifact = {
            "variant": self.variant,
            "manifest": manifest.to_dict()
            if self.variant != "full_context_original_sglang"
            else None,
            "candidate_blocks": (
                [value.to_dict() for value in contexts]
                if self.variant != "full_context_original_sglang"
                else []
            ),
            "route": result.route,
            "fallback_reason": result.fallback_reason,
            "tokenizer_fingerprint": self._tokenizer.fingerprint,
            "request_payload": result.request_payload,
            "request_execution_plan": (
                execution.execution_plan.to_dict()
                if execution.execution_plan is not None
                else None
            ),
            "stitch_equivalence": (
                execution.stitched.equivalence.to_dict()
                if execution.stitched is not None
                else None
            ),
            "preliminary_token_count": (
                execution.ingest_result.segmentation.preliminary_token_count.to_dict()
                if execution.ingest_result is not None
                and execution.ingest_result.segmentation.preliminary_token_count
                is not None
                else None
            ),
            "selected_content_sha256": hashlib.sha256(
                result.selected_content.encode("utf-8")
            ).hexdigest(),
            "logical_prompt_token_ids": list(logical_prompt_token_ids),
            "objective": prompt,
            "loop_stage": stage,
            "mutable_text": (
                f"Current {stage}: produce the next terminal-agent action "
                "for the selected objective."
            ),
            "generation_parameters": dict(self._adapter.generation),
            "message_history": [
                {
                    "role": _message_value(message, "role", "user"),
                    "content": _message_value(message, "content"),
                }
                for message in (message_history or [])
            ],
            "source_reconstruction": self._manager.source_reconstruction_evidence,
            "source_coverage": list(self._manager.source_coverage_evidence),
            "source_atomicity_adjustments": list(self._manager.source_atomicity_adjustments),
            "turn_events": [
                {
                    "event_type": EventType(event_type).value,
                    "source_id": source_id,
                    "raw_text": raw_text,
                    "metadata": metadata,
                }
                for event_type, source_id, raw_text, metadata in turn_events
            ],
            "active_source_events": [
                event.to_dict() for event in self._manager.collector.events()
            ],
        }
        choice = (result.response.get("choices") or [{}])[0]
        # The model's output is part of the request's record -- including a
        # response truncated at max_tokens, which used to be lost with the raise.
        artifact["model_output"] = result.content
        artifact["finish_reason"] = choice.get("finish_reason")
        artifact["output_token_ids"] = _output_token_ids(result.response)
        if self._record_logprobs:
            content = ((choice.get("logprobs") or {}).get("content")) or []
            artifact["output_logprobs"] = {
                "tokens": [item.get("token") for item in content],
                "token_bytes": [item.get("bytes") for item in content],
                "logprobs": [item.get("logprob") for item in content],
            }
        if logging_path:
            logging_path.parent.mkdir(parents=True, exist_ok=True)
            logging_path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if choice.get("finish_reason") == "length":
            raise OutputLengthExceededError(
                "SGLang response reached max_tokens", truncated_response=result.content
            )
        return result.content

    def _write_error_fixture(self, error, turn_events, logging_path) -> None:
        manager = self._manager
        manifest = getattr(manager, "last_manifest", None)
        try:
            candidates = manager.store.list_by_task(self.task_id) if manager else []
        except Exception:  # the fixture never replaces the original error
            candidates = []
        selected = set(manifest.selected_sub_context_ids) if manifest else set()
        fixture = {
            "error_fixture": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_cause_type": type(error.__cause__).__name__ if error.__cause__ else None,
            "task_id": self.task_id,
            "turn": self.turn,
            "variant": self.variant,
            "manifest": manifest.to_dict() if manifest else None,
            "candidate_blocks": [
                {
                    "sub_context_id": value.sub_context_id,
                    "context_field": value.context_field,
                    "cache_key": value.cache_key,
                    "source_spans": [span.to_dict() for span in value.source_spans],
                    "source_revision_identity": value.source_revision_identity,
                    "raw_text_sha256": hashlib.sha256(value.raw_text.encode("utf-8")).hexdigest(),
                    "token_count": value.token_count,
                    "token_ids": list(value.token_ids or ()),
                    "is_mutable": value.is_mutable,
                    "selected": value.sub_context_id in selected,
                }
                for value in candidates
            ],
            "source_coverage": getattr(manager, "source_coverage_evidence", None),
            "source_atomicity_adjustments": getattr(manager, "source_atomicity_adjustments", None),
            "coverage_failures": getattr(error, "failures", None),
            "turn_events": [
                {"event_type": EventType(event_type).value, "source_id": source_id,
                 "raw_text": raw_text, "metadata": metadata}
                for event_type, source_id, raw_text, metadata in turn_events
            ],
            "active_source_events": [
                event.to_dict() for event in manager.collector.events()
            ] if manager else [],
        }
        paths = [Path(logging_path)] if logging_path else []
        task_root = getattr(self, "_task_root", None)
        if task_root is not None:
            paths.append(Path(task_root) / "errors" / f"turn-{self.turn:03d}.json")
        for path in paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass


class NOCSGLangAgent(SGLangPICAgent):
    """Terminus policy with A/B/C context-management request variants."""

    @staticmethod
    def name() -> str:
        return "noc-sglang-terminus"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        noc_keys = {
            "variant",
            "original_api_base",
            "operation_mode",
            "segmentation_mode",
            "segmentation_api_base",
            "segmentation_model",
            "preserve_full_source_content",
            "selection_mode",
            "compile_policy_mode",
            "compile_cost_profile",
            "deployment_hardware_id",
            "compile_reuse_probability",
            "context_token_budget",
            "artifacts_root",
            "tokenizer_mode",
            "tokenizer_path",
            "tokenizer_revision",
            "tokenizer_local_files_only",
            "tokenizer_trust_remote_code",
            "tokenizer_api_base",
            # Same reason as stream_metrics below: a switch the agent accepts
            # but does not forward stops at the agent, and the run measures the
            # default while its command line says otherwise.
            "residency_recovery",
            "residency_recovery_attempts",
            # Terminal-Bench's BaseAgent accepts and silently drops unknown kwargs, so a switch missing
            # from this set never reaches NOCSGLangLLM: `--agent-kwarg semantic_classification_enabled=False`
            # used to leave Stage 2 on.
            "semantic_classification_enabled",
            "require_llm_segmentation",
            "segmentation_prompt_variant",
            "shared_catalog_path",
            "semantic_cache_namespace_mode",
            "global_physical_dedup",
            "cross_namespace_pairs",
            "repetition_penalty",
            "record_logprobs",
        }
        noc_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in noc_keys}
        kwargs.setdefault("api_base", "http://127.0.0.1:30000/v1")
        kwargs.setdefault("legolink_k", 256)
        kwargs.setdefault("max_tokens", 1024)
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("timeout", 1200)
        llm_keys = {
            "api_base",
            "recompute_mode",
            "legolink_k",
            "first_block_k",
            "max_tokens",
            "temperature",
            "block_chars",
            "timeout",
            "api_key",
            # Without this the switch reaches the agent and stops there, which
            # is how stream_metrics_status came back None on all 48 requests of
            # the 2026-09-03 paired run.
            "stream_metrics",
        }
        llm_kwargs = {key: value for key, value in kwargs.items() if key in llm_keys}
        super().__init__(model_name=model_name, **kwargs)
        self._llm = NOCSGLangLLM(
            model_name=model_name,
            **llm_kwargs,
            **noc_kwargs,
        )


class NOCSGLangTerminus2Agent(SGLangPICTerminus2Agent):
    """Terminus-2 parser/completion policy with the NOC/PIC transport."""

    @staticmethod
    def name() -> str:
        return "noc-sglang-terminus-2"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        noc_keys = {
            "variant",
            "original_api_base",
            "operation_mode",
            "segmentation_mode",
            "segmentation_api_base",
            "segmentation_model",
            "preserve_full_source_content",
            "selection_mode",
            "compile_policy_mode",
            "compile_cost_profile",
            "deployment_hardware_id",
            "compile_reuse_probability",
            "context_token_budget",
            "artifacts_root",
            "tokenizer_mode",
            "tokenizer_path",
            "tokenizer_revision",
            "tokenizer_local_files_only",
            "tokenizer_trust_remote_code",
            "tokenizer_api_base",
            # Same reason as stream_metrics below: a switch the agent accepts
            # but does not forward stops at the agent, and the run measures the
            # default while its command line says otherwise.
            "residency_recovery",
            "residency_recovery_attempts",
            # Terminal-Bench's BaseAgent accepts and silently drops unknown kwargs, so a switch missing
            # from this set never reaches NOCSGLangLLM: `--agent-kwarg semantic_classification_enabled=False`
            # used to leave Stage 2 on.
            "semantic_classification_enabled",
            "require_llm_segmentation",
            "segmentation_prompt_variant",
            "shared_catalog_path",
            "semantic_cache_namespace_mode",
            "global_physical_dedup",
            "cross_namespace_pairs",
            "repetition_penalty",
            "record_logprobs",
        }
        noc_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in noc_keys}
        kwargs.setdefault("api_base", "http://127.0.0.1:30000/v1")
        kwargs.setdefault("legolink_k", 256)
        kwargs.setdefault("max_tokens", 1024)
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("timeout", 1200)
        llm_keys = {
            "api_base",
            "recompute_mode",
            "legolink_k",
            "first_block_k",
            "max_tokens",
            "temperature",
            "block_chars",
            "timeout",
            "api_key",
            # Without this the switch reaches the agent and stops there, which
            # is how stream_metrics_status came back None on all 48 requests of
            # the 2026-09-03 paired run.
            "stream_metrics",
        }
        llm_kwargs = {key: value for key, value in kwargs.items() if key in llm_keys}
        super().__init__(model_name=model_name, **kwargs)
        self._llm = NOCSGLangLLM(
            model_name=model_name,
            **llm_kwargs,
            **noc_kwargs,
        )
