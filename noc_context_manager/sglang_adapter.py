"""NOC request routing over original and patched SGLang deployments.

ARCHITECTURE_STAGE: 9 -- NOCSGLang Adapter
CANONICAL_ENTRYPOINT: NOCSGLangAdapter (compile / compile_plan / link /
    _link_payload). This is the only module that sends a request to SGLang.
DELEGATED_HELPERS: pic_link_contract.LogicalPICRequestFramer (Stage 8: layout,
    repair windows and the logical request), stitch.ContextStitcher, scheduler
    planner (Stage 6), chat_transport.py. ``pipeline.NOCRequestPipeline``
    orchestrates the calls but holds no adapter semantics.
LEGACY: sglang_bridge.py is the plan-file microbenchmark bridge, not this path.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable

from .cache_namespace import (
    PHYSICAL_ENTRY_METADATA_KEY,
    CacheNamespacePolicy,
    NamespaceMode,
    NamespaceResolver,
    physical_kv_identity,
)
from .chat_transport import ChatStreamError, post_chat
from .compile_policy import CacheCatalog, CompileAction, CompileDecision, CompilePolicy
from .context_selection import AgentContextManifest
from .pic_link_contract import (
    LogicalPICRequestFramer,
    PICLinkRequestAudit,
)
from .physical_framing import (
    PHYSICAL_TOKEN_LAYOUT_VERSION,
    NormalizedChatTokenLayout,
    PhysicalBlockSegment,
    PhysicalTokenLayoutError,
)
from .repair_window import RepairWindowPlanner
from .residency_recovery import FAILED, OWNER, SETTLED, WAITER, ResidencyRecoveryCoordinator, default_coordinator
from .manager import TokenAccountingMode
from .token_accounting import require_authoritative_tokenizer
from .reuse_eligibility import ReuseEligibilityPolicy, StateOrigin
from .scheduler import NOCRequestExecutionPlan, NOCRequestPlanner
from .stitch import (  # noqa: F401
    ContextStitcher,
    StitchedContext,
    render_selected_context,
)
from .subcontext_store import SemanticSubContext
from .tokenizer import ChatTemplateTokenizer, TextTokenizer


class OperationMode(str, Enum):
    STRICT_EXPERIMENTAL = "strict_experimental"
    ROBUST_AGENT = "robust_agent"


class SGLangRequestError(RuntimeError):
    def __init__(
        self, message: str, *, reason: str = "request_error", status: int | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class AdapterResult:
    content: str
    route: str
    request_payload: dict[str, Any]
    response: dict[str, Any]
    compile_decisions: tuple[CompileDecision, ...] = ()
    compile_requests: int = 0
    duplicate_compile_avoided: int = 0
    compile_latency_s: float = 0.0
    request_latency_s: float = 0.0
    fallback_reason: str | None = None
    selected_content: str = ""
    # Residency recovery evidence. Non-zero means the runtime reported that a
    # source block the host planned to reuse was no longer physically resident,
    # and the host recompiled it under the same identity and resent.
    residency_recovery_attempts: int = 0
    residency_recovered_cache_keys: tuple[str, ...] = ()
    residency_recovery_compiles: int = 0
    # Single-flight evidence (residency_recovery.py): how this request took part in recoveries of keys the runtime reported gone.
    # A key this request recompiled is in `residency_recovered_cache_keys` (and counted as its Compile work); a key a peer
    # recompiled while this request waited, or had already recompiled when this request's abort arrived, is in
    # `residency_waited_cache_keys` and costs this request no Compile.
    residency_waited_cache_keys: tuple[str, ...] = ()
    residency_recovery_owner_count: int = 0
    residency_recovery_waiter_count: int = 0
    residency_recovery_settled_count: int = 0
    residency_recovery_wait_time_s: float = 0.0
    residency_duplicate_recovery_compile_count: int = 0


@dataclass(frozen=True)
class ResidencyRecoveryStep:
    """What one pass of recovery did for one request (internal to ``NOCSGLangAdapter.link``)."""

    recovered: tuple[str, ...] = ()
    waited: tuple[str, ...] = ()
    compiles: int = 0
    latency_s: float = 0.0
    owner_count: int = 0
    waiter_count: int = 0
    settled_count: int = 0
    wait_s: float = 0.0
    duplicates: int = 0
    failure: str | None = None  # set when a recovery this request depended on failed: the request must raise

    @property
    def retry(self) -> bool:
        return self.failure is None and bool(self.recovered or self.waited)


class JSONHTTPTransport:
    def __init__(
        self,
        api_base: str,
        *,
        api_key: str = "",
        timeout: float = 1200.0,
        stream: bool = False,
    ) -> None:
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.stream = bool(stream)
        # The last request's TTFT, or None when it was not streamed. The
        # transport keeps the call signature a plain payload-to-response
        # callable, because every route in the adapter depends on that shape.
        self.last_ttft_s: float | None = None
        # Diagnostics only, never read by a decision: the server's request id
        # (the SSE chunk id, which is the SGLang rid) and client wall-clock
        # anchors, so a client row can be joined to server-side timing.
        self.last_response_id: str | None = None
        self.last_timing: dict[str, Any] = {}

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.last_ttft_s = None
        self.last_response_id = None
        self.last_timing = {}
        try:
            result = post_chat(
                self.endpoint,
                payload,
                timeout=self.timeout,
                headers=headers,
                stream=self.stream,
            )
            self.last_ttft_s = result.ttft_s
            self.last_response_id = result.response.get("id")
            self.last_timing = {
                "client_request_started_unix_ns": result.request_started_unix_ns or None,
                "client_first_token_unix_ns": result.first_token_unix_ns or None,
                "client_request_completed_unix_ns": result.request_completed_unix_ns or None,
                "client_request_latency_s": result.latency_s,
            }
            return result.response
        except ChatStreamError as error:
            # A streamed abort arrives inside an HTTP 200. It is still an
            # error, and it carries the same message the non-streamed route
            # returns as a 400, so both paths classify identically.
            detail = str(error)
            raise SGLangRequestError(
                detail,
                reason=NOCSGLangAdapter.classify_failure(detail),
                status=error.status,
            ) from error
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            reason = NOCSGLangAdapter.classify_failure(detail)
            raise SGLangRequestError(
                detail, reason=reason, status=error.code
            ) from error
        except urllib.error.URLError as error:
            raise SGLangRequestError(str(error), reason="transport_error") from error


def response_content(response: dict[str, Any]) -> str:
    return str(
        (
            ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ""
        )
    )


class NOCSGLangAdapter:
    """Keep Compile, Link, and native requests explicit and separately measurable."""

    def __init__(
        self,
        *,
        model: str,
        patched_transport: Callable[[dict[str, Any]], dict[str, Any]],
        original_transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        mode: OperationMode = OperationMode.STRICT_EXPERIMENTAL,
        compile_policy: CompilePolicy | None = None,
        cache_catalog: CacheCatalog | None = None,
        recompute_initial_tokens: int = 256,
        generation: dict[str, Any] | None = None,
        tokenizer: TextTokenizer | None = None,
        require_physical_token_equivalence: bool | None = None,
        repair_window_planner: RepairWindowPlanner | None = None,
        execution_profile: dict[str, Any] | None = None,
        quality_profile_id: str | None = None,
        execution_mode: str = "legacy_debug",
        token_accounting_mode: TokenAccountingMode | str = (
            TokenAccountingMode.AUTHORITATIVE
        ),
        residency_recovery: bool = False,
        residency_recovery_attempts: int = 1,
        reuse_eligibility_policy: ReuseEligibilityPolicy | None = None,
        semantic_cache_namespace_mode: NamespaceMode | str = NamespaceMode.GLOBAL,
        cache_namespace_policy: CacheNamespacePolicy | None = None,
        global_physical_dedup: bool = True,
        residency_coordinator: ResidencyRecoveryCoordinator | None = None,
    ) -> None:
        self.execution_profile = execution_profile
        self.quality_profile_id = quality_profile_id
        self.execution_mode = execution_mode
        if execution_mode == "formal" and execution_profile is None:
            raise ValueError("formal adapter requires an execution profile")
        self.model = model
        self.patched_transport = patched_transport
        self.original_transport = original_transport or patched_transport
        self.mode = OperationMode(mode)
        self.compile_policy = compile_policy or CompilePolicy()
        self.cache_catalog = cache_catalog or CacheCatalog()
        # The mode is a statement about intended use, never an observation
        # about what happened to be supplied. An earlier version of this line
        # read `is_authoritative_tokenizer(tokenizer)`, which meant a missing
        # tokenizer silently downgraded the requirement and the runtime went on
        # planning with estimates -- the exact shape of failure the rest of this
        # module exists to prevent. Executing against a real server defaults to
        # AUTHORITATIVE and fails here if it cannot back that up; estimate-only
        # callers say so.
        self.token_accounting_mode = TokenAccountingMode(token_accounting_mode)
        if self.token_accounting_mode is TokenAccountingMode.AUTHORITATIVE:
            require_authoritative_tokenizer(
                tokenizer,
                purpose=(
                    f"NOCSGLangAdapter(model={model!r}) in AUTHORITATIVE token "
                    "accounting mode. Pass the serving model's tokenizer, or "
                    "opt in to token_accounting_mode=ESTIMATE_ONLY if this "
                    "caller genuinely does not execute against a server"
                ),
            )
        self.require_model_tokens = (
            self.token_accounting_mode is TokenAccountingMode.AUTHORITATIVE
        )
        # Host-side namespace lookup over the one runtime registry. The mode is
        # an explicit, logged experiment setting; GLOBAL is the pre-namespace
        # lookup behaviour (the resolver then only observes).
        self.namespace_resolver = NamespaceResolver(
            semantic_cache_namespace_mode,
            cache_namespace_policy,
            global_physical_dedup=global_physical_dedup,
        )
        self.planner = NOCRequestPlanner(
            compile_policy=self.compile_policy,
            cache_catalog=self.cache_catalog,
            require_model_tokens=self.require_model_tokens,
            reuse_eligibility_policy=reuse_eligibility_policy,
            namespace_resolver=self.namespace_resolver,
        )
        # Compatibility for callers written before the planner rename.
        self.scheduler = self.planner
        self.tokenizer = tokenizer
        self.chat_tokenizer = (
            tokenizer if isinstance(tokenizer, ChatTemplateTokenizer) else None
        )
        # Derived from execution intent, exactly like `require_model_tokens`.
        # This line used to read `self.chat_tokenizer is not None`, so an
        # adapter without a chat-capable tokenizer quietly stopped sending the
        # expected-token-hash evidence and the runtime had nothing left to
        # verify its own tokenization against. Missing evidence must not relax
        # the requirement that the evidence exists.
        #
        # Note this is a strictly stronger demand than authoritative token
        # *counting*: `require_authoritative_tokenizer` proves the tokenizer is
        # the model's, and chat-template rendering can still change the token
        # IDs independently of that. So AUTHORITATIVE additionally requires a
        # tokenizer that can reproduce the server's chat template.
        authoritative = (
            self.token_accounting_mode is TokenAccountingMode.AUTHORITATIVE
        )
        if authoritative and require_physical_token_equivalence is False:
            raise PhysicalTokenLayoutError(
                "AUTHORITATIVE execution cannot disable physical token "
                "equivalence: the host would send semantic reuse the runtime "
                "has no way to check against its own tokenization. Use "
                "token_accounting_mode=ESTIMATE_ONLY if this caller builds no "
                "host-derived physical plan."
            )
        self.require_physical_token_equivalence = (
            True if authoritative else bool(require_physical_token_equivalence)
        )
        if self.require_physical_token_equivalence and self.chat_tokenizer is None:
            raise PhysicalTokenLayoutError(
                "physical token equivalence requires a ChatTemplateTokenizer "
                f"that reproduces the server chat template; {type(tokenizer).__name__} "
                "cannot render one. An authoritative model tokenizer is not "
                "sufficient on its own -- chat-template rendering changes token "
                "IDs independently of the vocabulary."
            )
        self.stitcher = ContextStitcher(tokenizer=tokenizer)
        self.recompute_initial_tokens = recompute_initial_tokens
        # Defaults to the uniform validated K, so behaviour is unchanged until a
        # caller supplies a calibrated policy.
        self.repair_window_planner = repair_window_planner or RepairWindowPlanner()
        self.generation = {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "seed": 0,
            "max_tokens": 512,
            **(generation or {}),
        }
        self.duplicate_compile_avoided = 0
        # Experimental, default off: when the runtime reports that a planned
        # source block is no longer resident, invalidate that key, recompile it
        # under the same identity and resend once. The host's catalog records
        # reuse intent, never physical residency, so an eviction has to read as
        # a cache miss rather than a failed request
        # (results/cache_residency_correctness/residency_contract.md).
        self.residency_recovery = bool(residency_recovery)
        self.residency_recovery_attempts = max(0, int(residency_recovery_attempts))
        # Who recompiles a key the runtime reports gone is decided per (runtime, cache key) by one coordinator shared by every adapter of
        # the process that talks to that runtime: a catalog (or this adapter) is per agent, residency is per runtime.
        self.residency_coordinator = residency_coordinator or default_coordinator()

    @property
    def residency_runtime_id(self) -> object:
        transport = self.patched_transport
        return getattr(transport, "endpoint", None) or id(transport)

    @staticmethod
    def residency_recovery_detail(message: str) -> dict[str, tuple[str, ...]]:
        """The cache keys a source-preflight abort says are not resident.

        The runtime appends a JSON object to the abort message naming the
        missing, stale and incompatible keys (pic_runtime.reserve_link_sources).
        Anything without that object yields empty tuples, so a caller can tell
        "the runtime named no key" from "these keys are gone" and never retries
        a plan it has no reason to believe changed.
        """

        marker = '{"noc_residency_recovery"'
        start = message.find(marker)
        if start < 0:
            # A non-streamed abort arrives as an HTTP error body: a JSON document whose "message" holds the
            # text above, so the inner quotes are escaped and the marker is not found in the raw string.
            try:
                body = json.loads(message)
            except ValueError:
                body = None
            inner = body.get("message") if isinstance(body, dict) else None
            if isinstance(inner, str) and inner != message:
                return NOCSGLangAdapter.residency_recovery_detail(inner)
            return {"missing": (), "stale": (), "incompatible": ()}
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(message[start:])
        except ValueError:
            return {"missing": (), "stale": (), "incompatible": ()}
        detail = (parsed or {}).get("noc_residency_recovery") or {}
        return {
            name: tuple(
                str(value) for value in (detail.get(f"{name}_cache_keys") or ())
            )
            for name in ("missing", "stale", "incompatible")
        }

    @staticmethod
    def classify_failure(message: str) -> str:
        lowered = message.lower()
        if "source preflight failed" in lowered:
            # The preflight reports every kind of unavailability through one
            # abort. The structured detail says which, and a missing source is
            # the recoverable one, so it is the classification that must not be
            # lost behind the word "preflight".
            detail = NOCSGLangAdapter.residency_recovery_detail(message)
            if detail["incompatible"]:
                return "incompatible"
            if detail["stale"] and not detail["missing"]:
                return "stale"
            return "missing"
        # "generation" alone is not evidence of staleness: the patched runtime's own LegoLink-only refusal
        # reads "... Generation requests must either compile ... or link ...". Only the registry's
        # generation-mismatch wording (stale_generation / allocator_generation_mismatch) counts.
        if "stale" in lowered or re.search(r"generation[ _]mismatch", lowered):
            return "stale"
        if (
            "incompatible" in lowered
            or "token mismatch" in lowered
            or "fingerprint" in lowered
        ):
            return "incompatible"
        if (
            "missing" in lowered
            or "not compiled" in lowered
            or "unknown cache" in lowered
        ):
            return "missing"
        if "timeout" in lowered:
            return "timeout"
        return "link_exception"

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            **self.generation,
        }

    def _normalized_layout(
        self,
        stitched: StitchedContext,
    ) -> NormalizedChatTokenLayout | None:
        if self.chat_tokenizer is None:
            if self.require_physical_token_equivalence:
                raise SGLangRequestError(
                    "LegoLink requires a tokenizer that can reproduce the "
                    "server chat template",
                    reason="incompatible",
                )
            return None
        try:
            return LogicalPICRequestFramer.build_layout(
                stitched,
                self.chat_tokenizer,
                self.generation,
            )
        except PhysicalTokenLayoutError as error:
            raise SGLangRequestError(
                f"physical token framing failed: {error}",
                reason="incompatible",
            ) from error

    def _standalone_compile_segment(
        self,
        value: SemanticSubContext,
    ) -> PhysicalBlockSegment | None:
        if self.chat_tokenizer is None:
            if self.require_physical_token_equivalence:
                raise SGLangRequestError(
                    "Compile requires a tokenizer that can reproduce the "
                    "server token layout",
                    reason="incompatible",
                )
            return None
        token_ids = tuple(
            int(token_id)
            for token_id in self.chat_tokenizer.tokenize(
                value.raw_text + self.stitcher.separator
            )
        )
        if not token_ids:
            raise SGLangRequestError(
                f"compiled block {value.cache_key} has no physical tokens",
                reason="incompatible",
            )
        return PhysicalBlockSegment(
            sub_context_id=value.sub_context_id,
            semantic_block_id=value.semantic_identity or value.sub_context_id,
            cache_key=value.cache_key,
            token_ids=token_ids,
        )

    def _catalog_segment_matches(
        self,
        cache_key: str,
        segment: PhysicalBlockSegment,
    ) -> bool:
        existing = self.cache_catalog.metadata(cache_key) or {}
        return bool(
            self.cache_catalog.contains(cache_key)
            and existing.get("physical_token_layout_version")
            == PHYSICAL_TOKEN_LAYOUT_VERSION
            and existing.get("physical_token_ids_sha256") == segment.token_ids_sha256
            and existing.get("token_count") == segment.token_count
        )

    def compile(
        self,
        value: SemanticSubContext,
        *,
        physical_segment: PhysicalBlockSegment | None = None,
    ) -> tuple[dict[str, Any], float, bool]:
        physical_segment = physical_segment or self._standalone_compile_segment(value)
        if not self.cache_catalog.contains(value.cache_key) and self._join_recovery_flight(
            value.cache_key, observed=self.residency_coordinator.completions
        ):
            # A peer's residency recovery emptied the entry and is putting the block back (or just did): that is the Compile, this
            # request sends none. No catalog lock is held while waiting: the owner needs it.
            self.duplicate_compile_avoided += 1
            return {}, 0.0, False
        lock = self.cache_catalog.lock_for(value.cache_key)
        with lock:
            if self.cache_catalog.contains(value.cache_key):
                physical_entry_matches = (
                    physical_segment is None
                    or self._catalog_segment_matches(
                        value.cache_key,
                        physical_segment,
                    )
                )
                if physical_entry_matches:
                    self.duplicate_compile_avoided += 1
                    return {}, 0.0, False
                self.cache_catalog.invalidate(value.cache_key)
            if physical_segment is not None and (
                physical_segment.cache_key != value.cache_key
                or physical_segment.sub_context_id != value.sub_context_id
                or physical_segment.semantic_block_id
                != (value.semantic_identity or value.sub_context_id)
            ):
                raise SGLangRequestError(
                    "physical Compile segment does not match the semantic block",
                    reason="incompatible",
                )
            # A view that stands for another entry's physical block (an alias) compiles as that entry: the
            # runtime registers, and the catalog records, the owner's root and field, never the alias's.
            owner = (value.metadata or {}).get(PHYSICAL_ENTRY_METADATA_KEY) or {}
            compiled_field = owner.get("context_field") or value.context_field
            compiled_namespace = owner.get("namespace") or self.namespace_resolver.policy.namespace_of(value).value
            payload = self._payload(value.raw_text)
            payload["max_tokens"] = 1
            payload["extra_key"] = value.cache_key
            payload["custom_labels"] = {
                "prefill_unit_id": value.sub_context_id,
                "cache_key": value.cache_key,
                "pic_storage_mode": "reference",
            }
            payload["sub_context"] = {
                # Provenance stays the occurrence; the semantic block id is the
                # content identity, so Link can match this block from another
                # task or agent. Compile and Link must agree, and they do:
                # `StitchedContext.block_ids` sends the same value.
                "prefill_unit_id": value.sub_context_id,
                "semantic_block_id": value.semantic_identity or value.sub_context_id,
                "cache_key": value.cache_key,
                "task_id": value.task_id,
                "context_field": compiled_field,
                "functional_role": value.question_or_objective,
                "content_hash": value.content_hash,
                "version": value.version,
                "tokenizer_fingerprint": value.tokenizer_fingerprint,
                "model_fingerprint": value.model_fingerprint,
                "segmentation_policy_version": value.policy_version,
                "recompute_policy_version": value.recompute_policy_version,
                "token_count": (
                    physical_segment.token_count
                    if physical_segment is not None
                    else value.token_count
                ),
            }
            if physical_segment is not None:
                payload["sub_context"].update(
                    {
                        "physical_token_layout_version": (
                            PHYSICAL_TOKEN_LAYOUT_VERSION
                        ),
                        "legolink_policy_version": (PHYSICAL_TOKEN_LAYOUT_VERSION),
                        "physical_input_ids": list(physical_segment.token_ids),
                        **physical_segment.to_metadata(),
                    }
                )
            started = time.perf_counter()
            response = self.patched_transport(payload)
            latency = time.perf_counter() - started
            prompt_tokens = (
                physical_segment.token_count
                if physical_segment is not None
                else int(
                    (response.get("usage") or {}).get("prompt_tokens")
                    or value.token_count
                )
            )
            self.cache_catalog.mark_compiled(
                value.cache_key,
                # What Gate 1 reads: this state is the block alone from local
                # position 0, the canonical context-free state.
                state_origin=StateOrigin.STANDALONE_COMPILE.value,
                sub_context_id=value.sub_context_id,
                # Provenance only: which task/agent issued the Compile. Gate 1
                # does not read it and the key does not contain it, so a shared
                # catalog can serve another agent's Link from this entry while
                # an audit can still say whose Compile produced the state.
                producer_task_id=value.task_id,
                token_count=prompt_tokens,
                content_hash=value.content_hash,
                tokenizer_fingerprint=value.tokenizer_fingerprint,
                model_fingerprint=value.model_fingerprint,
                segmentation_policy_version=value.policy_version,
                recompute_policy_version=value.recompute_policy_version,
                physical_token_layout_version=(
                    payload["sub_context"].get("physical_token_layout_version")
                ),
                physical_token_ids_sha256=(
                    physical_segment.token_ids_sha256
                    if physical_segment is not None
                    else None
                ),
                # Namespace bookkeeping (cache_namespace). Host-side only: the
                # runtime registry never sees these, and no decision outside
                # the namespace resolver reads them.
                context_field=compiled_field,
                namespace=compiled_namespace,
                semantic_identity=value.semantic_identity or None,
                physical_kv_identity=(
                    None if (identity := physical_kv_identity(value)) is None
                    else identity.digest
                ),
                compiled_at=time.time(),
            )
            return response, latency, True

    def compile_selected(
        self,
        manifest: AgentContextManifest,
        contexts: list[SemanticSubContext],
        *,
        expected_future_reuse: int = 1,
        memory_pressure: float = 0.0,
    ) -> tuple[tuple[CompileDecision, ...], int, float]:
        plan = self.planner.plan(
            manifest,
            contexts,
            expected_future_reuse=expected_future_reuse,
            memory_pressure=memory_pressure,
        )
        stitched = self.stitcher.stitch(
            plan,
            generation_parameters=self.generation,
        )
        compiled_plan, requests, latency = self.compile_plan(
            plan,
            stitched=stitched,
        )
        return compiled_plan.decisions, requests, latency

    def compile_plan(
        self,
        plan: NOCRequestExecutionPlan,
        *,
        stitched: StitchedContext | None = None,
    ) -> tuple[NOCRequestExecutionPlan, int, float]:
        """Execute only compile-on-miss actions from an existing logical plan."""

        stitched = stitched or self.stitcher.stitch(
            plan,
            generation_parameters=self.generation,
        )
        physical_layout = (
            self._normalized_layout(stitched) if plan.linked_contexts else None
        )
        if physical_layout is not None:
            for value in plan.linked_contexts:
                # One segment per presented occurrence; a repeated immutable
                # block is several occurrences of one logical cache key.
                segment = physical_layout.segment_for_occurrence(value.sub_context_id)
                if self.cache_catalog.contains(
                    value.cache_key
                ) and not self._catalog_segment_matches(
                    value.cache_key,
                    segment,
                ):
                    self.cache_catalog.invalidate(value.cache_key)
            plan = replace(
                self.planner.bind_decisions(
                    plan.manifest,
                    plan.ordered_contexts,
                    plan.decisions,
                ),
                namespace_resolutions=plan.namespace_resolutions,
            )
        requests = 0
        latency = 0.0
        self.duplicate_compile_avoided += len(plan.reuse_cached_contexts)
        for value in plan.compile_on_miss_contexts:
            physical_segment = (
                physical_layout.segment_for_occurrence(value.sub_context_id)
                if physical_layout is not None
                else None
            )
            # A later occurrence of an already-compiled key finds it in the
            # catalog with the same compiled ids and issues no request.
            _response, elapsed, created = self.compile(
                value,
                physical_segment=physical_segment,
            )
            latency += elapsed
            requests += int(created)
        rebound = replace(
            self.planner.bind_decisions(
                plan.manifest,
                plan.ordered_contexts,
                plan.decisions,
            ),
            namespace_resolutions=plan.namespace_resolutions,
        )
        return rebound, requests, latency

    def native_selected(
        self,
        manifest: AgentContextManifest,
        contexts: list[SemanticSubContext],
        *,
        fallback_reason: str | None = None,
        plan: NOCRequestExecutionPlan | None = None,
        stitched: StitchedContext | None = None,
    ) -> AdapterResult:
        plan = plan or self.planner.plan(manifest, contexts)
        stitched = stitched or self.stitcher.stitch(
            plan,
            generation_parameters=self.generation,
        )
        selected = stitched.selected_text
        payload = self._payload(selected)
        started = time.perf_counter()
        response = self.original_transport(payload)
        latency = time.perf_counter() - started
        return AdapterResult(
            content=response_content(response),
            route="selected_context_original_sglang",
            request_payload=payload,
            response=response,
            request_latency_s=latency,
            fallback_reason=fallback_reason,
            selected_content=selected,
        )

    def native_full(self, full_text: str) -> AdapterResult:
        payload = self._payload(full_text)
        started = time.perf_counter()
        response = self.original_transport(payload)
        return AdapterResult(
            content=response_content(response),
            route="full_context_original_sglang",
            request_payload=payload,
            response=response,
            request_latency_s=time.perf_counter() - started,
            selected_content=full_text,
        )

    def _link_payload(
        self,
        plan: NOCRequestExecutionPlan,
        stitched: StitchedContext | None = None,
    ) -> tuple[dict[str, Any], StitchedContext]:
        stitched = stitched or self.stitcher.stitch(
            plan,
            generation_parameters=self.generation,
        )
        if stitched.generation_parameters != self.generation:
            raise ValueError(
                "stitched generation parameters differ from the adapter request"
            )
        try:
            self.stitcher.validate_link_suffix(stitched)
        except ValueError as error:
            raise SGLangRequestError(
                str(error),
                reason="empty_mutable_suffix",
            ) from error
        segmentation_policy_versions = {
            str(value.policy_version) for value in stitched.linked_contexts
        }
        if len(segmentation_policy_versions) > 1:
            raise SGLangRequestError(
                "linked semantic blocks must use one segmentation policy version",
                reason="incompatible",
            )
        if self.execution_mode == "formal" and not plan.canonical:
            # Foreign contextual state is experiment-only. It may be exercised
            # by explicit experiment machinery, never by a formal execution.
            raise SGLangRequestError(
                "formal execution refuses experimental foreign-state links: "
                + ", ".join(plan.experimental_links),
                reason="incompatible",
            )
        if not stitched.linked_contexts:
            # Nothing to link. `link()` turns this into a fallback or a strict
            # rejection; building a Link request naming no blocks would only
            # move that decision into the contract, which cannot express it.
            return self._payload(stitched.mutable_text), stitched
        physical_layout = self._normalized_layout(stitched)
        # Stage 8: the logical PIC request, its repair windows and its layout
        # evidence are produced by one entrypoint (pic_link_contract).
        framed = LogicalPICRequestFramer(
            repair_window_planner=self.repair_window_planner
        ).frame(
            stitched=stitched,
            plan=plan,
            physical_layout=physical_layout,
            model=self.model,
            generation=self.generation,
            execution_mode=self.execution_mode,
            execution_profile=self.execution_profile,
            quality_profile_id=self.quality_profile_id,
            recompute_initial_tokens=self.recompute_initial_tokens,
            segmentation_policy_version=(
                next(iter(segmentation_policy_versions))
                if segmentation_policy_versions
                else None
            ),
        )
        payload = framed.payload
        repair_windows = framed.repair_windows
        repair_metadata = framed.repair_metadata
        framing_spec = framed.framing_spec
        repair_decision = framed.repair_decision
        link_cache_keys = framed.link_cache_keys
        # Detailed planning evidence stays beside the request. Framing and repair
        # admission are first-class wire fields; hashes are an additional check.
        payload["pic_link_audit"] = PICLinkRequestAudit(
            request_id=plan.manifest.request_id,
            task_id=plan.manifest.task_id,
            selected_order=tuple(stitched.selected_order),
            expected_cache_keys=tuple(stitched.expected_cache_keys),
            framing_policy=asdict(stitched.legolink_framing),
            separator_token_ids=tuple(
                int(value)
                for value in stitched.separator_token_ids
                if isinstance(value, int) and not isinstance(value, bool)
            ),
            equivalence_validation=stitched.equivalence.to_dict(),
            expected_complete_selected_token_count=len(
                stitched.expected_complete_selected_token_ids
            ),
            expected_complete_selected_content_sha256=(
                stitched.selected_content_sha256
            ),
            generation_parameters_sha256=stitched.generation_parameters_sha256,
            repair_window=repair_metadata,
        ).to_dict()
        payload["pic_link_audit"]["execution_mode"] = self.execution_mode
        payload["pic_link_audit"]["framing_spec"] = framing_spec
        payload["pic_link_audit"]["repair_decision"] = repair_decision
        payload["pic_link_audit"]["layout_summary"] = plan.layout_summary.to_dict()
        # Evidence only; nothing reads it back.
        payload["pic_link_audit"]["token_stable_framing"] = (
            physical_layout.framing_diagnostics()
            if physical_layout is not None
            else None
        )
        payload["pic_link_audit"]["block_planning"] = [record.to_dict() for record in plan.planning_records]
        payload["pic_link_audit"]["reuse_eligibility_summary"] = plan.eligibility_summary()
        # Gate 3 accounting with the repair windows this request actually
        # carries. Without explicit windows the runtime applies the scalar K to
        # every linked block, so that is what is recorded. Observe-only.
        repair_by_key = dict(
            zip(
                link_cache_keys,
                repair_windows
                if repair_windows is not None
                else [self.recompute_initial_tokens] * len(link_cache_keys),
            )
        )
        accounting = plan.reuse_accounting(
            repair_tokens_by_cache_key=repair_by_key,
            linked_cache_keys=link_cache_keys,
            route="selected_context_radixcache_legolink",
        )
        payload["pic_link_audit"]["reuse_accounting"] = accounting.to_dict()
        payload["pic_link_audit"]["profitability"] = (
            self.planner.profitability_policy.evaluate(accounting).to_dict()
        )
        return payload, stitched

    def _recover_residency(
        self,
        detail: str,
        linked: list[SemanticSubContext],
        stitched: StitchedContext,
        *,
        only: frozenset[str] | None = None,
    ) -> tuple[tuple[str, ...], int, float]:
        """Turn a reported source miss into a cache miss the host can serve.

        The runtime is the only authority on physical residency, and it has just
        said that these cache keys are not resident. So drop exactly those
        entries from the catalog -- which is what makes the following Compile
        real rather than a duplicate-avoided no-op -- and compile those blocks
        again from the same tokens, under the same semantic identity, the same
        cache key and the same physical layout. Nothing else in the plan moves.

        Returns the keys that were recompiled, the number of Compile requests
        issued and their latency. An empty key tuple means there is nothing to
        retry: either the abort named no key of this request, or a Compile was
        refused.
        """

        classified = self.residency_recovery_detail(detail)
        named = tuple(
            dict.fromkeys(
                classified["missing"] + classified["stale"] + classified["incompatible"]
            )
        )
        if only is not None:  # single-flight: this request owns only some of the named keys
            named = tuple(key for key in named if key in only)
        # One Compile per named key: its first occurrence in prompt order. Every
        # occurrence of a key carries the same compiled ids (framing refuses
        # otherwise), so which occurrence compiles does not change the KV.
        by_key: dict[str, SemanticSubContext] = {}
        for value in linked:
            by_key.setdefault(value.cache_key, value)
        targets = [by_key[key] for key in named if key in by_key]
        if not targets:
            return (), 0, 0.0
        physical_layout = self._normalized_layout(stitched)
        recovered: list[str] = []
        requests = 0
        latency = 0.0
        for value in targets:
            # Residency is the runtime's answer, so the host's entry is wrong by
            # definition here; it is dropped before the Compile, never after.
            self.cache_catalog.invalidate(value.cache_key)
            segment = (
                physical_layout.segment_for_occurrence(value.sub_context_id)
                if physical_layout is not None
                else None
            )
            try:
                _response, elapsed, created = self.compile(
                    value, physical_segment=segment
                )
            except Exception:
                # A refused Compile leaves the key invalidated and recovery
                # incomplete: the caller must not retry the same plan.
                return (), requests, latency
            requests += int(created)
            latency += elapsed
            if created or self.cache_catalog.contains(value.cache_key):
                recovered.append(value.cache_key)
        return tuple(recovered), requests, latency

    def _join_recovery_flight(self, cache_key: str, *, observed: int, evidence: dict[str, Any] | None = None) -> bool:
        """A planned block is absent from the HOST catalog and no runtime abort has said why. Ask the recovery coordinator whether a residency
        recovery of that key explains it (a peer invalidated the entry and is recompiling it).

        Returns True when one does and has finished well (this request waited for it, or it completed after ``observed``): the request goes on
        to the strict runtime Link, whose source preflight is the residency check, and does not depend on this catalog's view of the entry.
        Returns False when nothing explains the miss, so the caller keeps its strict contract. Raises when the recovery that explains it failed,
        with the reason its owner recorded, the same for every request that depended on it. Never sends a Compile and never writes the catalog.
        ``evidence`` is where a successful join is recorded (``link`` passes its own; ``compile`` records on the adapter for ``link`` to pick up).
        """

        coordinator = self.residency_coordinator
        ticket = coordinator.join(self.residency_runtime_id, cache_key, observed=observed)
        if ticket is None:
            return False
        waited_s = 0.0
        if ticket.role == WAITER:
            ok, reason, waited_s = coordinator.wait(ticket)
            if not ok:
                raise SGLangRequestError(
                    f"residency recovery failed ({cache_key}: {reason}); strict link rejected missing cache key {cache_key}",
                    reason="missing",
                )
            self._note_prejoin(cache_key, waiter=True, settled=False, wait_s=waited_s, evidence=evidence)
            return True
        if ticket.role == SETTLED:
            self._note_prejoin(cache_key, waiter=False, settled=True, wait_s=0.0, evidence=evidence)
            return True
        raise SGLangRequestError(
            f"residency recovery failed ({cache_key}: {ticket.error}); strict link rejected missing cache key {cache_key}",
            reason="missing",
        )

    def _note_prejoin(self, cache_key: str, *, waiter: bool, settled: bool, wait_s: float, evidence: dict[str, Any] | None = None) -> None:
        """Evidence of waits that happen BEFORE the Link loop (compile and the strict guard); ``link`` folds it into its result.
        Only a join that went well is recorded: a failed one raises, and must not leak into the next request of this adapter."""

        if evidence is None:
            evidence = self.__dict__.setdefault("_residency_prejoin", {"keys": [], "waiters": 0, "settled": 0, "wait_s": 0.0})
        evidence["keys"].append(cache_key)
        evidence["waiters"] += int(waiter)
        evidence["settled"] += int(settled)
        evidence["wait_s"] += float(wait_s)

    def _take_prejoin(self) -> dict[str, Any]:
        return self.__dict__.pop("_residency_prejoin", None) or {"keys": [], "waiters": 0, "settled": 0, "wait_s": 0.0}

    def _recover_residency_singleflight(
        self,
        detail: str,
        linked: list[SemanticSubContext],
        stitched: StitchedContext,
        *,
        observed: int,
    ) -> ResidencyRecoveryStep:
        """``_recover_residency`` with one recompile per key, however many requests name it.

        Each named key is claimed on the process-wide coordinator: the first request to find it gone *owns* it and recompiles it
        (the unchanged ``_recover_residency`` body, restricted to that key); requests that find it gone while that happens *wait*
        for the owner and retry their Link, and a request whose abort is about a state a recovery has since replaced just retries.
        The owner's completion is in a ``finally``, so no waiter can outlive a failed or crashed owner. All owned keys are finished
        before any wait begins, which is what rules out two requests waiting on each other. Nothing here knows where a block lives.
        """

        classified = self.residency_recovery_detail(detail)
        named = tuple(dict.fromkeys(classified["missing"] + classified["stale"] + classified["incompatible"]))
        present = {value.cache_key for value in linked}
        keys = [key for key in named if key in present]
        if not keys:
            return ResidencyRecoveryStep()
        coordinator, runtime = self.residency_coordinator, self.residency_runtime_id
        tickets = [coordinator.claim(runtime, key, observed=observed) for key in keys]
        recovered: list[str] = []
        compiles, latency, duplicates = 0, 0.0, 0
        own_failure: str | None = None
        owners = [ticket for ticket in tickets if ticket.role == OWNER]
        try:
            for ticket in owners:
                requests, elapsed, ok = 0, 0.0, False
                error: str | None = "recovery owner stopped before the block was compiled again"
                try:
                    if own_failure is None:
                        duplicates += coordinator.note_compile(ticket)
                        healed, requests, elapsed = self._recover_residency(
                            detail, linked, stitched, only=frozenset({ticket.cache_key})
                        )
                        ok = bool(healed)
                        error = None if ok else "the Compile of the block was refused or named no block of this request"
                        recovered.extend(healed)
                    else:
                        error = f"recovery owner aborted after another of its keys failed: {own_failure}"
                except BaseException as raised:  # noqa: BLE001 - the flight must always end, see finally
                    error = f"{type(raised).__name__}: {raised}"
                    raise
                finally:
                    coordinator.complete(
                        ticket, ok=ok, error=error, compile_requests=requests, compile_latency_s=elapsed
                    )
                compiles += requests
                latency += elapsed
                if not ok and own_failure is None:
                    own_failure = error
        finally:
            # Every key this request claimed as OWNER was claimed before the first Compile. An exception out of the
            # first owner's Compile would otherwise leave the later owners' flights registered and never completed:
            # their waiters would sit out the full wait timeout and every later request for those keys would join
            # the dead flight instead of recovering. `complete` is idempotent, so keys already finished are untouched.
            for ticket in owners:
                coordinator.complete(
                    ticket, ok=False, error="recovery owner stopped before the block was compiled again"
                )
        waiters = [ticket for ticket in tickets if ticket.role == WAITER]
        settled = tuple(ticket.cache_key for ticket in tickets if ticket.role == SETTLED)
        counts = {
            "compiles": compiles, "latency_s": latency, "owner_count": len(owners), "waiter_count": len(waiters),
            "settled_count": len(settled), "duplicates": duplicates,
        }
        if own_failure is not None:
            # This request's own Compile was refused: historical behaviour, the original abort is raised (no retry of a stale plan).
            return ResidencyRecoveryStep(wait_s=0.0, **counts)
        waited: list[str] = []
        wait_s = 0.0
        waiter_failure: str | None = None
        for ticket in waiters:
            ok, error, waited_s = self.residency_coordinator.wait(ticket)
            wait_s += waited_s
            if ok:
                waited.append(ticket.cache_key)
            elif waiter_failure is None:
                waiter_failure = f"{ticket.cache_key}: {error}"
        return ResidencyRecoveryStep(
            recovered=tuple(recovered), waited=(*waited, *settled), wait_s=wait_s, failure=waiter_failure, **counts
        )

    def link(
        self,
        manifest: AgentContextManifest,
        contexts: list[SemanticSubContext],
        *,
        decisions: tuple[CompileDecision, ...] | None = None,
        execution_plan: NOCRequestExecutionPlan | None = None,
        schedule: NOCRequestExecutionPlan | None = None,
        stitched: StitchedContext | None = None,
    ) -> AdapterResult:
        if execution_plan is not None and schedule is not None:
            if execution_plan != schedule:
                raise ValueError(
                    "execution_plan and deprecated schedule argument differ"
                )
        plan = execution_plan or schedule
        if plan is None:
            if decisions is None:
                plan = self.planner.plan(manifest, contexts)
            else:
                plan = self.planner.bind_decisions(
                    manifest,
                    contexts,
                    decisions,
                )
        elif decisions is not None and decisions != plan.decisions:
            raise ValueError("link decisions do not match the supplied execution plan")
        decisions = plan.decisions
        # Read before anything else of this request: a recovery of a planned key that finishes after this point explains a later catalog miss.
        observed_at_start = self.residency_coordinator.completions
        prejoin = self._take_prejoin()  # waits of this request's compile step, recorded by compile() before link() was called
        payload, stitched = self._link_payload(plan, stitched)
        linked = list(stitched.linked_contexts)
        if not linked:
            if self.mode == OperationMode.STRICT_EXPERIMENTAL:
                raise SGLangRequestError(
                    "strict link requires at least one compiled block",
                    reason="missing",
                )
            return self.native_selected(
                manifest,
                contexts,
                fallback_reason="no_reusable_blocks",
                plan=plan,
                stitched=stitched,
            )
        by_id = {value.sub_context_id: value for value in contexts}
        # The plan's blocks win: a block served by another namespace's entry is
        # presented there under that entry's cache key, and that is the key the
        # Link names and the catalog must already hold.
        by_id.update({value.sub_context_id: value for value in plan.ordered_contexts})
        compile_requests = 0
        compile_latency = 0.0
        for decision in decisions:
            if decision.action != CompileAction.COMPILE:
                continue
            value = by_id[decision.sub_context_id]
            if not self.cache_catalog.contains(value.cache_key):
                if self.mode == OperationMode.STRICT_EXPERIMENTAL:
                    # A catalog miss that a concurrent residency recovery explains (its owner emptied the entry on purpose and is
                    # putting the block back) is not an invalid plan: wait for it and let the strict runtime preflight decide. Any
                    # other miss keeps the strict contract.
                    if self._join_recovery_flight(value.cache_key, observed=observed_at_start, evidence=prejoin):
                        continue
                    raise SGLangRequestError(
                        f"strict link rejected missing cache key {value.cache_key}",
                        reason="missing",
                    )
                try:
                    _response, elapsed, created = self.compile(value)
                except Exception:
                    return self.native_selected(
                        manifest,
                        contexts,
                        fallback_reason="compile_failure_native_prefill",
                        plan=plan,
                        stitched=stitched,
                    )
                compile_requests += int(created)
                compile_latency += elapsed
        selected_content = stitched.selected_text
        recovery_attempts = 0
        recovered_keys: tuple[str, ...] = ()
        waited_keys: tuple[str, ...] = ()
        recovery_compiles = 0
        owner_count = waiter_count = settled_count = duplicate_compiles = 0
        wait_time_s = 0.0
        response = None
        request_latency = 0.0
        last_error: Exception | None = None
        while response is None:
            # read before the Link is sent: a recovery that completes after this point makes a missing-source abort of this attempt stale
            observed = self.residency_coordinator.completions
            try:
                started = time.perf_counter()
                response = self.patched_transport(payload)
                request_latency = time.perf_counter() - started
            except Exception as error:  # noqa: BLE001 - classified below
                last_error = error
                reason = (
                    error.reason
                    if isinstance(error, SGLangRequestError)
                    else self.classify_failure(str(error))
                )
                if (
                    self.residency_recovery
                    and recovery_attempts < self.residency_recovery_attempts
                    and reason in {"missing", "stale", "incompatible"}
                ):
                    step = self._recover_residency_singleflight(
                        str(error), linked, stitched, observed=observed
                    )
                    compile_requests += step.compiles
                    compile_latency += step.latency_s
                    recovery_compiles += step.compiles
                    owner_count += step.owner_count
                    waiter_count += step.waiter_count
                    settled_count += step.settled_count
                    wait_time_s += step.wait_s
                    duplicate_compiles += step.duplicates
                    if step.failure is not None:
                        # a recovery this request waited on failed: every waiter of that flight raises the same reason
                        last_error = SGLangRequestError(
                            f"residency recovery failed ({step.failure}); {error}",
                            reason=reason,
                            status=getattr(error, "status", None),
                        )
                        break
                    if step.retry:
                        recovery_attempts += 1
                        recovered_keys += step.recovered
                        waited_keys += step.waited
                        # The plan is no longer the stale one: every key the
                        # runtime named as gone has been registered again under
                        # the same identity, so the same payload now describes
                        # blocks that exist. A retry without a completed
                        # recompile would just resend the stale plan, which is
                        # why `healed` gates it.
                        continue
                break
        if response is None:
            error = last_error
            reason = (
                error.reason
                if isinstance(error, SGLangRequestError)
                else self.classify_failure(str(error))
            )
            if self.mode == OperationMode.STRICT_EXPERIMENTAL:
                raise error
            if reason in {"missing", "stale", "incompatible"}:
                for value in linked:
                    if reason in {"stale", "incompatible"}:
                        self.cache_catalog.invalidate(value.cache_key)
                    try:
                        _response, elapsed, created = self.compile(value)
                    except Exception:
                        return self.native_selected(
                            manifest,
                            contexts,
                            fallback_reason=f"{reason}_recompile_failure_native_prefill",
                            plan=plan,
                            stitched=stitched,
                        )
                    compile_requests += int(created)
                    compile_latency += elapsed
                try:
                    started = time.perf_counter()
                    response = self.patched_transport(payload)
                    request_latency = time.perf_counter() - started
                except Exception:
                    return self.native_selected(
                        manifest,
                        contexts,
                        fallback_reason=f"{reason}_native_prefill",
                        plan=plan,
                        stitched=stitched,
                    )
            else:
                return self.native_selected(
                    manifest,
                    contexts,
                    fallback_reason=f"{reason}_native_prefill",
                    plan=plan,
                    stitched=stitched,
                )
        # The Link completed: register the names it used so each logical root shows which physical block it
        # references (the planner stays side-effect free, so this is the one place aliases are written).
        self.namespace_resolver.record_aliases(
            plan.namespace_resolutions,
            self.cache_catalog,
            linked_block_ids=[value.sub_context_id for value in linked],
        )
        return AdapterResult(
            content=response_content(response),
            route="selected_context_radixcache_legolink",
            request_payload=payload,
            response=response,
            compile_decisions=decisions,
            compile_requests=compile_requests,
            duplicate_compile_avoided=self.duplicate_compile_avoided,
            compile_latency_s=compile_latency,
            request_latency_s=request_latency,
            selected_content=selected_content,
            residency_recovery_attempts=recovery_attempts,
            residency_recovered_cache_keys=recovered_keys,
            residency_recovery_compiles=recovery_compiles,
            residency_waited_cache_keys=(*prejoin["keys"], *waited_keys),
            residency_recovery_owner_count=owner_count,
            residency_recovery_waiter_count=waiter_count + prejoin["waiters"],
            residency_recovery_settled_count=settled_count + prejoin["settled"],
            residency_recovery_wait_time_s=wait_time_s + prejoin["wait_s"],
            residency_duplicate_recovery_compile_count=duplicate_compiles,
        )
