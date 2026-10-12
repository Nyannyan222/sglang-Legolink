"""The one logical contract a host may send to the patched SGLang runtime.

Three builders used to construct `pic_link` by hand -- the bridge for plan-file
microbenchmarks, the adapter for the live NOC pipeline, the Terminal-Bench agent
for its own requests -- and their field sets barely overlapped. The performance
result and the correctness result were therefore measured through two different
request shapes, which is not a thing a paper can claim is one system.

The division of labour this module encodes:

* the **host** says which semantic blocks to reuse, in what order, under what
  repair-window policy, and (when it owns the chat template) exactly which
  tokens those blocks are;
* the **runtime** derives every physical consequence -- block ranges, global
  offsets, the mutable boundary, sparse-repair positions, KV slots.

That was already the runtime's behaviour: ``PICRequestTokenRegistry
._prepare_link`` overwrote every physical field a host sent. What was missing
was a contract saying so, which is why a stale ``imp_indices`` in a saved plan
file still looked like it was steering the run. The runtime now refuses those
fields; this module is how a host stays on the right side of that.

Correctness evidence -- what the host expected to reuse, how a repair window was
chosen, which framing policy produced the text -- is not part of the request. It
belongs in :class:`PICLinkRequestAudit`, which is written beside the payload
rather than into it. Formal execution requires FramingSpec and RepairDecision on the wire.
Token hashes additionally prove consistency; they cannot prove canonical framing.
The installed deployment profile supplies the independent admission evidence.

ARCHITECTURE_STAGE: 8 -- PIC Contract + Logical Request Framing
CANONICAL_ENTRYPOINT: LogicalPICRequestFramer.frame (bottom of this module); the
    field lists, ``PICLinkRequest`` and ``validate_pic_link_request`` above it are
    the contract it produces.
DELEGATED_HELPERS: physical_framing.py, repair_window.py (Stage 7),
    token_accounting.py, authoritative_tokens.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from .execution_governance import validate_execution_intent

# Kept in step with `pic_runtime.PIC_LINK_*` by
# `tests/test_pic_link_contract.py::test_the_host_and_runtime_field_lists_agree`,
# which loads the runtime file directly. The runtime copy is authoritative --
# it is the one that rejects -- and this copy exists because the host cannot
# import from a checkout that is gitignored and Linux-only.
PIC_LINK_IDENTITY_FIELDS = frozenset(
    {
        "request_id",
        "task_id",
        "variant",
        "profiling_warmup",
        "concurrency",
    }
)

PIC_LINK_SELECTION_FIELDS = frozenset({"cache_keys", "block_ids"})

PIC_LINK_COMPATIBILITY_FIELDS = frozenset(
    {
        "adapter_id",
        "segmentation_policy_version",
        "legolink_policy_version",
    }
)

PIC_LINK_REPAIR_POLICY_FIELDS = frozenset(
    {
        "recompute_mode",
        "recompute_initial_tokens",
        "recompute_first_block",
        "recompute_first_block_tokens",
        "recompute_tokens_per_block",
    }
)

PIC_LINK_TOKEN_LAYOUT_FIELDS = frozenset(
    {
        "physical_token_layout_version",
        "physical_input_ids",
        "physical_token_count",
        "physical_token_ids_sha256",
        "physical_prefix_token_count",
        "physical_block_segments",
        "expected_complete_token_count",
        "expected_complete_token_ids_sha256",
        # EXPERIMENTAL (indexed exact sub-context reuse), refused by a runtime
        # that has not enabled SGLANG_PIC_EXPERIMENTAL_INDEXED_MIXED_PREFILL.
        # How many host-owned fresh tokens sit between consecutive cached
        # blocks -- a count of host text, like the prefix count, not a slot or
        # a position.
        "physical_interblock_fresh_token_counts",
    }
)

PIC_LINK_DIAGNOSTIC_FIELDS = frozenset(
    {
        "diagnostic_flashinfer_reuse_paged_wrapper_across_plans",
        "diagnostic_flashinfer_skip_exact_paged_metadata_cache",
        "diagnostic_sparse_repair_record_function",
        "diagnostic_sparse_repair_row_batching",
        "diagnostic_sparse_repair_rowwise_ffn",
        "diagnostic_sparse_repair_rowwise_output_projection",
    }
)

#: EXPERIMENTAL, off by default in the runtime. A host may ask that a
#: successfully reconstructed execution group be promoted into RadixCache, and
#: name the group. Both are semantic intent: which members were grouped, and
#: what the group is called. The slots, the positions and the decision to
#: refuse remain the runtime's, which is why the key is sendable and the
#: layout is not.
PIC_LINK_EXPERIMENTAL_PROMOTION_FIELDS = frozenset(
    {
        "promote_reconstructed_group",
        "execution_group_key",
    }
)

#: Accepted by the runtime, never emitted here. `recompute_initial_tokens` says
#: the same thing, and one name for one quantity is the point of this module.
PIC_LINK_DEPRECATED_ALIASES = frozenset({"legolink_k"})

PIC_LINK_GOVERNANCE_FIELDS = frozenset({"execution_mode", "framing_spec", "repair_decision"})

PIC_LINK_REQUEST_FIELDS = (
    PIC_LINK_GOVERNANCE_FIELDS |
    PIC_LINK_IDENTITY_FIELDS
    | PIC_LINK_SELECTION_FIELDS
    | PIC_LINK_COMPATIBILITY_FIELDS
    | PIC_LINK_REPAIR_POLICY_FIELDS
    | PIC_LINK_TOKEN_LAYOUT_FIELDS
    | PIC_LINK_DIAGNOSTIC_FIELDS
    | PIC_LINK_EXPERIMENTAL_PROMOTION_FIELDS
    | PIC_LINK_DEPRECATED_ALIASES
)

#: The runtime derives these and rejects a request that carries one.
PIC_LINK_RUNTIME_OWNED_FIELDS = frozenset(
    {
        "attention_ranges",
        "block_ranges",
        "imp_indices",
        "kvlink",
        "mutable_start_token",
        "mutable_token_count",
        "recompute_positions",
        "token_layout_source",
        "total_token_count",
    }
)


class PICLinkContractError(ValueError):
    """A request that the runtime would reject, caught before it is sent."""


@dataclass(frozen=True)
class PICLinkRequest:
    """What one Link request asks for, and nothing about how it will execute.

    Every field here is either something the runtime cannot derive (which blocks,
    in what order, how much repair) or something it checks rather than trusts
    (the token layout hashes).
    """

    request_id: str
    task_id: str
    #: Compiled blocks to reuse, in the order they should appear. This ordering
    #: is the whole point of block-level reuse: RadixCache prefix matching
    #: cannot express it, and it is what the runtime turns into global offsets.
    cache_keys: tuple[str, ...]
    #: Semantic labels for the same blocks, used only to name them in the trace.
    #: Defaults to `cache_keys` when a caller has no separate identity for them.
    block_ids: tuple[str, ...] = ()

    recompute_initial_tokens: int = 16
    recompute_mode: str = "batched"
    recompute_first_block: bool | None = None
    recompute_first_block_tokens: int | None = None
    #: One repair window per cache key. When present the runtime honours it
    #: exactly; otherwise it derives windows from the two scalars above.
    recompute_tokens_per_block: tuple[int, ...] | None = None

    #: Declared so a Link binds only to a Compile that declared the same. The
    #: runtime folds these into the RadixCache lookup key.
    adapter_id: str | None = None
    segmentation_policy_version: str | None = None
    legolink_policy_version: str | None = None

    variant: str | None = None
    profiling_warmup: bool | None = None
    concurrency: int | None = None

    #: The exact chat-template tokenization, when the host owns it. Keys are
    #: `PIC_LINK_TOKEN_LAYOUT_FIELDS`; the runtime hashes what it holds and
    #: raises on any disagreement, so this is an assertion, not an instruction.
    token_layout: Mapping[str, Any] | None = None

    execution_mode: str = "legacy_debug"
    framing_spec: Mapping[str, Any] | None = None
    repair_decision: Mapping[str, Any] | None = None
    execution_profile: Mapping[str, Any] | None = field(default=None, repr=False)
    diagnostics: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cache_keys:
            raise PICLinkContractError("a Link request must name at least one block")
        if self.recompute_initial_tokens < 0:
            raise PICLinkContractError("recompute_initial_tokens must be non-negative")
        if self.recompute_tokens_per_block is not None and len(
            self.recompute_tokens_per_block
        ) != len(self.cache_keys):
            raise PICLinkContractError(
                "recompute_tokens_per_block must hold one window per cache key"
            )
        if self.block_ids and len(self.block_ids) != len(self.cache_keys):
            raise PICLinkContractError("block_ids must hold one label per cache key")
        if self.token_layout is not None:
            unexpected = sorted(set(self.token_layout) - PIC_LINK_TOKEN_LAYOUT_FIELDS)
            if unexpected:
                raise PICLinkContractError(
                    f"token_layout has non-layout fields: {', '.join(unexpected)}"
                )
        for name in self.diagnostics:
            if name not in PIC_LINK_DIAGNOSTIC_FIELDS:
                raise PICLinkContractError(f"unknown diagnostic switch: {name}")

    def to_wire(self) -> dict[str, Any]:
        """The `pic_link` object exactly as the runtime accepts it."""

        wire: dict[str, Any] = {
            "request_id": str(self.request_id),
            "task_id": str(self.task_id),
            "cache_keys": [str(value) for value in self.cache_keys],
            "block_ids": [
                str(value)
                for value in (self.block_ids or self.cache_keys)
            ],
            "recompute_mode": str(self.recompute_mode),
            "recompute_initial_tokens": int(self.recompute_initial_tokens),
        }
        if self.recompute_first_block is not None:
            wire["recompute_first_block"] = bool(self.recompute_first_block)
        if self.recompute_first_block_tokens is not None:
            wire["recompute_first_block_tokens"] = int(
                self.recompute_first_block_tokens
            )
        if self.recompute_tokens_per_block is not None:
            wire["recompute_tokens_per_block"] = [
                int(value) for value in self.recompute_tokens_per_block
            ]
        for name in ("variant", "adapter_id", "segmentation_policy_version",
                     "legolink_policy_version"):
            value = getattr(self, name)
            if value is not None:
                wire[name] = str(value)
        if self.profiling_warmup is not None:
            wire["profiling_warmup"] = bool(self.profiling_warmup)
        if self.concurrency is not None:
            wire["concurrency"] = int(self.concurrency)
        if self.token_layout:
            wire.update(dict(self.token_layout))
        wire.update({name: bool(value) for name, value in self.diagnostics.items()})
        wire["execution_mode"] = self.execution_mode
        if self.framing_spec is not None:
            wire["framing_spec"] = dict(self.framing_spec)
        if self.repair_decision is not None:
            wire["repair_decision"] = dict(self.repair_decision)
        validate_pic_link_request(wire, profile=self.execution_profile)
        return wire


@dataclass(frozen=True)
class PICLinkRequestAudit:
    """Host-side evidence about one Link request, written beside the payload.

    None of this reaches the runtime. It used to: `expected_cache_keys`,
    `equivalence_validation`, `framing_policy`, `separator_token_ids` and
    `selected_order` were all serialized into `pic_link` and read by nothing on
    the server. Keeping them is right -- they are how a run defends its own
    correctness -- but a request contract is the wrong place, because a field
    nobody reads looks like a field that works.
    """

    request_id: str
    task_id: str
    #: The order the selector chose, before it became `cache_keys`. Identical in
    #: practice; kept separately so a divergence is visible rather than assumed.
    selected_order: tuple[str, ...] = ()
    #: Blocks the host believed were already compiled. A mismatch against
    #: `cache_keys` is a reuse-accounting bug, not a runtime error.
    expected_cache_keys: tuple[str, ...] = ()
    #: How the selected text was framed. This already took effect during host
    #: tokenization; the runtime verifies the resulting token hash instead.
    framing_policy: Mapping[str, Any] | None = None
    separator_token_ids: tuple[int, ...] = ()
    equivalence_validation: Mapping[str, Any] | None = None
    expected_complete_selected_token_count: int | None = None
    expected_complete_selected_content_sha256: str | None = None
    generation_parameters_sha256: str | None = None
    #: Why each repair window was chosen. The windows themselves travel on the
    #: wire as `recompute_tokens_per_block`; the rationale stays here.
    repair_window: Mapping[str, Any] = field(default_factory=dict)
    #: The host's own estimate of the physical layout, for reports. The runtime
    #: recomputes all of it, so a difference is a measurement note, not a fault.
    estimated_layout: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "selected_order": list(self.selected_order),
            "expected_cache_keys": list(self.expected_cache_keys),
            "separator_token_ids": list(self.separator_token_ids),
        }
        if self.framing_policy is not None:
            record["framing_policy"] = dict(self.framing_policy)
        if self.equivalence_validation is not None:
            record["equivalence_validation"] = dict(self.equivalence_validation)
        for name in (
            "expected_complete_selected_token_count",
            "expected_complete_selected_content_sha256",
            "generation_parameters_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                record[name] = value
        if self.repair_window:
            record["repair_window"] = dict(self.repair_window)
        if self.estimated_layout:
            record["estimated_layout"] = dict(self.estimated_layout)
        return record


def validate_pic_link_request(wire: Mapping[str, Any], *, profile=None, require_formal=False) -> None:
    """Refuse here what the runtime would refuse there.

    Host-side so a malformed request fails in a unit test rather than as an
    HTTP 500 halfway through a benchmark.
    """

    physical = sorted(set(wire) & PIC_LINK_RUNTIME_OWNED_FIELDS)
    if physical:
        raise PICLinkContractError(
            "the runtime derives these and rejects a request that sends them: "
            + ", ".join(physical)
        )
    unknown = sorted(set(wire) - PIC_LINK_REQUEST_FIELDS)
    if unknown:
        raise PICLinkContractError(
            "unknown pic_link fields: "
            + ", ".join(unknown)
            + " (host evidence belongs in PICLinkRequestAudit)"
        )

    try:
        validate_execution_intent(wire, profile, require_formal=require_formal)
    except ValueError as error:
        raise PICLinkContractError(str(error)) from error


#: Fields the pre-contract builders wrote that belong nowhere on the wire.
#: Some were host evidence, some were a second name for a field that stayed, and
#: some declared a policy whose *result* the runtime verifies instead.
LEGACY_PIC_LINK_FIELDS = frozenset(
    {
        "compile_attention_policy",
        "decode_attention_policy",
        "destination_offsets",
        "equivalence_validation",
        "expected_cache_keys",
        "expected_complete_selected_content_sha256",
        "expected_complete_selected_token_count",
        "framing_policy",
        "generation_parameters_sha256",
        "legolink_k",
        "link_attention_policy",
        "model",
        "mutable_suffix_required",
        "mutable_text",
        "nonprefix_order_label",
        "nonprefix_permutation",
        "position_policy",
        "recompute_ranges",
        "rope_position_deltas",
        "selected_order",
        "semantic_block_ids",
        "separator_token_ids",
        "storage_mode",
        "stress_generation",
    }
)

#: What a block reordering invalidates. The token override was hashed against
#: one specific order; carrying it into a permuted plan makes the runtime raise
#: on a mismatch that is really a stale artefact.
PIC_LINK_ORDER_DEPENDENT_FIELDS = (
    PIC_LINK_TOKEN_LAYOUT_FIELDS
    | PIC_LINK_RUNTIME_OWNED_FIELDS
    | {"recompute_tokens_per_block"}
)


def strip_stale_link_fields(
    wire: dict[str, Any], *, order_changed: bool = True
) -> list[str]:
    """Drop what a rewritten plan may no longer assert. Returns what went.

    Used by the plan-surgery scripts, which take a prepared plan and change
    which blocks it links or in what order. Everything removed here is either
    derived by the runtime or an assertion about a layout that no longer holds.
    """

    removable = set(LEGACY_PIC_LINK_FIELDS) | set(PIC_LINK_RUNTIME_OWNED_FIELDS)
    if order_changed:
        removable |= PIC_LINK_ORDER_DEPENDENT_FIELDS
    removed = sorted(set(wire) & removable)
    for name in removed:
        wire.pop(name, None)
    return removed


#: Where each pre-contract field lands in the audit record. A saved plan is
#: migrated rather than regenerated: re-running the extractor could select
#: different blocks or segment them differently, and then the artefact a
#: published number was measured on would quietly stop being that artefact.
_MIGRATE_TO_ESTIMATE = {
    "block_ranges": "attention_ranges",
    "attention_ranges": "attention_ranges",
    "mutable_start_token": "mutable_start_token",
    "mutable_token_count": "mutable_token_count",
    "total_token_count": "total_token_count",
    "recompute_positions": "recompute_positions",
    "imp_indices": "recompute_positions",
    "kvlink": "recompute_positions",
    "recompute_ranges": "recompute_ranges",
    "position_policy": "position_policy",
    "compile_attention_policy": "compile_attention_policy",
    "link_attention_policy": "link_attention_policy",
    "decode_attention_policy": "decode_attention_policy",
    "destination_offsets": "destination_offsets",
    "rope_position_deltas": "rope_position_deltas",
}

_MIGRATE_TO_AUDIT = (
    "selected_order",
    "expected_cache_keys",
    "framing_policy",
    "separator_token_ids",
    "equivalence_validation",
    "expected_complete_selected_token_count",
    "expected_complete_selected_content_sha256",
    "generation_parameters_sha256",
    "nonprefix_permutation",
    "nonprefix_order_label",
    "stress_generation",
)

#: Carried nothing the runtime read and nothing a report needs.
_MIGRATE_DROP = ("mutable_suffix_required", "storage_mode", "model", "mutable_text")


class PICLinkMigrationError(ValueError):
    """A saved plan this migration would have to guess about."""


def migrate_legacy_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Upgrade one saved chat payload onto the contract. Returns (payload, notes).

    Idempotent: a payload already on the contract comes back unchanged with no
    notes. Nothing is invented -- every value either moves to the audit record
    or is dropped because it duplicated a field that stays.
    """

    result = {key: value for key, value in payload.items()}
    link = dict(result.get("pic_link") or {})
    if not link:
        return result, []

    audit = dict(result.get("pic_link_audit") or {})
    estimate = dict(audit.get("estimated_layout") or {})
    notes: list[str] = []

    # legolink_k is the older name for recompute_initial_tokens, and the runtime
    # read the latter with the former as fallback. Dropping it silently would be
    # wrong if a plan ever set only one, or set them to different values.
    if "legolink_k" in link:
        alias = link.pop("legolink_k")
        canonical = link.get("recompute_initial_tokens")
        if canonical is None:
            link["recompute_initial_tokens"] = alias
            notes.append("legolink_k promoted to recompute_initial_tokens")
        elif canonical != alias:
            raise PICLinkMigrationError(
                f"legolink_k={alias!r} disagrees with "
                f"recompute_initial_tokens={canonical!r}; the runtime used the "
                "latter, so this plan did not repair what it claimed"
            )
        else:
            notes.append("legolink_k dropped (duplicate)")

    if "semantic_block_ids" in link:
        duplicate = link.pop("semantic_block_ids")
        if list(duplicate) != list(link.get("block_ids") or []):
            raise PICLinkMigrationError(
                "semantic_block_ids disagrees with block_ids; the runtime read "
                "block_ids, so the two labelled different blocks"
            )
        notes.append("semantic_block_ids dropped (duplicate)")

    for name in sorted(set(link) & set(_MIGRATE_TO_ESTIMATE)):
        target = _MIGRATE_TO_ESTIMATE[name]
        value = link.pop(name)
        estimate.setdefault(target, value)
        notes.append(f"{name} -> pic_link_audit.estimated_layout.{target}")

    for name in _MIGRATE_TO_AUDIT:
        if name in link:
            audit.setdefault(name, link.pop(name))
            notes.append(f"{name} -> pic_link_audit.{name}")

    repair = {name: link.pop(name) for name in sorted(link) if name.startswith("repair_window_")}
    if repair:
        audit.setdefault("repair_window", {}).update(repair)
        notes.append(f"{len(repair)} repair_window_* -> pic_link_audit.repair_window")

    for name in _MIGRATE_DROP:
        if name in link:
            link.pop(name)
            notes.append(f"{name} dropped (no consumer)")

    if not notes:
        return result, []

    audit.setdefault("request_id", str(link.get("request_id") or ""))
    audit.setdefault("task_id", str(link.get("task_id") or ""))
    if estimate:
        audit["estimated_layout"] = estimate

    validate_pic_link_request(link)
    result["pic_link"] = link
    result["pic_link_audit"] = audit
    return result, notes


def build_pic_link_request(
    *,
    request_id: str,
    task_id: str,
    cache_keys: Sequence[str],
    block_ids: Sequence[str] | None = None,
    recompute_initial_tokens: int = 16,
    recompute_mode: str = "batched",
    recompute_first_block: bool | None = None,
    recompute_first_block_tokens: int | None = None,
    recompute_tokens_per_block: Sequence[int] | None = None,
    adapter_id: str | None = None,
    segmentation_policy_version: str | None = None,
    legolink_policy_version: str | None = None,
    variant: str | None = None,
    profiling_warmup: bool | None = None,
    concurrency: int | None = None,
    token_layout: Mapping[str, Any] | None = None,
    execution_mode: str = "legacy_debug",
    framing_spec: Mapping[str, Any] | None = None,
    repair_decision: Mapping[str, Any] | None = None,
    execution_profile: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, bool] | None = None,
) -> PICLinkRequest:
    """The canonical builder. Every producer goes through here."""

    return PICLinkRequest(
        execution_mode=execution_mode,
        framing_spec=framing_spec,
        repair_decision=repair_decision,
        execution_profile=execution_profile,
        request_id=str(request_id),
        task_id=str(task_id),
        cache_keys=tuple(str(value) for value in cache_keys),
        block_ids=tuple(str(value) for value in (block_ids or ())),
        recompute_initial_tokens=int(recompute_initial_tokens),
        recompute_mode=str(recompute_mode),
        recompute_first_block=recompute_first_block,
        recompute_first_block_tokens=recompute_first_block_tokens,
        recompute_tokens_per_block=(
            tuple(int(value) for value in recompute_tokens_per_block)
            if recompute_tokens_per_block is not None
            else None
        ),
        adapter_id=adapter_id,
        segmentation_policy_version=segmentation_policy_version,
        legolink_policy_version=legolink_policy_version,
        variant=variant,
        profiling_warmup=profiling_warmup,
        concurrency=concurrency,
        token_layout=dict(token_layout) if token_layout else None,
        diagnostics=dict(diagnostics or {}),
    )


def build_pic_link_payload(
    request: PICLinkRequest,
    *,
    model: str,
    mutable_text: str,
    generation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One chat-completions payload carrying one canonical Link request."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": mutable_text}],
    }
    payload.update(dict(generation or {}))
    payload["pic_link"] = request.to_wire()
    return payload


@dataclass(frozen=True)
class FramedPICLinkRequest:
    """What Stage 8 hands to Stage 9: the logical request and how it was framed.

    ``request`` and ``payload`` are the wire contract. Everything else is the
    framing evidence the adapter records beside the payload (never inside it):
    the repair windows and why, the framing spec and repair decision when a
    formal execution profile supplied them, and the cache keys / block ids the
    Link actually names (fewer than the selection under token-stable framing).
    """

    request: PICLinkRequest
    payload: dict[str, Any]
    repair_windows: tuple[int, ...] | None
    repair_metadata: dict[str, Any]
    framing_spec: Any
    repair_decision: Any
    link_cache_keys: tuple[str, ...]
    link_block_ids: tuple[str, ...]


class LogicalPICRequestFramer:
    """Stage 8 entrypoint: produce the complete *logical* ``pic_link`` request.

    ARCHITECTURE_STAGE: 8 -- PIC Contract + Logical Request Framing
    CANONICAL_ENTRYPOINT: LogicalPICRequestFramer.frame (and .build_layout for the
        token layout it frames against)
    DELEGATED_HELPERS: physical_framing.py (normalized chat-template layout),
        repair_window.py (Stage 7 -- the per-block windows are asked for here
        because the physical layout is the only place global offsets are known
        before the runtime rebuilds the plan), token_accounting.py,
        authoritative_tokens.py (the tokenizer identity the layout is built from).

    Ownership. The host may name: semantic block ids, cache keys, logical token
    order, compile/reuse intent, the repair policy and the authoritative token
    sequence. The host must not name physical KV slots, ``block_ranges``,
    ``attention_ranges``, global physical positions, ``imp_indices``, ``kvlink``
    or ``recompute_positions``; those are derived by the runtime and refused if
    sent (``validate_pic_link_request``). This class only ever builds a
    ``PICLinkRequest`` through ``build_pic_link_request``, so it cannot emit a
    runtime-owned field.

    This is the code that used to sit inline in
    ``NOCSGLangAdapter._link_payload``, moved without change of order or
    content; the adapter keeps request-error mapping and the audit block.
    """

    def __init__(self, *, repair_window_planner: Any) -> None:
        self.repair_window_planner = repair_window_planner

    @staticmethod
    def build_layout(
        stitched: Any,
        chat_tokenizer: Any,
        generation: Mapping[str, Any],
    ) -> Any:
        """The normalized chat token layout; raises ``PhysicalTokenLayoutError``."""

        from .physical_framing import build_normalized_chat_token_layout

        return build_normalized_chat_token_layout(
            stitched,
            chat_tokenizer,
            tools=generation.get("tools"),
            tool_choice=generation.get("tool_choice"),
            reasoning_effort=generation.get("reasoning_effort"),
            continue_final_message=bool(generation.get("continue_final_message", False)),
            chat_template_kwargs=(
                dict(generation["chat_template_kwargs"])
                if isinstance(generation.get("chat_template_kwargs"), dict)
                else None
            ),
        )

    def frame(
        self,
        *,
        stitched: Any,
        plan: Any,
        physical_layout: Any,
        model: str,
        generation: Mapping[str, Any],
        execution_mode: str,
        execution_profile: Mapping[str, Any] | None,
        quality_profile_id: str | None,
        recompute_initial_tokens: int,
        segmentation_policy_version: str | None,
    ) -> FramedPICLinkRequest:
        from .repair_window import build_repair_window_blocks

        repair_windows: tuple[int, ...] | None = None
        repair_metadata: dict[str, Any] = {}
        if physical_layout is not None:
            # Per-block repair windows. The exact physical layout is the only
            # place global offsets are known before the scheduler rebuilds the
            # plan, so the window decision is made here and carried across.
            repair_blocks = build_repair_window_blocks(
                physical_layout, stitched.linked_contexts
            )
            if repair_blocks:
                decision = self.repair_window_planner.plan(
                    repair_blocks,
                    validated_k=recompute_initial_tokens,
                    task_id=str(plan.manifest.task_id),
                )
                repair_windows = tuple(decision.windows)
                # Why each window was chosen is evidence, not instruction: the
                # windows themselves are what the runtime acts on.
                repair_metadata = decision.to_metadata(repair_blocks)

        framing_spec = None
        repair_decision = None
        if execution_profile is not None:
            framing_spec = execution_profile["framing_spec"]
            if (framing_spec["model_id"] != model or physical_layout is None
                    or framing_spec["tokenizer_fingerprint"] != physical_layout.tokenizer_fingerprint):
                raise ValueError("adapter/model/tokenizer framing profile mismatch")
            quality = execution_profile.get("repair_quality_profiles", {}).get(quality_profile_id)
            repair_decision = self.repair_window_planner.execution_decision(
                decision, framing_layout_version=physical_layout.version,
                quality_profile=quality, workload_class=str(plan.manifest.task_id),
            ).to_dict()
        link_cache_keys = stitched.cache_keys
        link_block_ids = stitched.block_ids
        if (
            physical_layout is not None
            and physical_layout.linked_block_indices is not None
        ):
            # Token-stable framing: the Link names only the blocks it reuses.
            # The others are fresh tokens in this request -- still selected,
            # still compiled and cached under their unchanged identity and key.
            link_cache_keys = tuple(
                stitched.cache_keys[index]
                for index in physical_layout.linked_block_indices
            )
            link_block_ids = tuple(
                stitched.block_ids[index]
                for index in physical_layout.linked_block_indices
            )
        request = build_pic_link_request(
            execution_mode=execution_mode,
            framing_spec=framing_spec,
            repair_decision=repair_decision,
            execution_profile=execution_profile,
            request_id=plan.manifest.request_id,
            task_id=plan.manifest.task_id,
            variant="selected_context_radixcache_legolink",
            cache_keys=link_cache_keys,
            block_ids=link_block_ids,
            recompute_mode="batched",
            recompute_initial_tokens=recompute_initial_tokens,
            recompute_first_block=(
                bool(repair_windows[0]) if repair_windows else True
            ),
            recompute_first_block_tokens=(
                repair_windows[0] if repair_windows else recompute_initial_tokens
            ),
            recompute_tokens_per_block=repair_windows,
            segmentation_policy_version=segmentation_policy_version,
            legolink_policy_version=(
                physical_layout.version if physical_layout is not None else None
            ),
            token_layout=(
                physical_layout.link_metadata() if physical_layout is not None else None
            ),
        )
        payload = build_pic_link_payload(
            request,
            model=model,
            mutable_text=stitched.mutable_text,
            generation=generation,
        )
        return FramedPICLinkRequest(
            request=request,
            payload=payload,
            repair_windows=repair_windows,
            repair_metadata=repair_metadata,
            framing_spec=framing_spec,
            repair_decision=repair_decision,
            link_cache_keys=link_cache_keys,
            link_block_ids=link_block_ids,
        )
