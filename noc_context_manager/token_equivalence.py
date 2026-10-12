"""Audit physical token equivalence across baseline and Compile/Link requests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .physical_framing import (
    PHYSICAL_TOKEN_LAYOUT_VERSION,
    build_normalized_chat_token_layout,
)
from .stitch import StitchedContext
from .tokenizer import ChatTemplateTokenizer


def _token_hash(values: Sequence[Any]) -> str:
    encoded = json.dumps(
        list(values),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_mismatch(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
) -> int | None:
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        if left != right:
            return index
    if len(baseline) != len(candidate):
        return min(len(baseline), len(candidate))
    return None


def _common_suffix_length(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    common_prefix_length: int,
) -> int:
    limit = min(len(baseline), len(candidate)) - common_prefix_length
    matched = 0
    while matched < limit and baseline[-1 - matched] == candidate[-1 - matched]:
        matched += 1
    return matched


@dataclass(frozen=True)
class TokenSequenceDigest:
    token_count: int
    token_ids_sha256: str

    @classmethod
    def from_tokens(cls, values: Sequence[Any]) -> "TokenSequenceDigest":
        return cls(token_count=len(values), token_ids_sha256=_token_hash(values))


@dataclass(frozen=True)
class PhysicalTokenEquivalenceAudit:
    """Result of comparing the exact baseline and current runtime token layouts."""

    request_id: str
    tokenizer_fingerprint: str
    selected_content_sha256: str
    baseline_chat: TokenSequenceDigest
    compile_chat_blocks: tuple[TokenSequenceDigest, ...]
    link_chat: TokenSequenceDigest
    runtime_concatenated: TokenSequenceDigest
    baseline_plain: TokenSequenceDigest
    runtime_plain_concatenated: TokenSequenceDigest
    physical_token_ids_match: bool
    plain_token_ids_match: bool
    first_mismatch_index: int | None
    common_prefix_tokens: int
    common_suffix_tokens: int
    runtime_token_delta: int
    diagnostics: tuple[str, ...]
    measurement_scope: str
    physical_token_layout_version: str = "legacy-independent-chat-v1"
    runtime_assembly_order: str = "compile-blocks-then-link-input"

    @property
    def status(self) -> str:
        return "PASS" if self.physical_token_ids_match else "FAIL"

    def to_dict(
        self,
        *,
        baseline_token_ids: Sequence[Any] | None = None,
        runtime_token_ids: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        value = {
            **asdict(self),
            "status": self.status,
        }
        if baseline_token_ids is not None:
            value["baseline_token_ids"] = list(baseline_token_ids)
        if runtime_token_ids is not None:
            value["runtime_token_ids"] = list(runtime_token_ids)
        return value


def audit_physical_token_sequences(
    *,
    request_id: str,
    tokenizer_fingerprint: str,
    selected_content_sha256: str,
    baseline_chat_token_ids: Sequence[Any],
    compile_chat_token_ids: Sequence[Sequence[Any]],
    link_chat_token_ids: Sequence[Any],
    baseline_plain_token_ids: Sequence[Any],
    runtime_plain_token_ids: Sequence[Any],
    runtime_chat_token_ids: Sequence[Any] | None = None,
    physical_token_layout_version: str = "legacy-independent-chat-v1",
    runtime_assembly_order: str = "compile-blocks-then-link-input",
) -> tuple[PhysicalTokenEquivalenceAudit, tuple[Any, ...], tuple[Any, ...]]:
    """Compare captured token sequences without assuming where wrappers occur."""

    baseline = tuple(baseline_chat_token_ids)
    compile_blocks = tuple(tuple(values) for values in compile_chat_token_ids)
    link = tuple(link_chat_token_ids)
    runtime = (
        tuple(runtime_chat_token_ids)
        if runtime_chat_token_ids is not None
        else tuple(token for values in (*compile_blocks, link) for token in values)
    )
    baseline_plain = tuple(baseline_plain_token_ids)
    runtime_plain = tuple(runtime_plain_token_ids)
    mismatch = _first_mismatch(baseline, runtime)
    common_prefix = mismatch if mismatch is not None else len(baseline)
    common_suffix = _common_suffix_length(
        baseline,
        runtime,
        common_prefix_length=common_prefix,
    )
    physical_match = baseline == runtime
    plain_match = baseline_plain == runtime_plain
    diagnostics: list[str] = []
    if not plain_match:
        diagnostics.append(
            "The independently tokenized block/mutable bodies do not reproduce "
            "the selected-text token sequence; separators or tokenizer boundary "
            "merges differ."
        )
    if not physical_match:
        diagnostics.append(
            "Current per-block Compile and Link requests apply their chat template "
            "independently, so concatenated runtime tokens differ from one baseline "
            "selected-context chat request."
        )
    if physical_match:
        diagnostics.append(
            "Captured baseline and concatenated Compile/Link token IDs are exact."
        )
    audit = PhysicalTokenEquivalenceAudit(
        request_id=request_id,
        tokenizer_fingerprint=tokenizer_fingerprint,
        selected_content_sha256=selected_content_sha256,
        baseline_chat=TokenSequenceDigest.from_tokens(baseline),
        compile_chat_blocks=tuple(
            TokenSequenceDigest.from_tokens(values) for values in compile_blocks
        ),
        link_chat=TokenSequenceDigest.from_tokens(link),
        runtime_concatenated=TokenSequenceDigest.from_tokens(runtime),
        baseline_plain=TokenSequenceDigest.from_tokens(baseline_plain),
        runtime_plain_concatenated=TokenSequenceDigest.from_tokens(runtime_plain),
        physical_token_ids_match=physical_match,
        plain_token_ids_match=plain_match,
        first_mismatch_index=mismatch,
        common_prefix_tokens=common_prefix,
        common_suffix_tokens=common_suffix,
        runtime_token_delta=len(runtime) - len(baseline),
        diagnostics=tuple(diagnostics),
        measurement_scope=(
            "Exact token IDs rendered by one tokenizer/chat-template contract. "
            "For formal remote evidence, use the running SGLang /v1/tokenize "
            "endpoint with the same model and chat-template settings."
        ),
        physical_token_layout_version=physical_token_layout_version,
        runtime_assembly_order=runtime_assembly_order,
    )
    return audit, baseline, runtime


def audit_stitched_context(
    stitched: StitchedContext,
    tokenizer: ChatTemplateTokenizer,
    *,
    request_id: str,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    reasoning_effort: str | None = None,
    continue_final_message: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[PhysicalTokenEquivalenceAudit, tuple[int, ...], tuple[int, ...]]:
    """Capture the token layout generated by the requests used today.

    The baseline is one user-message request containing all selected text. The
    candidate is the exact concatenation of every independently templated Compile
    request followed by the independently templated mutable Link request.
    """

    def chat_tokens(text: str) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in tokenizer.tokenize_messages(
                [{"role": "user", "content": text}],
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                continue_final_message=continue_final_message,
                chat_template_kwargs=chat_template_kwargs,
            )
        )

    baseline_chat = chat_tokens(stitched.selected_text)
    compile_chat = tuple(
        chat_tokens(value.raw_text) for value in stitched.linked_contexts
    )
    link_chat = chat_tokens(stitched.mutable_text)
    baseline_plain = tuple(tokenizer.tokenize(stitched.selected_text))
    runtime_plain = tuple(
        token
        for text in (
            *(value.raw_text for value in stitched.linked_contexts),
            stitched.mutable_text,
        )
        for token in tokenizer.tokenize(text)
    )
    return audit_physical_token_sequences(
        request_id=request_id,
        tokenizer_fingerprint=tokenizer.fingerprint,
        selected_content_sha256=stitched.selected_content_sha256,
        baseline_chat_token_ids=baseline_chat,
        compile_chat_token_ids=compile_chat,
        link_chat_token_ids=link_chat,
        baseline_plain_token_ids=baseline_plain,
        runtime_plain_token_ids=runtime_plain,
    )


def audit_normalized_stitched_context(
    stitched: StitchedContext,
    tokenizer: ChatTemplateTokenizer,
    *,
    request_id: str,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    reasoning_effort: str | None = None,
    continue_final_message: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[PhysicalTokenEquivalenceAudit, tuple[int, ...], tuple[int, ...]]:
    """Audit the exact Scheduler layout used by normalized Compile/Link."""

    layout = build_normalized_chat_token_layout(
        stitched,
        tokenizer,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        continue_final_message=continue_final_message,
        chat_template_kwargs=chat_template_kwargs,
    )
    runtime_plain = (
        tuple(
            token_id
            for segment in layout.block_segments
            for token_id in segment.token_ids
        )
        + layout.mutable_body_token_ids
    )
    return audit_physical_token_sequences(
        request_id=request_id,
        tokenizer_fingerprint=tokenizer.fingerprint,
        selected_content_sha256=stitched.selected_content_sha256,
        baseline_chat_token_ids=layout.baseline_token_ids,
        compile_chat_token_ids=[value.token_ids for value in layout.block_segments],
        link_chat_token_ids=layout.link_input_token_ids,
        baseline_plain_token_ids=layout.selected_body_token_ids,
        runtime_plain_token_ids=runtime_plain,
        runtime_chat_token_ids=layout.assembled_token_ids,
        physical_token_layout_version=PHYSICAL_TOKEN_LAYOUT_VERSION,
        runtime_assembly_order=(
            "chat-prefix-then-compiled-blocks-then-mutable-chat-suffix"
        ),
    )
