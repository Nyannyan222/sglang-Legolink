"""Exact single-chat-template token framing for NOC Compile and LegoLink."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

from .stitch import StitchedContext
from .tokenizer import ChatTemplateTokenizer

PHYSICAL_TOKEN_LAYOUT_VERSION = "noc-single-chat-template-split-v1"

#: Every selected block linked, nothing between them.
FRAMING_PATH_EXACT_SPLIT = "exact_split"
#: A token crossed a block boundary; non-composing blocks travel fresh.
FRAMING_PATH_BOUNDARY_ADJUSTED = "exact_boundary_adjusted"


class PhysicalTokenLayoutError(ValueError):
    """Raised when Compile/Link tokens cannot reproduce one baseline prompt."""


def token_ids_sha256(values: Sequence[int]) -> str:
    encoded = json.dumps(
        [int(value) for value in values],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _int_token_ids(values: Sequence[Any], *, label: str) -> tuple[int, ...]:
    values = tuple(values)
    # Every id exactly `int` (not bool, not a subclass): the loop below would
    # return these same values, so skip it. Anything else takes the loop.
    if set(map(type, values)) <= {int}:
        return values
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PhysicalTokenLayoutError(
                f"{label} contains a non-integer token id: {value!r}"
            )
        normalized.append(int(value))
    return tuple(normalized)


def _subsequence_starts(
    haystack: Sequence[int],
    needle: Sequence[int],
) -> tuple[int, ...]:
    """Return all KMP matches without copying long token slices."""

    if not needle:
        return ()
    prefix = [0] * len(needle)
    matched = 0
    for index in range(1, len(needle)):
        while matched and needle[index] != needle[matched]:
            matched = prefix[matched - 1]
        if needle[index] == needle[matched]:
            matched += 1
            prefix[index] = matched

    starts: list[int] = []
    matched = 0
    for index, token_id in enumerate(haystack):
        while matched and token_id != needle[matched]:
            matched = prefix[matched - 1]
        if token_id == needle[matched]:
            matched += 1
        if matched == len(needle):
            starts.append(index - len(needle) + 1)
            matched = prefix[matched - 1]
    return tuple(starts)


def _first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


@dataclass(frozen=True)
class PhysicalBlockSegment:
    sub_context_id: str
    semantic_block_id: str
    cache_key: str
    token_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    # Frozen, so the hash of its ids cannot change; computed once.
    @cached_property
    def token_ids_sha256(self) -> str:
        return token_ids_sha256(self.token_ids)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sub_context_id": self.sub_context_id,
            "semantic_block_id": self.semantic_block_id,
            "cache_key": self.cache_key,
            "physical_token_count": self.token_count,
            "physical_token_ids_sha256": self.token_ids_sha256,
        }


@dataclass(frozen=True)
class NormalizedChatTokenLayout:
    """One baseline chat prompt split into prefix, blocks, and mutable suffix.

    ``block_segments`` holds every selected reusable block's compiled unit,
    ``tokenize(raw_text + separator)``, which is a function of the block alone.
    Normally every one of them is linked, in order, with nothing between them.
    When a token of the authoritative prompt crosses a semantic block boundary
    (tokenizers are not compositional), ``linked_block_indices`` names the
    blocks whose compiled units still sit exactly where they belong, and the
    authoritative tokens between them travel as fresh tokens
    (``interblock_fresh_token_ids``). Either way the assembly is the
    authoritative prompt, token for token.
    """

    version: str
    tokenizer_fingerprint: str
    separator: str
    baseline_token_ids: tuple[int, ...]
    selected_body_token_ids: tuple[int, ...]
    chat_prefix_token_ids: tuple[int, ...]
    block_segments: tuple[PhysicalBlockSegment, ...]
    mutable_body_token_ids: tuple[int, ...]
    chat_suffix_token_ids: tuple[int, ...]
    # Token-stable framing. ``None`` means every block is linked (the plain
    # split); otherwise the indices, in order, of the blocks this Link reuses.
    linked_block_indices: tuple[int, ...] | None = None
    # One run of fresh tokens after each linked block except the last.
    interblock_fresh_token_ids: tuple[tuple[int, ...], ...] = ()
    framing_path: str = FRAMING_PATH_EXACT_SPLIT

    @property
    def link_segments(self) -> tuple[PhysicalBlockSegment, ...]:
        """The compiled blocks this Link reuses, in prompt order."""

        if self.linked_block_indices is None:
            return self.block_segments
        return tuple(self.block_segments[index] for index in self.linked_block_indices)

    @property
    def fresh_block_segments(self) -> tuple[PhysicalBlockSegment, ...]:
        """Selected blocks whose tokens this Link computes fresh instead."""

        if self.linked_block_indices is None:
            return ()
        linked = set(self.linked_block_indices)
        return tuple(
            segment
            for index, segment in enumerate(self.block_segments)
            if index not in linked
        )

    @cached_property
    def assembled_token_ids(self) -> tuple[int, ...]:
        assembled: list[int] = list(self.chat_prefix_token_ids)
        for index, segment in enumerate(self.link_segments):
            assembled.extend(segment.token_ids)
            if index < len(self.interblock_fresh_token_ids):
                assembled.extend(self.interblock_fresh_token_ids[index])
        assembled.extend(self.mutable_body_token_ids)
        assembled.extend(self.chat_suffix_token_ids)
        return tuple(assembled)

    @property
    def link_input_token_ids(self) -> tuple[int, ...]:
        """Tokens carried by Link before Scheduler inserts compiled blocks."""

        return (
            *self.chat_prefix_token_ids,
            *(token_id for gap in self.interblock_fresh_token_ids for token_id in gap),
            *self.mutable_body_token_ids,
            *self.chat_suffix_token_ids,
        )

    def framing_diagnostics(self) -> dict[str, Any]:
        """What framing did. Evidence only: nothing reads it back to decide."""

        linked = self.link_segments
        return {
            "framing_path": self.framing_path,
            "framing_parity": (
                "exact" if self.assembled_token_ids == self.baseline_token_ids else "mismatch"
            ),
            "selected_block_count": len(self.block_segments),
            "linked_block_count": len(linked),
            "boundary_adjustment_count": len(self.fresh_block_segments),
            "boundary_fresh_token_count": sum(
                segment.token_count for segment in self.fresh_block_segments
            ),
            "interblock_fresh_token_count": sum(
                len(gap) for gap in self.interblock_fresh_token_ids
            ),
            "reusable_interior_token_count": sum(segment.token_count for segment in linked),
        }

    @property
    def expected_token_count(self) -> int:
        return len(self.baseline_token_ids)

    @cached_property
    def expected_token_ids_sha256(self) -> str:
        return token_ids_sha256(self.baseline_token_ids)

    def block_for_cache_key(self, cache_key: str) -> PhysicalBlockSegment:
        """The single occurrence of ``cache_key``; refuses a repeated key.

        A prompt may present the same immutable block more than once (two
        selected sub-contexts, byte-identical, same field): two occurrences,
        one logical cache key. Callers that act on one occurrence use
        :meth:`segment_for_occurrence`; callers that need the compiled unit
        of a key use :meth:`compiled_token_ids_for_cache_key`.
        """

        matches = self.occurrences_for_cache_key(cache_key)
        if len(matches) != 1:
            raise PhysicalTokenLayoutError(
                f"expected one physical segment for cache key {cache_key}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def occurrences_for_cache_key(
        self, cache_key: str
    ) -> tuple[PhysicalBlockSegment, ...]:
        """Every presented occurrence of ``cache_key``, in prompt order."""

        return tuple(
            value for value in self.block_segments if value.cache_key == cache_key
        )

    def segment_for_occurrence(self, sub_context_id: str) -> PhysicalBlockSegment:
        """The physical segment of one selected occurrence.

        Occurrences are identified by ``sub_context_id`` (unique within a plan);
        several occurrences may reference the same logical cache key.
        """

        matches = [
            value for value in self.block_segments if value.sub_context_id == sub_context_id
        ]
        if len(matches) != 1:
            raise PhysicalTokenLayoutError(
                f"expected one physical segment for occurrence {sub_context_id}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def compiled_token_ids_for_cache_key(self, cache_key: str) -> tuple[int, ...]:
        """The compiled unit a key denotes; identical for every occurrence."""

        matches = self.occurrences_for_cache_key(cache_key)
        if not matches:
            raise PhysicalTokenLayoutError(
                f"no physical segment for cache key {cache_key}"
            )
        _require_consistent_occurrences(matches)
        return matches[0].token_ids

    def link_metadata(self) -> dict[str, Any]:
        metadata = {
            "physical_token_layout_version": self.version,
            "physical_input_ids": list(self.link_input_token_ids),
            "physical_token_count": len(self.link_input_token_ids),
            "physical_token_ids_sha256": token_ids_sha256(self.link_input_token_ids),
            "physical_prefix_token_count": len(self.chat_prefix_token_ids),
            "physical_block_segments": [
                value.to_metadata() for value in self.link_segments
            ],
            "expected_complete_token_count": self.expected_token_count,
            "expected_complete_token_ids_sha256": (self.expected_token_ids_sha256),
        }
        # Only a boundary-adjusted layout carries fresh tokens between blocks;
        # a plain split sends exactly the fields it always sent.
        if any(self.interblock_fresh_token_ids):
            metadata["physical_interblock_fresh_token_counts"] = [
                len(gap) for gap in self.interblock_fresh_token_ids
            ]
        return metadata


def _require_consistent_occurrences(
    segments: Sequence[PhysicalBlockSegment],
) -> None:
    """Occurrences are unique; occurrences sharing a key share their tokens.

    A repeated immutable block is legitimate: the runtime reserves one source
    block per distinct key and places it at every position that names it. That
    is only sound if the key denotes one compiled unit, so two occurrences of a
    key with different compiled ids are refused rather than resolved.
    """

    seen_occurrences: set[str] = set()
    by_key: dict[str, PhysicalBlockSegment] = {}
    for segment in segments:
        if segment.sub_context_id in seen_occurrences:
            raise PhysicalTokenLayoutError(
                f"occurrence {segment.sub_context_id} is presented twice"
            )
        seen_occurrences.add(segment.sub_context_id)
        first = by_key.setdefault(segment.cache_key, segment)
        if first.token_ids != segment.token_ids:
            raise PhysicalTokenLayoutError(
                f"occurrences {first.sub_context_id} and {segment.sub_context_id} "
                f"share cache key {segment.cache_key} but not compiled token ids"
            )


def build_normalized_chat_token_layout(
    stitched: StitchedContext,
    tokenizer: ChatTemplateTokenizer,
    *,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    reasoning_effort: str | None = None,
    continue_final_message: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> NormalizedChatTokenLayout:
    """Split one authoritative chat layout without adding per-block wrappers."""

    if not stitched.linked_contexts:
        raise PhysicalTokenLayoutError(
            "normalized LegoLink framing requires at least one linked block"
        )
    if not stitched.mutable_text:
        raise PhysicalTokenLayoutError(
            "normalized LegoLink framing requires a non-empty mutable suffix"
        )

    separator = stitched.legolink_framing.separator
    physical_body_text = (
        "".join(value.raw_text + separator for value in stitched.linked_contexts)
        + stitched.mutable_text
    )
    if physical_body_text != stitched.selected_text:
        raise PhysicalTokenLayoutError(
            "physical block framing does not reconstruct selected_text exactly"
        )

    baseline = _int_token_ids(
        tokenizer.tokenize_messages(
            [{"role": "user", "content": stitched.selected_text}],
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            continue_final_message=continue_final_message,
            chat_template_kwargs=chat_template_kwargs,
        ),
        label="baseline chat template",
    )
    selected_body = _int_token_ids(
        tokenizer.tokenize(stitched.selected_text),
        label="selected body",
    )
    if not selected_body:
        raise PhysicalTokenLayoutError("selected body token sequence is empty")

    occurrences = _subsequence_starts(baseline, selected_body)

    block_segments = tuple(
        PhysicalBlockSegment(
            sub_context_id=value.sub_context_id,
            semantic_block_id=(
                value.semantic_identity or value.sub_context_id
            ),
            cache_key=value.cache_key,
            token_ids=_int_token_ids(
                tokenizer.tokenize(value.raw_text + separator),
                label=f"compiled block {value.cache_key}",
            ),
        )
        for value in stitched.linked_contexts
    )
    _require_consistent_occurrences(block_segments)
    mutable_body = _int_token_ids(
        tokenizer.tokenize(stitched.mutable_text),
        label="mutable body",
    )
    split_body = (
        tuple(token_id for segment in block_segments for token_id in segment.token_ids)
        + mutable_body
    )
    if split_body != selected_body or len(occurrences) != 1:
        # A token of the authoritative prompt crosses a semantic boundary (or
        # the chat prefix). Place what still composes; refuse if nothing does.
        return _boundary_adjusted_layout(
            stitched,
            tokenizer,
            separator=separator,
            baseline=baseline,
            selected_body=selected_body,
            occurrences=occurrences,
            block_segments=block_segments,
            split_body=split_body,
        )
    body_start = occurrences[0]
    body_end = body_start + len(selected_body)

    layout = NormalizedChatTokenLayout(
        version=PHYSICAL_TOKEN_LAYOUT_VERSION,
        tokenizer_fingerprint=tokenizer.fingerprint,
        separator=separator,
        baseline_token_ids=baseline,
        selected_body_token_ids=selected_body,
        chat_prefix_token_ids=baseline[:body_start],
        block_segments=block_segments,
        mutable_body_token_ids=mutable_body,
        chat_suffix_token_ids=baseline[body_end:],
    )
    if layout.assembled_token_ids != baseline:
        raise PhysicalTokenLayoutError(
            "normalized Compile/Link assembly does not match baseline tokens"
        )
    if not layout.link_input_token_ids:
        raise PhysicalTokenLayoutError("normalized Link input is empty")
    return layout


def _boundary_adjusted_layout(
    stitched: StitchedContext,
    tokenizer: ChatTemplateTokenizer,
    *,
    separator: str,
    baseline: tuple[int, ...],
    selected_body: tuple[int, ...],
    occurrences: tuple[int, ...],
    block_segments: tuple[PhysicalBlockSegment, ...],
    split_body: tuple[int, ...],
) -> NormalizedChatTokenLayout:
    """Token-stable framing when the plain split does not compose.

    The authoritative prompt tokenization decides everything; the tokenizer is
    the only oracle, so no rule about which characters merge is encoded here.

    * A character position ``c`` of the selected text is a token boundary iff
      ``tokenize(text[:c]) + tokenize(text[c:]) == tokenize(text)``.
    * A selected block is linked iff both ends of its framing unit are token
      boundaries and the authoritative tokens between them are exactly its
      compiled ids. Compiled ids never depend on neighbours, so the block's
      identity and cache key keep denoting the same tokens.
    * Every other authoritative token is fresh: before the first linked block,
      between linked blocks, or after the last.

    The assembly is checked against the authoritative prompt and refused on
    any difference; the runtime checks the same hashes again.
    """

    text = stitched.selected_text
    starts: list[int] = []
    cursor = 0
    for value in stitched.linked_contexts:
        starts.append(cursor)
        cursor += len(value.raw_text) + len(separator)
    ends = starts[1:] + [cursor]

    body_boundaries: dict[int, int | None] = {0: 0, len(text): len(selected_body)}

    def body_index(position: int) -> int | None:
        if position not in body_boundaries:
            left = _int_token_ids(tokenizer.tokenize(text[:position]), label="boundary prefix")
            right = _int_token_ids(tokenizer.tokenize(text[position:]), label="boundary suffix")
            body_boundaries[position] = (
                len(left) if left + right == selected_body else None
            )
        return body_boundaries[position]

    suffix_length: int | None = None
    if len(occurrences) == 1:
        body_offset: int | None = occurrences[0]
        suffix_length = len(baseline) - occurrences[0] - len(selected_body)
    else:
        body_offset = None

    prompt_positions: dict[int, int | None] = {}

    def prompt_index(position: int) -> int | None:
        """Index in the complete chat prompt of a body token boundary."""

        nonlocal suffix_length
        if position in prompt_positions:
            return prompt_positions[position]
        found: int | None = None
        index = body_index(position)
        if index is not None:
            if body_offset is not None:
                found = body_offset + index
            elif position < len(text):
                # The chat prefix merged into the body, so the body is not a
                # contiguous run of the prompt. Locate this boundary by the
                # unique occurrence of everything after it.
                rest = selected_body[index:]
                matches = _subsequence_starts(baseline, rest)
                if len(matches) == 1:
                    tail = len(baseline) - matches[0] - len(rest)
                    if suffix_length is None:
                        suffix_length = tail
                    if tail == suffix_length:
                        found = matches[0]
        prompt_positions[position] = found
        return found

    linked: list[int] = []
    for block_index, segment in enumerate(block_segments):
        begin = prompt_index(starts[block_index])
        end = prompt_index(ends[block_index])
        if begin is None or end is None:
            # A token crosses one of this block's edges: the boundary effect
            # this path exists for. Its tokens travel fresh in this request.
            continue
        if baseline[begin:end] != segment.token_ids:
            # No token crosses either edge, so the authoritative tokens of this
            # span are the block's own tokens. Compiled ids that differ are not
            # a boundary effect but ids that do not encode the block's text;
            # carrying them fresh would hide that, so refuse.
            raise PhysicalTokenLayoutError(
                f"compiled block {segment.cache_key} ids do not reproduce its "
                "authoritative tokens although no token crosses its boundaries"
            )
        if linked and begin < prompt_positions[ends[linked[-1]]]:
            continue
        linked.append(block_index)

    mismatch = _first_mismatch(selected_body, split_body)
    detail = (
        f"first mismatch: {mismatch}, expected tokens: {len(selected_body)}, "
        f"split tokens: {len(split_body)}, body occurrences in chat prompt: "
        f"{len(occurrences)}"
    )
    if not linked:
        raise PhysicalTokenLayoutError(
            "tokenizer boundary behavior prevents an exact reusable block "
            f"split: no selected block composes with its neighbours ({detail})"
        )
    if suffix_length is None or suffix_length < 0:
        raise PhysicalTokenLayoutError(
            "token-stable framing could not locate the chat-template suffix "
            f"({detail})"
        )

    first_begin = prompt_positions[starts[linked[0]]]
    last_end = prompt_positions[ends[linked[-1]]]
    gaps = tuple(
        baseline[prompt_positions[ends[previous]]:prompt_positions[starts[following]]]
        for previous, following in zip(linked, linked[1:])
    )
    suffix_start = len(baseline) - suffix_length
    if last_end > suffix_start:
        raise PhysicalTokenLayoutError(
            f"token-stable framing placed a block inside the chat suffix ({detail})"
        )
    layout = NormalizedChatTokenLayout(
        version=PHYSICAL_TOKEN_LAYOUT_VERSION,
        tokenizer_fingerprint=tokenizer.fingerprint,
        separator=separator,
        baseline_token_ids=baseline,
        selected_body_token_ids=selected_body,
        chat_prefix_token_ids=baseline[:first_begin],
        block_segments=block_segments,
        mutable_body_token_ids=baseline[last_end:suffix_start],
        chat_suffix_token_ids=baseline[suffix_start:],
        linked_block_indices=tuple(linked),
        interblock_fresh_token_ids=gaps,
        framing_path=FRAMING_PATH_BOUNDARY_ADJUSTED,
    )
    if layout.assembled_token_ids != baseline:
        raise PhysicalTokenLayoutError(
            f"token-stable framing assembly does not match baseline tokens ({detail})"
        )
    if not layout.mutable_body_token_ids:
        raise PhysicalTokenLayoutError(
            "token-stable framing left no fresh mutable body after the last "
            f"linked block ({detail})"
        )
    return layout
