"""Compose source-backed blocks without turning semantic splits into text."""

from __future__ import annotations

from collections.abc import Iterable

from .subcontext_store import SemanticSubContext


def _source_extent(value: SemanticSubContext) -> tuple[str, int, int] | None:
    if not value.source_spans:
        return None
    source_ids = {span.source_id for span in value.source_spans}
    if len(source_ids) != 1:
        return None
    spans = sorted(value.source_spans, key=lambda span: (span.start, span.end))
    return next(iter(source_ids)), spans[0].start, spans[-1].end


def same_source_revision(
    previous: SemanticSubContext, current: SemanticSubContext
) -> bool:
    previous_extent = _source_extent(previous)
    current_extent = _source_extent(current)
    return bool(
        previous_extent
        and current_extent
        and previous_extent[0] == current_extent[0]
        and previous.source_revision_identity
        and previous.source_revision_identity == current.source_revision_identity
        and previous.context_field == current.context_field
        and previous.source_order == current.source_order
    )


def is_contiguous_source_sibling(
    previous: SemanticSubContext, current: SemanticSubContext
) -> bool:
    if not same_source_revision(previous, current):
        return False
    previous_extent = _source_extent(previous)
    current_extent = _source_extent(current)
    assert previous_extent is not None and current_extent is not None
    return previous_extent[2] == current_extent[1]


def source_boundary(
    previous: SemanticSubContext,
    current: SemanticSubContext,
    separator: str,
) -> str:
    if not same_source_revision(previous, current):
        return separator
    if not is_contiguous_source_sibling(previous, current):
        raise ValueError(
            "selected blocks from one source revision are not contiguous: "
            f"{previous.sub_context_id} -> {current.sub_context_id}"
        )
    return ""


def compose_source_text(
    contexts: Iterable[SemanticSubContext],
    *,
    separator: str,
    previous: SemanticSubContext | None = None,
    trailing_text: str = "",
) -> str:
    """Join exact source slices, separating source groups but not siblings."""

    values = list(contexts)
    pieces: list[str] = []
    prior = previous
    for value in values:
        if prior is not None:
            pieces.append(source_boundary(prior, value, separator))
        pieces.append(value.raw_text)
        prior = value
    if trailing_text:
        if prior is not None:
            pieces.append(separator)
        pieces.append(trailing_text)
    return "".join(pieces)
