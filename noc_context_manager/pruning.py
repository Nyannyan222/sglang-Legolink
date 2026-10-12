"""Optional provenance-preserving pruning and derived summarization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .subcontext_store import SemanticSubContext
from .tokenizer import TextTokenizer, count_tokens


@dataclass(frozen=True)
class PruneResult:
    sub_context: SemanticSubContext
    original_tokens: int
    result_tokens: int
    summarized: bool
    reason: str


class ContextPrunerSummarizer:
    _VERBATIM_PATTERN = re.compile(
        r"```|\{\s*[\"']|Traceback|(?:^|\n)\s*[$>]\s|(?:error|exception):",
        re.IGNORECASE,
    )

    def __init__(
        self,
        summarizer: Callable[[str, int], str] | None = None,
        *,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        """Summarize a block, counting the summary the way the block was counted.

        `tokenizer` must be the one that produced `value.token_count`. A derived
        block whose count came from a different tokenizer than its siblings
        would corrupt the context budget and the compile floor at once, which is
        why this no longer silently reaches for the regex estimator: with no
        tokenizer the pruner declines to summarize and keeps the original.
        """

        self.summarizer = summarizer
        self.tokenizer = tokenizer

    def prune(self, value: SemanticSubContext, target_tokens: int) -> PruneResult:
        if value.token_count <= target_tokens:
            return PruneResult(value, value.token_count, value.token_count, False, "within budget")
        if self.summarizer is None or self._VERBATIM_PATTERN.search(value.raw_text):
            return PruneResult(value, value.token_count, value.token_count, False, "verbatim content or summarizer unavailable")
        if self.tokenizer is None:
            return PruneResult(
                value,
                value.token_count,
                value.token_count,
                False,
                "no tokenizer supplied, so a derived block could not be counted "
                "in the same unit as the block it derives from",
            )
        try:
            summary = self.summarizer(value.raw_text, target_tokens).strip()
            if not summary or count_tokens(summary, self.tokenizer) >= value.token_count:
                raise ValueError("summary did not reduce context")
            derived = SemanticSubContext.create(
                task_id=value.task_id,
                session_revision=value.session_revision,
                context_field=value.context_field,
                title=f"Summary: {value.title}",
                question_or_objective=value.question_or_objective,
                raw_text=summary,
                source_spans=value.source_spans,
                token_count=count_tokens(summary, self.tokenizer),
                is_mutable=False,
                reuse_candidate=value.reuse_candidate,
                semantic_independence_reason="derived compression of source-backed block",
                segmentation_confidence=value.segmentation_confidence,
                policy_version=f"{value.policy_version}+summary-v1",
                derived_from=(value.sub_context_id,),
                tokenizer_fingerprint=value.tokenizer_fingerprint,
                model_fingerprint=value.model_fingerprint,
                metadata={"original_content_hash": value.content_hash},
            )
            return PruneResult(derived, value.token_count, derived.token_count, True, "selected context exceeded budget")
        except Exception:
            return PruneResult(value, value.token_count, value.token_count, False, "summary failed; original retained")
