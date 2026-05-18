"""Prototype context manager components for NOC semantic sub-contexts."""

from .registry import SubContextRegistry
from .reuse import ReuseDecisionEngine
from .runtime_isolation import (
    IndependentPrefillCache,
    IsolationReport,
    PrefillBlock,
    PrefillResult,
    block_isolation_report,
    build_blocks,
)
from .schema import RequestProfile, ReuseDecision, SubContext
from .selector import SubContextSelector
from .similarity import LexicalSimilarityJudge

__all__ = [
    "LexicalSimilarityJudge",
    "IndependentPrefillCache",
    "IsolationReport",
    "PrefillBlock",
    "PrefillResult",
    "RequestProfile",
    "ReuseDecision",
    "ReuseDecisionEngine",
    "SubContext",
    "SubContextRegistry",
    "SubContextSelector",
    "block_isolation_report",
    "build_blocks",
]
