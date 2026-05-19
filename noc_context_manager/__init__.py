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
from .sglang_bridge import (
    AttentionRange,
    SGLangBlockRuntimePlan,
    SGLangPrefillRequest,
    build_sglang_block_runtime_plan,
    validate_sglang_block_runtime_plan,
)
from .schema import RequestProfile, ReuseDecision, SubContext
from .selector import SubContextSelector
from .similarity import LexicalSimilarityJudge

__all__ = [
    "LexicalSimilarityJudge",
    "IndependentPrefillCache",
    "IsolationReport",
    "AttentionRange",
    "PrefillBlock",
    "PrefillResult",
    "RequestProfile",
    "ReuseDecision",
    "ReuseDecisionEngine",
    "SGLangBlockRuntimePlan",
    "SGLangPrefillRequest",
    "SubContext",
    "SubContextRegistry",
    "SubContextSelector",
    "block_isolation_report",
    "build_blocks",
    "build_sglang_block_runtime_plan",
    "validate_sglang_block_runtime_plan",
]
