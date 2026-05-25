"""Prototype context manager components for NOC semantic sub-contexts."""

from .block_kv_cache import (
    BlockKVEntry,
    BlockKVLookup,
    OrderInvariantBlockKVCache,
    prefix_reuse_tokens,
    summarize_lookups,
)
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
from .sglang_block_kv import (
    BlockKVLookupResult,
    BlockKVRecord,
    SemanticBlockKVStore,
    SemanticBlockMetadata,
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
    "BlockKVEntry",
    "BlockKVLookup",
    "BlockKVLookupResult",
    "BlockKVRecord",
    "PrefillBlock",
    "PrefillResult",
    "OrderInvariantBlockKVCache",
    "RequestProfile",
    "ReuseDecision",
    "ReuseDecisionEngine",
    "SGLangBlockRuntimePlan",
    "SGLangPrefillRequest",
    "SemanticBlockKVStore",
    "SemanticBlockMetadata",
    "SubContext",
    "SubContextRegistry",
    "SubContextSelector",
    "block_isolation_report",
    "build_blocks",
    "build_sglang_block_runtime_plan",
    "prefix_reuse_tokens",
    "summarize_lookups",
    "validate_sglang_block_runtime_plan",
]
