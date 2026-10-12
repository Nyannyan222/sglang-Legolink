"""Characterization tests for the behaviours the audit found and deliberately did NOT change (docs/correctness_audit.md, section U).

Each test pins what the code does today so that a later change is a decision, not an accident. A failing test here means a documented
limitation moved: read the matching U-xx entry, then update the test together with the entry. None of these tests says the behaviour is
desirable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_residency_consistency as base  # noqa: E402
from noc_context_manager.compile_policy import CacheCatalog, CompileAction, CompileDecision  # noqa: E402
from noc_context_manager.link_layout import LayoutAdjustment, LinkLayoutFeasibility  # noqa: E402


def _block(field: str, text: str, *, mutable: bool = False):
    from noc_context_manager.subcontext_store import SemanticSubContext

    tokens = tuple(range(len(text.split())))
    return SemanticSubContext.create(
        task_id="t", session_revision=1, context_field=field, title="x", question_or_objective="q", raw_text=text, source_spans=[],
        token_count=len(tokens), is_mutable=mutable, reuse_candidate=True, semantic_independence_reason="r", segmentation_confidence=1.0,
        token_ids=tokens,
    )


def _compile_into(catalog: CacheCatalog, block, namespace: str) -> None:
    from noc_context_manager.cache_namespace import physical_kv_identity

    catalog.mark_compiled(
        block.cache_key, namespace=namespace, context_field=block.context_field, semantic_identity=block.semantic_identity,
        physical_kv_identity=physical_kv_identity(block).digest, physical_token_ids_sha256="h", token_count=block.token_count, compiled_at=1.0,
    )


def test_u01_prefix_closure_promotes_a_block_even_when_the_later_compiled_block_is_itself_demoted():
    """U-01. X is promoted "to preserve selected order before later reusable blocks", but the only later compiled block, Z, is
    demoted by the same pass (H opened the mutable suffix). The promotion's own premise is gone; the result is still a linkable request."""

    x, h, z = (_block("tool_history_output", name * 80) for name in "xhz")
    decisions = [
        CompileDecision(x.sub_context_id, x.cache_key, CompileAction.KEEP_MUTABLE, "below floor"),
        CompileDecision(h.sub_context_id, h.cache_key, CompileAction.KEEP_MUTABLE, "not linkable"),
        CompileDecision(z.sub_context_id, z.cache_key, CompileAction.COMPILE, "policy"),
    ]
    result = LinkLayoutFeasibility().apply([x, h, z], decisions, link_permitted=[True, False, True])
    assert [value.effective_action for value in result] == [CompileAction.COMPILE, CompileAction.KEEP_MUTABLE, CompileAction.KEEP_MUTABLE]
    assert [value.adjustment for value in result] == [
        LayoutAdjustment.FORCE_COMPILE_FOR_PREFIX_CLOSURE, LayoutAdjustment.NONE, LayoutAdjustment.FORCE_MUTABLE_FOR_SUFFIX_CLOSURE,
    ]


def test_u02_resident_hit_count_and_reuse_accounting_are_a_post_compile_snapshot():
    """U-02. On a cold turn the host compiled one block, yet `resident_hit_count` and `reuse_accounting` already count it as cached.
    `scripts/analyze_agent_effectiveness.cache_state` compensates ("includes blocks compiled by this same request"); the namespace
    counters are the plan-time truth and the runtime trace is the runtime truth."""

    import tempfile

    tmp = Path(tempfile.mkdtemp())
    runtime = base.FakeRuntime()
    llm = base.make_llm(tmp, runtime)
    cold = base.step(llm, tmp)
    assert len(runtime.compile_payloads) == 1 and cold["compile_count"] == 1
    assert cold["resident_hit_count"] == 1
    counts = cold["reuse_accounting"]["counts"]
    assert counts["linked_cached_block_count"]["value"] == 1 and counts["compiled_block_count"]["value"] == 0
    totals = cold["namespace_metrics"]["totals"]
    assert totals["compiled_token_count"] > 0 and totals["reused_token_count"] == 0


def test_u03_global_mode_does_not_enforce_persistence_but_the_isolating_modes_do():
    """U-03. `no_cache` and a settled `user_prompt` are served from a persistent entry in GLOBAL mode (documented: GLOBAL is the legacy
    baseline arm). Hard / logical modes keep them out of the cache."""

    from noc_context_manager.cache_namespace import NamespaceResolver

    text = "alpha beta gamma delta " * 20
    owner = _block("tool_history_output", text)
    no_cache = _block("task_metadata", text)
    settled_prompt = _block("user_prompt", text)
    catalog = CacheCatalog()
    _compile_into(catalog, owner, "tool_output")
    for block in (no_cache, settled_prompt):
        assert NamespaceResolver("global").resolve(block, catalog).resident
        for mode in ("hard_namespace", "logical_namespace"):
            result = NamespaceResolver(mode).resolve(block, catalog)
            assert not result.resident and result.not_persistent
