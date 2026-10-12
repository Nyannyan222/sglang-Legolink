"""Regression tests for the defects confirmed by the architecture + correctness audit (docs/debugging_fixes.md).

Each test names the finding it pins (``F-xx``) and fails on the code as received in the bundle. CPU only: no server, no GPU, no
sglang import. They reuse the strict fake runtime of ``test_residency_consistency`` so they exercise the real adapter, planner,
catalog and coordinator rather than stand-ins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_residency_consistency as base  # noqa: E402
from noc_context_manager.persistent_catalog import PersistentCacheCatalog  # noqa: E402
from noc_context_manager.residency_recovery import OWNER, default_coordinator  # noqa: E402
from noc_context_manager.sglang_adapter import NOCSGLangAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_coordinator():
    default_coordinator().reset()
    yield default_coordinator()
    default_coordinator().reset()


# --- F-01: failure classification ---------------------------------------------------------------------------------------------------


def test_f01_the_legolink_only_refusal_is_not_a_stale_block():
    """The runtime's own refusal of a plain request says "Generation requests must either compile ... or link"; the word
    "generation" used to classify it as ``stale``, which a ROBUST_AGENT adapter answers by invalidating and recompiling every
    linked block."""

    refusal = (
        "This patched SGLang deployment is LegoLink-only. Generation requests must either compile one semantic "
        "sub-context with sub_context.cache_key or link previously compiled blocks with pic_link."
    )
    assert NOCSGLangAdapter.classify_failure(refusal) == "link_exception"


@pytest.mark.parametrize(
    "message",
    [
        "Cannot acquire stale LegoLink block: k",
        "radix_legolink_block_stale: allocator_generation_mismatch",
        "KV generation mismatch for block k",
        "stale_generation",
    ],
)
def test_f01_real_staleness_wording_is_still_stale(message):
    assert NOCSGLangAdapter.classify_failure(message) == "stale"


# --- F-02: single-flight cleanup ----------------------------------------------------------------------------------------------------


class _Block:
    def __init__(self, key: str) -> None:
        self.cache_key = key
        self.sub_context_id = "id-" + key


def _detail(*keys: str) -> str:
    body = {"noc_residency_recovery": {"missing_cache_keys": list(keys), "stale_cache_keys": [], "incompatible_cache_keys": []}}
    return "LegoLink source preflight failed | " + json.dumps(body)


def _adapter(tmp_path):
    runtime = base.FakeRuntime()
    llm = base.make_llm(tmp_path, runtime)
    base.step(llm, tmp_path)
    return llm._adapter


def test_f02_an_exception_out_of_the_first_owner_ends_every_flight_it_claimed(tmp_path, monkeypatch, fresh_coordinator):
    adapter = _adapter(tmp_path)
    coordinator = fresh_coordinator
    monkeypatch.setattr(adapter, "_recover_residency", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("compile transport died")))

    with pytest.raises(RuntimeError, match="compile transport died"):
        adapter._recover_residency_singleflight(_detail("A", "B"), [_Block("A"), _Block("B")], None, observed=coordinator.completions)

    # Key B was claimed as OWNER before the first Compile and was never reached. Its flight must not outlive the request.
    assert coordinator.stats()["in_flight"] == 0
    ticket = coordinator.claim(adapter.residency_runtime_id, "B", observed=coordinator.completions)
    assert ticket.role == OWNER, "a later request joined a flight whose owner is gone"


# --- F-03: persistent catalog ------------------------------------------------------------------------------------------------------


def test_f03_mark_compiled_survives_a_peers_unread_invalidate_of_the_same_key(tmp_path):
    path = tmp_path / "catalog.jsonl"
    mine = PersistentCacheCatalog(path)
    # A peer session invalidated the key after my last read of the file; I have not looked at the file since.
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"op": "invalidate", "cache_key": "k"}) + "\n")

    mine.mark_compiled("k", namespace="instruction", physical_kv_identity="id-1", token_count=5)

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    compiled = [event for event in events if event["op"] == "compiled"]
    assert len(compiled) == 1
    assert compiled[0]["metadata"].get("namespace") == "instruction", "the compiled event was written with empty metadata"
    assert compiled[0]["metadata"].get("physical_kv_identity") == "id-1"

    # A third session replaying the file sees the full row, and so does mine after its next read.
    other = PersistentCacheCatalog(path)
    for catalog in (mine, other):
        assert catalog.contains("k")
        assert (catalog.metadata("k") or {}).get("token_count") == 5


# --- F-04: hard_namespace and logical_namespace are one treatment unless a cross-root policy is configured ---------------------------


def _blk(field, text, *, mutable=False, reuse=True):
    from noc_context_manager.subcontext_store import SemanticSubContext

    tokens = tuple(range(len(text.split())))
    return SemanticSubContext.create(
        task_id="t", session_revision=1, context_field=field, title="x", question_or_objective="q", raw_text=text, source_spans=[],
        token_count=len(tokens), is_mutable=mutable, reuse_candidate=reuse, semantic_independence_reason="r", segmentation_confidence=1.0,
        token_ids=tokens,
    )


def _compile_into(catalog, block, namespace):
    from noc_context_manager.cache_namespace import physical_kv_identity

    catalog.mark_compiled(
        block.cache_key, namespace=namespace, context_field=block.context_field, semantic_identity=block.semantic_identity,
        physical_kv_identity=physical_kv_identity(block).digest, physical_token_ids_sha256="h", token_count=block.token_count, compiled_at=1.0,
    )


def test_f04_hard_and_logical_without_pairs_resolve_every_block_identically():
    from noc_context_manager.cache_namespace import NamespaceResolver
    from noc_context_manager.compile_policy import CacheCatalog

    text = "alpha beta gamma delta " * 20
    owner = _blk("tool_history_output", text)
    cross_root_twin = _blk("workspace_evidence", text)
    unrelated = _blk("workspace_evidence", "something else entirely " * 20)
    settled_prompt = _blk("user_prompt", text)
    no_cache = _blk("task_metadata", text)
    probes = [owner, cross_root_twin, unrelated, settled_prompt, no_cache]

    outcomes = {}
    for mode in ("hard_namespace", "logical_namespace"):
        catalog = CacheCatalog()
        _compile_into(catalog, owner, "tool_output")
        resolver = NamespaceResolver(mode)
        outcomes[mode] = [
            {k: v for k, v in resolver.resolve(block, catalog).to_dict().items() if k not in {"mode", "lookup_latency_s", "entry_age_s", "physical_block_residency_age_s"}}
            for block in probes
        ]
    assert outcomes["hard_namespace"] == outcomes["logical_namespace"]
    # and the one thing that would make them differ is absent: no cross-root alias was ever taken
    assert not any(row["cross_namespace_hit"] for rows in outcomes.values() for row in rows)


def test_f04_a_configured_pair_makes_logical_namespace_a_different_treatment():
    from noc_context_manager.cache_namespace import CacheNamespace, CacheNamespacePolicy, NamespaceResolver
    from noc_context_manager.compile_policy import CacheCatalog

    text = "alpha beta gamma delta " * 20
    owner, twin = _blk("tool_history_output", text), _blk("workspace_evidence", text)
    catalog = CacheCatalog()
    _compile_into(catalog, owner, "tool_output")
    pair = ((CacheNamespace.WORKSPACE, CacheNamespace.TOOL_OUTPUT),)
    assert NamespaceResolver("logical_namespace", CacheNamespacePolicy(cross_namespace_pairs=pair)).resolve(twin, catalog).cross_namespace_hit
    assert not NamespaceResolver("hard_namespace").resolve(twin, catalog).cross_namespace_hit


# --- F-05: the Native baseline row must not print a speedup under a regime that did not pass ---------------------------------------------


def _summarizer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("summarize_h200_multi_agent_reuse", ROOT / "scripts" / "summarize_h200_multi_agent_reuse.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _group(ttft):
    return {"derived": {"semantic_cache_hit_rate": 0.5, "cross_agent_reused_tokens": 1, "avoided_compile_tokens": 1}, "stats": {"ttft_e2e_s": {"mean": ttft}}}


# --- F-06: runtime-labelled token counts must come from the runtime when the trace has them ----------------------------------------------


# --- F-07: a half-written line of a peer is read once it is whole, never dropped ------------------------------------------------------------


def test_f07_a_fragment_of_a_peers_append_is_not_consumed_until_its_newline_arrives(tmp_path):
    path = tmp_path / "catalog.jsonl"
    reader = PersistentCacheCatalog(path)
    whole = json.dumps({"op": "compiled", "cache_key": "k", "metadata": {"namespace": "instruction", "token_count": 7}}, separators=(",", ":"))
    cut = len(whole) // 2
    with path.open("ab") as handle:  # the first half of the line is visible, the second is not yet
        handle.write(whole[:cut].encode("utf-8"))
    assert not reader.contains("k")

    with path.open("ab") as handle:
        handle.write(whole[cut:].encode("utf-8") + b"\n")
    assert reader.contains("k"), "the event was lost: its first half was consumed as a corrupt line"
    assert reader.metadata("k")["token_count"] == 7


def test_f07_a_corrupt_whole_line_is_still_skipped_without_poisoning_the_lines_after_it(tmp_path):
    path = tmp_path / "catalog.jsonl"
    reader = PersistentCacheCatalog(path)
    good = json.dumps({"op": "compiled", "cache_key": "ok", "metadata": {"token_count": 1}})
    with path.open("ab") as handle:
        handle.write(b"{not json\n" + good.encode("utf-8") + b"\n")
    assert reader.contains("ok")


# --- F-08: the launch scripts respect the scheduler-owned GPU environment ------------------------------------------------------------------


def _bash_runs(tmp_path):
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        return None
    probe = tmp_path / "probe.txt"
    probe.write_text("x", encoding="utf-8")
    try:
        done = subprocess.run([bash, "-c", f"test -f '{probe.as_posix()}' && echo ok"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return bash if done.stdout.strip() == "ok" else None


def _source_lib(tmp_path, *, environment: dict, snippet: str) -> "subprocess.CompletedProcess":
    import os
    import subprocess

    bash = _bash_runs(tmp_path)
    if bash is None:
        pytest.skip("no POSIX bash that can read the temporary directory")
    config = tmp_path / "cfg.env"
    config.write_text("PYTHON_BIN=/bin/true\nCUDA_VISIBLE_DEVICES=0\n", encoding="utf-8", newline="\n")  # the shipped example, copied verbatim
    # A Windows checkout may hold the script with CRLF endings, which bash cannot source: test a line-ending-normalized copy of the same text.
    script = tmp_path / "scripts" / "h200_reuse" / "lib.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes((ROOT / "scripts" / "h200_reuse" / "lib.sh").read_bytes().replace(b"\r\n", b"\n"))
    env = {k: v for k, v in os.environ.items() if k not in {"CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "ALLOW_NON_SLURM", "H200_REUSE_ENV"}}
    env.update({"H200_REUSE_ENV": config.as_posix(), **environment})
    return subprocess.run([bash, "-c", f"source scripts/h200_reuse/lib.sh; {snippet}"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


