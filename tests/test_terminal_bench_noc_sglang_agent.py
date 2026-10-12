# ruff: noqa: E402

import json
import sys
import types
from pathlib import Path


class FakeBaseLLM:
    def __init__(self, **kwargs):
        if kwargs:
            raise TypeError(f"unexpected BaseLLM kwargs: {sorted(kwargs)}")


class FakeOutputLengthExceededError(Exception):
    def __init__(self, message, truncated_response=None):
        super().__init__(message)
        self.truncated_response = truncated_response


class FakeTerminus:
    def __init__(self, **kwargs):
        self._llm = None


def install_stubs():
    definitions = {
        "terminal_bench": {},
        "terminal_bench.agents": {},
        "terminal_bench.agents.base_agent": {
            "AgentResult": type("AgentResult", (), {})
        },
        "terminal_bench.agents.terminus_1": {"Terminus": FakeTerminus},
        "terminal_bench.llms": {},
        "terminal_bench.llms.base_llm": {
            "BaseLLM": FakeBaseLLM,
            "OutputLengthExceededError": FakeOutputLengthExceededError,
        },
        "terminal_bench.terminal": {},
        "terminal_bench.terminal.tmux_session": {
            "TmuxSession": type("TmuxSession", (), {})
        },
    }
    for name, attrs in definitions.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


install_stubs()

from terminal_bench_integration.noc_sglang_agent import (
    NOCSGLangAgent,
    NOCSGLangLLM,
    NOCSGLangTerminus2Agent,
)
from noc_context_manager.compile_cost import (
    CompileCostProfile,
    LatencyCurve,
    ProfileProvenance,
)
from noc_context_manager.compile_policy import CostBenefitCompilePolicy


def transport(payload):
    content = (
        '{"commands":[],"is_task_complete":true}'
        if "pic_link" in payload or "sub_context" not in payload
        else "compiled"
    )
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 10},
    }


def make_llm(tmp_path: Path, variant="selected_context_radixcache_legolink"):
    llm = NOCSGLangLLM(
        model_name="model",
        api_base="http://patched/v1",
        original_api_base="http://original/v1",
        variant=variant,
        segmentation_mode="deterministic",
        legolink_k=16,
        max_tokens=64,
        timeout=1,
        artifacts_root=str(tmp_path),
        tokenizer_mode="regex",
    )
    llm._patched_transport = transport
    llm._original_transport = transport
    llm.set_task_context("task", tmp_path / "metrics.jsonl")
    return llm


def test_noc_llm_compiles_and_links_selected_context(tmp_path):
    llm = make_llm(tmp_path)
    output = llm.call(
        "fix parser",
        message_history=[{"role": "system", "content": "preserve behavior " * 100}],
    )
    assert "is_task_complete" in output
    rows = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")
    assert "selected_context_radixcache_legolink" in rows
    metric = json.loads(rows.splitlines()[-1])
    assert metric["linked_tokens"] == 200
    # Source order puts the long reusable instruction before the tiny current
    # prompt.  Prefix closure therefore compiles only the reusable block.
    assert metric["recompute_tokens"] == 16
    assert metric["reused_tokens"] == 184
    assert metric["context_cache_ratio"] == 0.92
    assert any(
        value.context_field == "task_instruction"
        for value in llm._manager.store.list_by_task("task")
    )


def test_full_context_baseline_skips_segmentation(tmp_path):
    llm = make_llm(tmp_path, "full_context_original_sglang")
    llm.call("fix parser", message_history=[])
    assert llm._manager is not None
    assert llm._manager.store.list_by_task("task") == []
    assert len(llm._manager.collector.events()) == 1


def test_noc_agent_filters_context_kwargs_before_terminus():
    agent = NOCSGLangAgent(
        model_name="model",
        api_base="http://patched/v1",
        original_api_base="http://original/v1",
        variant="selected_context_original_sglang",
        max_episodes=2,
        version="local-smoke",
        tokenizer_mode="regex",
    )
    assert isinstance(agent._llm, NOCSGLangLLM)


def test_noc_terminus2_agent_uses_noc_llm_transport():
    agent = object.__new__(NOCSGLangTerminus2Agent)
    assert agent.name() == "noc-sglang-terminus-2"
    assert issubclass(NOCSGLangTerminus2Agent, NOCSGLangAgent.__bases__[0])


def test_cost_benefit_mode_requires_a_measured_profile(tmp_path):
    try:
        NOCSGLangLLM(
            model_name="model",
            api_base="http://patched/v1",
            original_api_base="http://original/v1",
            tokenizer_mode="regex",
            artifacts_root=str(tmp_path),
            compile_policy_mode="cost_benefit",
        )
    except ValueError as error:
        assert "requires compile_cost_profile" in str(error)
    else:
        raise AssertionError("cost_benefit silently ran without measurements")


def test_llm_segmentation_never_inherits_generation_endpoint(tmp_path):
    try:
        NOCSGLangLLM(
            model_name="model",
            api_base="http://patched/v1",
            original_api_base="http://original/v1",
            tokenizer_mode="regex",
            artifacts_root=str(tmp_path),
            segmentation_mode="llm",
        )
    except ValueError as error:
        assert "explicit segmentation_api_base" in str(error)
    else:
        raise AssertionError("LLM segmentation inherited a generation endpoint")


def test_s1k_keeps_native_segmentation_endpoint_separate(tmp_path):
    llm = NOCSGLangLLM(
        model_name="model",
        api_base="http://semantic-kv/v1",
        original_api_base="http://native-generation/v1",
        variant="selected_context_radixcache_legolink",
        tokenizer_mode="regex",
        artifacts_root=str(tmp_path),
        segmentation_mode="llm",
        segmentation_api_base="http://native-segmentation/v1",
    )
    assert llm._segmenter.endpoint == "http://native-segmentation/v1/chat/completions"
    assert llm._patched_transport.endpoint == "http://semantic-kv/v1/chat/completions"


def test_cost_benefit_mode_loads_the_declared_profile(tmp_path):
    curve = LatencyCurve(
        name="latency",
        points=((1, 1.0), (4096, 2.0)),
        samples_per_point=3,
    )
    profile = CompileCostProfile(
        provenance=ProfileProvenance(
            profile_version="test-v1",
            hardware_id="test-gpu",
            model_id="model",
        ),
        prefill_ms=curve,
        compile_ms=curve,
        link_ms=curve,
    )
    profile_path = profile.write_json(tmp_path / "profile.json")

    llm = NOCSGLangLLM(
        model_name="model",
        api_base="http://patched/v1",
        original_api_base="http://original/v1",
        tokenizer_mode="regex",
        artifacts_root=str(tmp_path),
        compile_policy_mode="cost_benefit",
        compile_cost_profile=str(profile_path),
        deployment_hardware_id="test-gpu",
        compile_reuse_probability=0.25,
    )

    assert isinstance(llm._compile_policy, CostBenefitCompilePolicy)
    assert llm._compile_policy.profile.profile_id == profile.profile_id
    assert llm._compile_policy.reuse_expectation.probability == 0.25


def test_metrics_row_records_eligibility_and_reuse_accounting(tmp_path):
    """Gates 1-3 per request, for both serving arms of the formal matrix."""

    for variant in ("selected_context_radixcache_legolink", "selected_context_original_sglang"):
        root = tmp_path / variant
        root.mkdir()
        llm = make_llm(root, variant)
        llm.call("fix parser", message_history=[{"role": "system", "content": "preserve behavior " * 100}])
        metric = json.loads((root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        summary = metric["reuse_eligibility_summary"]
        assert summary["canonical"] is True and summary["experimental_link_count"] == 0
        accounting = metric["reuse_accounting"]
        counts = accounting["counts"]
        assert {"boundary_count", "linked_block_count", "repair_tokens", "recomputed_tokens"} <= set(counts)
        assert counts["link_setup_time_s"]["availability"] == "unavailable"
        if variant.endswith("legolink"):
            assert counts["recomputed_tokens"]["value"] == metric["recompute_tokens"]
            decomposition = accounting["recompute_decomposition"]
            assert decomposition["recomputed_includes_repair"] is True
            assert decomposition["recomputed_tokens"] == metric["recompute_tokens"]
            assert decomposition["repair_tokens"] <= decomposition["recomputed_tokens"]
        else:
            assert counts["linked_block_count"]["value"] == 0
            assert counts["repair_tokens"]["reason"] == "not_applicable_native_route"
        assert all(record["reuse_eligibility"] for record in metric["block_planning"])
