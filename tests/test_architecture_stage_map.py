"""The 13-stage architecture maps one-to-one onto canonical source files.

`noc_context_manager/architecture_stages.py` is data. These tests are what keep it
honest: every canonical file and symbol must exist, every host stage must carry its
own `ARCHITECTURE_STAGE` marker, and every recorded call edge must really appear
in its caller's body (read with `ast`, so a comment or a string cannot satisfy it).

Runtime stages (10-13) are checked against `external/sglang` when that checkout is
present and skipped when it is not -- the same dependency the rest of the suite
already has. Nothing here imports the runtime.
"""

from __future__ import annotations

import ast
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noc_context_manager.architecture_stages import (  # noqa: E402
    CALL_EDGES,
    HOST,
    RUNTIME,
    RUNTIME_ROOT,
    STAGES,
)

RUNTIME_DIR = ROOT / RUNTIME_ROOT
if not (RUNTIME_DIR / "pic_runtime.py").is_file():
    # The standalone package ships the patched runtime files (not a git checkout) under patched_sglang_files/.
    RUNTIME_DIR = ROOT / "patched_sglang_files" / "python" / "sglang" / "srt"
HAS_RUNTIME = (RUNTIME_DIR / "pic_runtime.py").is_file()
needs_runtime = pytest.mark.skipif(
    not HAS_RUNTIME, reason="external/sglang checkout is not present"
)


def _skip_if_unshipped(path: Path) -> None:
    """The package ships only the runtime files the patch series modifies; an upstream file it leaves alone is not part of it."""

    if not path.is_file() and "patched_sglang_files" in RUNTIME_DIR.parts:
        pytest.skip(f"{path.name} is an unmodified upstream SGLang file; the package ships only the files the patch series changes")


UPSTREAM_SUPPORT_DIR = ROOT / "upstream_support_files" / "python" / "sglang" / "srt"


def _path(side: str, relative: str) -> Path:
    if side != RUNTIME:
        return ROOT / relative
    candidate = RUNTIME_DIR / relative
    if not candidate.is_file():  # an unmodified upstream file the tests read: shipped verbatim under upstream_support_files/
        return UPSTREAM_SUPPORT_DIR / relative
    return candidate


@lru_cache(maxsize=None)
def _tree(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _find(tree: ast.Module, qualname: str):
    """The class or function node named by a dotted `Class.method` path."""

    scope: list[ast.AST] = list(tree.body)
    node = None
    for part in qualname.split("."):
        node = next(
            (
                item
                for item in scope
                if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == part
            ),
            None,
        )
        if node is None:
            return None
        scope = list(getattr(node, "body", []))
    return node


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            target = child.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _runtime_stages():
    return [value for value in STAGES if value.side == RUNTIME]


def _host_stages():
    return [value for value in STAGES if value.side == HOST]


# --- the table itself -------------------------------------------------------


def test_there_are_exactly_thirteen_stages_numbered_one_to_thirteen() -> None:
    assert [value.step for value in STAGES] == list(range(1, 14))


def test_every_stage_has_exactly_one_canonical_file_and_no_file_serves_two_stages() -> None:
    files = [(value.side, value.canonical_file) for value in STAGES]
    assert all(isinstance(value.canonical_file, str) and value.canonical_file for value in STAGES)
    assert len(set(files)) == len(files), "two stages share a canonical file"


def test_host_stages_are_steps_one_to_nine_and_runtime_stages_ten_to_thirteen() -> None:
    assert [value.step for value in _host_stages()] == list(range(1, 10))
    assert [value.step for value in _runtime_stages()] == list(range(10, 14))


# --- host: file, symbol, marker --------------------------------------------


@pytest.mark.parametrize("stage", _host_stages(), ids=lambda value: f"stage{value.step}")
def test_host_canonical_file_defines_its_entrypoint_and_names_its_stage(stage) -> None:
    path = _path(stage.side, stage.canonical_file)
    assert path.is_file(), path
    text = path.read_text(encoding="utf-8")
    assert f"ARCHITECTURE_STAGE: {stage.step}" in text, "missing stage marker"
    assert "CANONICAL_ENTRYPOINT:" in text
    if stage.step == 3:
        # Defined elsewhere on purpose; exposed here by a lazy module attribute.
        assert "def __getattr__" in text
        return
    assert _find(_tree(str(path)), stage.canonical_symbol) is not None, (
        f"{stage.canonical_symbol} is not defined in {stage.canonical_file}"
    )


@pytest.mark.parametrize("stage", _host_stages(), ids=lambda value: f"stage{value.step}")
def test_host_helper_files_exist(stage) -> None:
    for helper in stage.helpers:
        assert (ROOT / helper).is_file(), helper


def test_stage_three_entrypoint_is_the_same_class_the_validator_module_defines() -> None:
    from noc_context_manager import segmentation_validation, semantic_segmentation

    for name in ("SemanticSegmentationPipeline", "SemanticSegmentationValidator"):
        assert getattr(semantic_segmentation, name) is getattr(segmentation_validation, name)
    with pytest.raises(AttributeError):
        semantic_segmentation.NotAStageThreeName  # noqa: B018


def test_the_first_block_repair_policy_is_reported_as_not_in_this_baseline() -> None:
    """Refactoring must not quietly pull branch-only behaviour into the baseline."""

    assert not (ROOT / "noc_context_manager/legolink_repair_policy.py").exists()
    seven = next(value for value in STAGES if value.step == 7)
    assert "NOT_IN_CURRENT_BASELINE" in seven.note
    assert "legolink_repair_policy.py" in seven.note


# --- runtime: file and symbol (no markers: the runtime source is untouched) --


@needs_runtime
@pytest.mark.parametrize("stage", _runtime_stages(), ids=lambda value: f"stage{value.step}")
def test_runtime_canonical_file_defines_its_entrypoint(stage) -> None:
    path = _path(stage.side, stage.canonical_file)
    _skip_if_unshipped(path)
    assert path.is_file(), path
    assert _find(_tree(str(path)), stage.canonical_symbol) is not None, stage.canonical_symbol
    for helper in stage.helpers:
        _skip_if_unshipped(_path(RUNTIME, helper))
        assert _path(RUNTIME, helper).is_file(), helper


@needs_runtime
def test_radix_cache_keeps_native_prefix_matching_and_the_registry_in_one_class() -> None:
    tree = _tree(str(RUNTIME_DIR / "mem_cache/radix_cache.py"))
    radix = _find(tree, "RadixCache")
    methods = {item.name for item in radix.body if isinstance(item, ast.FunctionDef)}
    assert {"match_prefix", "cache_finished_req", "cache_unfinished_req"} <= methods
    assert {
        "register_legolink_block",
        "lookup_legolink_blocks",
        "acquire_legolink_blocks",
        "release_legolink_blocks",
    } <= methods


# --- call edges -------------------------------------------------------------


def _edge_id(edge) -> str:
    return f"{edge.from_step}-{edge.to_step}:{edge.caller}->{edge.callee}"


@pytest.mark.parametrize(
    "edge", [value for value in CALL_EDGES if value.side == HOST], ids=_edge_id
)
def test_every_host_call_edge_appears_in_its_caller(edge) -> None:
    path = _path(edge.side, edge.caller_file)
    node = _find(_tree(str(path)), edge.caller)
    assert node is not None, f"{edge.caller} not found in {edge.caller_file}"
    assert edge.callee in _called_names(node), (
        f"{edge.caller} does not call {edge.callee}"
    )


@needs_runtime
@pytest.mark.parametrize(
    "edge", [value for value in CALL_EDGES if value.side == RUNTIME], ids=_edge_id
)
def test_every_runtime_call_edge_appears_in_its_caller(edge) -> None:
    path = _path(edge.side, edge.caller_file)
    _skip_if_unshipped(path)
    node = _find(_tree(str(path)), edge.caller)
    assert node is not None, f"{edge.caller} not found in {edge.caller_file}"
    assert edge.callee in _called_names(node), (
        f"{edge.caller} does not call {edge.callee}"
    )


def test_the_edge_checker_can_fail() -> None:
    """A checker that cannot say no proves nothing: the planner never links."""

    node = _find(_tree(str(ROOT / "noc_context_manager/scheduler.py")), "NOCRequestPlanner.plan")
    assert "provisional_decisions" in _called_names(node)
    assert "link" not in _called_names(node)
    assert "run_link" not in _called_names(node)


def test_every_consecutive_pair_of_stages_has_a_recorded_edge_or_a_stated_gap() -> None:
    """6 -> 7 is the one pair with no call: Stage 7 is asked for from Stage 8."""

    covered = {(value.from_step, value.to_step) for value in CALL_EDGES}
    missing = [(a, a + 1) for a in range(1, 13) if (a, a + 1) not in covered]
    assert missing == [(6, 7)]


# --- ownership boundary survives the move -----------------------------------


def test_the_stage_eight_framer_cannot_name_a_runtime_owned_field() -> None:
    from noc_context_manager.pic_link_contract import (
        PIC_LINK_RUNTIME_OWNED_FIELDS,
        LogicalPICRequestFramer,
    )

    source = (ROOT / "noc_context_manager/pic_link_contract.py").read_text(encoding="utf-8")
    body = source.split("class LogicalPICRequestFramer", 1)[1]
    # The docstring names them in prose (as the things the host must not own);
    # the executable part -- everything after the docstring -- must not.
    code = body.split('"""', 2)[2]
    for field in sorted(PIC_LINK_RUNTIME_OWNED_FIELDS):
        assert f'"{field}"' not in code and f"{field}=" not in code, field
    assert LogicalPICRequestFramer.frame is not None


def test_the_host_never_imports_the_gitignored_runtime() -> None:
    for relative in (
        "noc_context_manager/architecture_stages.py",
        "noc_context_manager/pic_link_contract.py",
        "noc_context_manager/sglang_adapter.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("sglang"), (relative, node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("sglang"), (relative, alias.name)
