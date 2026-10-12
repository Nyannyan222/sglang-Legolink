"""The runtime is reproducible and identifiable from what this repository ships: the patch series, the shipped runtime files, the manifest and the
fingerprint tool agree with each other. These tests run offline and need no SGLang. They prove consistency of the shipped data; rebuilding the
tree from upstream (``scripts/build_patched_sglang.sh``) is what proves the series applies to the pinned base, and needs the network."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "external_patches" / "sglang"
SHIPPED = ROOT / "patched_sglang_files"
SUPPORT = ROOT / "upstream_support_files"
SPEC = json.loads((ROOT / "PATCHED_SGLANG.json").read_text(encoding="utf-8"))
UPSTREAM = json.loads((ROOT / "UPSTREAM_SGLANG.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "RUNTIME_MANIFEST.json").read_text(encoding="utf-8"))

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_pinned_identity_is_the_documented_one():
    assert SPEC["patched_tree"] == UPSTREAM["patched_tree_id"] == MANIFEST["patched_tree"] == "d0c440cfb49a775c52a3eb88437a56c1050e4428"
    assert UPSTREAM["base_commit_full"] == MANIFEST["base_commit"] and UPSTREAM["base_commit_full"].startswith(SPEC["base_commit"])
    assert SPEC["upstream"] == UPSTREAM["upstream"] == MANIFEST["upstream"] == "https://github.com/sgl-project/sglang"


def test_the_patch_series_is_complete_ordered_and_unaltered():
    patches = sorted(p for p in PATCH_DIR.glob("*.patch"))
    assert len(patches) == SPEC["patch_count"] == len(SPEC["patches"]) == 28
    assert [p.name[:4] for p in patches] == [f"{index:04d}" for index in range(1, 29)]  # no gap, no duplicate, apply order = name order
    recorded = {entry["file"]: entry["sha256"] for entry in SPEC["patches"]}
    assert set(recorded) == {p.name for p in patches}
    for patch in patches:
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == recorded[patch.name], patch.name  # byte hash: line endings are part of a patch


def test_the_patches_touch_exactly_the_files_in_the_manifest():
    touched: set[str] = set()
    for patch in PATCH_DIR.glob("*.patch"):
        touched.update(re.findall(r"^diff --git a/(\S+) b/", patch.read_text(encoding="utf-8", errors="replace"), flags=re.MULTILINE))
    assert touched == set(MANIFEST["files"])
    assert len(touched) == 27
    for module in UPSTREAM["patched_only_modules"]:
        assert f"python/{module}" in touched


def test_the_shipped_runtime_files_are_exactly_the_patched_files():
    shipped = {path.relative_to(SHIPPED).as_posix() for path in SHIPPED.rglob("*.py")}
    assert shipped == set(MANIFEST["files"])
    for name, expected in MANIFEST["files"].items():
        assert sha256_lf(SHIPPED / name) == expected, name


def test_the_unmodified_upstream_support_files_are_the_base_commits_files():
    shipped = {path.relative_to(SUPPORT).as_posix() for path in SUPPORT.rglob("*.py")}
    assert shipped == set(MANIFEST["unmodified_upstream_files"])
    assert not shipped & set(MANIFEST["files"])  # a file is either patched or verbatim upstream, never both
    for name, expected in MANIFEST["unmodified_upstream_files"].items():
        assert sha256_lf(SUPPORT / name) == expected, name


def run_verify(directory: Path) -> tuple[int, dict]:
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_runtime.py"), "--sglang-dir", str(directory), "--json"], capture_output=True, text=True, timeout=300, check=False)
    return done.returncode, json.loads(done.stdout)


def test_verify_runtime_accepts_the_shipped_files():
    code, report = run_verify(SHIPPED)
    assert code == 0, report
    checks = report["checks"]
    assert checks["files"]["status"] == "pass" and checks["files"]["checked"] == 27
    assert checks["new_modules"]["status"] == "pass"
    assert checks["tree"]["status"] == "not_applicable"  # a directory of files has no tree id; it must not be reported as a pass
    assert checks["contract"]["status"] in {"pass", "skipped"}
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    assert checks["contract"]["status"] == "pass" and checks["contract"]["runtime_owned_fields_equal"] is True


def test_verify_runtime_names_an_altered_a_missing_and_a_stale_runtime(tmp_path):
    copy = tmp_path / "python"
    shutil.copytree(SHIPPED / "python", copy)
    target = copy / "sglang" / "srt" / "mem_cache" / "radix_cache.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# local edit\n", encoding="utf-8")
    code, report = run_verify(copy)
    assert code == 1 and report["checks"]["files"]["changed"] == ["python/sglang/srt/mem_cache/radix_cache.py"]

    target.write_bytes((SHIPPED / "python/sglang/srt/mem_cache/radix_cache.py").read_bytes())
    (copy / "sglang" / "srt" / "pic_runtime.py").unlink()
    code, report = run_verify(copy)
    assert code == 1 and "python/sglang/srt/pic_runtime.py" in report["checks"]["files"]["missing"]
    assert "sglang/srt/pic_runtime.py" in report["checks"]["new_modules"]["missing"]


def test_verify_runtime_tolerates_windows_line_endings(tmp_path):
    copy = tmp_path / "python"
    shutil.copytree(SHIPPED / "python", copy)
    for path in copy.rglob("*.py"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    code, report = run_verify(copy)
    assert code == 0 and report["checks"]["files"]["status"] == "pass"


def test_a_directory_without_sglang_is_reported_not_passed(tmp_path):
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_runtime.py"), "--sglang-dir", str(tmp_path)], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 2 and '"status": "fail"' in done.stdout
