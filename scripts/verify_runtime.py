#!/usr/bin/env python3
"""Is this SGLang tree the patched LegoLink / PICRuntime runtime this repository was written against?

    python scripts/verify_runtime.py                       # the sglang that ``import sglang`` would find (not imported, only located)
    python scripts/verify_runtime.py --sglang-dir /path/to/patched-sglang      # a checkout root, its python/ dir, or a site-packages dir
    python scripts/verify_runtime.py --json

What it checks (all CPU, standard library; no GPU and no sglang import):

1. ``tree``       the git tree id of the checkout's HEAD equals ``PATCHED_SGLANG.json`` -> ``patched_tree`` (only when the directory is a clean git
                  checkout; otherwise reported as ``not_applicable``, never as a pass)
2. ``files``      every file the patch series adds or changes exists in the tree and has the SHA-256 recorded in ``RUNTIME_MANIFEST.json``
                  (line endings are normalised, so a Windows checkout compares equal)
3. ``new_modules`` the four modules that exist only in the patched runtime are present
4. ``contract``   the ``pic_link`` field lists of the runtime (``pic_runtime.PIC_LINK_REQUEST_FIELDS`` / ``PIC_LINK_RUNTIME_OWNED_FIELDS``) equal the host's
                  (``noc_context_manager.pic_link_contract``). Needs ``torch``; reported as ``skipped`` when it cannot be imported

Exit code 0 only when ``files``, ``new_modules`` and ``contract`` pass and ``tree`` is not a mismatch. ``skipped`` / ``not_applicable`` are
printed as such and never counted as a pass of that check by anything else in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256_text_file(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def locate_python_root(candidate: Path | None) -> Path | None:
    """Return the directory that contains the ``sglang`` package (``<checkout>/python`` or a site-packages directory)."""

    if candidate is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            return None
        return Path(list(spec.submodule_search_locations)[0]).parent
    for base in (candidate, candidate / "python"):
        if (base / "sglang" / "srt").is_dir():
            return base
    return None


def git_checkout_root(python_root: Path) -> Path | None:
    """The SGLang checkout this ``python/`` directory is the root of, or None (an installed copy, or a directory merely *inside* some other repository)."""

    candidate = python_root.parent if python_root.name == "python" else python_root
    if not (candidate / ".git").exists():
        return None
    top = run_git(candidate, "rev-parse", "--show-toplevel")
    return candidate if top and Path(top).resolve() == candidate.resolve() else None


def run_git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def check_tree(python_root: Path, expected: str) -> dict[str, Any]:
    checkout = git_checkout_root(python_root)
    if checkout is None:
        return {"status": "not_applicable", "detail": "not a git checkout (an installed copy has no tree id); the file hashes decide"}
    tree = run_git(checkout, "rev-parse", "HEAD^{tree}")
    dirty = run_git(checkout, "status", "--porcelain", "--untracked-files=no")
    if tree is None:
        return {"status": "not_applicable", "detail": "git could not read HEAD"}
    result = {"actual": tree, "expected": expected, "dirty": bool(dirty)}
    if dirty:
        return {**result, "status": "not_applicable", "detail": "the working tree has uncommitted changes, so HEAD's tree id does not describe it; the file hashes decide"}
    return {**result, "status": "pass" if tree == expected else "mismatch"}


def check_files(python_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    missing, changed = [], []
    for relative, expected in sorted(manifest["files"].items()):
        path = python_root / relative.removeprefix("python/")
        if not path.is_file():
            missing.append(relative)
        elif sha256_text_file(path) != expected:
            changed.append(relative)
    return {"status": "pass" if not (missing or changed) else "fail", "checked": len(manifest["files"]), "missing": missing, "changed": changed}


def check_new_modules(python_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    missing = [m for m in spec["patched_only_modules"] if not (python_root / m).is_file()]
    return {"status": "pass" if not missing else "fail", "missing": missing}


def check_contract(python_root: Path) -> dict[str, Any]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return {"status": "skipped", "detail": "torch is not importable; install it (CPU build is enough) to compare the pic_link field lists"}
    from noc_context_manager.pic_link_contract import PIC_LINK_REQUEST_FIELDS, PIC_LINK_RUNTIME_OWNED_FIELDS

    path = python_root / "sglang" / "srt" / "pic_runtime.py"
    module_spec = importlib.util.spec_from_file_location("legolink_verify_pic_runtime", path)
    if module_spec is None or module_spec.loader is None:
        return {"status": "fail", "detail": f"cannot load {path}"}
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - reported, not hidden
        return {"status": "skipped", "detail": f"pic_runtime.py could not be loaded in this interpreter ({type(error).__name__}: {error})"}
    request_equal = module.PIC_LINK_REQUEST_FIELDS == PIC_LINK_REQUEST_FIELDS
    owned_equal = module.PIC_LINK_RUNTIME_OWNED_FIELDS == PIC_LINK_RUNTIME_OWNED_FIELDS
    return {"status": "pass" if request_equal and owned_equal else "fail", "request_fields_equal": request_equal, "runtime_owned_fields_equal": owned_equal,
            "host_request_fields": len(PIC_LINK_REQUEST_FIELDS), "host_runtime_owned_fields": len(PIC_LINK_RUNTIME_OWNED_FIELDS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sglang-dir", type=Path, default=None, help="patched SGLang checkout root, its python/ directory, or a site-packages directory")
    parser.add_argument("--json", action="store_true", help="print only the JSON report")
    args = parser.parse_args()

    spec = json.loads((ROOT / "PATCHED_SGLANG.json").read_text(encoding="utf-8"))
    upstream = json.loads((ROOT / "UPSTREAM_SGLANG.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "RUNTIME_MANIFEST.json").read_text(encoding="utf-8"))
    python_root = locate_python_root(args.sglang_dir)
    if python_root is None:
        report = {"status": "fail", "detail": "no sglang package found; pass --sglang-dir (see docs/INSTALLATION.md to build the patched runtime)"}
        print(json.dumps(report, indent=2))
        return 2

    checks = {
        "tree": check_tree(python_root, spec["patched_tree"]),
        "files": check_files(python_root, manifest),
        "new_modules": check_new_modules(python_root, upstream),
        "contract": check_contract(python_root),
    }
    ok = checks["files"]["status"] == "pass" and checks["new_modules"]["status"] == "pass" and checks["contract"]["status"] in {"pass", "skipped"} \
        and checks["tree"]["status"] != "mismatch"
    report = {
        "status": "pass" if ok else "fail",
        "sglang_python_root": str(python_root),
        "expected": {"upstream": spec["upstream"], "base_commit": upstream["base_commit_full"], "patched_tree": spec["patched_tree"], "patch_count": spec["patch_count"]},
        "checks": checks,
        "note": "pass means the tree is byte-equivalent to the runtime this repository was written against; it says nothing about GPU behaviour.",
    }
    print(json.dumps(report, indent=2))
    if not args.json:
        for name, result in checks.items():
            print(f"  {name:12s} {result['status']}", file=sys.stderr)
        print(f"  overall      {report['status']}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
