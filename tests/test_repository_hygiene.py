"""The published repository is self-consistent and clean: every file the documentation tells you to use exists, relative links resolve, and no
file carries a private path, credential, or large artifact. Runs over the working tree (``git ls-files`` when available, else a directory walk)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".tmp", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}
pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def repository_files() -> list[Path]:
    try:
        done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True, timeout=60, check=False)
        if done.returncode == 0 and done.stdout:
            return [ROOT / name for name in done.stdout.decode("utf-8").split("\0") if name and (ROOT / name).is_file()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and not p.name.endswith(".egg-info")]


FILES = repository_files()
TEXT_SUFFIXES = {".py", ".md", ".sh", ".json", ".txt", ".toml", ".ini", ".cfg", ".yml", ".yaml", ".patch", ""}

#: patterns that must not appear anywhere in the published tree. Built from fragments so this file does not match itself.
FORBIDDEN = {
    "a Windows user path": re.compile(r"[A-Za-z]:[\\/]+" + "Users" + r"[\\/]"),
    "a home directory": re.compile(r"/(?:home|Users)/" + r"[a-z][a-z0-9_.-]+/"),
    "a cloud-sync folder path": re.compile("One" + "Drive"),
    "a GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|github_pat_[A-Za-z0-9_]{20,}"),
    "a Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "an API key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b|AKIA[0-9A-Z]{16}"),
    "a private key": re.compile("-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "a cluster name": re.compile("neno" + "5", re.IGNORECASE),
}


def text_files() -> list[Path]:
    return [p for p in FILES if p.suffix in TEXT_SUFFIXES and p.stat().st_size < 4_000_000]


def test_no_private_paths_credentials_or_cluster_names():
    hits = []
    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN.items():
            match = pattern.search(text)
            if match:
                hits.append(f"{path.relative_to(ROOT)}: {label}: {match.group(0)[:60]!r}")
    assert not hits, "\n".join(hits)


def test_no_large_or_binary_artifacts():
    too_big = [f"{p.relative_to(ROOT)} ({p.stat().st_size} bytes)" for p in FILES if p.stat().st_size > 3_000_000]
    assert not too_big, too_big
    forbidden_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".gguf", ".ckpt", ".npz", ".npy", ".sqlite", ".db", ".nsys-rep", ".tar", ".gz", ".zip", ".pptx", ".docx", ".pyc"}
    bad = [str(p.relative_to(ROOT)) for p in FILES if p.suffix.lower() in forbidden_suffixes]
    assert not bad, bad


def test_the_old_research_content_is_not_in_the_tree():
    top_level = {p.relative_to(ROOT).parts[0] for p in FILES}
    for name in ("benchmark_results", "results", "artifacts", "dist", "deployment", "local_validation", "corpus", "openclaw_sglang_diff_bundle"):
        assert name not in top_level, name
    assert not [p for p in FILES if p.name.lower().endswith((".log", ".out")) or "slurm" in p.name.lower()]


LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
BACKTICK_PATH = re.compile(r"`((?:\.?/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*-]+\.(?:py|sh|json|md|txt|patch|ini|toml))`")
#: paths a document mentions that are produced at run time or live outside the repository
RUNTIME_OUTPUTS = ("run/", "./run/", ".tmp/")
#: names inside documents that are placeholders, upstream-relative or other-repository paths
NOT_OURS_PREFIXES = ("python/", "sglang/", "managers/", "mem_cache/", "model_executor/", "entrypoints/", "layers/", "models/", "utils/", "tests/pic/support")


def markdown_files() -> list[Path]:
    return [p for p in FILES if p.suffix == ".md"]


def test_markdown_relative_links_resolve():
    broken = []
    for doc in markdown_files():
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if path and not (doc.parent / path).exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, "\n".join(broken)


def test_every_repository_file_the_documentation_names_exists():
    missing = []
    for doc in markdown_files():
        if doc.relative_to(ROOT).as_posix().startswith(("docs/logical_radix_roots", "docs/semantic_cache_namespaces", "external_patches/")):
            continue  # carry a provenance note: they cite research tooling that is deliberately not in this repository
        for token in BACKTICK_PATH.findall(doc.read_text(encoding="utf-8")):
            relative = token[2:] if token.startswith("./") else token
            if relative.startswith(RUNTIME_OUTPUTS) or relative.startswith(NOT_OURS_PREFIXES) or "*" in relative:
                continue
            if not ((ROOT / relative).exists() or (doc.parent / relative).exists()):
                missing.append(f"{doc.relative_to(ROOT)}: {token}")
    assert not missing, "\n".join(missing)


def test_the_files_the_installation_commands_use_exist():
    for relative in ("scripts/build_patched_sglang.sh", "scripts/verify_runtime.py", "scripts/serve_sglang.sh", "scripts/check_server.py", "scripts/run_openai_proxy.py",
                     "examples/compile_link_decode.py", "configs/constraints-runtime.txt", "configs/requirements-runtime-reference.txt", "PATCHED_SGLANG.json", "UPSTREAM_SGLANG.json",
                     "RUNTIME_MANIFEST.json", "LICENSE", "NOTICE", "requirements.txt", "pyproject.toml"):
        assert (ROOT / relative).is_file(), relative
    assert len(list((ROOT / "external_patches" / "sglang").glob("*.patch"))) == 28


def test_shell_scripts_are_lf_and_have_a_shebang():
    for script in (p for p in FILES if p.suffix == ".sh"):
        data = script.read_bytes()
        assert data.startswith(b"#!/usr/bin/env bash\n"), script
        assert b"\r\n" not in data, f"{script} has CRLF line endings; bash would fail on it"
