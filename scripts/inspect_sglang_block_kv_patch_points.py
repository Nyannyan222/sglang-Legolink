#!/usr/bin/env python3
"""Inspect SGLang source files for OpenClaw block-KV patch points.

Run this inside the same Python environment that launches SGLang, or pass
`--source-root` pointing at a checked-out / installed SGLang source tree.

The output is a concrete patch-point map for the next implementation step:

- where request metadata such as `extra_key` / `custom_labels` is stored,
- where prefix-cache lookup is performed,
- where cache-hit token counters are assigned,
- where the KV/memory cache implementation lives.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MODULE_CANDIDATES = {
    "schedule_policy": [
        ("sglang.srt.managers.schedule_policy", ("sglang", "srt", "managers", "schedule_policy.py")),
    ],
    "schedule_batch": [
        ("sglang.srt.managers.schedule_batch", ("sglang", "srt", "managers", "schedule_batch.py")),
    ],
    "io_struct": [
        ("sglang.srt.managers.io_struct", ("sglang", "srt", "managers", "io_struct.py")),
    ],
    "radix_cache": [
        ("sglang.srt.mem_cache.radix_cache", ("sglang", "srt", "mem_cache", "radix_cache.py")),
        ("sglang.srt.mem_cache.radix_cache", ("python", "sglang", "srt", "mem_cache", "radix_cache.py")),
    ],
    "memory_pool": [
        ("sglang.srt.mem_cache.memory_pool", ("sglang", "srt", "mem_cache", "memory_pool.py")),
        ("sglang.srt.mem_cache.memory_pool", ("python", "sglang", "srt", "mem_cache", "memory_pool.py")),
    ],
    "scheduler": [
        ("sglang.srt.managers.scheduler", ("sglang", "srt", "managers", "scheduler.py")),
    ],
}


PATTERNS = [
    "extra_key",
    "custom_labels",
    "prefix_indices",
    "cache_protected_len",
    "match_prefix",
    "cache_lookup",
    "Radix",
    "radix",
    "req_to_token_pool",
    "token_to_kv_pool",
    "extend",
    "prefill",
    "cached_tokens",
]


@dataclass(frozen=True)
class PatternHit:
    pattern: str
    line_number: int
    line: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleInspection:
    role: str
    module_name: str
    path: str
    found: bool
    reason: str = ""
    pattern_hits: list[PatternHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["pattern_hits"] = [hit.to_dict() for hit in self.pattern_hits]
        return row


def find_by_import(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        return None
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin)
    return path if path.exists() else None


def find_by_source_root(source_root: Path, relative_path: tuple[str, ...]) -> Path | None:
    candidate = source_root.joinpath(*relative_path)
    return candidate if candidate.exists() else None


def locate_module(role: str, source_root: Path | None) -> tuple[str, Path | None, str]:
    candidates = MODULE_CANDIDATES[role]
    for module_name, relative_path in candidates:
        if source_root is not None:
            path = find_by_source_root(source_root, relative_path)
            if path is not None:
                return module_name, path, "source_root"
        path = find_by_import(module_name)
        if path is not None:
            return module_name, path, "import"
    return candidates[0][0], None, "not_found"


def scan_patterns(path: Path) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.strip()
        for pattern in PATTERNS:
            if pattern in line:
                hits.append(PatternHit(pattern=pattern, line_number=line_number, line=stripped))
    return hits


def inspect(source_root: Path | None) -> list[ModuleInspection]:
    inspections: list[ModuleInspection] = []
    for role in MODULE_CANDIDATES:
        module_name, path, reason = locate_module(role, source_root)
        if path is None:
            inspections.append(
                ModuleInspection(
                    role=role,
                    module_name=module_name,
                    path="",
                    found=False,
                    reason=reason,
                )
            )
            continue
        inspections.append(
            ModuleInspection(
                role=role,
                module_name=module_name,
                path=str(path),
                found=True,
                reason=reason,
                pattern_hits=scan_patterns(path),
            )
        )
    return inspections


def patch_readiness(inspections: list[ModuleInspection]) -> dict[str, Any]:
    found_roles = {inspection.role for inspection in inspections if inspection.found}
    hits_by_role = {
        inspection.role: {hit.pattern for hit in inspection.pattern_hits}
        for inspection in inspections
        if inspection.found
    }
    required_roles = {"schedule_policy", "schedule_batch", "radix_cache"}
    required_patterns = {
        "schedule_policy": {"extra_key"},
        "schedule_batch": {"prefix_indices", "cache_protected_len"},
        "radix_cache": {"match_prefix"},
    }
    missing_roles = sorted(required_roles - found_roles)
    missing_patterns: dict[str, list[str]] = {}
    for role, patterns in required_patterns.items():
        if role not in hits_by_role:
            continue
        missing = sorted(patterns - hits_by_role[role])
        if missing:
            missing_patterns[role] = missing
    ready = not missing_roles and not missing_patterns
    return {
        "ready_for_block_kv_patch": ready,
        "missing_required_roles": missing_roles,
        "missing_required_patterns": missing_patterns,
        "next_patch_targets": [
            "Attach OpenClaw block metadata to request objects before prefix matching.",
            "Use openclaw_cache_key/extra_key as the semantic block cache namespace.",
            "Add block-level cache hit/miss counters beside existing prefix counters.",
            "Keep block lookup/store separate from prompt-order prefix lookup.",
        ],
    }


def write_report(path: Path, inspections: list[ModuleInspection], readiness: dict[str, Any]) -> None:
    lines = [
        "# SGLang Block-KV Patch Point Inspection",
        "",
        "## Readiness",
        "",
        f"- Ready for block-KV patch: {readiness['ready_for_block_kv_patch']}",
        f"- Missing required roles: {', '.join(readiness['missing_required_roles']) or 'none'}",
        f"- Missing required patterns: {json.dumps(readiness['missing_required_patterns'], ensure_ascii=False)}",
        "",
        "## Module Map",
        "",
        "| role | found | module | path | hits |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for inspection in inspections:
        lines.append(
            f"| {inspection.role} | {inspection.found} | `{inspection.module_name}` | "
            f"`{inspection.path}` | {len(inspection.pattern_hits)} |"
        )
    lines.extend(["", "## Suggested Patch Targets", ""])
    lines.extend(f"- {target}" for target in readiness["next_patch_targets"])
    lines.extend(["", "## Relevant Pattern Hits", ""])
    for inspection in inspections:
        if not inspection.found:
            lines.extend([f"### {inspection.role}", "", f"- Not found: {inspection.module_name}", ""])
            continue
        lines.extend([f"### {inspection.role}", "", f"Source: `{inspection.path}`", ""])
        for hit in inspection.pattern_hits[:80]:
            safe_line = hit.line.replace("|", "\\|")
            lines.append(f"- `{hit.pattern}` line {hit.line_number}: `{safe_line}`")
        if len(inspection.pattern_hits) > 80:
            lines.append(f"- ... {len(inspection.pattern_hits) - 80} more hits omitted")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/sglang_block_kv_patch_points"))
    parser.add_argument(
        "--require-sglang",
        action="store_true",
        help="Exit non-zero if required SGLang patch points are not found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve() if args.source_root else None
    inspections = inspect(source_root)
    readiness = patch_readiness(inspections)
    payload = {
        "python_executable": sys.executable,
        "source_root": str(source_root) if source_root else "",
        "readiness": readiness,
        "modules": [inspection.to_dict() for inspection in inspections],
    }
    (args.output_dir / "sglang_block_kv_patch_points.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "sglang_block_kv_patch_points_report.md", inspections, readiness)

    print("SGLang block-KV patch point inspection")
    print("=====================================")
    print(f"Ready for block-KV patch: {readiness['ready_for_block_kv_patch']}")
    print(f"Missing roles: {', '.join(readiness['missing_required_roles']) or 'none'}")
    print(f"Output: {args.output_dir}")
    if args.require_sglang and not readiness["ready_for_block_kv_patch"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
