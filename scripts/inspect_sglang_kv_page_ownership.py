#!/usr/bin/env python3
"""Inspect SGLang KV-page ownership and semantic block attach points.

This script is intentionally read-only.  It maps the installed or checked-out
SGLang source to answer one implementation question:

    Where can OpenClaw attach semantic sub-context ownership metadata to the
    request -> token -> KV-page path without pretending that prefix-cache order
    reuse is semantic reuse?

Run it inside an SGLang environment, or pass --source-root pointing at an
SGLang checkout / installed package root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MODULE_CANDIDATES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "scheduler": [
        ("sglang.srt.managers.scheduler", ("sglang", "srt", "managers", "scheduler.py")),
        ("sglang.srt.managers.scheduler", ("python", "sglang", "srt", "managers", "scheduler.py")),
    ],
    "schedule_batch": [
        ("sglang.srt.managers.schedule_batch", ("sglang", "srt", "managers", "schedule_batch.py")),
        ("sglang.srt.managers.schedule_batch", ("python", "sglang", "srt", "managers", "schedule_batch.py")),
    ],
    "schedule_policy": [
        ("sglang.srt.managers.schedule_policy", ("sglang", "srt", "managers", "schedule_policy.py")),
        ("sglang.srt.managers.schedule_policy", ("python", "sglang", "srt", "managers", "schedule_policy.py")),
    ],
    "io_struct": [
        ("sglang.srt.managers.io_struct", ("sglang", "srt", "managers", "io_struct.py")),
        ("sglang.srt.managers.io_struct", ("python", "sglang", "srt", "managers", "io_struct.py")),
    ],
    "radix_cache": [
        ("sglang.srt.mem_cache.radix_cache", ("sglang", "srt", "mem_cache", "radix_cache.py")),
        ("sglang.srt.mem_cache.radix_cache", ("python", "sglang", "srt", "mem_cache", "radix_cache.py")),
    ],
    "memory_pool": [
        ("sglang.srt.mem_cache.memory_pool", ("sglang", "srt", "mem_cache", "memory_pool.py")),
        ("sglang.srt.mem_cache.memory_pool", ("python", "sglang", "srt", "mem_cache", "memory_pool.py")),
    ],
    "allocator": [
        ("sglang.srt.mem_cache.allocator", ("sglang", "srt", "mem_cache", "allocator.py")),
        ("sglang.srt.mem_cache.allocator", ("python", "sglang", "srt", "mem_cache", "allocator.py")),
    ],
    "forward_batch_info": [
        (
            "sglang.srt.model_executor.forward_batch_info",
            ("sglang", "srt", "model_executor", "forward_batch_info.py"),
        ),
        (
            "sglang.srt.model_executor.forward_batch_info",
            ("python", "sglang", "srt", "model_executor", "forward_batch_info.py"),
        ),
    ],
    "model_runner": [
        ("sglang.srt.model_executor.model_runner", ("sglang", "srt", "model_executor", "model_runner.py")),
        ("sglang.srt.model_executor.model_runner", ("python", "sglang", "srt", "model_executor", "model_runner.py")),
    ],
}


PATTERN_GROUPS: dict[str, list[str]] = {
    "request_identity": [
        "rid",
        "req_pool_idx",
        "req_pool_indices",
        "Req(",
        "ScheduleBatch",
    ],
    "request_metadata": [
        "extra_key",
        "custom_labels",
        "custom_params",
        "openclaw",
        "sub_context",
        "canonical_key",
        "content_hash",
        "reuse_policy",
    ],
    "prompt_token_mapping": [
        "req_to_token_pool",
        "ReqToTokenPool",
        "req_to_token",
        "fill_ids",
        "origin_input_ids",
        "extend_input_len",
        "extend_prefix_len",
        "extend_input_logprob_token_ids",
        "seq_lens",
        "prefix_indices",
    ],
    "kv_page_storage": [
        "token_to_kv_pool",
        "TokenToKVPool",
        "out_cache_loc",
        "alloc(",
        "free(",
        "alloc_extend",
        "alloc_decode",
        "available_size",
        "KVCache",
        "kv_cache",
    ],
    "prefix_cache_flow": [
        "match_prefix",
        "cache_finished_req",
        "cache_unfinished_req",
        "cached_tokens",
        "matched_prefix_len",
        "last_device_node",
        "device_indices",
        "RadixCache",
    ],
    "prefill_execution": [
        "prepare_for_extend",
        "forward_extend",
        "prefill",
        "extend_num_tokens",
        "extend_seq_lens",
        "extend_start_loc",
        "positions",
    ],
}


ROLE_GROUP_HINTS: dict[str, set[str]] = {
    "scheduler": {"request_identity", "request_metadata", "prompt_token_mapping", "prefix_cache_flow", "prefill_execution"},
    "schedule_batch": {"request_identity", "request_metadata", "prompt_token_mapping", "kv_page_storage", "prefix_cache_flow", "prefill_execution"},
    "schedule_policy": {"request_metadata", "prefix_cache_flow"},
    "io_struct": {"request_metadata", "request_identity"},
    "radix_cache": {"prefix_cache_flow", "kv_page_storage"},
    "memory_pool": {"prompt_token_mapping", "kv_page_storage"},
    "allocator": {"kv_page_storage"},
    "forward_batch_info": {"prompt_token_mapping", "kv_page_storage", "prefill_execution"},
    "model_runner": {"kv_page_storage", "prefill_execution"},
}


REQUIRED_GROUPS: dict[str, set[str]] = {
    "schedule_batch": {"request_identity", "prompt_token_mapping", "kv_page_storage", "prefill_execution"},
    "radix_cache": {"prefix_cache_flow"},
    "memory_pool": {"prompt_token_mapping", "kv_page_storage"},
    "allocator": {"kv_page_storage"},
}


FUNCTION_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")


@dataclass(frozen=True)
class PatternHit:
    group: str
    pattern: str
    line_number: int
    owner: str
    line: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleInspection:
    role: str
    module_name: str
    path: str
    found: bool
    locate_method: str = ""
    pattern_hits: list[PatternHit] = field(default_factory=list)
    group_counts: dict[str, int] = field(default_factory=dict)
    notable_owners: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pattern_hits"] = [hit.to_dict() for hit in self.pattern_hits]
        return payload


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
    candidates = [
        source_root.joinpath(*relative_path),
        source_root.joinpath("python", *relative_path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def locate_module(role: str, source_root: Path | None) -> tuple[str, Path | None, str]:
    for module_name, relative_path in MODULE_CANDIDATES[role]:
        if source_root is not None:
            path = find_by_source_root(source_root, relative_path)
            if path is not None:
                return module_name, path, "source_root"
        path = find_by_import(module_name)
        if path is not None:
            return module_name, path, "import"
    module_name = MODULE_CANDIDATES[role][0][0]
    return module_name, None, "not_found"


def scan_patterns(path: Path, max_hits_per_pattern: int) -> tuple[list[PatternHit], dict[str, int], dict[str, list[str]]]:
    hits: list[PatternHit] = []
    raw_counts: Counter[tuple[str, str]] = Counter()
    owner_counts: dict[str, Counter[str]] = defaultdict(Counter)
    current_owner = "<module>"

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        class_match = CLASS_RE.match(line)
        function_match = FUNCTION_RE.match(line)
        if class_match:
            current_owner = f"class {class_match.group(1)}"
        elif function_match:
            current_owner = f"def {function_match.group(1)}"

        stripped = line.strip()
        for group, patterns in PATTERN_GROUPS.items():
            for pattern in patterns:
                if pattern in line:
                    key = (group, pattern)
                    raw_counts[key] += 1
                    owner_counts[group][current_owner] += 1
                    if raw_counts[key] <= max_hits_per_pattern:
                        hits.append(
                            PatternHit(
                                group=group,
                                pattern=pattern,
                                line_number=line_number,
                                owner=current_owner,
                                line=stripped,
                            )
                        )

    group_counts: Counter[str] = Counter()
    for (group, _pattern), count in raw_counts.items():
        group_counts[group] += count
    notable_owners = {
        group: [owner for owner, _count in counter.most_common(6)]
        for group, counter in owner_counts.items()
    }
    return hits, dict(group_counts), notable_owners


def inspect(source_root: Path | None, max_hits_per_pattern: int) -> list[ModuleInspection]:
    inspections: list[ModuleInspection] = []
    for role in MODULE_CANDIDATES:
        module_name, path, locate_method = locate_module(role, source_root)
        if path is None:
            inspections.append(
                ModuleInspection(
                    role=role,
                    module_name=module_name,
                    path="",
                    found=False,
                    locate_method=locate_method,
                )
            )
            continue
        hits, group_counts, notable_owners = scan_patterns(path, max_hits_per_pattern=max_hits_per_pattern)
        inspections.append(
            ModuleInspection(
                role=role,
                module_name=module_name,
                path=str(path),
                found=True,
                locate_method=locate_method,
                pattern_hits=hits,
                group_counts=group_counts,
                notable_owners=notable_owners,
            )
        )
    return inspections


def readiness(inspections: list[ModuleInspection]) -> dict[str, Any]:
    by_role = {inspection.role: inspection for inspection in inspections}
    missing_roles = sorted(role for role in REQUIRED_GROUPS if not by_role.get(role) or not by_role[role].found)
    missing_groups: dict[str, list[str]] = {}
    for role, groups in REQUIRED_GROUPS.items():
        inspection = by_role.get(role)
        if inspection is None or not inspection.found:
            continue
        present = {group for group, count in inspection.group_counts.items() if count > 0}
        missing = sorted(groups - present)
        if missing:
            missing_groups[role] = missing

    attach_plan = [
        {
            "step": 1,
            "name": "carry_subcontext_metadata",
            "anchor": "request object / ScheduleBatch construction",
            "reason": "semantic block identity must travel with each request before prefix matching",
        },
        {
            "step": 2,
            "name": "record_token_span_ownership",
            "anchor": "req_to_token_pool and prefix_indices assignment",
            "reason": "each sub-context needs a stable token span mapped to request token slots",
        },
        {
            "step": 3,
            "name": "record_kv_page_ownership",
            "anchor": "out_cache_loc allocation in token_to_kv_pool",
            "reason": "semantic reuse needs ownership over the KV pages produced by each sub-context span",
        },
        {
            "step": 4,
            "name": "semantic_lookup_before_prefill",
            "anchor": "RadixCache match_prefix / scheduler prefill path",
            "reason": "exact prefix lookup remains unchanged, semantic lookup is an additional block-level path",
        },
        {
            "step": 5,
            "name": "safe_attach_policy",
            "anchor": "reuse_policy, canonical_key, content_hash, tool metadata",
            "reason": "only exact duplicate or judged semantic-equivalent evidence should attach cached KV pages",
        },
    ]

    return {
        "ready_for_kv_page_patch": not missing_roles and not missing_groups,
        "missing_required_roles": missing_roles,
        "missing_required_groups": missing_groups,
        "required_groups": {role: sorted(groups) for role, groups in REQUIRED_GROUPS.items()},
        "attach_plan": attach_plan,
    }


def ownership_map(inspections: list[ModuleInspection]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, hinted_groups in ROLE_GROUP_HINTS.items():
        inspection = next((item for item in inspections if item.role == role), None)
        if inspection is None:
            continue
        for group in sorted(hinted_groups):
            rows.append(
                {
                    "role": role,
                    "group": group,
                    "found": bool(inspection.found and inspection.group_counts.get(group, 0)),
                    "hit_count": inspection.group_counts.get(group, 0),
                    "path": inspection.path,
                    "owners": inspection.notable_owners.get(group, [])[:4],
                }
            )
    return rows


def write_report(path: Path, payload: dict[str, Any], max_report_hits: int) -> None:
    ready = payload["readiness"]
    inspections = payload["modules"]
    ownership_rows = payload["ownership_map"]

    lines = [
        "# SGLang KV Page Ownership Inspection",
        "",
        "## Readiness",
        "",
        f"- Ready for KV-page patch: {ready['ready_for_kv_page_patch']}",
        f"- Missing required roles: {', '.join(ready['missing_required_roles']) or 'none'}",
        f"- Missing required groups: `{json.dumps(ready['missing_required_groups'], ensure_ascii=False)}`",
        "",
        "## Ownership Map",
        "",
        "| role | group | found | hits | likely owners |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in ownership_rows:
        owners = ", ".join(f"`{owner}`" for owner in row["owners"]) or ""
        lines.append(f"| {row['role']} | {row['group']} | {row['found']} | {row['hit_count']} | {owners} |")

    lines.extend(
        [
            "",
            "## Attach Plan",
            "",
            "| step | name | anchor | reason |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for item in ready["attach_plan"]:
        lines.append(f"| {item['step']} | {item['name']} | {item['anchor']} | {item['reason']} |")

    lines.extend(
        [
            "",
            "## Module Summary",
            "",
            "| role | found | locate method | path | groups | hits |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for inspection in inspections:
        groups = ", ".join(sorted(group for group, count in inspection["group_counts"].items() if count))
        lines.append(
            f"| {inspection['role']} | {inspection['found']} | {inspection['locate_method']} | "
            f"`{inspection['path']}` | {groups} | {len(inspection['pattern_hits'])} |"
        )

    lines.extend(["", "## Pattern Hits", ""])
    for inspection in inspections:
        lines.extend([f"### {inspection['role']}", ""])
        if not inspection["found"]:
            lines.extend(["- Not found.", ""])
            continue
        lines.append(f"Source: `{inspection['path']}`")
        lines.append("")
        for hit in inspection["pattern_hits"][:max_report_hits]:
            safe_line = hit["line"].replace("|", "\\|")
            lines.append(
                f"- `{hit['group']}` / `{hit['pattern']}` line {hit['line_number']} "
                f"({hit['owner']}): `{safe_line}`"
            )
        omitted = len(inspection["pattern_hits"]) - max_report_hits
        if omitted > 0:
            lines.append(f"- ... {omitted} more hits omitted")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/sglang_kv_page_ownership_inspection"))
    parser.add_argument("--max-hits-per-pattern", type=int, default=8)
    parser.add_argument("--max-report-hits", type=int, default=120)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve() if args.source_root else None
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inspections = inspect(source_root=source_root, max_hits_per_pattern=args.max_hits_per_pattern)
    ready = readiness(inspections)
    payload = {
        "python_executable": sys.executable,
        "source_root": str(source_root) if source_root else "",
        "readiness": ready,
        "ownership_map": ownership_map(inspections),
        "modules": [inspection.to_dict() for inspection in inspections],
    }
    json_path = output_dir / "sglang_kv_page_ownership.json"
    report_path = output_dir / "sglang_kv_page_ownership_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, payload=payload, max_report_hits=args.max_report_hits)

    print("SGLang KV page ownership inspection")
    print("===================================")
    print(f"Ready for KV-page patch: {ready['ready_for_kv_page_patch']}")
    print(f"Missing roles: {', '.join(ready['missing_required_roles']) or 'none'}")
    print(f"Missing groups: {json.dumps(ready['missing_required_groups'], ensure_ascii=False)}")
    print(f"Output: {output_dir}")

    if args.require_ready and not ready["ready_for_kv_page_patch"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
