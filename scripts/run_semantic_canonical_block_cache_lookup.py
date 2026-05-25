#!/usr/bin/env python3
"""Exercise OpenClaw semantic/canonical block-level KV lookup policy.

This is the implementation-facing test for:
1. SGLang internal block-level KV storage semantics, modelled as independent
   block records keyed by content hash and canonical key.
2. semantic/canonical block-level cache lookup, where different raw text can hit
   the same stored block when the canonical key matches and the policy allows it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noc_context_manager.sglang_block_kv import SemanticBlockKVStore, SemanticBlockMetadata, stable_hash

DEFAULT_METADATA = Path("benchmark_results/wildclaw_tool_aware_sglang_metadata/field_sglang_metadata.jsonl")
DEFAULT_CANONICAL_MANIFEST = Path("benchmark_results/canonical_semantic_positive_eval/canonical_semantic_positive_runtime_manifest.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9_./:+-]+|[\u4e00-\u9fff]|[^\s]", text)))


def metadata_rows(path: Path) -> list[SemanticBlockMetadata]:
    rows = []
    for row in read_jsonl(path):
        rows.append(SemanticBlockMetadata.from_row(row))
    return rows


def canonical_positive_rows(path: Path) -> list[SemanticBlockMetadata]:
    rows = []
    for row in read_jsonl(path):
        condition = str(row.get("condition", ""))
        if not condition.startswith("original_raw_text_"):
            continue
        prompt_path = Path(str(row.get("prompt_path", "")))
        raw_text = prompt_path.read_text(encoding="utf-8", errors="replace") if prompt_path.exists() else condition
        canonical_text = str(row.get("canonical_text") or raw_text)
        rows.append(
            SemanticBlockMetadata(
                block_id=str(row.get("id", "")),
                context_field="canonical_semantic_positive",
                canonical_key=str(row.get("canonical_key") or stable_hash(canonical_text)),
                content_hash=stable_hash(raw_text),
                token_count=token_count(raw_text),
                reuse_policy="canonical",
                canonical_text=canonical_text,
                raw_text=raw_text,
                task_id=str(row.get("task_id", "")),
            )
        )
    return rows


def run_sequence(name: str, rows: list[SemanticBlockMetadata], store: SemanticBlockKVStore) -> list[dict[str, Any]]:
    out = []
    for position, meta in enumerate(rows, start=1):
        result = store.lookup_or_store(meta)
        out.append(
            {
                "sequence": name,
                "position": position,
                "block_id": meta.block_id,
                "context_field": meta.context_field,
                "tool_name": meta.tool_name,
                "tool_type": meta.tool_type,
                "is_mutating_tool": meta.is_mutating_tool,
                "reuse_policy": meta.reuse_policy,
                "canonical_key": meta.canonical_key,
                "content_hash": meta.content_hash,
                **result.to_dict(),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["sequence"], row["relation"]), []).append(row)
    summary = []
    for (sequence, relation), items in sorted(groups.items()):
        tokens = sum(int(item["reused_tokens"]) + int(item["prefill_tokens"]) for item in items)
        reused = sum(int(item["reused_tokens"]) for item in items)
        prefill = sum(int(item["prefill_tokens"]) for item in items)
        summary.append(
            {
                "sequence": sequence,
                "relation": relation,
                "rows": len(items),
                "tokens": tokens,
                "reused_tokens": reused,
                "prefill_tokens": prefill,
                "reuse_ratio": round(reused / tokens, 4) if tokens else 0.0,
                "cache_hits": sum(1 for item in items if item["cache_hit"]),
            }
        )
    return summary


def write_report(path: Path, lookup_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Semantic / Canonical Block KV Lookup Report",
        "",
        "## What This Tests",
        "",
        "- block-level KV storage by independent sub-context block metadata,",
        "- exact content lookup by `content_hash`,",
        "- semantic/canonical lookup by `canonical_key`,",
        "- conservative no-reuse behavior for mutating tools.",
        "",
        "## Summary",
        "",
        "| sequence | relation | rows | tokens | reused | prefill | reuse ratio | hits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['sequence']}` | `{row['relation']}` | {row['rows']} | {row['tokens']} | {row['reused_tokens']} | {row['prefill_tokens']} | {row['reuse_ratio']:.1%} | {row['cache_hits']} |"
        )
    canonical_hits = [row for row in lookup_rows if row["relation"] == "canonical_semantic"]
    lines.extend(
        [
            "",
            "## Canonical Semantic Hits",
            "",
            f"- Canonical semantic hits: {len(canonical_hits)}",
            "- These are cases where raw text differs but the canonical key matches a stored block.",
            "",
            "## Boundary",
            "",
            "This is the OpenClaw block-KV policy and storage layer. The SGLang patch hook can log/store the same metadata internally, but reusing actual paged KV tensors inside SGLang still requires scheduler/model-runner integration.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-jsonl", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/semantic_canonical_block_kv_lookup"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = SemanticBlockKVStore()
    lookups: list[dict[str, Any]] = []

    base_rows = metadata_rows(args.metadata_jsonl)
    lookups.extend(run_sequence("tool_aware_store", base_rows, store))
    lookups.extend(run_sequence("tool_aware_exact_replay", base_rows, store))

    canonical_rows = canonical_positive_rows(args.canonical_manifest)
    lookups.extend(run_sequence("canonical_positive_raw_text", canonical_rows, store))

    summary = summarize(lookups)
    write_csv(args.output_dir / "block_kv_lookup_rows.csv", lookups)
    write_csv(args.output_dir / "block_kv_lookup_summary.csv", summary)
    (args.output_dir / "block_kv_store_records.json").write_text(
        json.dumps([record.to_dict() for record in store.records()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "block_kv_events.json").write_text(
        json.dumps(store.events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "semantic_canonical_block_kv_lookup_report.md", lookups, summary)

    canonical_hits = sum(1 for row in lookups if row["relation"] == "canonical_semantic")
    print("Semantic/canonical block KV lookup")
    print("==================================")
    print(f"Lookup rows: {len(lookups)}")
    print(f"Canonical semantic hits: {canonical_hits}")
    print(f"Report: {args.output_dir / 'semantic_canonical_block_kv_lookup_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
