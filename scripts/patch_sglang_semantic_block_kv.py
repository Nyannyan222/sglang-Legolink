#!/usr/bin/env python3
"""Patch SGLang cache lookup logging with OpenClaw semantic block KV index.

This is an experimental internal metadata store for OpenClaw block-KV work. It
adds a scheduler-side block cache index keyed by exact content/canonical cache
key from `openclaw_sub_context` metadata and reports exact/canonical hits in
cache lookup logs.

Boundary: this patch indexes and reports block-level semantic KV handles. It does
not yet splice/reuse paged KV tensors in the model runner; that requires a deeper
scheduler/model-runner integration.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys


MARKER = "# OPENCLAW_SEMANTIC_BLOCK_KV_INDEX"


def find_schedule_policy_path(source_root: pathlib.Path | None) -> pathlib.Path:
    if source_root is not None:
        for rel in (
            ("sglang", "srt", "managers", "schedule_policy.py"),
            ("python", "sglang", "srt", "managers", "schedule_policy.py"),
        ):
            candidate = source_root.joinpath(*rel)
            if candidate.exists():
                return candidate
    spec = importlib.util.find_spec("sglang.srt.managers.schedule_policy")
    if spec is not None and spec.origin is not None:
        return pathlib.Path(spec.origin)
    raise RuntimeError("Could not locate sglang.srt.managers.schedule_policy")


def helper_text() -> str:
    return f'''

{MARKER}
class _OpenClawSemanticBlockKVIndex:
    def __init__(self):
        self.by_content_hash = {{}}
        self.by_canonical_key = {{}}

    def lookup_or_store(self, metadata, token_count, kv_handle):
        content_hash = metadata.get("openclaw_content_hash")
        canonical_key = metadata.get("openclaw_cache_key") or metadata.get("openclaw_canonical_key")
        reuse_policy = metadata.get("openclaw_reuse_policy")
        is_mutating = str(metadata.get("openclaw_is_mutating_tool", "")).lower() == "true"
        block_id = metadata.get("openclaw_prefill_unit_id") or metadata.get("openclaw_source_sub_context_id")

        if is_mutating or reuse_policy == "no_reuse":
            return {{
                "openclaw_block_cache_hit": False,
                "openclaw_block_lookup_relation": "policy_no_reuse",
                "openclaw_block_matched_id": None,
                "openclaw_block_kv_handle": None,
            }}

        if content_hash is not None and content_hash in self.by_content_hash:
            record = self.by_content_hash[content_hash]
            return {{
                "openclaw_block_cache_hit": True,
                "openclaw_block_lookup_relation": "exact_content",
                "openclaw_block_matched_id": record.get("block_id"),
                "openclaw_block_kv_handle": record.get("kv_handle"),
            }}

        if canonical_key is not None and canonical_key in self.by_canonical_key:
            record = self.by_canonical_key[canonical_key]
            return {{
                "openclaw_block_cache_hit": True,
                "openclaw_block_lookup_relation": "canonical_semantic",
                "openclaw_block_matched_id": record.get("block_id"),
                "openclaw_block_kv_handle": record.get("kv_handle"),
            }}

        record = {{
            "block_id": block_id,
            "content_hash": content_hash,
            "canonical_key": canonical_key,
            "token_count": token_count,
            "kv_handle": kv_handle,
        }}
        if content_hash is not None:
            self.by_content_hash[content_hash] = record
        if canonical_key is not None:
            self.by_canonical_key.setdefault(canonical_key, record)
        return {{
            "openclaw_block_cache_hit": False,
            "openclaw_block_lookup_relation": "miss_store",
            "openclaw_block_matched_id": None,
            "openclaw_block_kv_handle": kv_handle,
        }}


_OPENCLAW_SEMANTIC_BLOCK_KV_INDEX = _OpenClawSemanticBlockKVIndex()


def _openclaw_semantic_block_kv_lookup(req, token_ids, match_result):
    metadata = _openclaw_block_metadata(req)
    cache_key = metadata.get("openclaw_cache_key") or metadata.get("openclaw_canonical_key")
    if not cache_key:
        return {{}}
    kv_handle = _openclaw_prefix_cache_node_id(getattr(match_result, "last_device_node", None))
    payload = _OPENCLAW_SEMANTIC_BLOCK_KV_INDEX.lookup_or_store(
        metadata, len(token_ids), kv_handle
    )
    payload["openclaw_block_lookup_key"] = cache_key
    return payload
'''


def patch_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    if "def _openclaw_block_metadata(req):" not in text or "def _openclaw_log_prefix_cache_lookup" not in text:
        raise RuntimeError(
            "OpenClaw prefix cache logging patch is required first. Run patch_sglang_cache_lookup_logging.py."
        )

    insert_anchor = "\ndef _openclaw_log_prefix_cache_lookup(req, token_ids, match_result):\n"
    if insert_anchor not in text:
        raise RuntimeError("Could not locate _openclaw_log_prefix_cache_lookup anchor")
    text = text.replace(insert_anchor, helper_text() + insert_anchor, 1)

    log_anchor = "    payload.update(_openclaw_block_metadata(req))\n    logger.info(json.dumps(payload, ensure_ascii=False, separators=(\",\", \":\")))\n"
    if log_anchor not in text:
        raise RuntimeError("Could not locate cache lookup payload log anchor")
    text = text.replace(
        log_anchor,
        "    payload.update(_openclaw_block_metadata(req))\n"
        "    payload.update(_openclaw_semantic_block_kv_lookup(req, token_ids, match_result))\n"
        "    logger.info(json.dumps(payload, ensure_ascii=False, separators=(\",\", \":\")))\n",
        1,
    )
    return text, True


def patch_file(path: pathlib.Path, dry_run: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    patched, changed = patch_text(original)
    if not changed:
        print(f"Already patched: {path}")
        return False
    compile(patched, str(path), "exec")
    if dry_run:
        print(f"Patch validates: {path}")
        return True
    backup = path.with_suffix(path.suffix + ".openclaw-semantic-block-kv.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Backup:  {backup}")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    path = find_schedule_policy_path(args.source_root)
    patch_file(path, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
