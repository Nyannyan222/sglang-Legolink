#!/usr/bin/env python3
"""Patch installed SGLang to log Radix prefix-cache lookup details.

Run this with the same Python environment that will launch SGLang:

    python scripts/patch_sglang_cache_lookup_logging.py

The patch is intentionally small and guarded by:

    SGLANG_PREFIX_CACHE_DEBUG_LOG=1

When enabled, SGLang logs JSON lines with event="cache_lookup" from the
prefix lookup paths used by schedule_policy and schedule_batch.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys


MARKER = "# OPENCLAW_PREFIX_CACHE_LOOKUP_LOGGING"
SCHEDULE_BATCH_MARKER = "# OPENCLAW_PREFIX_CACHE_LOOKUP_LOGGING_SCHEDULE_BATCH"
BLOCK_METADATA_MARKER = "# OPENCLAW_BLOCK_METADATA_CACHE_LOOKUP_LOGGING"


def find_schedule_policy_path() -> pathlib.Path:
    try:
        spec = importlib.util.find_spec("sglang.srt.managers.schedule_policy")
    except Exception:
        spec = None
    if spec is not None and spec.origin is not None:
        return pathlib.Path(spec.origin)

    for entry in sys.path:
        if not entry:
            entry = "."
        candidate = (
            pathlib.Path(entry)
            / "sglang"
            / "srt"
            / "managers"
            / "schedule_policy.py"
        )
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not locate sglang.srt.managers.schedule_policy")


def find_module_path(module_name: str, relative_path: tuple[str, ...]) -> pathlib.Path:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        spec = None
    if spec is not None and spec.origin is not None:
        return pathlib.Path(spec.origin)

    for entry in sys.path:
        if not entry:
            entry = "."
        candidate = pathlib.Path(entry).joinpath(*relative_path)
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Could not locate {module_name}")


def block_metadata_helper() -> str:
    return f'''

{BLOCK_METADATA_MARKER}
def _openclaw_request_custom_labels(req):
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    if isinstance(custom_params, dict):
        metadata = custom_params.get("openclaw_metadata")
        if isinstance(metadata, dict):
            return metadata

    labels = getattr(req, "custom_labels", None)
    if labels is None:
        return {{}}
    if isinstance(labels, dict):
        return labels
    if hasattr(labels, "model_dump"):
        return labels.model_dump()
    try:
        return dict(labels)
    except Exception:
        return {{}}


def _openclaw_block_metadata(req):
    labels = _openclaw_request_custom_labels(req)
    cache_key = labels.get("openclaw_cache_key")
    extra_key = getattr(req, "extra_key", None)
    return {{
        "openclaw_prefill_unit_id": labels.get("openclaw_prefill_unit_id"),
        "openclaw_cache_key": cache_key,
        "openclaw_context_field": labels.get("openclaw_context_field"),
        "openclaw_extra_key_matches_cache_key": (
            str(extra_key) == str(cache_key) if cache_key is not None else None
        ),
    }}
'''


def add_block_metadata_logging(text: str) -> tuple[str, bool]:
    if BLOCK_METADATA_MARKER in text:
        return text, False

    helper_anchor = """def _openclaw_prefix_cache_node_id(node):
    return None if node is None else hex(id(node))
"""
    if helper_anchor not in text:
        raise RuntimeError("Could not find OpenClaw prefix-cache helper anchor")
    text = text.replace(helper_anchor, helper_anchor + block_metadata_helper(), 1)

    log_anchor = '    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))\n'
    if log_anchor not in text:
        raise RuntimeError("Could not find OpenClaw cache-lookup logger anchor")
    text = text.replace(
        log_anchor,
        '    payload.update(_openclaw_block_metadata(req))\n' + log_anchor,
        1,
    )
    return text, True


def patch_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return add_block_metadata_logging(text)

    if "import json\n" not in text:
        text = text.replace("import logging\n", "import logging\nimport json\n", 1)

    env_marker = '_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")'
    if env_marker not in text:
        raise RuntimeError("Could not find routing-key debug env marker")
    text = text.replace(
        env_marker,
        env_marker
        + '\n_PREFIX_CACHE_LOOKUP_DEBUG_LOG = get_bool_env_var("SGLANG_PREFIX_CACHE_DEBUG_LOG")',
        1,
    )

    helper_anchor = "IGNORE_EOS_RESERVE_TOKENS = 1\n"
    if helper_anchor not in text:
        raise RuntimeError("Could not find helper insertion anchor")
    helper = f'''

{MARKER}
def _openclaw_prefix_cache_node_id(node):
    return None if node is None else hex(id(node))


{BLOCK_METADATA_MARKER}
def _openclaw_request_custom_labels(req):
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    if isinstance(custom_params, dict):
        metadata = custom_params.get("openclaw_metadata")
        if isinstance(metadata, dict):
            return metadata

    labels = getattr(req, "custom_labels", None)
    if labels is None:
        return {{}}
    if isinstance(labels, dict):
        return labels
    if hasattr(labels, "model_dump"):
        return labels.model_dump()
    try:
        return dict(labels)
    except Exception:
        return {{}}


def _openclaw_block_metadata(req):
    labels = _openclaw_request_custom_labels(req)
    cache_key = labels.get("openclaw_cache_key")
    extra_key = getattr(req, "extra_key", None)
    return {{
        "openclaw_prefill_unit_id": labels.get("openclaw_prefill_unit_id"),
        "openclaw_cache_key": cache_key,
        "openclaw_context_field": labels.get("openclaw_context_field"),
        "openclaw_extra_key_matches_cache_key": (
            str(extra_key) == str(cache_key) if cache_key is not None else None
        ),
    }}


def _openclaw_log_prefix_cache_lookup(req, token_ids, match_result):
    if not _PREFIX_CACHE_LOOKUP_DEBUG_LOG:
        return

    input_token_len = len(token_ids)
    matched_prefix_len = len(match_result.device_indices)
    uncached_tokens = max(input_token_len - matched_prefix_len, 0)
    first_mismatch_token_position = (
        matched_prefix_len if matched_prefix_len < input_token_len else None
    )
    payload = {{
        "event": "cache_lookup",
        "rid": getattr(req, "rid", None),
        "input_token_len": input_token_len,
        "matched_prefix_len": matched_prefix_len,
        "matched_node_id": _openclaw_prefix_cache_node_id(
            match_result.last_device_node
        ),
        "matched_host_node_id": _openclaw_prefix_cache_node_id(
            match_result.last_host_node
        ),
        "cached_tokens": matched_prefix_len,
        "uncached_tokens": uncached_tokens,
        "first_mismatch_token_position": first_mismatch_token_position,
        "host_hit_length": match_result.host_hit_length,
        "cache_protected_len": match_result.cache_protected_len,
        "extra_key": repr(getattr(req, "extra_key", None)),
    }}
    payload.update(_openclaw_block_metadata(req))
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
'''
    text = text.replace(helper_anchor, helper_anchor + helper, 1)

    call_anchor = """    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    return match_result
"""
    if call_anchor in text:
        text = text.replace(
            call_anchor,
            """    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    _openclaw_log_prefix_cache_lookup(req, token_ids, match_result)
    return match_result
""",
            1,
        )
        return text, True

    method_anchor = """            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
"""
    if method_anchor not in text:
        raise RuntimeError("Could not find prefix-match assignment anchor")
    text = text.replace(
        method_anchor,
        """            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
            )
            _openclaw_log_prefix_cache_lookup(r, prefix_ids, match_result)

            # NOTE(sang): This logic is for in-batch prefix caching;
""",
        1,
    )
    return text, True


def patch_schedule_batch_text(text: str) -> tuple[str, bool]:
    imports_changed = False
    missing_imports = []
    if "import json\n" not in text:
        missing_imports.append("import json")
    if "import os\n" not in text:
        missing_imports.append("import os")
    if missing_imports:
        import_anchor = "import logging\n"
        if import_anchor not in text:
            raise RuntimeError("Could not find schedule_batch import anchor")
        text = text.replace(
            import_anchor,
            import_anchor + "\n".join(missing_imports) + "\n",
            1,
        )
        imports_changed = True

    if SCHEDULE_BATCH_MARKER in text:
        text, metadata_changed = add_block_metadata_logging(text)
        return text, imports_changed or metadata_changed

    helper_anchor = "logger = logging.getLogger(__name__)\n"
    if helper_anchor not in text:
        raise RuntimeError("Could not find schedule_batch logger anchor")
    helper = f'''

{SCHEDULE_BATCH_MARKER}
_PREFIX_CACHE_LOOKUP_DEBUG_LOG = (
    os.environ.get("SGLANG_PREFIX_CACHE_DEBUG_LOG", "").lower()
    in ("1", "true", "yes", "on")
)


def _openclaw_prefix_cache_node_id(node):
    return None if node is None else hex(id(node))


{BLOCK_METADATA_MARKER}
def _openclaw_request_custom_labels(req):
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    if isinstance(custom_params, dict):
        metadata = custom_params.get("openclaw_metadata")
        if isinstance(metadata, dict):
            return metadata

    labels = getattr(req, "custom_labels", None)
    if labels is None:
        return {{}}
    if isinstance(labels, dict):
        return labels
    if hasattr(labels, "model_dump"):
        return labels.model_dump()
    try:
        return dict(labels)
    except Exception:
        return {{}}


def _openclaw_block_metadata(req):
    labels = _openclaw_request_custom_labels(req)
    cache_key = labels.get("openclaw_cache_key")
    extra_key = getattr(req, "extra_key", None)
    return {{
        "openclaw_prefill_unit_id": labels.get("openclaw_prefill_unit_id"),
        "openclaw_cache_key": cache_key,
        "openclaw_context_field": labels.get("openclaw_context_field"),
        "openclaw_extra_key_matches_cache_key": (
            str(extra_key) == str(cache_key) if cache_key is not None else None
        ),
    }}


def _openclaw_log_prefix_cache_lookup(req, token_ids, match_result):
    if not _PREFIX_CACHE_LOOKUP_DEBUG_LOG:
        return

    input_token_len = len(token_ids)
    matched_prefix_len = len(match_result.device_indices)
    uncached_tokens = max(input_token_len - matched_prefix_len, 0)
    first_mismatch_token_position = (
        matched_prefix_len if matched_prefix_len < input_token_len else None
    )
    payload = {{
        "event": "cache_lookup",
        "rid": getattr(req, "rid", None),
        "input_token_len": input_token_len,
        "matched_prefix_len": matched_prefix_len,
        "matched_node_id": _openclaw_prefix_cache_node_id(
            match_result.last_device_node
        ),
        "matched_host_node_id": _openclaw_prefix_cache_node_id(
            match_result.last_host_node
        ),
        "cached_tokens": matched_prefix_len,
        "uncached_tokens": uncached_tokens,
        "first_mismatch_token_position": first_mismatch_token_position,
        "host_hit_length": match_result.host_hit_length,
        "cache_protected_len": len(match_result.device_indices),
        "extra_key": repr(getattr(req, "extra_key", None)),
    }}
    payload.update(_openclaw_block_metadata(req))
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
'''
    text = text.replace(helper_anchor, helper_anchor + helper, 1)

    old_call_anchor = """            self.cache_protected_len = len(self.prefix_indices)

        if (
"""
    if old_call_anchor in text:
        text = text.replace(
            old_call_anchor,
            """            self.cache_protected_len = len(self.prefix_indices)
            _openclaw_log_prefix_cache_lookup(self, token_ids, match_result)

        if (
""",
            1,
        )
        return text, True

    current_call_anchor = """            else:
                self.cache_protected_len = len(self.prefix_indices)

            if self.is_dllm():
"""
    if current_call_anchor in text:
        text = text.replace(
            current_call_anchor,
            """            else:
                self.cache_protected_len = len(self.prefix_indices)
            _openclaw_token_ids = (
                token_ids_to_match if "token_ids_to_match" in locals() else token_ids
            )
            _openclaw_log_prefix_cache_lookup(self, _openclaw_token_ids, match_result)

            if self.is_dllm():
""",
            1,
        )
        return text, True

    raise RuntimeError("Could not find schedule_batch prefix assignment anchor")


def patch_file(target: pathlib.Path, patcher) -> bool:
    original = target.read_text(encoding="utf-8")
    patched, changed = patcher(original)
    if not changed:
        print(f"Already patched: {target}")
        return False

    compile(patched, str(target), "exec")
    backup = target.with_suffix(target.suffix + ".openclaw-prefix-log.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the patch without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.dry_run:
        policy_target = find_schedule_policy_path()
        policy_patched, _ = patch_text(policy_target.read_text(encoding="utf-8"))
        compile(policy_patched, str(policy_target), "exec")

        batch_target = find_module_path(
            "sglang.srt.managers.schedule_batch",
            ("sglang", "srt", "managers", "schedule_batch.py"),
        )
        batch_patched, _ = patch_schedule_batch_text(
            batch_target.read_text(encoding="utf-8")
        )
        compile(batch_patched, str(batch_target), "exec")
        print(f"Patch validates: {policy_target}")
        print(f"Patch validates: {batch_target}")
        return 0

    patch_file(find_schedule_policy_path(), patch_text)
    patch_file(
        find_module_path(
            "sglang.srt.managers.schedule_batch",
            ("sglang", "srt", "managers", "schedule_batch.py"),
        ),
        patch_schedule_batch_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
