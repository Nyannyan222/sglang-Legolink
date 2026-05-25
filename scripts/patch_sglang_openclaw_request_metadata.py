#!/usr/bin/env python3
"""Patch SGLang OpenAI request parsing to preserve OpenClaw block metadata.

Stock SGLang keeps `extra_key` from the OpenAI request body, but custom labels
are normally extracted from a metrics header. This patch allows block-prefill
payloads to carry OpenClaw metadata through the request body:

- `custom_labels.openclaw_*`
- `openclaw_sub_context`

The metadata is attached to `sampling_params.custom_params.openclaw_metadata`
so the scheduler cache-lookup patch can observe it without polluting SGLang
Prometheus metric labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from collections.abc import Callable


MARKER = "# OPENCLAW_REQUEST_METADATA_INGESTION"


def find_module_path(
    module_name: str,
    relative_paths: tuple[tuple[str, ...], ...],
    source_root: pathlib.Path | None,
) -> pathlib.Path:
    if source_root is not None:
        for relative_path in relative_paths:
            candidate = source_root.joinpath(*relative_path)
            if candidate.exists():
                return candidate

    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        spec = None
    if spec is not None and spec.origin is not None:
        return pathlib.Path(spec.origin)

    for entry in sys.path:
        base = pathlib.Path(entry or ".")
        for relative_path in relative_paths:
            candidate = base.joinpath(*relative_path)
            if candidate.exists():
                return candidate
    raise RuntimeError(f"Could not locate {module_name}")


def openai_path(name: str, source_root: pathlib.Path | None) -> pathlib.Path:
    return find_module_path(
        f"sglang.srt.entrypoints.openai.{name}",
        (
            ("sglang", "srt", "entrypoints", "openai", f"{name}.py"),
            ("python", "sglang", "srt", "entrypoints", "openai", f"{name}.py"),
        ),
        source_root,
    )


def patch_protocol_text(text: str) -> tuple[str, bool]:
    changed = False
    completion_anchor = "    # For custom metric labels\n    custom_labels: Optional[Dict[str, str]] = None\n"
    completion_insert = (
        completion_anchor
        + f"\n    {MARKER}\n"
        + "    # OpenClaw semantic sub-context metadata for block-level KV experiments.\n"
        + "    openclaw_sub_context: Optional[Dict[str, Any]] = None\n"
    )
    completion_start = text.find("class CompletionRequest(BaseModel):")
    completion_end = text.find("\n\nclass SglExt", completion_start)
    if completion_start < 0 or completion_end < 0:
        raise RuntimeError("Could not locate CompletionRequest block")
    completion_block = text[completion_start:completion_end]
    if "openclaw_sub_context" not in completion_block:
        if completion_anchor not in completion_block:
            raise RuntimeError("Could not find CompletionRequest custom_labels anchor")
        text = (
            text[:completion_start]
            + completion_block.replace(completion_anchor, completion_insert, 1)
            + text[completion_end:]
        )
        changed = True

    chat_anchor = "    # Priority for the request\n    priority: Optional[int] = None\n"
    chat_insert = (
        chat_anchor
        + "\n    # For custom metric labels\n"
        + "    custom_labels: Optional[Dict[str, str]] = None\n"
        + f"\n    {MARKER}\n"
        + "    # OpenClaw semantic sub-context metadata for block-level KV experiments.\n"
        + "    openclaw_sub_context: Optional[Dict[str, Any]] = None\n"
    )
    chat_start = text.find("class ChatCompletionRequest(BaseModel):")
    chat_end = text.find("    # OpenAI/SGLang default sampling parameters", chat_start)
    if chat_start < 0 or chat_end < 0:
        raise RuntimeError("Could not locate ChatCompletionRequest block")
    chat_block = text[chat_start:chat_end]
    if "openclaw_sub_context" not in chat_block:
        if chat_anchor not in chat_block:
            raise RuntimeError("Could not find ChatCompletionRequest priority anchor")
        text = (
            text[:chat_start]
            + chat_block.replace(chat_anchor, chat_insert, 1)
            + text[chat_end:]
        )
        changed = True

    return text, changed


def openclaw_helper() -> str:
    return f'''

    {MARKER}
    def extract_openclaw_custom_labels(self, request, raw_request):
        custom_labels = self.extract_custom_labels(raw_request) or {{}}

        body_labels = getattr(request, "custom_labels", None)
        if isinstance(body_labels, dict):
            for key, value in body_labels.items():
                if value is not None and str(key).startswith("openclaw_"):
                    custom_labels[str(key)] = str(value)

        block = getattr(request, "openclaw_sub_context", None)
        if isinstance(block, dict):
            block_labels = {{
                "openclaw_prefill_unit_id": block.get("prefill_unit_id"),
                "openclaw_cache_key": block.get("cache_key") or block.get("canonical_key"),
                "openclaw_canonical_key": block.get("canonical_key") or block.get("cache_key"),
                "openclaw_content_hash": block.get("content_hash"),
                "openclaw_reuse_policy": block.get("reuse_policy"),
                "openclaw_source_sub_context_id": block.get("source_sub_context_id") or block.get("sub_context_id"),
                "openclaw_context_field": block.get("context_field"),
                "openclaw_functional_role": block.get("functional_role"),
                "openclaw_attention_policy": block.get("attention_policy"),
                "openclaw_cache_scope": block.get("cache_scope"),
                "openclaw_kv_reuse_granularity": block.get("kv_reuse_granularity"),
                "openclaw_task_id": block.get("task_id"),
                "openclaw_tool_name": block.get("tool_name"),
                "openclaw_tool_type": block.get("tool_type"),
                "openclaw_tool_call_id": block.get("tool_call_id"),
                "openclaw_is_mutating_tool": block.get("is_mutating_tool"),
            }}
            for key, value in block_labels.items():
                if value is not None:
                    custom_labels[key] = str(value)

        cache_key = custom_labels.get("openclaw_cache_key")
        extra_key = self._compute_extra_key(request)
        if cache_key is not None and extra_key is not None:
            custom_labels["openclaw_extra_key_matches_cache_key"] = str(
                str(extra_key) == str(cache_key)
            ).lower()

        return custom_labels or None
'''


def patch_serving_base_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    anchor = "    def extract_routing_key(self, raw_request):\n"
    if anchor not in text:
        raise RuntimeError("Could not find extract_routing_key anchor")
    text = text.replace(anchor, openclaw_helper() + "\n" + anchor, 1)
    return text, True


def replace_once(text: str, old: str, new: str, description: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"Could not find {description}")
    return text.replace(old, new, 1), True


def attach_openclaw_metadata_to_sampling_params(
    text: str, sampling_anchor: str, description: str
) -> tuple[str, bool]:
    metadata_block = (
        sampling_anchor
        + '        openclaw_metadata = self.extract_openclaw_custom_labels(request, raw_request)\n'
        + '        if openclaw_metadata:\n'
        + '            if not isinstance(sampling_params.get("custom_params"), dict):\n'
        + '                sampling_params["custom_params"] = {}\n'
        + '            sampling_params["custom_params"]["openclaw_metadata"] = openclaw_metadata\n'
    )
    if 'sampling_params["custom_params"]["openclaw_metadata"] = openclaw_metadata' in text:
        return text, False
    if sampling_anchor not in text:
        raise RuntimeError(f"Could not find {description} sampling_params anchor")
    return text.replace(sampling_anchor, metadata_block, 1), True


def restore_metrics_custom_labels(text: str, description: str) -> tuple[str, bool]:
    changed = False
    old = "        custom_labels = self.extract_openclaw_custom_labels(request, raw_request)\n"
    new = "        custom_labels = self.extract_custom_labels(raw_request)\n"
    if old in text:
        text = text.replace(old, new, 1)
        changed = True
    elif new not in text:
        raise RuntimeError(f"Could not find {description} custom_labels assignment")
    return text, changed


def patch_serving_chat_text(text: str) -> tuple[str, bool]:
    changed = False
    sampling_anchor = """        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )
"""
    text, did_change = attach_openclaw_metadata_to_sampling_params(
        text, sampling_anchor, "serving_chat"
    )
    changed = changed or did_change
    text, did_change = restore_metrics_custom_labels(text, "serving_chat")
    changed = changed or did_change
    return text, changed


def patch_serving_completions_text(text: str) -> tuple[str, bool]:
    changed = False
    sampling_anchor = "        sampling_params = self._build_sampling_params(request)\n"
    text, did_change = attach_openclaw_metadata_to_sampling_params(
        text, sampling_anchor, "serving_completions"
    )
    changed = changed or did_change
    text, did_change = restore_metrics_custom_labels(text, "serving_completions")
    changed = changed or did_change
    return text, changed


def patch_file(target: pathlib.Path, patcher: Callable[[str], tuple[str, bool]], dry_run: bool) -> bool:
    original = target.read_text(encoding="utf-8")
    patched, changed = patcher(original)
    if not changed:
        print(f"Already patched: {target}")
        return False

    compile(patched, str(target), "exec")
    if dry_run:
        print(f"Patch validates: {target}")
        return True

    backup = target.with_suffix(target.suffix + ".openclaw-request-metadata.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-root", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    source_root = args.source_root.resolve() if args.source_root else None
    targets = [
        (openai_path("protocol", source_root), patch_protocol_text),
        (openai_path("serving_base", source_root), patch_serving_base_text),
        (openai_path("serving_chat", source_root), patch_serving_chat_text),
        (openai_path("serving_completions", source_root), patch_serving_completions_text),
    ]
    for target, patcher in targets:
        patch_file(target, patcher, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
