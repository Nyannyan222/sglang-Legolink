# LLM Trace Field Segmentation

## Purpose

This step converts a complete WildClaw/OpenClaw agent trace or task context into
field/function sub-contexts before any timeline-based splitting.

The target fields are:

- `user_prompt`
- `tool_definitions`
- `tool_history_output`
- `conversation_history`
- `workspace_evidence`
- `task_metadata`

This matches the current semantic-independence direction: sub-contexts should
represent independently stored and managed context blocks, not simply adjacent
time slices from a trace.

## Script

```bash
python scripts/llm_trace_field_segmenter.py
```

Default input:

```text
benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl
```

Default output:

```text
benchmark_results/wildclaw_llm_trace_field_segmentation/
```

## LLM Mode

Use OpenAI/OpenRouter-compatible chat completions:

```bash
python scripts/llm_trace_field_segmenter.py \
  --mode llm \
  --base-url https://openrouter.ai/api/v1 \
  --api-key $OPENROUTER_API_KEY \
  --model openai/gpt-5.4
```

The script renders the complete task context as marked `TRACE_BLOCK`s, then asks
the LLM to classify content by field/function:

```text
TRACE_BLOCK task.prompt -> user_prompt
TRACE_BLOCK task.skills -> tool_definitions
TRACE_BLOCK source_files[...] -> workspace_evidence
TRACE_BLOCK tool_outputs/messages -> tool_history_output or conversation_history
```

The LLM must return strict JSON. The output rows then carry:

- `context_field`
- `functional_role`
- `source_spans`
- `content_hash`
- `sglang_storage_plan.prefill_unit_id`
- `sglang_storage_plan.cache_key`
- `attention_policy_target = no_pairwise_attention_between_subcontexts`

## Offline Fallback

When no API key is available, `--mode auto` falls back to a deterministic field
classifier based on the marked trace block metadata. This is not the final LLM
result, but it lets the pipeline and validation run locally.

Offline local smoke result:

```text
Rows: 27
Used LLM: False
Validation failed checks: 0
```

The local SGLang `Qwen/Qwen2.5-0.5B-Instruct` endpoint was reachable, but it did
not reliably return the required strict JSON for this extraction task. Use a
stronger OpenAI/OpenRouter model for the formal LLM field segmentation run.

## OpenAI LLM Run

The formal OpenAI-backed run completed on 2026-05-21 using `gpt-4.1-mini`:

```text
Tasks: 3
Rows: 12
Used LLM: True
Validation failed checks: 0
```

Outputs:

```text
benchmark_results/wildclaw_llm_trace_field_segmentation_openai/
benchmark_results/wildclaw_llm_trace_field_segmentation_openai_validation/
```

The output is more selective than the offline fallback: it groups or omits
non-evidence blocks instead of mechanically exporting every workspace file.
This is expected for an LLM field classifier, but the selected rows should still
be manually reviewed before being treated as the final semantic-independent
context set.

## Why This Helps

This gives the project a cleaner first-stage segmentation:

```text
complete trace
  -> LLM field/function segmentation
  -> independent sub-context rows
  -> optional internal splitting only for long fields
  -> SGLang block prefill/cache metadata
```

That is closer to the final goal than timeline trace chunking, because each
field block can later be independently prefetched, cached, reused, and isolated
with no pairwise attention to unrelated sub-context blocks.
