# Field/Function Semantic Sub-contexts

## Direction

The next version should not start from timeline trace segmentation.

Instead, it should first split WildClawBench / OpenClaw context into semantic
sub-contexts by field and functional role:

- `user_prompt`
- `tool_definitions`
- `tool_history_output`
- `conversation_history`
- `workspace_evidence`
- `task_metadata`

Only when one field is too long should it be split internally. That internal
split is a fallback for length control, not the primary segmentation principle.

## Why This Matters

Timeline trace segmentation answers the question:

> What happened first, second, third?

The target system needs a different unit:

> What context block can be stored, prefetched, cached, and reused
> independently?

For SGLang integration, a sub-context should correspond to an independently
managed context block rather than a chronological slice of an execution trace.

## Sub-context Contract

Each sub-context row should include:

- stable id
- context field
- functional role
- source span
- content hash
- SGLang prefill unit id
- SGLang cache key
- target attention policy

The intended runtime policy is:

```text
Sub-context A: prefill/cache independently
Sub-context B: prefill/cache independently
Sub-context C: prefill/cache independently

No pairwise attention between A, B, and C during independent prefill.
```

The answer stage can select which independent sub-context blocks to read or
compose, but the sub-context prefill/cache stage should not mix their KV states.

## Current Implementation

Script:

```bash
python scripts/prepare_field_function_subcontexts.py
```

Default input:

```text
benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl
```

Outputs:

```text
benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts.jsonl
benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts_summary.csv
benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts_report.md
```

## Runtime Target

The generated metadata is not yet a full SGLang runtime implementation. It is a
bridge format for the next step.

The runtime target is:

1. Store each sub-context as a separate prefill unit.
2. Use `content_hash` as a reusable cache key.
3. Keep one `prefill_unit_id` per isolated block.
4. Add block-isolation attention so sub-contexts do not attend to each other.
5. Reuse only selected sub-context KV blocks at request time.

## Chinese Summary

這一版不先照時間軸切 trace，而是先按欄位和功能切：

```text
user prompt
tool definitions
tool history output
conversation history
workspace evidence
task metadata
```

只有欄位太長時，才在欄位內部再切。每個 sub-context 都會帶上
`prefill_unit_id`、`cache_key`、`attention_policy_target`，目標是之後能在
SGLang 裡獨立 prefill、cache、reuse，並做到 sub-context 之間 no pairwise
attention。
