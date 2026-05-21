# Field/Function Sub-context Validation Report

- Source JSONL: `benchmark_results/wildclaw_llm_trace_field_segmentation_openai/wildclaw_llm_trace_field_subcontexts.jsonl`
- Rows: 12
- Unique ids: 12
- Unique prefill units: 12
- Unique cache keys: 11
- Internally split rows: 0
- Failed checks: 0

## Field Counts

| context_field | rows |
| --- | ---: |
| `conversation_history` | 3 |
| `task_metadata` | 3 |
| `tool_definitions` | 2 |
| `user_prompt` | 3 |
| `workspace_evidence` | 1 |

## Interpretation

- Passing this validation means the artifact is field/function-first and SGLang-metadata-ready.
- It does not prove SGLang already enforces block-isolated attention at runtime.
- Runtime no-pairwise attention still requires an SGLang attention-mask / independent-prefill implementation.

## Result

All validation checks passed.
