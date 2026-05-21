# Field/Function Sub-context Validation Report

- Source JSONL: `benchmark_results/wildclaw_tool_history_trace_segmentation/wildclaw_llm_trace_field_subcontexts.jsonl`
- Rows: 33
- Unique ids: 33
- Unique prefill units: 33
- Unique cache keys: 31
- Internally split rows: 14
- Failed checks: 0

## Field Counts

| context_field | rows |
| --- | ---: |
| `conversation_history` | 6 |
| `task_metadata` | 3 |
| `tool_definitions` | 3 |
| `tool_history_output` | 3 |
| `user_prompt` | 3 |
| `workspace_evidence` | 15 |

## Interpretation

- Passing this validation means the artifact is field/function-first and SGLang-metadata-ready.
- It does not prove SGLang already enforces block-isolated attention at runtime.
- Runtime no-pairwise attention still requires an SGLang attention-mask / independent-prefill implementation.

## Result

All validation checks passed.
