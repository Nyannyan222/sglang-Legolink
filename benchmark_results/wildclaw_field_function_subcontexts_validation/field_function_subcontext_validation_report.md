# Field/Function Sub-context Validation Report

- Source JSONL: `benchmark_results/wildclaw_field_function_subcontexts/wildclaw_field_function_subcontexts.jsonl`
- Rows: 87
- Unique ids: 87
- Unique prefill units: 87
- Unique cache keys: 85
- Internally split rows: 77
- Failed checks: 0

## Field Counts

| context_field | rows |
| --- | ---: |
| `task_metadata` | 3 |
| `tool_definitions` | 3 |
| `user_prompt` | 3 |
| `workspace_evidence` | 78 |

## Interpretation

- Passing this validation means the artifact is field/function-first and SGLang-metadata-ready.
- It does not prove SGLang already enforces block-isolated attention at runtime.
- Runtime no-pairwise attention still requires an SGLang attention-mask / independent-prefill implementation.

## Result

All validation checks passed.
