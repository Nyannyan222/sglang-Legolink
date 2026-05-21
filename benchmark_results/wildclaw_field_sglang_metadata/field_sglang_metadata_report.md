# SGLang Field Metadata Manifest

- Input JSONL: `benchmark_results/wildclaw_tool_history_trace_segmentation/wildclaw_llm_trace_field_subcontexts.jsonl`
- Metadata rows: 33
- Runtime manifest rows: 33

## Field Counts

| context_field | rows |
| --- | ---: |
| `conversation_history` | 6 |
| `task_metadata` | 3 |
| `tool_definitions` | 3 |
| `tool_history_output` | 3 |
| `user_prompt` | 3 |
| `workspace_evidence` | 15 |

## Metadata Contract

Each manifest row carries `openclaw_metadata` with `sub_context_id`, `context_field`, `raw_text`, `canonical_text`, `canonical_key`, `attention_policy`, and `reuse_policy`.
The runtime replay runner forwards this metadata into the SGLang OpenAI-compatible request body.
