# SGLang Field Metadata Manifest

- Input JSONL: `benchmark_results/wildclaw_tool_aware_trace_segmentation/wildclaw_tool_aware_field_subcontexts.jsonl`
- Metadata rows: 39
- Runtime manifest rows: 39

## Field Counts

| context_field | rows |
| --- | ---: |
| `conversation_history` | 6 |
| `task_metadata` | 3 |
| `tool_definitions` | 3 |
| `tool_history_output` | 9 |
| `user_prompt` | 3 |
| `workspace_evidence` | 15 |

## Tool Counts

| tool_name | tool_type | reuse_policy | rows |
| --- | --- | --- | ---: |
| `read_file` | `filesystem_read` | `exact` | 3 |
| `web_search` | `retrieval` | `exact` | 3 |
| `write_file` | `filesystem_write` | `no_reuse` | 3 |

## Metadata Contract

Each manifest row carries `openclaw_metadata` with `sub_context_id`, `context_field`, `raw_text`, `canonical_text`, `canonical_key`, `attention_policy`, `reuse_policy`, and optional tool identity fields.
The runtime replay runner forwards this metadata into the SGLang OpenAI-compatible request body.
