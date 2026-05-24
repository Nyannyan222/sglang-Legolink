# Tool-aware Field Segmentation

- Input segmentation JSONL: `benchmark_results/wildclaw_tool_history_trace_segmentation/wildclaw_llm_trace_field_subcontexts.jsonl`
- Input trace JSONL: `benchmark_results/wildclaw_tool_history_trace_fixture/wildclaw_tool_history_trace_contexts.jsonl`
- Output rows: 39
- Tool-history output rows: 9

## Tool Counts

| tool_name | tool_type | reuse policy hint | rows |
| --- | --- | --- | ---: |
| `read_file` | `filesystem_read` | `candidate_reuse_after_content_or_same_answer_judge` | 3 |
| `web_search` | `retrieval` | `candidate_reuse_after_content_or_same_answer_judge` | 3 |
| `write_file` | `filesystem_write` | `no_reuse_mutating_tool` | 3 |

## Interpretation

`tool_history_output` is now split by individual tool identity instead of one aggregate field block.
Read-only retrieval/file-read tools can become semantic reuse candidates after content or same-answer judging; mutating tools are marked `no_reuse_mutating_tool`.
