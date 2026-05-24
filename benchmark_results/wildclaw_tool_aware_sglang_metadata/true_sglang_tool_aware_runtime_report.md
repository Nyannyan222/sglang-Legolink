# Tool-aware SGLang Runtime Report

## Runtime

- Endpoint: `http://127.0.0.1:30000/v1`
- Container: `openclaw-sglang-local`
- Image: `lmsysorg/sglang:latest-cu130-runtime`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Manifest: `benchmark_results/wildclaw_tool_aware_sglang_metadata/field_sglang_runtime_manifest.jsonl`
- Runtime CSV: `benchmark_results/wildclaw_tool_aware_sglang_metadata/runtime_runs_true_sglang/wildclaw_sglang_runtime_20260525_011935/wildclaw_sglang_runtime_results.csv`
- Tool-wise report: `benchmark_results/wildclaw_tool_aware_sglang_metadata/true_sglang_toolwise_report/toolwise_cache_report.md`

## Result

- Tool-aware metadata rows: 39
- Tool-history metadata rows: 9
- Runtime replay rows: 78
- Errors: 0
- Rows with metadata sent by replay runner: 78
- Runtime tool-history rows: 18
- Tool identities covered: `read_file`, `web_search`, `write_file`

## Tool Policy

| tool_name | tool_type | is_mutating_tool | reuse policy |
| --- | --- | --- | --- |
| `read_file` | `filesystem_read` | `False` | candidate reuse after content or same-answer judge |
| `web_search` | `retrieval` | `False` | candidate reuse after content or same-answer judge |
| `write_file` | `filesystem_write` | `True` | no reuse |

## Field Summary

| field | rows | avg tokens | avg cached | avg prefill | cached ratio | avg latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `conversation_history` | 12 | 307.5 | 153.25 | 154.25 | 49.8% | 0.0895 |
| `task_metadata` | 6 | 241.67 | 120.33 | 121.33 | 49.8% | 0.0901 |
| `tool_definitions` | 6 | 140.0 | 88.5 | 51.5 | 63.2% | 0.0961 |
| `tool_history_output` | 18 | 293.22 | 146.11 | 147.11 | 49.8% | 0.099 |
| `user_prompt` | 6 | 331.67 | 198.4 | 127.2 | 49.9% | 1.4746 |
| `workspace_evidence` | 30 | 1391.87 | 695.43 | 696.43 | 50.0% | 0.099 |

## Tool-history Output Summary

| tool_name | tool_type | is_mutating_tool | rows | avg tokens | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `read_file` | `filesystem_read` | `False` | 6 | 404.0 | 201.5 | 202.5 | 49.9% | 0.1249 |
| `web_search` | `retrieval` | `False` | 6 | 258.33 | 128.67 | 129.67 | 49.8% | 0.085 |
| `write_file` | `filesystem_write` | `True` | 6 | 217.33 | 108.17 | 109.17 | 49.8% | 0.0872 |

## Interpretation

This run proves we can distinguish tools inside `tool_history_output`, attach tool identity to SGLang request metadata, and report cache behavior by tool type. Read-only tools are marked as reuse candidates after semantic/content judging, while mutating tools are marked no-reuse.

This still uses stock SGLang prefix-cache behavior. Runtime-level semantic KV reuse and no-pairwise-attention between sub-contexts remain future SGLang runtime work.
