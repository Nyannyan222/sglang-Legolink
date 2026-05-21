# True SGLang Field Metadata Runtime Report

## Runtime

- Endpoint: `http://127.0.0.1:30000/v1`
- Container: `openclaw-sglang-local`
- Image: `lmsysorg/sglang:latest-cu130-runtime`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Manifest: `benchmark_results/wildclaw_field_sglang_metadata/field_sglang_runtime_manifest.jsonl`
- Runtime CSV: `benchmark_results/wildclaw_field_sglang_metadata/runtime_runs_true_sglang/wildclaw_sglang_runtime_20260522_033857/wildclaw_sglang_runtime_results.csv`
- Field-wise report: `benchmark_results/wildclaw_field_sglang_metadata/true_sglang_fieldwise_report/fieldwise_cache_report.md`

## Result

- Rows: 66
- Errors: 0
- Rows with metadata sent by replay runner: 66
- Rows with non-empty SGLang `cached_tokens`: 65
- Context fields covered: `user_prompt`, `tool_definitions`, `conversation_history`, `tool_history_output`, `workspace_evidence`, `task_metadata`

## Field-wise Cache Summary

| field | rows | avg tokens | avg cached | avg prefill | cached ratio | avg latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `user_prompt` | 6 | 331.67 | 198.4 | 127.2 | 49.9% | 0.2674 |
| `tool_definitions` | 6 | 140.0 | 88.5 | 51.5 | 63.2% | 0.0899 |
| `conversation_history` | 12 | 307.5 | 153.25 | 154.25 | 49.8% | 0.0928 |
| `tool_history_output` | 6 | 576.0 | 287.5 | 288.5 | 49.9% | 0.1266 |
| `workspace_evidence` | 30 | 1391.87 | 695.43 | 696.43 | 50.0% | 0.0983 |
| `task_metadata` | 6 | 241.67 | 120.33 | 121.33 | 49.8% | 0.0896 |

## Log Evidence

SGLang logs show the request `extra_key` carrying field-level canonical keys, for example:

```text
extra_key: field:tool_history_output:external_observations_and_tool_outputs:fb3b6a33f4759bcdb7332cee
extra_key: field:conversation_history:prior_user_assistant_state:db5270d48d9c239b45934475
```

They also show prefix-cache behavior during the same run:

```text
Prefill batch, #new-token: 380, #cached-token: 0
Prefill batch, #new-token: 1, #cached-token: 379
Prefill batch, #new-token: 539, #cached-token: 0
Prefill batch, #new-token: 1, #cached-token: 538
```

## Interpretation

This is a real SGLang Docker run, not the mock endpoint. It confirms the field metadata manifest can be replayed through SGLang successfully and that SGLang reports cached-token behavior for repeated field sub-context prompts.

This still uses stock SGLang prefix-cache mechanics. It does not yet prove runtime-level semantic KV reuse or no-pairwise-attention between sub-contexts inside SGLang.
