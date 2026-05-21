# Field-wise Cache Report

- Source CSV: `C:/Users/Administrator/Desktop/New project/benchmark_results/wildclaw_field_sglang_metadata/runtime_runs_true_sglang/wildclaw_sglang_runtime_20260522_033857/wildclaw_sglang_runtime_results.csv`

| field | rows | avg tokens | avg cached | avg prefill | cached ratio | avg latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `user_prompt` | 6 | 331.67 | 198.4 | 127.2 | 49.9% | 0.2674 |
| `tool_definitions` | 6 | 140.0 | 88.5 | 51.5 | 63.2% | 0.0899 |
| `conversation_history` | 12 | 307.5 | 153.25 | 154.25 | 49.8% | 0.0928 |
| `tool_history_output` | 6 | 576.0 | 287.5 | 288.5 | 49.9% | 0.1266 |
| `workspace_evidence` | 30 | 1391.87 | 695.43 | 696.43 | 50.0% | 0.0983 |
| `task_metadata` | 6 | 241.67 | 120.33 | 121.33 | 49.8% | 0.0896 |
