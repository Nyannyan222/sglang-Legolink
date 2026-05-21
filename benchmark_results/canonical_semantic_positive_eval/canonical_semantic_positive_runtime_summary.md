# WildClaw SGLang Runtime Summary

- Source CSV: `C:/Users/Administrator/OneDrive/文件/New project/benchmark_results/canonical_semantic_positive_eval/runtime_runs/wildclaw_sglang_runtime_20260521_160250/wildclaw_sglang_runtime_results.csv`
- Rows: 12
- Completed: 12
- Errors: 0

## Condition And Replay Summary

| condition | replay | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openclaw_canonical_text | 1 | 2 | 202.5 | 187 | 15.5 | 92.3% | 0.2348 |
| openclaw_canonical_text_repeat | 1 | 2 | 202.5 | 201.5 | 1 | 99.5% | 0.2343 |
| original_raw_text_left | 1 | 2 | 203 | 0 | 203 | 0.0% | 0.2839 |
| original_raw_text_right | 1 | 2 | 204 | 196.5 | 7.5 | 96.3% | 0.1958 |
| sglang_exact_repeat | 1 | 2 | 203 | 202 | 1 | 99.5% | 0.1996 |
| sglang_exact_repeat_repeat | 1 | 2 | 203 | 202 | 1 | 99.5% | 0.2028 |

## Replay-1 Context Reduction Vs Fixed Size

| condition | tasks | fixed prompt sum | condition prompt sum | weighted reduction | mean task reduction |
| --- | --- | --- | --- | --- | --- |

## Notes

- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.
- Order-permutation manifests may encode the repeat as a separate `real_semantic_order_repeat` row with `REPEAT=1`.
- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.
- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.
