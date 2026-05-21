# WildClaw SGLang Runtime Summary

- Source CSV: `C:/Users/Administrator/OneDrive/文件/New project/benchmark_results/canonical_text_sglang_replay/runtime_runs/wildclaw_sglang_runtime_20260521_155137/wildclaw_sglang_runtime_results.csv`
- Rows: 20
- Completed: 20
- Errors: 0

## Condition And Replay Summary

| condition | replay | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| block_canonical_text | 1 | 1 | 63 | 45 | 18 | 71.4% | 0.1633 |
| block_canonical_text_repeat | 1 | 1 | 63 | 62 | 1 | 98.4% | 0.1643 |
| canonical_text | 1 | 3 | 66 | 57.7 | 8.3 | 87.4% | 0.1869 |
| canonical_text_repeat | 1 | 3 | 66 | 65 | 1 | 98.5% | 0.1740 |
| exact_canonical_text | 1 | 3 | 108.7 | 47.7 | 61 | 43.9% | 0.1110 |
| exact_canonical_text_repeat | 1 | 3 | 108.7 | 107.7 | 1 | 99.1% | 0.1412 |
| raw_parameterized_left | 1 | 3 | 69 | 68 | 1 | 65.7% | 0.3661 |
| raw_parameterized_right | 1 | 3 | 71.3 | 54.7 | 16.7 | 76.6% | 0.1999 |

## Replay-1 Context Reduction Vs Fixed Size

| condition | tasks | fixed prompt sum | condition prompt sum | weighted reduction | mean task reduction |
| --- | --- | --- | --- | --- | --- |

## Notes

- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.
- Order-permutation manifests may encode the repeat as a separate `real_semantic_order_repeat` row with `REPEAT=1`.
- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.
- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.
