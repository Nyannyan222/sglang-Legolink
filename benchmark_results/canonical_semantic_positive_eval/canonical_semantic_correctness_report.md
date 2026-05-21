# Canonical Semantic-Positive Correctness And Runtime Report

- Runtime CSV: `C:/Users/Administrator/OneDrive/文件/New project/benchmark_results/canonical_semantic_positive_eval/runtime_runs/wildclaw_sglang_runtime_20260521_160250/wildclaw_sglang_runtime_results.csv`
- Rows: 12
- Pair groups: 2

## Baseline Comparison

| condition | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s | answer correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openclaw_canonical_text | 2 | 202.5 | 187.0 | 15.5 | 0.923 | 0.2348 | 2 |
| openclaw_canonical_text_repeat | 2 | 202.5 | 201.5 | 1.0 | 0.995 | 0.2343 | 2 |
| original_raw_text_left | 2 | 203.0 | 0.0 | 203.0 | 0.000 | 0.2839 | 2 |
| original_raw_text_right | 2 | 204.0 | 196.5 | 7.5 | 0.963 | 0.1958 | 2 |
| sglang_exact_repeat | 2 | 203.0 | 202.0 | 1.0 | 0.995 | 0.1996 | 2 |
| sglang_exact_repeat_repeat | 2 | 203.0 | 202.0 | 1.0 | 0.995 | 0.2028 | 2 |

## Correctness Validation

| pair | answer equivalence | evidence preservation | no hallucinated facts | avg judge score |
| --- | --- | --- | --- | --- |
| prompt_internal_pair_0004 | True | True | True | 4.00 |
| prompt_internal_pair_0010 | True | True | True | 4.00 |

## Interpretation

- `original_raw_text_left/right` are different raw texts that share a canonical key.
- `sglang_exact_repeat` is the ordinary SGLang exact-prompt repeat baseline.
- `openclaw_canonical_text` uses the shared canonical text generated from semantic-equivalent templates.
- Correctness is measured as intent equivalence plus evidence preservation and no new concrete facts in the canonical text.
