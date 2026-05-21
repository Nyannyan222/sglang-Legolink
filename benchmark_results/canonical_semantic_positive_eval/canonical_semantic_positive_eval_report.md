# Canonical Semantic-Positive Evaluation

- Manifest: `benchmark_results/canonical_semantic_positive_eval/canonical_semantic_positive_runtime_manifest.jsonl`
- Runtime run: `benchmark_results/canonical_semantic_positive_eval/runtime_runs/wildclaw_sglang_runtime_20260521_160250/`
- Rows: 12
- Errors: 0
- Semantic positive pairs: 2

## What This Adds

This experiment moves beyond exact canonical repeats. The selected pairs have
different raw text, the same semantic template, and the same `canonical_key`:

1. `Save the results into /tmp_workspace/results` vs `Save the results into /tmp_workspace/results/results.md`
   canonicalizes to `save the results into <CODE>.`
2. `searches must not exceed 5` vs `searches must not exceed 4`
   canonicalizes to `<NUM>. the total number of searches must not exceed <NUM>.`

The goal is to test:

different raw text -> same canonical_text -> SGLang cache reuse on canonical text -> answer utility preserved.

## Runtime Comparison

| method | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s | answer correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original SGLang raw text, left side | 2 | 203.0 | 0.0 | 203.0 | 0.0% | 0.2839 | 2/2 |
| Original SGLang raw text, semantic counterpart | 2 | 204.0 | 196.5 | 7.5 | 96.3% | 0.1958 | 2/2 |
| SGLang exact repeat | 2 | 203.0 | 202.0 | 1.0 | 99.5% | 0.1996 | 2/2 |
| SGLang exact repeat, repeat row | 2 | 203.0 | 202.0 | 1.0 | 99.5% | 0.2028 | 2/2 |
| OpenClaw canonical text | 2 | 202.5 | 187.0 | 15.5 | 92.3% | 0.2348 | 2/2 |
| OpenClaw canonical text, repeat row | 2 | 202.5 | 201.5 | 1.0 | 99.5% | 0.2343 | 2/2 |

## Correctness Validation

| pair | answer equivalence | evidence preservation | no hallucinated facts | judge score |
| --- | --- | --- | --- | ---: |
| `prompt_internal_pair_0004` | pass | pass | pass | 4.00 / 4 |
| `prompt_internal_pair_0010` | pass | pass | pass | 4.00 / 4 |

Checks:

- `answer_equivalence`: raw-left, raw-right, exact-repeat, and canonical prompts all return the same intended answer label.
- `evidence_preservation`: canonical text keeps the reusable intent and replaces task-specific values with placeholders.
- `no_hallucinated_facts`: canonical text does not introduce concrete paths, numbers, or facts absent from the raw evidence.
- `judge_score`: rule-based score over the three checks.

## Interpretation

The experiment shows that canonicalization can turn different raw texts into the
same cacheable canonical prompt while preserving answer utility for the tested
semantic template level. The `openclaw_canonical_text_repeat` condition reaches
99.5% cached ratio and 1 average prefill token, matching the exact-repeat
baseline.

This still does not implement runtime-level semantic KV reuse inside SGLang.
It demonstrates a valid application-level path:

semantic-equivalent sub-contexts -> canonical_text/canonical_key -> stable
prefix-cacheable prompt -> correctness preserved on the tested intent task.

The high cache ratio on `original_raw_text_right` should be interpreted with
care: those prompts share a large validation wrapper, so native prefix cache can
reuse that wrapper even when the raw sub-context differs. The cleaner comparison
is between exact repeat and OpenClaw canonical repeat, both of which reach 99.5%
cached ratio.
