# Semantic Reuse Positive/Negative Benchmark

- Generated: `2026-05-20T17:42:58+00:00`
- Positive source rows: `benchmark_results\wildclaw_field_function_subcontexts\wildclaw_field_function_subcontexts.jsonl`
- Negative source CSV: `benchmark_results\semantic_reuse_candidate_eval\semantic_reuse_candidate_eval.csv`
- Positive pairs: 3
- Negative pool pairs: 111
- Benchmark rows: 12

## Why This Benchmark Matters

This benchmark tests the teacher's requested distinction:

- positive rows are semantic sub-context pairs that are actually safe to reuse;
- negative rows may look related, but they must not be reused because they are
  only partial-overlap, broad-topic, or unrelated.

It does not test prompt order tricks. The positive side currently comes from
real WildClaw field/function context blocks with exact cross-task content
matches. The negative side comes from strict LLM-judged semantic-similarity
candidates that were rejected for reuse.

## Benchmark Relation Counts

| relation_type | pairs |
| --- | --- |
| broad_topic | 3 |
| exact_duplicate | 3 |
| partial_overlap | 3 |
| unrelated | 3 |

## Positive Field Counts

| context_field | positive_pairs |
| --- | --- |
| tool_definitions | 3 |

## Current Finding

The current safe positives are exact duplicate `tool_definitions` blocks shared
across WildClaw Search Retrieval tasks. This is a small but clean positive set:
it proves the pipeline can identify truly reusable sub-contexts without
mistaking partial topical overlap for reuse.

The next expansion should mine additional WildClaw categories and repeated
workspace/tool-output facts, then pass near-duplicate and same-answer-utility
candidates through the same strict judge.
