# Semantic Reuse Candidate Evaluation

- Generated: `2026-05-20T17:32:34+00:00`
- Input CSVs: 2
- Total candidate pairs: 111
- Safe-to-reuse pairs: 0
- Judge-required pairs: 0
- Do-not-reuse pairs: 111
- Judge threshold: 3.0
- Embedding threshold: 0.7

## Main Verdict

No pair is currently safe to reuse under the strict semantic policy. This is a useful negative result: embedding similarity found related candidates, but the judge separated reusable evidence from partial overlap or broad topic matches.

This evaluation is intentionally different from a prefix-order experiment. The
goal is to test whether two WildClaw semantic sub-contexts have equivalent
answer utility, not whether prompt ordering makes SGLang's native prefix cache
hit.

## Decision Counts

| reuse_decision | pairs |
| --- | --- |
| do_not_reuse | 111 |

## Relation Counts

| relation_type | pairs |
| --- | --- |
| broad_topic | 19 |
| partial_overlap | 32 |
| unrelated | 60 |

## Source Summary

| source | pairs | safe_to_reuse | judge_required | do_not_reuse |
| --- | --- | --- | --- | --- |
| semantic_similarity_discovery_mixed_strict | 66 | 0 | 0 | 66 |
| semantic_similarity_discovery_openai_strict | 45 | 0 | 0 | 45 |

## Top Review Rows

| source | relation | decision | embedding | judge | same_answer | left_task | right_task |
| --- | --- | --- | --- | --- | --- | --- | --- |
| semantic_similarity_discovery_openai_strict | partial_overlap | do_not_reuse | 0.726 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_openai_strict | partial_overlap | do_not_reuse | 0.726 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_mixed_strict | partial_overlap | do_not_reuse | 0.640 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_openai_strict | partial_overlap | do_not_reuse | 0.621 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_openai_strict | partial_overlap | do_not_reuse | 0.611 | 2.0 | False | 04_Search_Retrieval_task_4_efficient_search | 04_Search_Retrieval_task_4_efficient_search |
| semantic_similarity_discovery_mixed_strict | partial_overlap | do_not_reuse | 0.605 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_mixed_strict | partial_overlap | do_not_reuse | 0.604 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_mixed_strict | broad_topic | do_not_reuse | 0.602 | 1.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_openai_strict | partial_overlap | do_not_reuse | 0.591 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |
| semantic_similarity_discovery_mixed_strict | partial_overlap | do_not_reuse | 0.589 | 2.0 | False | 04_Search_Retrieval_task_2_conflicting_handling | 04_Search_Retrieval_task_2_conflicting_handling |

## Interpretation

Embedding similarity is used only to find candidates. The reuse decision is made
after checking relation type and same-answer utility:

- `exact_duplicate`, `near_duplicate`, and `same_answer_utility` can be reused
  only when the judge supports equivalent answer utility.
- `partial_overlap`, `broad_topic`, and `unrelated` are rejected because they may
  be relevant but are not safe replacements.
- If no safe pair is found, the correct action is not to force reuse. The next
  step is to mine more WildClaw tasks/workspaces for repeated tool outputs,
  duplicated workspace evidence, and near-equivalent facts, while keeping this
  reuse filter strict.
