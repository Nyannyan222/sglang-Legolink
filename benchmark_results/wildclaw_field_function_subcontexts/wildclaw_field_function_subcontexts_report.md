# Field/Function Semantic Sub-contexts

This artifact segments WildClaw/OpenClaw context by fields and functional blocks first, not by timeline trace order.

## Policy

- First split by `user_prompt`, `tool_definitions`, `tool_history_output`, `conversation_history`, `workspace_evidence`, and `task_metadata`.
- Only overlong fields are split internally.
- Each output row carries SGLang-oriented metadata for independent prefill/cache/reuse.
- The target runtime policy is no pairwise attention between sub-context blocks.

## Inputs

- Input JSONL: `benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl`
- Max field chars before internal split: 4000
- Max source files per task: 12

## Summary

| task_id | context_field | sub-contexts | total chars | avg chars |
| --- | --- | ---: | ---: | ---: |
| `04_Search_Retrieval_task_2_conflicting_handling` | `task_metadata` | 1 | 414 | 414.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `tool_definitions` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `user_prompt` | 1 | 1088 | 1088.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `workspace_evidence` | 78 | 279306 | 3580.8 |
| `04_Search_Retrieval_task_4_efficient_search` | `task_metadata` | 1 | 369 | 369.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `tool_definitions` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `user_prompt` | 1 | 1340 | 1340.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `task_metadata` | 1 | 358 | 358.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `tool_definitions` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `user_prompt` | 1 | 313 | 313.0 |

## Runtime Interpretation

Each row is intended to become an SGLang sub-context block with its own `prefill_unit_id` and `cache_key`.
A future runtime implementation should prefill these blocks independently and use a block-isolation attention mask so tokens inside one sub-context do not attend to tokens inside another sub-context.
