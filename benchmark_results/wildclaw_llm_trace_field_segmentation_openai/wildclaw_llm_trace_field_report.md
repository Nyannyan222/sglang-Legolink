# LLM Trace Field Segmentation

This artifact uses an LLM-style field classifier to split complete agent trace/task context into field/function sub-contexts before any timeline slicing.

## Run

- Input JSONL: `benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl`
- Mode: `llm`
- Used LLM: `True`
- Model: `gpt-4.1-mini`
- Tasks: 3
- Output rows: 12

## Field Counts

| context_field | rows |
| --- | ---: |
| `user_prompt` | 3 |
| `tool_definitions` | 2 |
| `tool_history_output` | 0 |
| `conversation_history` | 3 |
| `workspace_evidence` | 1 |
| `task_metadata` | 3 |

## Method Counts

| method | rows |
| --- | ---: |
| `llm_trace_field_segmentation` | 12 |

## Task / Field Summary

| task_id | context_field | method | sub-contexts | total chars | avg chars |
| --- | --- | --- | ---: | ---: | ---: |
| `04_Search_Retrieval_task_2_conflicting_handling` | `conversation_history` | `llm_trace_field_segmentation` | 1 | 367 | 367.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `task_metadata` | `llm_trace_field_segmentation` | 1 | 375 | 375.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `user_prompt` | `llm_trace_field_segmentation` | 1 | 1083 | 1083.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `workspace_evidence` | `llm_trace_field_segmentation` | 1 | 769 | 769.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `conversation_history` | `llm_trace_field_segmentation` | 1 | 269 | 269.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `task_metadata` | `llm_trace_field_segmentation` | 1 | 330 | 330.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `tool_definitions` | `llm_trace_field_segmentation` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `user_prompt` | `llm_trace_field_segmentation` | 1 | 1340 | 1340.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `conversation_history` | `llm_trace_field_segmentation` | 1 | 321 | 321.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `task_metadata` | `llm_trace_field_segmentation` | 1 | 319 | 319.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `tool_definitions` | `llm_trace_field_segmentation` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `user_prompt` | `llm_trace_field_segmentation` | 1 | 313 | 313.0 |

## Runtime Interpretation

Each output row is intended to be stored as an independent SGLang prefill/cache block.
The target attention policy remains `no_pairwise_attention_between_subcontexts`; long fields are split only after field classification.
