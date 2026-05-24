# LLM Trace Field Segmentation

This artifact uses an LLM-style field classifier to split complete agent trace/task context into field/function sub-contexts before any timeline slicing.

## Run

- Input JSONL: `benchmark_results/wildclaw_tool_history_trace_fixture/wildclaw_tool_history_trace_contexts.jsonl`
- Mode: `heuristic`
- Used LLM: `False`
- Model: ``
- Tasks: 3
- Output rows: 33

## Field Counts

| context_field | rows |
| --- | ---: |
| `user_prompt` | 3 |
| `tool_definitions` | 3 |
| `tool_history_output` | 3 |
| `conversation_history` | 6 |
| `workspace_evidence` | 15 |
| `task_metadata` | 3 |

## Method Counts

| method | rows |
| --- | ---: |
| `heuristic_trace_field_fallback` | 33 |

## Task / Field Summary

| task_id | context_field | method | sub-contexts | total chars | avg chars |
| --- | --- | --- | ---: | ---: | ---: |
| `04_Search_Retrieval_task_2_conflicting_handling` | `conversation_history` | `heuristic_trace_field_fallback` | 2 | 1679 | 839.5 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `task_metadata` | `heuristic_trace_field_fallback` | 1 | 375 | 375.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `tool_definitions` | `heuristic_trace_field_fallback` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `tool_history_output` | `heuristic_trace_field_fallback` | 1 | 2420 | 2420.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `user_prompt` | `heuristic_trace_field_fallback` | 1 | 1088 | 1088.0 |
| `04_Search_Retrieval_task_2_conflicting_handling` | `workspace_evidence` | `heuristic_trace_field_fallback` | 15 | 32066 | 2137.7 |
| `04_Search_Retrieval_task_4_efficient_search` | `conversation_history` | `heuristic_trace_field_fallback` | 2 | 1868 | 934.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `task_metadata` | `heuristic_trace_field_fallback` | 1 | 330 | 330.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `tool_definitions` | `heuristic_trace_field_fallback` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `tool_history_output` | `heuristic_trace_field_fallback` | 1 | 1455 | 1455.0 |
| `04_Search_Retrieval_task_4_efficient_search` | `user_prompt` | `heuristic_trace_field_fallback` | 1 | 1340 | 1340.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `conversation_history` | `heuristic_trace_field_fallback` | 2 | 847 | 423.5 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `task_metadata` | `heuristic_trace_field_fallback` | 1 | 319 | 319.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `tool_definitions` | `heuristic_trace_field_fallback` | 1 | 21 | 21.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `tool_history_output` | `heuristic_trace_field_fallback` | 1 | 1415 | 1415.0 |
| `04_Search_Retrieval_task_5_fuzzy_search` | `user_prompt` | `heuristic_trace_field_fallback` | 1 | 313 | 313.0 |

## Runtime Interpretation

Each output row is intended to be stored as an independent SGLang prefill/cache block.
The target attention policy remains `no_pairwise_attention_between_subcontexts`; long fields are split only after field classification.
