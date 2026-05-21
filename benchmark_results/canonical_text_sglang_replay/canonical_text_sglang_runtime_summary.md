# WildClaw SGLang Runtime Summary

- Source CSV: `C:/Users/Administrator/OneDrive/文件/New project/benchmark_results/canonical_text_sglang_replay/runtime_runs/wildclaw_sglang_runtime_20260521_152152/wildclaw_sglang_runtime_results.csv`
- Rows: 20
- Completed: 0
- Errors: 20

## Condition And Replay Summary

| condition | replay | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Replay-1 Context Reduction Vs Fixed Size

| condition | tasks | fixed prompt sum | condition prompt sum | weighted reduction | mean task reduction |
| --- | --- | --- | --- | --- | --- |

## Notes

- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.
- Order-permutation manifests may encode the repeat as a separate `real_semantic_order_repeat` row with `REPEAT=1`.
- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.
- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.

## Errors

| task_id | condition | replay | error |
| --- | --- | --- | --- |
| 01_Productivity_Flow_task_2_table_tex_download | raw_parameterized_left | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_11_fuzzy_repo_search | raw_parameterized_right | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 01_Productivity_Flow_task_2_table_tex_download | canonical_text | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_11_fuzzy_repo_search | canonical_text_repeat | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 01_Productivity_Flow_task_2_table_tex_download | raw_parameterized_left | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_1_google_scholar_search | raw_parameterized_right | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 01_Productivity_Flow_task_2_table_tex_download | canonical_text | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_1_google_scholar_search | canonical_text_repeat | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 01_Productivity_Flow_task_2_table_tex_download | raw_parameterized_left | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_5_fuzzy_search | raw_parameterized_right | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 01_Productivity_Flow_task_2_table_tex_download | canonical_text | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_5_fuzzy_search | canonical_text_repeat | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_10_tomllib_trace | exact_canonical_text | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_4_efficient_search | exact_canonical_text_repeat | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_10_tomllib_trace | exact_canonical_text | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_4_efficient_search | exact_canonical_text_repeat | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_10_tomllib_trace | exact_canonical_text | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 04_Search_Retrieval_task_4_efficient_search | exact_canonical_text_repeat | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 01_Productivity_Flow_task_1_arxiv_digest | block_canonical_text | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
| 01_Productivity_Flow_task_4_2022_conference_papers | block_canonical_text_repeat | 1 | <urlopen error [WinError 10061] 無法連線，因為目標電腦拒絕連線。> |
