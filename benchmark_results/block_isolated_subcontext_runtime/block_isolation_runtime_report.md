# Block-Isolated Sub-context Runtime Prototype

## Purpose

This prototype models the runtime contract needed before patching SGLang internals:

- each field/function sub-context is an independent prefill unit,
- each unit has its own cache key,
- different sub-context blocks have zero allowed pairwise attention edges.

## Selected Blocks

- Blocks: 8
- Context fields: `tool_definitions`=1, `user_prompt`=1, `workspace_evidence`=6

## Independent Prefill / Cache

| replay | blocks | tokens | prefill tokens | reused tokens | cache hit ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 23712 | 23712 | 0 | 0.0% |
| 2 | 8 | 23712 | 0 | 23712 | 100.0% |

## Block-Isolated Attention

- Total tokens: 23712
- Intra-block allowed edges: 46129126
- Cross-block allowed edges: 0
- Cross-block disallowed edges: 470024404
- No pairwise attention check: PASS

## Output Files

- `benchmark_results/block_isolated_subcontext_runtime/prefill_results.csv`
- `benchmark_results/block_isolated_subcontext_runtime/block_pair_attention_matrix.csv`
- `benchmark_results/block_isolated_subcontext_runtime/block_isolation_runtime_prototype.json`

## Caveat

This is a runtime-facing prototype and validation harness. It proves the block contract and attention mask semantics, but it is not yet an SGLang kernel/scheduler patch.
