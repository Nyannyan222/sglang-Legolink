# SGLang Block-Isolated Sub-context Runtime Plan

## What This Adds

This artifact maps WildClaw field/function sub-contexts into SGLang-facing
runtime inputs. Each sub-context becomes an independent prefill request
with its own `prefill_unit_id`, `cache_key`, and `extra_key`.

## Runtime Summary

- Request id: `openclaw_block_isolated_demo`
- Task id: `04_Search_Retrieval_task_2_conflicting_handling`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Blocks: 8
- Tokens: 23712
- Cross-block allowed attention edges: 0
- No pairwise attention: PASS
- Validation errors: 0

## Scheduler Requirements

- Parse openclaw_sub_context metadata into each SGLang request object.
- Treat every prefill_unit_id as an independent prefill/cache unit.
- Use cache_key/extra_key to lookup and store block-local KV entries.
- Never concatenate independent sub-context blocks before prefill.
- When composing selected blocks, apply a block-diagonal attention mask.
- Expose per-block cached/prefill token counters for evaluation.

## KV Cache Requirements

- KV entries are keyed by cache_key, not by prompt position alone.
- Reused KV blocks must keep block-local positions or an explicit offset map.
- A cache hit for block A must not require block A to appear as a prefix.

## Attention Mask Contract

- Same block: `causal_attention_allowed`
- Different block: `attention_disallowed`
- Validation metric: `cross_block_allowed_edges == 0`

## Implementation Boundary

This is the concrete SGLang integration contract and request metadata
producer. It is still not the final SGLang scheduler/KV-cache patch.
The next patch should consume `openclaw_sub_context` and `extra_key`
inside SGLang request objects, then store/reuse KV by `cache_key`.
