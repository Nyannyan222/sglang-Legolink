# SGLang Block KV Reuse Policy

## Purpose

This step defines and tests the target cache policy for semantic sub-context KV
reuse.

Native SGLang prefix cache is order-sensitive: if the same semantic blocks are
reordered, reuse can drop because the prompt prefix no longer matches.

The OpenClaw target policy is order-invariant:

```text
lookup/store KV by semantic block cache_key
not by prompt prefix position alone
```

## Local Policy Simulation

Run:

```bash
python scripts/simulate_sglang_block_kv_reuse.py
```

Input:

```text
benchmark_results/sglang_block_runtime_plan/sglang_prefill_request_payloads.jsonl
```

Output:

```text
benchmark_results/sglang_block_kv_reuse_simulation/
```

## Current Result

| sequence | prefix reuse | block-key KV reuse |
| --- | ---: | ---: |
| original | 0.0% | 0.0% |
| repeat_same_order | 100.0% | 100.0% |
| permuted_reverse_order | 0.0% | 100.0% |

The important row is `permuted_reverse_order`: the same semantic blocks appear
in a different order. Prefix reuse drops, but block-key reuse still hits every
block by `cache_key`.

## Meaning

This is the behavior the SGLang runtime patch should implement internally:

1. receive one semantic sub-context block with `openclaw_cache_key`,
2. prefill/store its KV independently,
3. reuse the block when the same `cache_key` appears again,
4. allow reuse regardless of where that block appears in the next request.

## Boundary

This is a cache policy simulation using real WildClaw-derived block payloads.
It is not yet a measured SGLang scheduler/KV-cache speedup.

The next SGLang internal patch should move this policy into the real cache
lookup/store path.
