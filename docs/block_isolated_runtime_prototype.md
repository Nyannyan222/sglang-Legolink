# Block-Isolated Sub-context Runtime Prototype

## Goal

This step addresses the runtime meaning of semantic independence:

> no pairwise attention between sub-contexts.

The prototype uses field/function semantic sub-contexts and models them as
independent prefill/cache units.

## What Is Implemented

- `PrefillBlock`: one runtime block per sub-context.
- `IndependentPrefillCache`: in-memory cache keyed by each sub-context
  `cache_key`.
- `block_isolation_report`: verifies block-diagonal causal attention semantics.
- `scripts/run_block_isolated_subcontext_prototype.py`: runs the prototype and
  writes a report.

## Attention Contract

For sub-context blocks A, B, and C:

```text
A tokens can attend only to A tokens.
B tokens can attend only to B tokens.
C tokens can attend only to C tokens.

A <-> B cross attention: disallowed
A <-> C cross attention: disallowed
B <-> C cross attention: disallowed
```

The validation metric is:

```text
cross_block_allowed_edges = 0
```

## Cache Contract

Each sub-context has:

- `prefill_unit_id`
- `cache_key`
- `cache_scope = sub_context_block`
- `kv_reuse_granularity = sub_context`

The prototype simulates two prefill passes:

1. First pass: cache miss, each block is independently prefetched.
2. Second pass: cache hit, each block is reused by `cache_key`.

## Current Limitation

This is not yet an SGLang internal patch. It proves the required runtime
contract before changing SGLang attention masks or KV cache managers.

The next implementation step is to map `PrefillBlock` to real SGLang request /
KV cache objects and enforce the block isolation mask inside the runtime.
