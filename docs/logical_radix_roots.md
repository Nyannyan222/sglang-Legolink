# Logical RadixTree roots over one shared physical cache

> **Provenance note.** This document was written inside the research repository this code was cut from. Scripts it names under `scripts/run_*` and the
> `experiment_harness/` package are research tooling that is **not** part of this repository; they are cited as the origin of the evidence, not as something you can run here.

Status: **IMPLEMENTED** and **TESTED** on CPU (`tests/test_logical_radix_roots.py`) and validated against a
live patched SGLang on a small model (`benchmark_results/local_small_model_namespace_validation/20261002_logical_roots/`).
The runtime and the LegoLink kernels are unchanged.

> One category has one logical RadixTree namespace/root,
> but all logical roots share the same physical SGLang RadixCache and KV pool.

Logical classification helps organize lookup. It must not become a hard physical reuse barrier unless
`HARD_NAMESPACE` mode is selected.

## Architecture

```
Host / CPU

Functional Classification
        |
        v
Cache Namespace
        |
        +-- instruction
        +-- tool_schema
        +-- conversation
        +-- tool_output
        +-- workspace
        |
        v
Logical RadixTree Roots            (semantic_cache_index.SemanticCacheIndex)
        |
        v
--------------------------------
SGLang Runtime / GPU
        |
        v
Shared Physical RadixCache         (one RadixCache, one allocator, one KV pool)
        |
        v
Shared Semantic KV Blocks          (RadixCache.legolink_blocks, keyed by runtime cache key)
        |
        v
LegoLink
        |
        v
Repair
        |
        v
Native Decode
```

`mutable_current_turn` and `no_cache` have **no root**. They are never persistent, whatever stage looks at them.

## Three identities, kept apart

| Identity | Question | Where |
| --- | --- | --- |
| functional field (`context_field`) | what kind of context is this? | Step 2, `session.CONTEXT_FIELDS` |
| cache namespace / logical root | which root does the host look this block's *name* up in? | `cache_namespace.CacheNamespacePolicy` |
| physical KV identity | may this compiled KV stand in for that block? | `cache_namespace.physical_kv_identity` |

`physical_kv_identity` hashes the exact authoritative token ids, the source bytes, the model and tokenizer
fingerprints and the segmentation and recompute policy versions. It contains **no** namespace, field, task or
session, so changing a block's namespace alone never invalidates an otherwise identical compiled block.

## What a logical root is

`semantic_cache_index.SemanticCacheIndex` maps `namespace -> logical root -> logical entries`:

```
SemanticCacheIndex
|-- instruction   : [ instruction/A  --\
|                                       +--> runtime cache key K1 --> RadixCache block K1
|-- workspace     : [ workspace/A (alias) --/
|-- tool_schema   : [ tool_schema/C ] ----> runtime cache key K2 --> RadixCache block K2
|-- conversation  : [ ... ]
|-- tool_output   : [ ... ]
```

* A **primary entry** is a row of the host `CacheCatalog` (the host's record of what it compiled). The index
  derives primary entries from the catalog, so it cannot drift from it.
* An **alias entry** is another name, in another (or the same) root, for a physical block an existing entry
  compiled. It is recorded when a Link has actually used it, and it is the only state the index holds.
* Every entry references a **runtime cache key** -- the key the patched runtime registered the block under.
  Nothing in the host constructs a slot, a range, a position, `imp_indices`, `kvlink` or a recompute position.

### Why this needs no second physical tree

The runtime registry is keyed by one `cache_key` per physical block; a Link names blocks by that key, and presenting
a block under another entry's key is the same shape as a repeated occurrence of one compiled block, which the
runtime already supports (its lookup returns one handle per requested position, and the reservation pins and
releases the *distinct* handles once). Compile also uses the cache key as the radix `extra_key`, so every compiled
block already lives in its own subtree of the single radix root. Per-namespace roots are therefore a *host* lookup
layer, not a tree the runtime would have to grow. A second `RadixCache`, allocator, pool, eviction manager or
reference counter would be a second source of truth for the same bytes.

## Namespace-first lookup

For one block, `NamespaceResolver.resolve`:

1. looks in **the block's own root**: the entry named by its own cache key. A hit is used.
2. looks for **another name for the same physical KV inside its own root** (two fields that map to one namespace,
   for example an opted-in `task_metadata` and `task_instruction`). Exact `physical_kv_identity` only.
3. on a miss, applies the mode's cross-root rule:

| Mode | Cross-root fallback |
| --- | --- |
| `global` | any root, if the physical identity is exact (`global_physical_dedup`, on by default) |
| `hard_namespace` | none; the block is compiled under its own name |
| `logical_namespace` | only for a pair listed in `CacheNamespacePolicy.cross_namespace_pairs` (default: none) |

The fallback requires the **exact** physical identity and a target that carries the token hash it was compiled
under. Similarity is never evidence for sharing KV. A pair is directional: `workspace <- instruction` does not
permit `instruction <- workspace`.

`global` used to be a field-partitioned registry, because the cache key contains the field: identical text under
two fields was two physical copies. It now lets an exact physical match serve the block, which is what "the
namespace never blocks exact-compatible reuse" means. `NOCSGLangAdapter(global_physical_dedup=False)` reproduces
the field-partitioned registry for a control run.

The three modes use the same underlying physical RadixCache.

## Physical deduplication

When two entries carry the same `physical_kv_identity` the second is an alias of the first's block. No second
Compile is issued. `physical_kv_dedup_count` counts the aliases a request used and
`cross_namespace_second_compile_avoided` the ones that crossed roots.

## Alias lifecycle

An alias is derived state, valid while its target is. The host never decides residency; the runtime does, and the
host learns of an eviction through the catalog (`mark_stale` / `invalidate`, fed by the runtime's own
"missing / stale" answer to a Link, which residency recovery turns into a recompile).

* When a physical block is removed, **every alias of it goes stale in the same call**
  (`SemanticCacheIndex.on_physical_removed`), and the removal is appended to an eviction log with the namespace,
  token count, residency age and the aliases it invalidated.
* An alias is also checked against its target's physical identity, so a key recompiled with different content never
  keeps old aliases; a removal another process made to a shared (persistent) catalog is noticed by `sweep()`.
* Recompiling an alias view registers the **owner's** root and field (`physical_entry` metadata on the view), not
  the alias's.
* Runtime reference counts are untouched: repeated names of one block in one request are one reservation and one
  release (`RadixCache.acquire_legolink_blocks` de-duplicates; the pin/unpin in `PICRuntime` uses distinct handles).
  No KV is duplicated to avoid this.

## Eviction is shared

There are no per-namespace GPU quotas and no per-namespace eviction policies. The runtime's single eviction
decides. The index only *exposes* namespace metadata so a later study can measure which namespace's logical blocks
are evicted, how long a physical block stayed resident, and how many reusable tokens survived:
`semantic_index.snapshot()["physical_removals_by_namespace"]`, `["removed_tokens_by_namespace"]`, `eviction_log`.

## Metrics

Per block (`NamespaceLookupResult.to_dict`): `context_field`, `cache_namespace`, `logical_root`,
`logical_entry_key`, `logical_lookup_hit/miss`, `cross_namespace_attempt`, `cross_namespace_hit`,
`physical_kv_identity`, `physical_kv_reused`, `second_compile_avoided`,
`cross_namespace_second_compile_avoided`, `physical_alias_count`, `lookup_latency_s`,
`physical_block_residency_age_s`.

Per request (`summarize_namespace_request` totals): the same counters plus `reused_token_count`,
`compiled_token_count`, `repair_token_count`, `physical_kv_dedup_count`.

The Terminal-Bench agent records the catalog's `snapshot()` as `namespace_logical_index`.

## Files

| File | Role |
| --- | --- |
| `noc_context_manager/semantic_cache_index.py` | the logical roots, entries, aliases and their lifecycle |
| `noc_context_manager/cache_namespace.py` | field -> namespace, physical identity, namespace-first lookup, metrics |
| `noc_context_manager/compile_policy.py` | `CacheCatalog` owns an index; `mark_stale` / `invalidate` feed it |
| `noc_context_manager/persistent_catalog.py` | `live_entries()` for the shared catalog |
| `noc_context_manager/sglang_adapter.py` | records aliases after a completed Link; compiles an alias view as its owner |
| `scripts/run_logical_root_validation.py` | the local runtime validation |
| `tests/test_logical_radix_roots.py` | the CPU invariants |

## Local runtime validation (Qwen2.5-0.5B-Instruct, RTX 5070)

`benchmark_results/local_small_model_namespace_validation/20261002_logical_roots/README.md`. Against a live patched
SGLang, in all three modes at K=256 and K=1024, with a cold-prefill Native reference and explicit sampling:

* 20/20 structural checks: same-root reuse (no second Compile); hard cross-root pays a second Compile; logical and
  global cross-root issue none and name the producer's key; one request naming one block twice acquires and releases
  once; an evicted block takes its alias with it and recovery re-derives it.
* Full repair equals Native on 21/21 requests (logprob diff 0.0); K=256 is the sparse approximation (6/21 exact on
  these half-repaired blocks) and is reported separately.
* A block linked through an alias is bit-identical to the same tokens compiled under their own name.

Limits: one small model, one GPU, 24 generated tokens, two-block prompts. It validates the host layer and its
interaction with the runtime's registry, not residency pressure at scale.

## Not done, on purpose

No new categories, no change to Jev, segmentation, the boundary lock, repair math, LegoLink kernels or native
decode, no per-namespace quotas, no second physical cache, no global sampling-default change, no H100 run.
