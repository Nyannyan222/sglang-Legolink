# Semantic cache namespaces and Step 2 field classification

> **Provenance note.** This document was written inside the research repository this code was cut from. Scripts it names under `scripts/run_*` and the
> `experiment_harness/` package are research tooling that is **not** part of this repository; they are cited as the origin of the evidence, not as something you can run here.

Status: **IMPLEMENTED** and **TESTED** (`tests/test_semantic_cache_namespaces.py`, 56 tests).
**No measurement is reported here.** The code makes it possible to compare
`global` against `hard_namespace` against `logical_namespace`; whether either
namespace arm improves reuse is an experimental question this page does not answer.

## Three concepts, three places

| Concept | Question it answers | Where it lives |
| --- | --- | --- |
| Functional field | What kind of context is this? | `session.CONTEXT_FIELDS`, decided in Step 2 (`field_classification.py`) |
| Cache namespace | Where does the *host* look this block up, and may it be retained? | `cache_namespace.CacheNamespacePolicy` (the only place field -> namespace is decided) |
| Physical KV identity | May this compiled KV tensor stand in for that block? | `cache_namespace.physical_kv_identity`, and the runtime's own `LegoLinkBlockKey` |

A namespace is host bookkeeping. There is still one RadixCache, one allocator and
one KV pool, owned by SGLang. The runtime source is unchanged.

## Audit (what the code did before this change)

1. **Step 1 -> 2 -> 3.** `NOCSGLangLLM.call` -> `NOCRequestPipeline.execute` ->
   `IncrementalContextManager.ingest` -> `SessionFieldPipeline.collect/extract` ->
   `SemanticSegmentationPipeline.run`.
2. **`SessionEvent`.** `event_id` (digest of type, source and text), `task_id`,
   `session_revision`, `event_type` (an `EventType`), `source_id`, `raw_text`,
   `timestamp`, `sequence_number`, `metadata`.
3. **Classification rules.** One table, `FieldFirstExtractor._FIELD_MAP`, from event
   type to field; sources whose metadata or path marks them oracle/grader/solution
   are dropped before classification. No ambiguity concept existed.
4. **Cache key.** `semantic_cache_key` hashes `semantic_identity`, **`context_field`**,
   model and tokenizer fingerprints, segmentation policy version and recompute policy
   version. `semantic_identity` also hashes `context_field`, plus the authoritative
   token ids (basis `tok`) or the text (basis `txt`).
5. **Task/session isolation.** Neither `task_id` nor `session_id` is in the semantic
   identity, the cache key, or the runtime `LegoLinkBlockKey`. Cross-task reuse is
   not blocked by identity. What bounds it is the *host* `CacheCatalog`: one per
   agent session unless `shared_catalog_path` / `shared_block_catalog` is set. This
   change does not touch that.
6. **Registry.** `RadixCache.register/lookup/acquire/release_legolink_block[s]`, a
   dict keyed by cache key whose `LegoLinkBlockKey` adds the token hash, model,
   tokenizer, kv dtype, rope, attention backend, adapter, tp size and the two policy
   versions. `context_field` is stored as metadata, not as part of the key.
7. **Ownership.** Registered blocks carry the radix node that pins their KV slots in
   the same `token_to_kv_pool_allocator` as native prefix KV. There is one allocator.
8. **Can namespaces exist without multiple physical caches?** Yes, host-side: lookup
   and retention policy are decisions about *which already-compiled cache key the
   Link names*, and the runtime is already agnostic to why a key was chosen.
9. **Consequence for the design.** Because the cache key contains the field, identical
   text under two fields already has two cache keys and two physical KV copies.
   `GLOBAL` is therefore a field-partitioned registry today, not a content-global one.

## Physical KV identity, and whether it can be shared across namespaces

`physical_kv_identity(block)` hashes: the exact authoritative token ids, the exact
source bytes, model and tokenizer fingerprints, and the segmentation and recompute
policy versions. It deliberately **excludes** the functional field, the namespace,
the task and the session, because none of them enters the compiled KV: a Compile
request carries only the block's own tokens, prefilled standalone from position 0.

Sharing a compiled block across namespaces is therefore *physically* plausible, and
`LOGICAL_NAMESPACE` can do it as an alias onto the existing entry (below). It is
**off by default** and only happens when all of these hold:

* the two blocks have equal `physical_kv_identity` (same token ids, bytes, model,
  tokenizer, policy versions) -- a block with **no recorded token ids has no identity
  and never aliases**, since equal bytes do not prove equal tokens;
* the existing entry is live (not stale), recorded the `physical_token_ids_sha256` it
  was compiled under, and recorded its `semantic_identity` -- the runtime validates
  `block_ids` against the registered block, so the alias must present the target's;
* `CacheNamespacePolicy.cross_namespace_pairs` names the `(block, entry)` namespace
  pair explicitly.

Conditions that **prevent** sharing, or that this change cannot establish:

* The runtime-side fingerprints (kv dtype, rope, attention backend, tensor-parallel
  size, adapter) are constant for one server and checked by the runtime's registry
  key; the host cannot see them. Two servers must not share a catalog.
* The Link's token-stable framing may carry a block fresh at a non-composing
  boundary. That affects the Link, not the compiled KV, and is unchanged.
* A catalog shared between adapters with **different stitch separators** could name a
  block compiled from different tokens. The runtime's per-block token hash check
  refuses that; the host's `_catalog_segment_matches` also invalidates a mismatch.
* The alias path has not been run against a live runtime in this change (no GPU
  here). It reuses the shape the runtime already supports -- a repeated occurrence of
  one compiled block -- but "supported shape" is not "measured".

## Namespaces

| Functional field | Namespace | Persistent |
| --- | --- | --- |
| `task_instruction` | `instruction` | yes |
| `tool_definitions` | `tool_schema` | yes |
| `conversation_history` | `conversation` once finalized (block immutable); otherwise `mutable_current_turn` | yes / no |
| `tool_history_output` | `tool_output` | yes |
| `workspace_evidence` | `workspace` | yes |
| `user_prompt` | `mutable_current_turn` | no |
| `task_metadata` | `no_cache`; `instruction` only if the block's metadata sets `model_visible`, `stable` and `reusable` all true | no / yes |

`CacheNamespacePolicy.profile(namespace)` exposes `expected_reuse` and `lifecycle`
for later residency policy. Nothing acts on them yet and eviction is unchanged.

## Modes (`semantic_cache_namespace_mode`)

| Mode | Lookup | Persistence | Alias |
| --- | --- | --- | --- |
| `global` (default) | `catalog.contains(cache_key)`; on a miss, an **exact physical-identity** match in any logical root (see `docs/logical_radix_roots.md`) | not enforced | any root, exact identity only (`global_physical_dedup=False` restores the old field-partitioned registry) |
| `hard_namespace` | a block is resident only through an entry recorded under its **own** namespace | non-persistent namespaces are kept out of the cache (`KEEP_MUTABLE`, reason `namespace is not persistent`) | never across roots; a same-root alias (two fields mapped to one namespace, e.g. `task_instruction` and an opted-in `task_metadata`) is allowed in every mode |
| `logical_namespace` | namespace first, then a policy-gated alias | as `hard_namespace` | per `cross_namespace_pairs`, exact physical identity only. **With the default policy (no pairs) this mode is behaviourally identical to `hard_namespace`** (audit F-04). **Since Q-1 the H200 runner's `logical_namespace` arm names the vetted pair `workspace<->tool_output`**, so the two arms are different treatments by configuration; whether they differ on a workload is measured (`observable_equivalence` in `summary.json`; on the recorded Terminal-Bench corpus the pair has no opportunity). See `docs/q1_cross_namespace_policy.md` |

Set it with `NOCSGLangAdapter(semantic_cache_namespace_mode=...)`,
`NOCSGLangLLM(semantic_cache_namespace_mode=...)`, the agent kwarg of the same
name, or `noc_openai_proxy.py --semantic-cache-namespace-mode`. The value is
validated at construction and written on every metrics row.

Because the cache key already contains the field, `global` and `hard_namespace`
differ in *persistence enforcement* (and in accounting), not in which same-key
entries they can see. `logical_namespace` resolves a block to an entry of another
root only for a named pair; `global` does so for any exact physical match (the
namespace never blocks exact-compatible reuse). Each namespace is a logical root of
one shared physical cache: see `docs/logical_radix_roots.md`.

Where it runs: `NOCRequestPlanner.plan` calls `NamespaceResolver.apply` first. A
block served by another namespace's entry is **presented** under that entry's cache
key and semantic identity (its own `sub_context_id` and spans are kept), so Stage 5,
Gate 1, layout feasibility, the stitcher and Stage 8 all plan over the presented
block. Nothing downstream knows an alias happened; the Link names the existing key
and no second Compile is issued. `NOCSGLangAdapter.link` reads the plan's blocks
rather than the caller's list for the catalog check, so the presented key is the one
checked.

## Step 2 classification

```
SessionEvent
  -> DeterministicFieldClassifier      event type in the table -> field, confidence 1.0
  -> (only when the table declines)
     JevFieldClassifier                injected callable, closed schema, no network code
  -> field, or FieldClassificationError
```

* `HybridFieldClassifier` never calls Jev for a structurally known event, remembers
  its answer per `event_id` (extraction runs several times a turn, and a block must
  not change field), and never memoizes a failure, so a retry asks again.
* Jev picks one value of `CONTEXT_FIELDS` and a confidence. Extra keys, an unknown
  field, a non-numeric confidence, a call that raises, or a confidence below
  `min_confidence` (default 0.8) is **not guessed at**: it raises, unless the caller
  configured an explicit `fallback_field`, which is then recorded as
  `source: explicit_fallback`.
* Hidden (oracle/grader/solution) sources are filtered before classification and are
  never sent to Jev.
* A new `EventType.UNCLASSIFIED` is the way a producer says "role unknown". Without a
  classifier it fails extraction. The `noc_session_event` schema lists it.
* Not wired: a concrete Jev endpoint. The interface takes a callable; binding it to
  a deployment is configuration this repository has no definition for.

## Step 3 stays host-constrained, and boundaries are locked

Unchanged: source -> deterministic atomic units -> the LLM returns `split_points`
only -> the host validates and rebuilds exact text. The LLM never writes block text.

**Boundary locking** (new): when a changed source's new text starts with the text its
stored, finalized leading blocks cover, `IncrementalContextManager._frozen_prefixes`
hands those blocks to `SemanticSegmentationPipeline.run(frozen_prefixes=...)`. They are
replayed as proposals exactly as stored (same spans, same text, so the same
identity and cache key) and only the appended remainder is segmented, with its spans
shifted back into the whole source. A block is frozen only if it is immutable and a
reuse candidate and contiguous from the start of the source. An edit (not an append),
a mutable block, or an unfinalized block is segmented as before.

Note the scope: a source id's text only changes in the agent when a producer reuses a
source id (a growing file or log). Message sources are keyed by content and never
change. Append-only conversation and tool-output blocks stay mutable until the *next*
ingest settles them, so they become lockable one revision later.

The existing guarantee that the same source, field and policy give the same
boundaries is the response cache (`cache_dir`), temperature 0, and the deterministic
atomic-unit builder; a test pins it. The atomic-unit policy is not part of the block
cache key, and this change does not add it (that would change every key).

## Metrics

Per request, on every `metrics.jsonl` row (`namespace_metrics`, plus
`semantic_cache_namespace_mode`, `namespace_active_blocks`, `namespace_removals`),
and per block on the plan under `namespace_resolution`:

`namespace`, `context_field`, `namespace_lookup_hit`, `namespace_lookup_miss`,
`cross_namespace_fallback_attempted`, `cross_namespace_hit`,
`cross_namespace_missed_reuse_tokens`, `reused_token_count`, `compiled_token_count`,
`repair_token_count`, `lookup_latency_s`, `entry_age_s`.

`cross_namespace_missed_reuse_tokens` is the block's tokens when a live entry with
the same physical identity existed in **another** namespace and the block was not
served from it. It is computed in every mode, so `global` and `hard_namespace` show
how much a namespace boundary (or, in `global`, the field in the key) leaves behind.
Tokens are host-accounted block tokens, not runtime measurements. Host-observed
removals by namespace are counted when the host invalidates or marks stale an entry
(e.g. residency recovery); the runtime's own evictions are not visible to the host.

## Experiment

Run the same workload three times, changing only
`--semantic-cache-namespace-mode`. What to compare: `compile_count`,
`compiled_token_count`, `reused_token_count`, `cross_namespace_missed_reuse_tokens`
and the end-to-end latencies already on the row. `logical_namespace` needs a
non-empty `cross_namespace_pairs`, or it is identical to `hard_namespace`.

The headline-metric fail-closed rules in `experiment_harness/` are unchanged: these
modes compare LegoLink against LegoLink and the counters above are raw
measurements, not regime-governed ratios.

## What was deliberately not done

* No runtime change, no patch, no per-namespace pool or allocator, no change to
  PICRuntime, kernels, repair math or physical layout.
* No change to the cache key, the semantic identity or task/session isolation.
* Jev is not mandatory and not bound to an endpoint.
* A runtime-side namespace index (per-namespace eviction or pinning) would be a
  patch to `radix_cache.py`; it is not needed to run the three-arm comparison.

## Non-persistence is a hard constraint on the layout stage

`LinkLayoutFeasibility` promotes an immutable reusable block that precedes a compiled one
(`force_compile_for_prefix_closure`). Before this constraint that promotion reached blocks in the `no_cache` and
`mutable_current_turn` namespaces and registered them in the runtime: the compile *policy* was told they were
non-persistent, the layout stage was not. `NamespaceResolver.layout_permissions` now hands the layout stage a "may
not be linked" answer for those blocks (the same channel Gate 1 already uses), so prefix closure never promotes
them and the *later* compiled blocks are demoted to the mutable suffix instead (`force_mutable_for_suffix_closure`).

Consequences, measured on the runtime:

* A settled earlier `user_prompt` (namespace `mutable_current_turn`) or a `task_metadata` block (`no_cache`) in the
  middle of a session makes every reusable block *after* it unlinkable. In the local N3 case the workspace block
  that follows a settled prompt was demoted: 259 reused tokens (hard/logical) against 518 (global).
* A request in which every reusable block is demoted is refused by the strict server
  (`strict link requires at least one compiled block`) instead of being served.
* `global` mode does not enforce the constraint and keeps the legacy behaviour.

`tests/test_prefix_closure_persistence.py` pins the shapes (`[NC][P]`, `[MC][P]`, `[P][NC]`, `[P][NC][P]`, ...) and
checks every ordering of up to four blocks; `scripts/run_full_repair_diagnosis.py --stage prefix-closure` writes the
before/after record (`prefix_closure_before_after.json`).

## Local runtime validation (Qwen2.5-0.5B-Instruct, RTX 5070)

`scripts/run_local_small_model_namespace_validation.py` and `scripts/run_full_repair_diagnosis.py` ran the host
pipeline against a real patched SGLang (Docker, `lmsysorg/sglang:latest-cu130-runtime`, runtime working tree at
`7e61b59`) in all three modes, with the runtime trace as the evidence. Artifacts:
`benchmark_results/local_small_model_namespace_validation/`. It is a correctness-and-behavior smoke, not a performance
or accuracy result.

* **Cross-namespace alias reaches the real runtime and is numerically exact.** In `logical_namespace` with only
  `(workspace -> instruction)` enabled, the consumer block issued no second Compile; the Link named the producer's cache
  key; the runtime looked it up and hit the producer's block; the request completed with output and logprobs
  bit-identical to the same block compiled under its own key (`global`) and to Native.
* **Full repair equals Native; sparse repair does not.** With a *cold* Native reference and identical explicit sampling,
  Link at a repair window covering every block (K=1024) reproduced Native exactly on all 28 cases (max |logprob diff|
  0.0). K=256 matched on 16 of 28; the first differing token is where stale (unrepaired) KV matters, and the post-repair
  K/V of unrepaired rows is 5-10x above the bf16 noise floor while repaired rows are inside it.
* **A Native reference must be a cold prefill.** Native requests that share the chat-template prefix with an earlier
  request are prefilled as an extend over cached tokens, which is a different computation from the full recompute a
  Link performs. Use a distinct `extra_key` per reference request. References made without it moved by up to 0.04 in
  logprob and flipped near-tied tokens, which earlier runs mistook for a Link defect.
* **Sampling defaults differ between the paths.** A Link request that omits `repetition_penalty` ran with 1.0 while
  the same native request inherited the model default 1.1. Send sampling parameters explicitly on every request when
  comparing arms. The root cause was not isolated and the runtime was not modified.
