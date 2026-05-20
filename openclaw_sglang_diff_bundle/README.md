# OpenClaw vs Original SGLang Difference Bundle

This folder collects the code and evidence that make the current local SGLang
runtime different from original SGLang.

## What Is Different

Original SGLang:

- Accepts a complete prompt/request.
- Uses RadixAttention prefix cache based on token-prefix equality.
- Does not know whether a prompt segment is an OpenClaw semantic sub-context.
- Does not log cache hit/miss by semantic block id, context field, or cache key.

Current OpenClaw-patched SGLang:

- Accepts OpenClaw block metadata from request body fields:
  - `openclaw_sub_context`
  - `custom_labels.openclaw_*`
- Preserves that metadata in:
  - `sampling_params.custom_params.openclaw_metadata`
- Emits OpenClaw metadata in scheduler `cache_lookup` logs:
  - `openclaw_prefill_unit_id`
  - `openclaw_cache_key`
  - `openclaw_context_field`
  - `openclaw_extra_key_matches_cache_key`
- Lets us verify cache behavior per semantic sub-context block.

## Evidence

Successful local run:

- `evidence/sglang_block_prefill_results.csv`
- rows: 4
- errors: 0

Cache lookup verification:

- `evidence/openclaw_cache_lookup_summary.json`
- total cache lookup events: 5
- OpenClaw cache lookup events: 4
- events with block id: 4
- events with cache key: 4
- extra-key mismatches: 0
- metadata complete: true

Observed block-level replay behavior:

| block | replay 1 cached | replay 1 uncached | replay 2 cached | replay 2 uncached |
| --- | ---: | ---: | ---: | ---: |
| `user_prompt` | 0 | 254 | 254 | 0 |
| `tool_definitions` | 0 | 35 | 35 | 0 |

This proves the patch is effective for metadata propagation, block-level
cache observability, and repeated block replay under SGLang's existing prefix
cache.

Additional validation summary:

- `docs/openclaw_method_effect_validation_report.md`
- `evidence/openclaw_method_effect_validation.json`

The validation summary explicitly separates runtime-proven facts from the
still-simulated block-key KV reuse policy.

## What This Does Not Prove Yet

This does not yet prove true semantic non-prefix KV reuse. The underlying
SGLang KV reuse is still prefix-based. The current improvement over original
SGLang is observability and OpenClaw semantic block awareness, not an
order-independent KV-cache mechanism.

## Files

- `scripts/patch_sglang_openclaw_request_metadata.py`
  Patches SGLang OpenAI request parsing and attaches OpenClaw metadata to
  sampling params.
- `scripts/patch_sglang_cache_lookup_logging.py`
  Patches scheduler cache lookup logging to emit OpenClaw block metadata.
- `scripts/verify_sglang_openclaw_block_logs.py`
  Verifies that patched logs contain complete OpenClaw block metadata.
- `scripts/summarize_openclaw_method_effect.py`
  Builds the current proof/gap report for the method.
- `scripts/start_local_sglang_docker.ps1`
  Starts the local patched Docker runtime with debug cache lookup logging.
- `docs/openclaw_sglang_request_metadata_patch_result.md`
  Short report of the implementation and local verification.
- `docs/block_kv_reuse_simulation_report.md`
  Shows the expected gain of semantic block-key KV reuse under order
  permutation.
