# SGLang OpenClaw Block Metadata Patch

## Purpose

This step connects the generated OpenClaw block-prefill payloads to the SGLang
runtime observation path.

The patch extends the existing SGLang cache-lookup logging so each lookup event
can report:

- `openclaw_prefill_unit_id`
- `openclaw_cache_key`
- `openclaw_context_field`
- `openclaw_extra_key_matches_cache_key`

This proves whether block metadata reaches the scheduler/cache lookup path.

## How To Patch SGLang

Run this inside the same Python environment that launches SGLang:

```bash
python scripts/patch_sglang_cache_lookup_logging.py
export SGLANG_PREFIX_CACHE_DEBUG_LOG=1
```

Then start SGLang normally and replay block-prefill payloads:

```bash
python scripts/run_sglang_block_prefill_payloads.py \
  --payloads benchmark_results/sglang_block_runtime_plan/sglang_prefill_request_payloads.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --repeat 2
```

Verify the SGLang server log:

```bash
python scripts/verify_sglang_openclaw_block_logs.py /path/to/sglang.log
```

## What Counts As Success

The verifier should report:

```text
Events with block id > 0
Events with cache key > 0
Extra key mismatches = 0
```

If `openclaw_extra_key_matches_cache_key` is true, the generated block payload
metadata is reaching the runtime request object consistently.

## Boundary

This still does not implement final non-prefix KV reuse. It is the observation
and metadata plumbing layer before modifying SGLang scheduler/KV-cache behavior.

The next implementation step is:

1. use `openclaw_cache_key` as a block-level KV cache lookup key,
2. store each prefilled block independently,
3. compose selected blocks with block-diagonal attention,
4. report per-block cached and prefill tokens.
