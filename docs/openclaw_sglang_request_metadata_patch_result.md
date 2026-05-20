# OpenClaw SGLang Request Metadata Patch Result

Date: 2026-05-20

## Goal

Patch local SGLang request parsing and cache-lookup logging so OpenClaw
field/function sub-context block metadata can travel with each request and be
observed at the scheduler prefix-cache lookup path.

## Implemented Path

OpenClaw block metadata is accepted from OpenAI-compatible request bodies:

- `custom_labels.openclaw_*`
- `openclaw_sub_context`

To avoid crashing SGLang Prometheus metrics with undeclared label names, the
OpenClaw metadata is not stored as SGLang metrics `custom_labels`. It is
attached to:

```text
sampling_params.custom_params.openclaw_metadata
```

The cache-lookup logging patch then reads this metadata from the scheduler
request's sampling params and emits it in `cache_lookup` log events.

## Local Verification

Runtime:

- Docker image: `openclaw-sglang:openclaw-metadata`
- Container: `openclaw-sglang-local`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Endpoint: `http://127.0.0.1:30000/v1`

Replay output:

- `benchmark_results/sglang_block_prefill_runs/sglang_block_prefill_20260520_153110/sglang_block_prefill_results.csv`
- Rows: 4
- Errors: 0

Log verification:

- Total cache lookup events: 5
- OpenClaw cache lookup events: 4
- Events with `openclaw_prefill_unit_id`: 4
- Events with `openclaw_cache_key`: 4
- Extra-key mismatches: 0
- Unique OpenClaw blocks: 2

The extra cache lookup event is SGLang warmup and is intentionally excluded
from OpenClaw metadata completeness checks.

## What This Proves

This verifies the plumbing needed before true runtime-level block KV reuse:

1. OpenClaw block metadata can be sent through OpenAI-compatible chat requests.
2. SGLang request parsing can preserve the block metadata without breaking
   metrics.
3. Scheduler-side cache lookup logs can identify which semantic sub-context
   block is being prefetched or reused.
4. Repeated block requests show prefix-cache reuse at the block payload level.

## Remaining Work

This is still metadata/logging plumbing. It does not yet implement true
non-prefix, order-independent KV reuse between semantic sub-context blocks.
The next SGLang patch should introduce runtime-owned block KV objects keyed by
semantic `cache_key`, then compose selected blocks without requiring them to
share a prompt prefix.
