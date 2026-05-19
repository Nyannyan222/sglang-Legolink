# Local SGLang Docker Runtime Result

## Setup

- Host GPU: NVIDIA GeForce RTX 5070
- Docker backend: Docker Desktop WSL2 `desktop-linux`
- Image: `lmsysorg/sglang:latest-cu130-runtime`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Endpoint: `http://127.0.0.1:30000/v1`
- Container: `openclaw-sglang-local`

## Status

The local SGLang Docker server started successfully and returned:

```text
/v1/models -> Qwen/Qwen2.5-0.5B-Instruct
```

The block-prefill payload runner completed successfully:

```text
Rows: 16
Errors: 0
Output: benchmark_results/sglang_block_prefill_runs/sglang_block_prefill_20260520_032133
```

## Runtime Observation

The OpenAI-compatible response did not expose `cached_tokens` in the `usage`
field, so the CSV cache columns are empty for this run.

SGLang server logs did show prefix-cache reuse on replay 2:

```text
#cached-token: 254
#cached-token: 35
#cached-token: 2123
#cached-token: 2143
#cached-token: 2119
#cached-token: 2220
#cached-token: 2191
#cached-token: 2268
```

This confirms the local SGLang Docker runtime is working and ordinary SGLang
prefix cache is active for repeated block-prefill payloads.

## Boundary

This is still stock SGLang behavior. It validates the local Docker runtime and
payload plumbing, but it does not yet implement OpenClaw semantic block-level
KV reuse.

The next implementation step is to patch the SGLang runtime so OpenClaw block
metadata is preserved beyond `extra_key` and exposed in cache-lookup logs.
