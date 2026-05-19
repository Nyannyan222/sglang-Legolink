# SGLang Block-KV Patch Point Inspection

## Readiness

- Ready for block-KV patch: False
- Missing required roles: radix_cache, schedule_batch, schedule_policy
- Missing required patterns: {}

## Module Map

| role | found | module | path | hits |
| --- | --- | --- | --- | ---: |
| schedule_policy | False | `sglang.srt.managers.schedule_policy` | `` | 0 |
| schedule_batch | False | `sglang.srt.managers.schedule_batch` | `` | 0 |
| io_struct | False | `sglang.srt.managers.io_struct` | `` | 0 |
| radix_cache | False | `sglang.srt.mem_cache.radix_cache` | `` | 0 |
| memory_pool | False | `sglang.srt.mem_cache.memory_pool` | `` | 0 |
| scheduler | False | `sglang.srt.managers.scheduler` | `` | 0 |

## Suggested Patch Targets

- Attach OpenClaw block metadata to request objects before prefix matching.
- Use openclaw_cache_key/extra_key as the semantic block cache namespace.
- Add block-level cache hit/miss counters beside existing prefix counters.
- Keep block lookup/store separate from prompt-order prefix lookup.

## Relevant Pattern Hits

### schedule_policy

- Not found: sglang.srt.managers.schedule_policy

### schedule_batch

- Not found: sglang.srt.managers.schedule_batch

### io_struct

- Not found: sglang.srt.managers.io_struct

### radix_cache

- Not found: sglang.srt.mem_cache.radix_cache

### memory_pool

- Not found: sglang.srt.mem_cache.memory_pool

### scheduler

- Not found: sglang.srt.managers.scheduler
