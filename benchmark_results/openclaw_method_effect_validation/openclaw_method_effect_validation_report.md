# OpenClaw Method Effect Validation

## Current Verdict

The current implementation proves runtime feasibility and observability,
but it does not yet prove that OpenClaw is faster than original SGLang
at the actual KV-cache implementation level.

What is proven:

- OpenClaw semantic block metadata reaches SGLang request parsing.
- The scheduler cache-lookup log can identify OpenClaw block id, cache key, and context field.
- Replaying the same OpenClaw block produces full prefix-cache hits under SGLang's existing cache.
- Native prefix-cache reuse drops when semantic blocks are reordered.

What is not proven yet:

- True runtime-level semantic non-prefix KV reuse.
- Order-independent KV reuse inside SGLang's scheduler/KV cache.
- End-to-end latency improvement over original SGLang after a real non-prefix patch.

## Local OpenClaw Metadata Replay

- Cache lookup events: 5
- OpenClaw cache lookup events: 4
- Metadata complete: True
- Extra-key mismatches: 0

| context field | events | tokens | first cached | last cached | first ratio | last ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tool_definitions | 2 | 35 | 0 | 35 | 0.0% | 100.0% |
| user_prompt | 2 | 254 | 0 | 254 | 0.0% | 100.0% |

## Native SGLang Prefix Limitation

This is an actual SGLang runtime replay using reordered semantic contexts.

| condition | rows | prompt tokens | cached tokens | prefill tokens | cached ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| real_semantic_order_original | 2 | 2079 | 151 | 1928 | 7.3% |
| real_semantic_order_permuted | 2 | 2079 | 816 | 1263 | 39.2% |
| real_semantic_order_repeat | 2 | 2079 | 2077 | 2 | 99.9% |

## Intended Block-Key KV Policy Simulation

This is a policy simulation of the target implementation, not a completed SGLang patch.

| sequence | blocks | tokens | prefix reuse | block-key reuse | block-key prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 8 | 23712 | 0.0% | 0.0% | 23712 |
| repeat_same_order | 8 | 23712 | 100.0% | 100.0% | 0 |
| permuted_reverse_order | 8 | 23712 | 0.0% | 100.0% | 0 |

## Next Patch Required

The next implementation step is to patch SGLang internals so that
`openclaw_cache_key` indexes runtime-owned KV blocks independently
from prompt order. The expected acceptance test is:

- same semantic blocks in same order: original SGLang and OpenClaw both hit;
- same semantic blocks in different order: original SGLang prefix reuse drops;
- patched OpenClaw block KV reuse still hits by semantic `cache_key`.
