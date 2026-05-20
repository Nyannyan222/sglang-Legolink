# SGLang Block KV Reuse Simulation

## Purpose

This compares native prefix-style reuse with the intended OpenClaw
block-key KV reuse policy. The key test is the permuted sequence:
the same semantic blocks appear in a different order.

## Result

| sequence | blocks | tokens | prefix reuse | block KV reuse | block KV prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 8 | 23712 | 0.0% | 0.0% | 23712 |
| repeat_same_order | 8 | 23712 | 100.0% | 100.0% | 0 |
| permuted_reverse_order | 8 | 23712 | 0.0% | 100.0% | 0 |

## Interpretation

- `repeat_same_order` should be easy for ordinary prefix cache.
- `permuted_reverse_order` is the important case: prefix reuse drops
  because order changed, while block-key KV reuse should still hit by
  `cache_key`.
- This is a policy simulation, not an SGLang internal patch yet.
