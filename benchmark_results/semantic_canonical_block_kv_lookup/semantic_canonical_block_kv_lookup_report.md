# Semantic / Canonical Block KV Lookup Report

## What This Tests

- block-level KV storage by independent sub-context block metadata,
- exact content lookup by `content_hash`,
- semantic/canonical lookup by `canonical_key`,
- conservative no-reuse behavior for mutating tools.

## Summary

| sequence | relation | rows | tokens | reused | prefill | reuse ratio | hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `canonical_positive_raw_text` | `canonical_semantic` | 2 | 288 | 288 | 0 | 100.0% | 2 |
| `canonical_positive_raw_text` | `miss` | 2 | 288 | 0 | 288 | 0.0% | 0 |
| `tool_aware_exact_replay` | `exact_content` | 36 | 3128 | 3128 | 0 | 100.0% | 36 |
| `tool_aware_exact_replay` | `policy_no_reuse` | 3 | 72 | 0 | 72 | 0.0% | 0 |
| `tool_aware_store` | `exact_content` | 2 | 6 | 6 | 0 | 100.0% | 2 |
| `tool_aware_store` | `miss` | 34 | 3122 | 0 | 3122 | 0.0% | 0 |
| `tool_aware_store` | `policy_no_reuse` | 3 | 72 | 0 | 72 | 0.0% | 0 |

## Canonical Semantic Hits

- Canonical semantic hits: 2
- These are cases where raw text differs but the canonical key matches a stored block.

## Boundary

This is the OpenClaw block-KV policy and storage layer. The SGLang patch hook can log/store the same metadata internally, but reusing actual paged KV tensors inside SGLang still requires scheduler/model-runner integration.
