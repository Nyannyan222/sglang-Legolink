# WildClaw Reusable Evidence Pair Mining

- Generated: `2026-05-20T18:45:49+00:00`
- Tasks scanned: 60
- Blocks built: 291
- Safe exact positive pairs: 36
- Near-duplicate review candidates: 4
- Benchmark rows: 108

## What Changed

This step expands positive mining from the small pilot artifacts to the local
WildClawBench task/workspace tree. Exact cross-task duplicate sub-contexts are
safe positive pairs. High-overlap non-identical pairs are kept as review
candidates rather than being treated as reusable.

## Positive Field Counts

| context_field | positive_pairs |
| --- | --- |
| tool_definitions | 36 |

## Benchmark Relation Counts

| relation_type | pairs |
| --- | --- |
| broad_topic | 19 |
| exact_duplicate | 36 |
| partial_overlap | 27 |
| unrelated | 26 |

## Interpretation

The current safe positives are still mostly reusable interface/context blocks,
especially repeated tool definitions. This is valid but not yet enough for the
final claim. The useful next step is to run LLM or manual judging over the
near-duplicate review candidates and promote only pairs with same-answer utility
into the positive set.
