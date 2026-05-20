# WildClaw Prompt Internal Segmentation

- Generated: `2026-05-20T18:45:49+00:00`
- Tasks scanned: 60
- Prompt-internal blocks: 625
- Safe exact prompt-template positive pairs: 3
- Parameterized template review candidates: 7

## Why This Step Matters

The previous near-duplicate whole prompts were rejected because they mixed
shared instruction templates with task-specific facts. This step splits the
`user_prompt` field into smaller semantic blocks before reuse mining.

## Block Types

| block_type | rows |
| --- | --- |
| output_format_template | 20 |
| parameterized_instruction_template | 106 |
| reusable_instruction_template | 34 |
| section_heading_or_short_label | 307 |
| task_specific_parameter | 158 |

## Pair Types

| relation_type | pairs |
| --- | --- |
| exact_instruction_template | 3 |
| parameterized_template_candidate | 7 |

## Interpretation

Exact repeated prompt-internal instruction blocks are safe reusable positives.
Parameterized template matches are not promoted yet: they need a task-specific
delta and manual/LLM judging before they can be used for reuse.
