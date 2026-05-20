# Semantic Reuse Review Candidate Judging

- Generated: `2026-05-20T18:46:21+00:00`
- Review candidates: 4
- Promoted positives: 0
- Rejected candidates: 4
- Existing safe positives: 36
- Promoted benchmark rows: 108

## Judgment Counts

| judge_relation | rows |
| --- | --- |
| partial_overlap | 4 |

## Interpretation

The current near-duplicate candidates are not promoted. They share task
templates, output requirements, or problem structure, but their task-specific
facts differ. Under the semantic reuse policy, that is `partial_overlap`, not
`same_answer_utility`.

This is a useful negative result: it shows the pipeline can find candidates
without over-claiming reuse. The next technical improvement is finer
field-internal segmentation so repeated instruction templates and task-specific
parameters can be separated before judging.
