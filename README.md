# OpenClaw Semantic Sub-context Prototype

This repository focuses on WildClawBench real-context semantic sub-context
management for the NOC project.

The current direction is to extract task-relevant evidence from real
WildClawBench task workspaces, segment it into semantic sub-contexts, find
semantically similar reusable candidates, and separate request relevance from
safe reuse eligibility.

## Current Goals

- Define semantic sub-contexts from real WildClawBench task prompts and
  workspace files.
- Identify reusable evidence candidates with embedding similarity, LLM judging,
  summary similarity, and manual review.
- Separate "relevant to the request" from "safe to reuse".
- Avoid reuse for broad topical similarity or partial overlap unless the reuse
  decision is explicitly supported.
- Compare semantic sub-contexts with fixed-size and file-based chunking on
  independence, relevance, completeness, redundancy, and context reduction.

## Core Components

- `noc_context_manager/`
  Prototype package for the NOC semantic sub-context manager. It includes the
  sub-context schema, registry, request-time selector, similarity utilities,
  and conservative reuse-decision engine.

- `scripts/run_noc_subcontext_manager.py`
  Loads WildClaw semantic JSONL artifacts into the NOC manager and exports
  selected contexts plus reuse decisions.

- `scripts/real_context_extractor.py`
  Builds leakage-checked long-context JSONL rows from WildClawBench task
  prompts and workspace files.

- `scripts/llm_semantic_segmenter.py`
  Extracts LLM semantic sub-contexts and writes fixed-size/file-based baselines
  plus comparison tables.

- `scripts/find_semantic_similar_subcontexts.py`
  Finds genuinely semantically similar WildClaw sub-context pairs/groups.
  Lexical overlap is only a prefilter; embeddings and optional LLM judging
  provide the semantic signal when `OPENAI_API_KEY` is available.

- `scripts/prepare_field_function_subcontexts.py`
  Builds field/function-first sub-contexts from WildClaw/OpenClaw context
  fields such as user prompt, tool definitions, tool history output,
  conversation history, workspace evidence, and task metadata. Overlong fields
  are split internally, and each row carries SGLang-oriented metadata for
  independent prefill/cache/reuse.

- `scripts/run_block_isolated_subcontext_prototype.py`
  Runs a runtime-facing prototype where each field/function sub-context is an
  independent prefill/cache unit and the simulated attention contract allows
  zero pairwise attention edges between different sub-context blocks.

- `scripts/prepare_sglang_block_runtime_plan.py`
  Converts field/function sub-context blocks into SGLang-facing prefill request
  payloads, cache keys, attention ranges, and a scheduler/KV-cache contract.

- `scripts/run_sglang_block_prefill_payloads.py`
  Sends generated block-prefill payloads to an SGLang OpenAI-compatible
  endpoint and records per-block cached/prefill token observations.

- `scripts/verify_sglang_openclaw_block_logs.py`
  Parses patched SGLang cache-lookup logs and checks whether OpenClaw block
  metadata reached the runtime cache lookup path.

- `scripts/select_wildclaw_similarity_tasks.py`
  Ranks WildClawBench tasks/workspaces by how likely they are to contain
  repeated or near-repeated real-context evidence.

- `scripts/wildclaw_candidate_evidence_extractor.py`
  Builds candidate-evidence contexts before semantic segmentation.

- `scripts/prepare_wildclaw_next_experiment.py`
  Converts reviewed semantic outputs into a framework-evaluation manifest.

- `scripts/prepare_wildclaw_mixed_category_eval.py`
  Builds a mixed-category WildClawBench manifest across Productivity, Search
  Retrieval, and Safety tasks.

- `scripts/run_wildclaw_framework_eval.py`
  Runs or stages framework-evaluation prompts and records answers, retrieval
  evidence, token estimates, and result logs.

- `scripts/run_wildclaw_sglang_runtime_replay.py`
  Replays WildClaw prompts through an SGLang OpenAI-compatible endpoint and
  records cached-token, prefill-token estimate, and latency metrics.

- `scripts/summarize_wildclaw_runtime_results.py`
  Summarizes WildClaw runtime replay CSV files into report-ready Markdown.

- `scripts/start_local_sglang_docker.ps1`
  Starts a local Docker Desktop SGLang runtime for machines that can run the
  required GPU container.

- `scripts/slurm_setup_env.sh`
  One-time neno5 setup job for the SGLang runtime environment.

- `scripts/slurm_run_wildclaw_runtime_replay.sh`
  Runs WildClaw runtime replay on neno5 through SLURM.

## Key Docs

- `docs/noc_semantic_subcontext_manager.md`
  Step-1 design notes for turning the WildClaw semantic sub-context pipeline
  into a formal NOC context-management component.

- `docs/wildclaw_semantic_subcontext_weekly_report.md`
  Weekly report centered on defining sub-contexts, finding reusable candidates,
  and avoiding unsafe reuse.

- `docs/wildclaw_semantic_subcontext_pilot.md`
  First pilot plan and output summary for WildClawBench real-context semantic
  sub-context extraction.

- `docs/semantic_subcontext_similarity_discovery.md`
  Current method for finding semantically similar WildClawBench sub-context
  pairs/groups.

- `docs/field_function_semantic_subcontexts.md`
  Design notes for splitting context by field/function first and targeting
  SGLang sub-context blocks with no pairwise attention.

- `docs/block_isolated_runtime_prototype.md`
  Runtime-facing proof plan for independent sub-context prefill/cache and
  block-isolated attention before patching SGLang internals.

- `docs/sglang_openclaw_block_metadata_patch.md`
  Instructions for patching SGLang cache-lookup logs so OpenClaw block metadata
  can be verified at the runtime request/cache lookup path.

- `docs/wildclaw_similarity_task_selector.md`
  Usage notes and current selector result for choosing better WildClawBench
  semantic-similarity source tasks.

- `docs/wildclaw_mixed_category_eval.md`
  Mixed-category evaluation notes for expanding beyond the first Search
  Retrieval pilot.

- `docs/wildclaw_retrieval_execution_report.md`
  Report-ready summary of the Search Retrieval pilot after adding retrieval
  execution.

- `docs/local_rtx5070_neno5_execution_plan.md`
  Execution split for local RTX 5070 work and neno5 GPU jobs.

- `docs/neno5_runtime_result_summary.md`
  Notes from the current neno5 WildClaw runtime replay.

## Key Artifacts

- `benchmark_results/noc_subcontext_manager_smoke/`
  Smoke-test outputs from the NOC semantic sub-context manager.

- `benchmark_results/wildclaw_semantic_subcontext_pilot_v3/`
  Reviewed WildClaw semantic sub-context pilot artifacts.

- `benchmark_results/wildclaw_mixed_category_eval/`
  Mixed-category WildClaw preparation artifacts.

- `benchmark_results/semantic_similarity_discovery_openai_strict/`
  Strict semantic-similarity discovery outputs using OpenAI-backed signals.

- `benchmark_results/semantic_similarity_discovery_mixed_strict/`
  Strict semantic-similarity discovery outputs over mixed-category data.

- `benchmark_results/wildclaw_similarity_task_selection/`
  Candidate task/workspace selection results for future similarity pilots.

- `benchmark_results/sglang_block_runtime_plan/`
  SGLang-facing block runtime plan, prefill request payloads, attention ranges,
  and scheduler/KV-cache contract for the block-isolation integration step.

## Reuse Relation Taxonomy

The manager separates retrieval relevance from reuse eligibility:

| relation type | meaning | reusable |
| --- | --- | --- |
| `exact_duplicate` | Content is effectively identical. | yes |
| `near_duplicate` | Wording differs, but the information is equivalent. | yes, after judge |
| `same_answer_utility` | Both contexts can answer the same question. | yes |
| `partial_overlap` | Some information overlaps, but each side has unique details. | no / maybe |
| `broad_topic` | Topic is similar, but intended use differs. | no |
| `unrelated` | No useful semantic relationship. | no |

Embedding similarity finds candidates. LLM or manual judging separates reusable
evidence from context that is merely related.

## Quick Smoke Check

```powershell
.\.venv\Scripts\python.exe -B scripts\run_noc_subcontext_manager.py `
  --input benchmark_results\wildclaw_semantic_subcontext_pilot_v3\wildclaw_semantic_subcontext_pilot.jsonl `
  --output-dir benchmark_results\noc_subcontext_manager_smoke
```

The output directory contains selected sub-contexts, reuse decisions, registry
summary, and a short report.
