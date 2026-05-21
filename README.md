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

- `scripts/build_semantic_reuse_candidate_eval.py`
  Converts semantic-similarity candidates into reuse decisions. Embedding
  similarity is treated as candidate discovery only; relation type,
  same-answer utility, judge score, and exact content hashes decide whether a
  pair is actually safe to reuse.

- `scripts/build_semantic_reuse_benchmark.py`
  Builds a positive/negative semantic reuse benchmark. Positive pairs are
  mined from real WildClaw field/function sub-contexts when they are safe
  cross-task duplicates; negative pairs come from strict judge-rejected
  semantic candidates.

- `scripts/mine_wildclaw_reusable_evidence_pairs.py`
  Scans the local WildClawBench task/workspace tree for cross-task reusable
  evidence pairs. Exact duplicate sub-contexts become safe positives; high
  overlap non-identical pairs become manual/LLM review candidates.

- `scripts/judge_semantic_reuse_review_candidates.py`
  Applies the promotion gate for near-duplicate review candidates. Only pairs
  with same-answer utility and a safe reuse label are promoted; template-level
  overlap with different task facts is rejected as partial overlap.

- `scripts/segment_wildclaw_prompt_internal_blocks.py`
  Splits WildClaw `user_prompt` fields into finer prompt-internal semantic
  blocks. Exact repeated instruction blocks become safe positives, while
  parameterized template matches are exported for judge/manual review.

- `scripts/prepare_canonical_text_sglang_replay.py`
  Builds an SGLang replay manifest from `canonical_text` fields. The manifest
  contains raw parameterized blocks, repeated canonical templates, and repeated
  exact reusable canonical blocks for cached-token and latency measurement.

- `scripts/prepare_canonical_semantic_positive_eval.py`
  Builds a three-way comparison manifest for semantic-positive pairs: original
  raw text, SGLang exact repeat, and OpenClaw canonical text.

- `scripts/evaluate_canonical_semantic_correctness.py`
  Validates canonicalization correctness by checking answer equivalence,
  evidence preservation, no hallucinated facts, and a rule-based judge score.

- `scripts/prepare_field_function_subcontexts.py`
  Builds field/function-first sub-contexts from WildClaw/OpenClaw context
  fields such as user prompt, tool definitions, tool history output,
  conversation history, workspace evidence, and task metadata. Overlong fields
  are split internally, and each row carries SGLang-oriented metadata for
  independent prefill/cache/reuse.

- `scripts/llm_trace_field_segmenter.py`
  Uses an OpenAI-compatible LLM to classify a complete WildClaw/OpenClaw agent
  trace into field/function sub-contexts first: user prompt, tool definitions,
  tool history output, conversation history, workspace evidence, and task
  metadata. Offline auto mode falls back to deterministic marked-block
  classification so the output schema and validation can still run locally.

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

- `scripts/summarize_openclaw_method_effect.py`
  Produces the current evidence summary that separates proven SGLang runtime
  behavior from the still-simulated block-key KV reuse policy.

- `scripts/patch_sglang_openclaw_request_metadata.py`
  Patches SGLang OpenAI request parsing so body-level OpenClaw block metadata
  is preserved in `sampling_params.custom_params.openclaw_metadata` without
  polluting SGLang metrics `custom_labels`.

- `docs/openclaw_sglang_request_metadata_patch_result.md`
  Records the local SGLang Docker patch and verification result for OpenClaw
  block metadata request parsing and scheduler cache-lookup logging.

- `scripts/simulate_sglang_block_kv_reuse.py`
  Simulates the target block-level KV reuse policy: reused by semantic
  `cache_key` even when sub-context blocks appear in a different order.

- `scripts/inspect_sglang_block_kv_patch_points.py`
  Runs inside an installed SGLang environment or source tree and maps the exact
  scheduler, prefix-cache, and KV-cache files/anchors needed for the internal
  block-KV patch.

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

- `docs/llm_trace_field_segmentation.md`
  Documents the complete-trace LLM field segmentation step: trace blocks are
  classified into user-prompt/tool-definition/tool-history/conversation/
  workspace/metadata fields before any length-based internal splitting.

- `docs/block_isolated_runtime_prototype.md`
  Runtime-facing proof plan for independent sub-context prefill/cache and
  block-isolated attention before patching SGLang internals.

- `docs/sglang_openclaw_block_metadata_patch.md`
  Instructions for patching SGLang cache-lookup logs so OpenClaw block metadata
  can be verified at the runtime request/cache lookup path.

- `docs/sglang_block_kv_reuse_policy.md`
  Defines the order-invariant semantic block KV reuse policy and reports the
  local order-permutation simulation result.

- `docs/sglang_block_kv_patch_point_inspection.md`
  Explains how to inspect an installed SGLang source tree before applying the
  internal semantic block-KV cache patch.

- `docs/local_sglang_source_install.md`
  Records the local SGLang source checkout path, current revision, and Windows
  runtime boundary.

- `docs/local_sglang_docker_runtime_result.md`
  Records the local WSL2/Docker SGLang runtime startup and block-prefill replay
  result on the RTX 5070 machine.

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

- `benchmark_results/semantic_reuse_candidate_eval/`
  Reuse-candidate evaluation over the strict semantic-similarity outputs. The
  current result rejects all 111 candidate pairs as not safe to reuse, because
  they are only partial-overlap, broad-topic, or unrelated matches.

- `benchmark_results/semantic_reuse_benchmark/`
  Positive/negative semantic reuse benchmark. The current local build contains
  3 safe positive exact-duplicate pairs from WildClaw tool-definition
  sub-contexts and 9 selected negative pairs balanced across broad-topic,
  partial-overlap, and unrelated relations.

- `benchmark_results/wildclaw_reusable_evidence_mining/`
  Expanded local WildClawBench scan over 60 tasks. The current output contains
  36 safe exact positive pairs, 4 near-duplicate review candidates, and a
  108-row positive/negative semantic reuse benchmark.

- `benchmark_results/semantic_reuse_candidate_judging/`
  Judging outputs for the near-duplicate review candidates. The current result
  rejects all 4 candidates as partial-overlap template matches and promotes no
  new positives, preserving the 36 safe exact positives.

- `benchmark_results/wildclaw_prompt_internal_segmentation/`
  Finer segmentation of WildClaw prompt fields. The current run scans 60 tasks,
  builds 625 prompt-internal blocks, finds 3 safe exact instruction-template
  positive pairs, and exports 7 parameterized template review candidates.

- `benchmark_results/canonical_text_sglang_replay/`
  Canonical-text SGLang replay manifest and runtime result. The successful
  local run has 20 rows, 0 errors, and 98-99% cached ratio on repeated
  canonical text conditions.

- `benchmark_results/canonical_semantic_positive_eval/`
  Semantic-positive canonicalization evaluation over two different-raw-text
  pairs. The successful local SGLang run has 12 rows, 0 errors, 99.5% cached
  ratio for OpenClaw canonical repeat, and 4/4 correctness judge scores for
  both semantic pairs.

- `benchmark_results/wildclaw_llm_trace_field_segmentation/`
  Complete-trace field segmentation smoke output. The current local run uses
  the offline fallback because no OpenAI/OpenRouter key is set; the same script
  runs true LLM classification when an API key is provided.

- `benchmark_results/wildclaw_llm_trace_field_segmentation_validation/`
  Validation report for the trace-field sub-context contract. The current
  27-row local smoke output has 0 failed checks.

- `benchmark_results/wildclaw_llm_trace_field_segmentation_openai/`
  OpenAI-backed LLM trace field segmentation over 3 WildClaw tasks using
  `gpt-4.1-mini`. The run produced 12 field/function sub-context rows with
  `llm_trace_field_segmentation` method.

- `benchmark_results/wildclaw_llm_trace_field_segmentation_openai_validation/`
  Validation report for the OpenAI-backed trace-field output. The current
  12-row LLM output has 0 failed checks and is SGLang-metadata-ready.

- `benchmark_results/wildclaw_similarity_task_selection/`
  Candidate task/workspace selection results for future similarity pilots.

- `benchmark_results/sglang_block_runtime_plan/`
  SGLang-facing block runtime plan, prefill request payloads, attention ranges,
  and scheduler/KV-cache contract for the block-isolation integration step.

- `benchmark_results/sglang_block_kv_reuse_simulation/`
  Order-permutation simulation comparing native prefix-style reuse with the
  intended semantic block-key KV reuse policy.

- `benchmark_results/openclaw_method_effect_validation/`
  Current validation report explaining what is proven, what remains simulated,
  and what SGLang runtime patch is needed next.

- `benchmark_results/sglang_block_kv_patch_points/`
  Source inspection report for locating SGLang scheduler, prefix-cache, and
  KV-cache patch points.

## Reuse Relation Taxonomy

The manager separates retrieval relevance from reuse eligibility:

Each reusable or reviewable sub-context now carries:

- `canonical_text`: the normalized text used for semantic reuse grouping. For
  exact positives this is the normalized instruction/evidence text; for
  parameterized review candidates it is the placeholder-normalized template.
- `canonical_key`: a stable key derived from the context field, functional
  role, reuse kind, and canonical text hash. This is the intended lookup key
  for later registry/KV-cache experiments.

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
