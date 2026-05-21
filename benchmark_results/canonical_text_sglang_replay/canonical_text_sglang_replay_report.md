# Canonical Text SGLang Replay

- Manifest: `benchmark_results/canonical_text_sglang_replay/canonical_text_sglang_runtime_manifest.jsonl`
- Manifest rows: 20
- Prompt directory: `benchmark_results/canonical_text_sglang_replay/prompts/`
- Runtime attempt: `benchmark_results/canonical_text_sglang_replay/runtime_runs/wildclaw_sglang_runtime_20260521_152152/`
- Runtime rows: 20
- Runtime errors: 20

## Purpose

This replay uses `canonical_text` rather than full task prompts. It is designed
to measure ordinary SGLang prefix-cache behavior on canonicalized semantic
sub-contexts:

- `raw_parameterized_left/right`: original task-specific prompt-internal blocks.
- `canonical_text` and `canonical_text_repeat`: repeated canonical template text.
- `exact_canonical_text` and `exact_canonical_text_repeat`: repeated exact reusable instruction text.
- `block_canonical_text` and `block_canonical_text_repeat`: repeated canonical tool-definition block text.

If SGLang is running, the repeat rows should show higher `cached_tokens` and
lower estimated prefill tokens for identical canonical prompts.

## Current Runtime Status

The manifest was generated successfully, but local runtime measurement could not
complete because no SGLang endpoint was available:

- `http://127.0.0.1:8000/v1/models`: connection refused.
- `http://127.0.0.1:30000/v1/models`: connection refused.
- Docker Desktop engine was unavailable: `dockerDesktopLinuxEngine` pipe not found.

Therefore the current CSV records the failed runtime attempt only. It does not
contain valid cached-token or latency measurements yet.

## Run Commands

Start local SGLang after Docker Desktop is running:

```powershell
.\scripts\start_local_sglang_docker.ps1 `
  -Model Qwen/Qwen2.5-0.5B-Instruct `
  -Port 30000 `
  -MemFractionStatic 0.35 `
  -ReadyChecks 120
```

Run canonical replay:

```powershell
.\.venv\Scripts\python.exe -B scripts\run_wildclaw_sglang_runtime_replay.py `
  --manifest benchmark_results\canonical_text_sglang_replay\canonical_text_sglang_runtime_manifest.jsonl `
  --base-url http://127.0.0.1:30000/v1 `
  --metrics-url http://127.0.0.1:30000/metrics `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --repeat 1 `
  --max-tokens 16 `
  --timeout 120 `
  --metrics-timeout 5 `
  --output-dir benchmark_results\canonical_text_sglang_replay\runtime_runs
```

Summarize the newest run:

```powershell
$run = (Get-ChildItem benchmark_results\canonical_text_sglang_replay\runtime_runs -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).FullName

.\.venv\Scripts\python.exe -B scripts\summarize_wildclaw_runtime_results.py `
  (Join-Path $run 'wildclaw_sglang_runtime_results.csv') `
  --output benchmark_results\canonical_text_sglang_replay\canonical_text_sglang_runtime_summary.md
```
