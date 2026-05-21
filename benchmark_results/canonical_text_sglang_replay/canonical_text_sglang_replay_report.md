# Canonical Text SGLang Replay

- Manifest: `benchmark_results/canonical_text_sglang_replay/canonical_text_sglang_runtime_manifest.jsonl`
- Manifest rows: 20
- Prompt directory: `benchmark_results/canonical_text_sglang_replay/prompts/`
- Successful runtime run: `benchmark_results/canonical_text_sglang_replay/runtime_runs/wildclaw_sglang_runtime_20260521_155137/`
- Runtime rows: 20
- Runtime errors: 0

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

## Runtime Result

After Docker Desktop and the local SGLang container were started, the canonical
text replay completed successfully with 20/20 completed rows.

Key observations from
`benchmark_results/canonical_text_sglang_replay/canonical_text_sglang_runtime_summary.md`:

| condition | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- |
| `canonical_text` | 3 | 66 | 57.7 | 8.3 | 87.4% | 0.1869 |
| `canonical_text_repeat` | 3 | 66 | 65 | 1 | 98.5% | 0.1740 |
| `exact_canonical_text` | 3 | 108.7 | 47.7 | 61 | 43.9% | 0.1110 |
| `exact_canonical_text_repeat` | 3 | 108.7 | 107.7 | 1 | 99.1% | 0.1412 |
| `block_canonical_text` | 1 | 63 | 45 | 18 | 71.4% | 0.1633 |
| `block_canonical_text_repeat` | 1 | 63 | 62 | 1 | 98.4% | 0.1643 |

The repeat rows show that identical canonical text can be reused by SGLang's
native prefix cache. This still does not implement semantic non-prefix KV reuse;
it validates the canonical-text replay path and confirms that canonicalization
can create stable prefix-cacheable inputs.

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
