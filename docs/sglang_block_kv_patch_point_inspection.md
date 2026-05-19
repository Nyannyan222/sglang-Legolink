# SGLang Block-KV Patch Point Inspection

## Purpose

Before modifying SGLang internals, we need to locate the exact upstream source
files and anchors for the installed SGLang version.

This step adds an inspector that maps:

- request metadata fields such as `extra_key` and `custom_labels`,
- prefix-cache lookup logic,
- cache-hit token accounting,
- Radix / KV memory cache modules.

## Run Locally Or On neno5

Local environment without SGLang:

```bash
python scripts/inspect_sglang_block_kv_patch_points.py
```

Installed SGLang venv:

```bash
python scripts/inspect_sglang_block_kv_patch_points.py --require-sglang
```

Checked-out SGLang source tree:

```bash
python scripts/inspect_sglang_block_kv_patch_points.py \
  --source-root /path/to/sglang
```

## Output

```text
benchmark_results/sglang_block_kv_patch_points/
```

The report contains:

- located module paths,
- relevant line hits,
- missing modules or patterns,
- readiness for the internal block-KV patch.

## Why This Is The Next Step

The policy simulation already shows the desired behavior:

```text
permuted_reverse_order:
  prefix reuse = 0.0%
  block-key KV reuse = 100.0%
```

The inspector bridges that policy to SGLang internals by identifying where to
attach the implementation:

1. request metadata ingestion,
2. semantic `cache_key` lookup,
3. block-level KV store/reuse,
4. block-level cached/prefill metrics.

## Boundary

This is not yet the final SGLang internal implementation. It is a version-aware
source inspection step so the next patch edits the correct installed SGLang
files instead of guessing anchors.
