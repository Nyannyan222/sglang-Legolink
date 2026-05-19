# Local SGLang Source Install

## Installed Source

SGLang source is cloned locally at:

```text
external/sglang
```

Current local source revision:

```text
cd012ad [Fix] Fix extra uninstall of cutlass packages (#25756)
```

The directory is ignored by Git via:

```text
external/sglang/
```

This keeps the upstream SGLang source available for inspection and patch
development without vendoring thousands of upstream files into this project.

## Source Inspection

The local source tree works with the block-KV patch-point inspector:

```powershell
.\.venv\Scripts\python.exe -B scripts\inspect_sglang_block_kv_patch_points.py `
  --source-root external\sglang `
  --require-sglang `
  --output-dir benchmark_results\sglang_block_kv_patch_points_local_source
```

Current result:

```text
Ready for block-KV patch: True
Missing roles: none
```

## Windows Runtime Boundary

Importing `sglang` directly in the native Windows `.venv` is not expected to
work cleanly because SGLang imports Linux runtime modules such as `resource`.

For real SGLang serving, use:

- WSL2 + Docker Desktop SGLang container, or
- neno5 SGLang runtime venv.

For local source reading and patch-point inspection, `external/sglang` is enough.
