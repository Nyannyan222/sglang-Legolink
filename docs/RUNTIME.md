# The patched SGLang runtime

The LegoLink runtime is **not** a stock SGLang. It is upstream SGLang plus a series of 28 patches. This repository does not vendor upstream
SGLang; it ships the series, the 27 resulting files (for the CPU tests), and the tools to rebuild and fingerprint the runtime.

## Identity

| | |
| --- | --- |
| Upstream | `https://github.com/sgl-project/sglang` |
| Base commit | `cd012ada58ac317dd2e9df4ac87b71c104874e39` (`cd012ad`) — "[Fix] Fix extra uninstall of cutlass packages (#25756)" |
| Series | `external_patches/sglang/0001-…` … `0028-…` (28 patches, applied in order with `git am`) |
| Patched head (author's checkout) | `7e61b597166b7ec07863dd42e830e77f6d5fd6f8` — patch 0028 |
| **Patched tree id (the fingerprint)** | **`d0c440cfb49a775c52a3eb88437a56c1050e4428`** |
| Scope | 27 files: 23 modified upstream files + 4 new modules (`pic_runtime.py`, `legolink_triton.py`, `flashinfer_sparse_repair.py`, `sparse_repair_compute.py`) |

The git **tree id**, not a commit id, is the fingerprint: a commit id depends on the committer and date of whoever ran `git am`; the tree id depends only on content.
Rebuilding with `scripts/build_patched_sglang.sh` produces a different commit id from the author's but the same tree id.

`PATCHED_SGLANG.json` records the base, the head, the tree id and the SHA-256 of every patch file. `RUNTIME_MANIFEST.json` records the SHA-256 (line endings normalised)
of each of the 27 files in the tree the series produces. `UPSTREAM_SGLANG.json` records the base commit's `python/` tree id and the names of the patched-only modules.

## Checking a runtime

```bash
python scripts/verify_runtime.py --sglang-dir <patched checkout | its python/ dir | a site-packages dir>
```

| Check | Passes when | Applies to |
| --- | --- | --- |
| `tree` | `git rev-parse HEAD^{tree}` equals the pinned tree id | a clean git checkout (otherwise `not_applicable`, never a pass) |
| `files` | all 27 files exist with the recorded SHA-256 | a checkout or an installed copy |
| `new_modules` | the four patched-only modules are present | both |
| `contract` | `pic_runtime.PIC_LINK_REQUEST_FIELDS` / `PIC_LINK_RUNTIME_OWNED_FIELDS` equal the host's `noc_context_manager.pic_link_contract` | needs `torch` (else `skipped`) |

Exit 0 means the tree is byte-equivalent to the runtime this repository was written against. It says nothing about GPU behaviour.
The test suite performs the same comparisons offline: `tests/test_runtime_integrity.py` checks the patch hashes, that `patched_sglang_files/` equals the manifest, and that the contract lists agree.

## Rebuilding

```bash
bash scripts/build_patched_sglang.sh <target-dir>      # git clone upstream → checkout cd012ad → git am the 28 patches → compare tree id
```

It needs `git` and network access (`SGLANG_UPSTREAM=<url or path>` selects a mirror), never overwrites an existing directory, and exits 2 if the tree id differs. When it was
run for this publication against the real upstream repository, the tree id was `d0c440cfb49a775c52a3eb88437a56c1050e4428` and all 27 files matched
`patched_sglang_files/` byte for byte ([VALIDATION.md](VALIDATION.md)).

## What the series adds

* **`pic_runtime.py`** — `PICRuntime.run_link` (RadixCache lookup, KV relocation, RoPE global-position remap, request-private write-back, sparse boundary repair, native-decode handoff); `PICRequestTokenRegistry` (`_prepare_link` rebuilds the authoritative link plan from RadixCache block tokens; `_admit_compile`, `pin_finished_compile`); the `pic_link` request contract (`PIC_LINK_REQUEST_FIELDS`, `PIC_LINK_RUNTIME_OWNED_FIELDS`, `validate_pic_link_request`, `ResolvedLinkPlan`); runtime trace events.
* **`mem_cache/radix_cache.py`** — the block registry: `LegoLinkBlockKey`, `LegoLinkBlockHandle`, `register_/lookup_/acquire_/release_/unregister_legolink_block(s)`, pins and active-link refcounts, eviction integration.
* **`managers/scheduler.py`, `schedule_policy.py`, `io_struct.py`, `scheduler_components/*`** — Link admission (a Link is admitted alone; sources are reserved and pinned before destination KV is allocated), Compile registration at request finish, in-flight Compile records, abort with the structured residency detail.
* **`model_executor/model_runner.py`, `forward_batch_info.py`, `piecewise_cuda_graph_runner.py`** — dispatch to `run_link`, compile capture.
* **`entrypoints/*`** — the `sub_context` and `pic_link` request fields on `/v1/chat/completions`.
* **`layers/radix_attention.py`, `models/qwen2.py`, `layers/logits_processor.py`, `layers/sampler.py`** — sparse-repair attention hooks and row-wise FFN/projection slicing. **`models/qwen2.py` is the only model file instrumented**, which is why other model families are unverified.
* **`legolink_triton.py`, `flashinfer_sparse_repair.py`, `sparse_repair_compute.py`** — optional fused KV-link kernel; FlashInfer repair attention; repair-row batching.
* Patches 0014 and 0016 are **debug instrumentation** (capture of post-remap K and post-repair K/V for selected rows), enabled only by `SGLANG_PIC_DEBUG_*` environment variables; they are part of the fingerprinted tree. Patches 0017–0020 and 0022 belong to an **experimental** "group promotion" path (off: `SGLANG_PIC_EXPERIMENTAL_GROUP_PROMOTION=0`). Patch 0028 hardens the `pic_link` input against malformed requests and Compile races.

The full list of patch subjects is in [`external_patches/sglang/README.md`](../external_patches/sglang/README.md).

## Environment flags that select the architecture

`scripts/serve_sglang.sh patched` sets exactly these (the combination the maintainers launched on the H200):

| Variable | Value | Meaning |
| --- | --- | --- |
| `SGLANG_PIC_LEGOLINK_ONLY` | `1` | serve Compile + Link only; refuse plain generation; a failed `run_link` aborts instead of falling back to native prefill |
| `SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS` | `1` | the RadixCache owns the block registry |
| `SGLANG_PIC_BATCH_BLOCK_KV_LINK` | `1` | batched per-block KV copy/remap |
| `SGLANG_PIC_EXPERIMENTAL_INDEXED_MIXED_PREFILL` | `1` | EXPERIMENTAL indexed cached/fresh prefill plan |
| `SGLANG_PIC_EXPERIMENTAL_GROUP_PROMOTION` | `0` | EXPERIMENTAL group promotion off |
| `SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND` | `native` (default) | `native` \| `flashinfer` \| `auto` \| `hybrid` |
| `SGLANG_PIC_MAX_BLOCKS` | `256` | the scheduler token registry's block cap |
| `SGLANG_PIC_TRACE_PATH` | file | per-request JSONL runtime trace |

Server flags: `--attention-backend triton --sampling-backend pytorch --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size -1 --enable-cache-report --skip-server-warmup` (the LegoLink-only server refuses SGLang's plain-generation startup warmup).
Other `SGLANG_PIC_*` variables are tuning or diagnostics (Triton launch parameters, FlashInfer workspace/caches, host timing, NVTX, CUDA events) and are not needed to run.
