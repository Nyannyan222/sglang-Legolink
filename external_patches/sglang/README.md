# SGLang LegoLink / PICRuntime patch series

Upstream SGLang is **not** vendored in this repository. This directory is the authoritative, version-controlled copy of every change made to it, so the
runtime can be reconstructed from this repository alone. Background, fingerprint and checks: [docs/RUNTIME.md](../../docs/RUNTIME.md).

## Upstream base

| field | value |
| --- | --- |
| upstream repo | `https://github.com/sgl-project/sglang` |
| base commit | `cd012ada58ac317dd2e9df4ac87b71c104874e39` (`cd012ad`) — `[Fix] Fix extra uninstall of cutlass packages (#25756)` |
| branch used by the author | `noc/legolink-pic-runtime` |
| patches | ordered series `0001` … `0028` (28 files; the order is the file name order) |
| scope | 27 files: 23 modified upstream files, 4 new modules |
| result | git tree id `d0c440cfb49a775c52a3eb88437a56c1050e4428` |

## Applying

The supported way is the script, which also proves the result:

```bash
bash scripts/build_patched_sglang.sh <target-dir>
```

By hand, from the repository root:

```bash
git clone https://github.com/sgl-project/sglang <target-dir>
cd <target-dir>
git checkout cd012ada58ac317dd2e9df4ac87b71c104874e39
git checkout -b noc/legolink-pic-runtime
git -c user.name=build -c user.email=build@localhost.invalid am <this-repo>/external_patches/sglang/*.patch
git rev-parse 'HEAD^{tree}'      # must print d0c440cfb49a775c52a3eb88437a56c1050e4428
```

`git apply` also works if you only want the working-tree changes, but then there is no tree id to compare; use `scripts/verify_runtime.py` to compare file hashes instead.
On Windows, run `git config core.longpaths true` before checking out upstream (some upstream paths are long); the runtime itself only runs on Linux.

## Patches

| # | subject |
| ---: | --- |
| 0001 | NOC: semantic sub-context KV reuse and position-independent KV linking |
| 0002 | Treat FlashInfer workspace overflow as capacity, not correctness |
| 0003 | Carry capacity degradation across execution units of one request |
| 0004 | Stop evaluating disabled trace arguments in the KV-link path |
| 0005 | Suppress overlap for PIC batches, not for the deployment |
| 0006 | Do not fence a PIC batch when nothing is in flight |
| 0007 | Emit the decode span around CUDA graph replay too |
| 0008 | Hold the trace descriptor open instead of reopening per event |
| 0009 | Warm a LegoLink-only server with the requests it accepts |
| 0010 | Warm the shape the traffic will actually send |
| 0011 | Ask the warmup for eight tokens so decode runs |
| 0012 | Let the host say what to reuse, not how to lay it out |
| 0013 | Govern formal PIC framing and repair decisions before execution |
| 0014 | DEBUG: capture post-remap cached K before sparse repair |
| 0015 | Remap a linked block by its position offset, not its ordinal |
| 0016 | DEBUG: capture post-repair cached K and V for selected rows |
| 0017 | EXPERIMENTAL: promote a fully reconstructed execution group into RadixCache |
| 0018 | Declare a promoted group's compatibility the way a compiled block does |
| 0019 | Give a promoted group the reference snapshot the link path reads |
| 0020 | fix: make LegoLink group duplicate registration fail safe |
| 0021 | fix: keep the dataclass decorator on LegoLinkBlockKey |
| 0022 | runtime: let group promotion participate in cache eviction |
| 0023 | runtime: support indexed cached-fresh prefill plan |
| 0024 | fix: tie the Link reservation marker to its pins; fail closed on retraction |
| 0025 | experiment: default-off flag to admit a Link while other requests decode |
| 0026 | fix: keep same-key Compiles out of one prefill batch |
| 0027 | fix: name the missing source blocks when a Link preflight fails |
| 0028 | security: contain malformed PIC requests and compile races |

0014 and 0016 are debug instrumentation gated by `SGLANG_PIC_DEBUG_*`; 0017–0020 and 0022 are the experimental group-promotion path (off by default).
Each patch file keeps its original author and message; the SHA-256 of every file is in `PATCHED_SGLANG.json`.

## Modified and new files, by layer

New modules under `python/sglang/srt/`: `pic_runtime.py`, `legolink_triton.py`, `flashinfer_sparse_repair.py`, `sparse_repair_compute.py`.

| layer | files |
| --- | --- |
| RadixCache + LegoLink registry | `mem_cache/radix_cache.py`, `mem_cache/base_prefix_cache.py` |
| Scheduler admission and block lifetime | `managers/scheduler.py`, `managers/schedule_policy.py`, `managers/io_struct.py`, `managers/scheduler_components/batch_result_processor.py`, `managers/tokenizer_control_mixin.py` |
| PIC dispatch and instrumentation | `model_executor/model_runner.py`, `model_executor/forward_batch_info.py`, `model_executor/piecewise_cuda_graph_runner.py` |
| Request API | `entrypoints/openai/protocol.py`, `serving_base.py`, `serving_chat.py`, `serving_completions.py`, `entrypoints/http_server.py`, `entrypoints/grpc_server.py` |
| Sparse repair execution | `layers/radix_attention.py`, `models/qwen2.py`, `layers/logits_processor.py`, `layers/sampler.py` |
| Profiling | `utils/nvtx_pytorch_hooks.py`, `utils/profile_utils.py`, `managers/scheduler_components/profiler_manager.py` |

The complete, verbatim list of the 27 resulting files is `patched_sglang_files/`; their hashes are in `RUNTIME_MANIFEST.json`.
