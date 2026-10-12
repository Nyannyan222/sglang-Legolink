# Validation status

This page keeps three things apart that are easy to blur: **(1)** what the maintainers report from their own H200 runs, **(2)** what was re-checked when this repository was
published, and **(3)** what is not established. Nothing in (1) is reproduced by, or included in, this repository.

## 1. Reported by the maintainers (H200) — not included, not re-run here

* The patched runtime tree `d0c440cfb49a775c52a3eb88437a56c1050e4428` (base `cd012ad` + 28 patches) is the tree the maintainers deployed for their H200 job series (Jobs 802–809, a sequence of
  pilots, comparison suites and a K sweep with a Qwen3-32B model). This repository ships exactly that tree's patch series so it can be rebuilt and fingerprinted.
* The artifacts of those jobs (request rows, runtime traces, summaries) live on the maintainers' cluster and are **not** part of this repository. Their numbers cannot be checked from here.
* The exact Python package versions that ran on the H200 were not recorded here either. `configs/constraints-runtime.txt` pins the versions the patched tree itself declares plus the reference image's pins; confirm with `pip freeze` on your machine.

## 2. Re-checked when this repository was published (2026-10-12)

Everything in this section was executed by the publishing tooling on a CPU-only host (no GPU was started, no model was loaded). "Reference container" is the Linux image
`lmsysorg/sglang@sha256:98223b99…` (Python 3.12.3, torch 2.11.0+cu130) run without a GPU.

| Check | How | Result |
| --- | --- | --- |
| The patch series applies to real upstream and yields the pinned tree | fresh `git clone` of `sgl-project/sglang`, checkout `cd012ad`, `git am` of the 28 patches; also `scripts/build_patched_sglang.sh` in the reference container | tree id `d0c440cfb49a775c52a3eb88437a56c1050e4428` — **equal to the pin**; 27 files changed |
| `patched_sglang_files/` is exactly the patched files | byte comparison against the built tree (line endings normalised) | 27 / 27 identical |
| `upstream_support_files/` is unmodified upstream | byte comparison against `cd012ad` | 5 / 5 identical |
| `scripts/verify_runtime.py` | on the built tree (reference container and Windows) and on `patched_sglang_files/` | `tree`, `files`, `new_modules`, `contract` all pass on the built tree; the host and runtime `pic_link` field lists are equal (36 request fields, 9 runtime-owned) when `pic_runtime.py` is loaded from the built tree |
| Patched modules import on CPU | `PYTHONPATH=<built tree>/python python -c "import sglang.srt.pic_runtime …"` in the reference container | `pic_runtime`, `legolink_triton`, `sparse_repair_compute` import; importing `radix_cache` there fails with `No module named 'distro'` because the image's own SGLang is a different version — see [INSTALLATION.md](INSTALLATION.md) B.3 |
| The documented install resolves | `pip install --dry-run -e <built tree>/python -c configs/constraints-runtime.txt` in the reference container | resolves; would add `distro`, `kernels`, `nvidia-cutlass-dsl-libs-cu13`, `transformers 5.8.1` and the patched `sglang`; nothing was installed |
| CPU test suite | `python -m pytest` (Windows 11, Python 3.13.5, torch 2.11.0+cu130 with CUDA hidden by `CUDA_VISIBLE_DEVICES=-1`) | see the table below |
| CPU test suite, Linux | same, in the reference container | see the table below |
| Example, three transports | `examples/compile_link_decode.py` in-process, `--fake-http`; proxy + fake runtime (in the tests) | exit 0; Compile on first sight, reuse (no Compile) on the next turn, no runtime-owned field on the wire, recovery after a full eviction, no plain-generation request ever sent |
| Hygiene | `tests/test_repository_hygiene.py`: no private path/credential/cluster name, no large or binary artifact, relative links resolve, every repository file the docs name exists | pass |
| The repository as cloned from GitHub | fresh clone of the pushed `main` into a clean directory, install and run as in the README | see the final section |

CPU test suite results (`CUDA_VISIBLE_DEVICES=-1`):

| Environment | Result |
| --- | --- |
| Windows 11, Python 3.13.5 | 777 passed, 2 skipped, 5 subtests passed (about 20 s) |
| Reference container (Linux, Python 3.12.3) | 777 passed, 2 skipped, 5 subtests passed (about 33 s) |

The two skipped tests need a CUDA device (`tests/pic/test_kv_relocation.py`, `tests/pic/test_sparse_attention_reference.py`). The tests exercise the host pipeline, the contract, residency recovery, the proxy,
and the runtime's registry, admission, KV-relocation, RoPE-remap and write-back math by loading the shipped runtime *source files* by path on CPU tensors. They do **not** run the model, the scheduler loop, an
attention kernel on a GPU, or `PICRuntime.run_link` end to end.

## 3. What is not established

* **GPU execution of anything in this repository.** The launch scripts (`serve_sglang.sh`, `check_server.py`), the real-server mode of `examples/compile_link_decode.py`, the install on a CUDA machine, and every runtime
  behaviour that needs a GPU were **not run** for this publication. Treat them as unverified until you run them; the fake-runtime modes only prove the host side of the protocol.
* **Sparse-repair quality and performance.** Job 809 (the K sweep) carries the status `RESULTS_VERIFIED` in the maintainers' tooling. That status is **not** evidence that sparse-repair
  output quality or latency passed, and this repository must not be read that way. The maintainers' own post-hoc analysis of that sweep ([KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)) reports exact-output agreement of about 32 %
  for K = 256 and K = 1024 against a 95 % gate that is never relaxed, a time-to-first-token slower than a cold upstream prefill at every K, and no reuse opportunity on that workload. Those figures are quoted from the analysis, not
  reproduced here.
* **Speed-ups.** No latency or throughput ratio against native SGLang is claimed; none can be derived from the CPU checks.
* **Other models, TP > 1, overlap scheduling, CUDA graphs, long-context behaviour, and the experimental flags** (indexed mixed prefill, group promotion, admit-Link-while-decoding, the FlashInfer/Triton paths).
* **Cross-process residency coordination** and multiple physical RadixCaches (not implemented).

## 4. Reproducing the checks

```bash
bash scripts/build_patched_sglang.sh /tmp/legolink-sglang          # needs git + network; exits 0 only if the tree id is d0c440cf…
python scripts/verify_runtime.py --sglang-dir /tmp/legolink-sglang
python scripts/verify_runtime.py --sglang-dir patched_sglang_files
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q
python examples/compile_link_decode.py && python examples/compile_link_decode.py --fake-http
```
