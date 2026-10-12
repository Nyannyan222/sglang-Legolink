# Installation

There are two independent installs. Do the first one whatever you plan to do; do the second only if you have a suitable GPU.

| Install | Gives you | Needs |
| --- | --- | --- |
| **A. Host + tests** (CPU) | the NOC Context Manager, the OpenAI-compatible proxy, the example against a fake runtime, the CPU test suite | Python ≥ 3.10 |
| **B. Patched SGLang runtime** (GPU, Linux) | the LegoLink-only server that serves Compile and Link | a CUDA 13 stack, an NVIDIA GPU, network access to upstream SGLang |

## A. Host and tests (CPU)

```bash
git clone https://github.com/Nyannyan222/sglang-Legolink.git
cd sglang-Legolink
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[tests]"                           # pytest + torch (a CPU build is enough)
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests -q   # expected: all pass; the CUDA-only tests skip
python examples/compile_link_decode.py              # expected: exit 0, JSON report
```

* The host core (`noc_context_manager`) uses only the Python standard library. Extras: `pip install -e ".[tokenizers]"` adds `transformers`, `tokenizers`, `jinja2`, needed for tokenizer modes other than `regex`
  (the `sglang` mode asks the server's own `/v1/tokenize`, which is what real runs use). `".[agent]"` adds the real `terminal-bench` package; without it a small labelled compatibility shim is
  installed automatically (`terminal_bench_integration/replay.py`), which is enough for the proxy, the example and the tests.
* The tests never import SGLang. The runtime tests load the shipped runtime *source files* (`patched_sglang_files/`, plus `upstream_support_files/`) by path and run them on CPU tensors.
* Platform notes: FlashInfer is Linux-only, so the FlashInfer repair tests skip elsewhere; the two CUDA tests skip without a GPU. Windows, macOS and Linux all run the CPU suite.

## B. The patched SGLang runtime

### B.1 Requirements

| | |
| --- | --- |
| OS | Linux (x86-64 or aarch64) |
| GPU / driver | NVIDIA GPU; the patched tree pins `flashinfer_python[cu13]==0.6.11.post1`, `cuda-python>=13.0`, i.e. a **CUDA 13** stack and a driver that supports it. The target was an H200 (1 GPU, TP = 1) |
| Python | ≥ 3.10 (reference environment: 3.12.3) |
| Key pins (from the patched tree's own `python/pyproject.toml`) | `torch==2.11.0`, `sglang-kernel==0.4.2.post2`, `flashinfer_python==0.6.11.post1`, `flashinfer_cubin==0.6.11.post1`, `transformers==5.8.1`, `xgrammar==0.2.0`, `triton==3.6.0` (the constraints file also pins `numpy`, `pydantic`, `fastapi`, `uvicorn`, `safetensors`, `tokenizers`, `huggingface_hub`) |
| Tools | `git`, `ninja` (the runtime JIT-compiles kernels at first use; `pip install ninja` or activate the virtualenv that has it) |
| Model | default `Qwen/Qwen2.5-7B-Instruct` (the model file the patch series instruments); weights are **not** part of this repository — they are downloaded from Hugging Face by SGLang or taken from `MODEL_PATH` |
| Memory | 7B bf16 weights ≈ 15 GB plus the KV pool; `MEM_FRACTION` (default 0.80 in `serve_sglang.sh`) is a share of the memory free at launch |

### B.2 Build the runtime from upstream and prove it

```bash
bash scripts/build_patched_sglang.sh ~/legolink-sglang
```

This clones `https://github.com/sgl-project/sglang`, checks out the pinned base commit `cd012ada58ac317dd2e9df4ac87b71c104874e39`, applies the 28 patches in
`external_patches/sglang/` with `git am`, and compares the resulting git tree id with `PATCHED_SGLANG.json`. It exits 2 and tells you not to use the build if they differ; it refuses to overwrite an existing directory.
Set `SGLANG_UPSTREAM=<url-or-path>` to use a mirror or a local clone. It needs no file from the author's machines or cluster.

### B.3 Install it

Install the patched checkout into a CUDA 13 environment, constrained by the pins recorded for the reference environment:

```bash
python -m venv ~/legolink-venv && . ~/legolink-venv/bin/activate
pip install -e ~/legolink-sglang/python -c configs/constraints-runtime.txt
pip install -e ".[tests]"                      # the host packages, in the same environment
```

`configs/constraints-runtime.txt` is a *constraint*, not a requirement list: it only limits what pip may choose. Its header explains the one change from the reference image
(`transformers` 5.8.1, `kernels<0.13`, taken from the patched tree's own metadata). The exact package versions that ran on the author's H200 were **not recorded in this repository**;
`configs/requirements-runtime-reference.txt` is the full `pip freeze` of a *different*, locally validated container image. If `pip` cannot satisfy the pins for your CUDA/driver combination, install
the `torch==2.11.0` build that matches your driver first (see pytorch.org for the wheel index of your CUDA version), then rerun the install.

**Alternative — the reference container.** The audited reference environment is the image `lmsysorg/sglang@sha256:98223b99a77330dc4bc04cbfc8341e55d874ab911a57d2f508da5a7ef5fb8475`
(`lmsysorg/sglang:latest-cu130-runtime`; Python 3.12.3, torch 2.11.0+cu130, CUDA 13.0, flashinfer-python 0.6.11.post1, transformers 5.6.0). It already contains **its own, different SGLang**
(0.5.12 at `/sgl-workspace/sglang`), so putting the patched tree on `PYTHONPATH` is *not* enough — the patched tree needs a few packages the image lacks (importing its RadixCache fails with
`No module named 'distro'`). Install the patched tree inside the container instead:

```bash
docker run --rm --gpus all -it -v ~/legolink-sglang:/patched -v "$PWD":/repo -w /repo   --entrypoint bash lmsysorg/sglang@sha256:98223b99a77330dc4bc04cbfc8341e55d874ab911a57d2f508da5a7ef5fb8475
pip install -e /patched/python -c configs/constraints-runtime.txt     # inside the container
python scripts/verify_runtime.py --sglang-dir /patched
```

When this was checked for the publication (`pip install --dry-run`, resolution only, nothing installed), pip planned to add exactly `distro 1.9.0`, `kernels 0.12.3`, `nvidia-cutlass-dsl-libs-cu13 4.5.0`,
`transformers 5.8.1` and the patched `sglang`, and found every other pin (torch 2.11.0, triton 3.6.0, sglang-kernel 0.4.2.post2, xgrammar 0.2.0, numpy 2.3.5, …) already satisfied. The real install, and a server launch from it, were not executed.

### B.4 Verify the install

```bash
python scripts/verify_runtime.py --sglang-dir ~/legolink-sglang        # tree, files, new_modules, contract
python -c "import sglang, sglang.srt.pic_runtime as p; print(sglang.__file__, p.__file__)"   # must point into the patched checkout
```

`verify_runtime.py` exits 0 only if the tree is byte-equivalent to the pinned runtime. It does not start a GPU process.

### B.5 Launch and check

```bash
bash scripts/serve_sglang.sh patched                                   # LegoLink-only server, port 30000
python scripts/check_server.py --role patched --api-base http://127.0.0.1:30000/v1
```

`check_server.py` passes only if the server answers `/v1/models` **and refuses** a plain chat request — the LegoLink-only contract. For a baseline, launch an unmodified or "native" server with
`bash scripts/serve_sglang.sh native` (port 30001, plain generation, prefix cache on) and check it with `--role native`. Two 7B servers on one GPU need `MEM_FRACTION` values that fit together
(the H200 runs used 0.45 for the patched and 0.35 for the native server, patched first).

If you use a scheduler (for example SLURM), start the scripts inside the allocation that owns the GPU; they never set `CUDA_VISIBLE_DEVICES`.

### Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `build_patched_sglang.sh` exits 2 | the checkout is not the pinned base or a patch applied differently; do not use the build |
| `git checkout` fails on Windows with "Filename too long" | `git config --global core.longpaths true` (the runtime itself is Linux-only; use WSL2 or a Linux host) |
| server dies at start with a `ninja` error | `ninja` is not on `PATH` of the server process; activate the venv or `pip install ninja` |
| `check_server.py --role patched` says the role does not hold | the server was started without the `SGLANG_PIC_*` switches (use `serve_sglang.sh patched`) or is an unpatched SGLang |
| startup warmup fails with HTTP 400 | the LegoLink-only server refuses plain generation; `--skip-server-warmup` (set by `serve_sglang.sh`) is required |
