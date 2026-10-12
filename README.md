# LegoLink — semantic sub-context KV reuse for agents on a patched SGLang

LegoLink lets an LLM agent reuse the KV cache of *parts* of its context, not only of an identical prefix.
A host-side manager (**NOC Context Manager**) cuts an agent's context into byte-exact *semantic sub-contexts*
(task instruction, tool schema, one conversation turn, one tool output, …), gives each one a content-derived cache key, and
decides which of them to **compile** (prefill once, register the KV under that key) and which to **reuse**. A patched
**SGLang** runtime (RadixCache block registry + **PICRuntime / LegoLink**) then **links** the registered blocks into a new request in any
order — copying their KV, re-rotating RoPE to the new global positions, recomputing only a sparse set of boundary positions — and
continues with ordinary **native decode**.

```
Agent / OpenAI-compatible request
  → NOC Context Manager          (host, CPU, Python)
      → semantic field extraction → semantic sub-context segmentation (+ validation, exact reconstruction)
      → sub-context identity / cache key → context selection
      → compile / reuse policy → link-layout feasibility → request plan
  → SGLang adapter               (HTTP: Compile requests and Link requests)
  → patched SGLang scheduler     (runtime derives the physical plan itself)
      → RadixCache / KV block registry
      → PICRuntime / LegoLink    (KV lookup → link → RoPE position remap → sparse boundary repair)
      → native decode → model output
```

**Who decides what.** The host decides *what* is reused: which blocks, in which order, with which repair-window sizes. The runtime decides *how*:
KV slots, block ranges, global positions, `imp_indices`, `kvlink`, `mutable_start_token`, RoPE deltas. These runtime-owned fields are never sent by the
host — the host refuses to emit them and the runtime refuses to accept them ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

> **Research code, honestly labelled.** This repository is the executable architecture of a research project, published so that others can run
> and examine it. It is not a production serving library. [docs/VALIDATION.md](docs/VALIDATION.md) separates what the maintainers report from
> H200 runs, what was re-verified when this repository was published, and what is *not* established — in particular, **sparse-repair output
> quality and latency benefit are not claimed**.

## What is in the repository

| Path | What it is |
| --- | --- |
| `noc_context_manager/` | The host pipeline (pure Python, standard library for the core): events → extraction → segmentation → sub-context store → selection → namespaces → compile policy → layout → planner → stitcher → framer → SGLang adapter, residency recovery |
| `terminal_bench_integration/` | Agent glue: the agent LLM object (`NOCSGLangLLM`), an **OpenAI-compatible proxy** that keeps any OpenAI-style agent on the LegoLink path, session registry, replay driver |
| `external_patches/sglang/` | The **28-patch series** that turns upstream SGLang `cd012ad` into the runtime (`git am`) |
| `patched_sglang_files/` | The 27 runtime files the series adds or changes, verbatim (used by the CPU tests; **not** a runnable SGLang) |
| `upstream_support_files/` | Five *unmodified* upstream files some runtime tests read, verbatim |
| `PATCHED_SGLANG.json`, `UPSTREAM_SGLANG.json`, `RUNTIME_MANIFEST.json` | Runtime identity: upstream repo + base commit, patched tree id `d0c440cfb49a775c52a3eb88437a56c1050e4428`, per-patch and per-file SHA-256 |
| `scripts/` | `build_patched_sglang.sh` (rebuild + prove the runtime), `verify_runtime.py` (fingerprint/compatibility check), `serve_sglang.sh` (launch patched/native server), `check_server.py` (role check), `run_openai_proxy.py` |
| `examples/` | `compile_link_decode.py` — minimal Compile → Reuse/Link → Decode walk-through (fake runtime, fake HTTP runtime, or a real server) |
| `tests/` | CPU correctness tests: host pipeline, contract, residency recovery, and the runtime's registry/link math loaded from the shipped runtime files |
| `schemas/`, `configs/`, `docs/` | Wire/plan JSON schemas, pinned dependency constraints, documentation |

## Requirements

| | CPU-only use (host code, tests, example) | Running the patched runtime on a GPU |
| --- | --- | --- |
| OS | any (Windows, macOS, Linux) | Linux |
| Python | ≥ 3.10 (CPU tests run on 3.13; reference runtime environment is 3.12.3) | ≥ 3.10 (the patched SGLang declares `>=3.10`; reference: 3.12.3) |
| Packages | `pytest`, `torch` (CPU build is enough) for the tests; `transformers`, `tokenizers`, `jinja2` only for tokenizer modes other than `regex` | the patched SGLang's own pins: `torch==2.11.0`, `flashinfer-python[cu13]==0.6.11.post1`, `sglang-kernel==0.4.2.post2`, `transformers==5.8.1`, `triton==3.6.0` (see `configs/constraints-runtime.txt`) |
| GPU | none | NVIDIA GPU with a **CUDA 13** stack (the pinned FlashInfer extra is `cu13`); an H200 was the target. A 7B model in bf16 needs ≳ 16 GB for weights plus KV pool |
| Model | none | `Qwen/Qwen2.5-7B-Instruct` is the default: the patch series instruments exactly one model file, `models/qwen2.py`. Other models (e.g. `Qwen/Qwen3-32B`, which the maintainers report using on the H200) are **not** claimed to work here ([docs/VALIDATION.md](docs/VALIDATION.md)) |

## Quick start (CPU, 1 minute)

```bash
git clone https://github.com/Nyannyan222/sglang-Legolink.git && cd sglang-Legolink
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[tests]"
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests -q     # host pipeline + runtime registry/link math, no GPU
python examples/compile_link_decode.py                # Compile → Link → Decode against a strict fake runtime
python scripts/verify_runtime.py --sglang-dir patched_sglang_files   # the shipped runtime files match RUNTIME_MANIFEST.json
```

## Quick start (GPU, real runtime)

```bash
# 1. Rebuild the runtime from upstream + the patch series; exits 2 unless the tree id equals d0c440cf…
bash scripts/build_patched_sglang.sh ~/legolink-sglang
# 2. Install it (inside a CUDA 13 environment) and verify
pip install -e ~/legolink-sglang/python -c configs/constraints-runtime.txt
python scripts/verify_runtime.py --sglang-dir ~/legolink-sglang
# 3. Launch the LegoLink-only server, then confirm its role
bash scripts/serve_sglang.sh patched &
python scripts/check_server.py --role patched --api-base http://127.0.0.1:30000/v1
# 4. Compile → Link → Decode against it
python examples/compile_link_decode.py --api-base http://127.0.0.1:30000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

Full instructions, alternatives (Docker reference image, scheduler allocations) and troubleshooting: [docs/INSTALLATION.md](docs/INSTALLATION.md), [docs/QUICKSTART.md](docs/QUICKSTART.md).

To put an OpenAI-style agent on the LegoLink path, run the proxy in front of the patched server (`scripts/run_openai_proxy.py --help`) and point the agent's `base_url` at it.

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data flow, module responsibilities, the host/runtime contract, state and lifetime
* [docs/INSTALLATION.md](docs/INSTALLATION.md) — environment, dependencies, building and verifying the patched SGLang
* [docs/QUICKSTART.md](docs/QUICKSTART.md) — the smallest end-to-end runs, CPU and GPU
* [docs/RUNTIME.md](docs/RUNTIME.md) — the patch series, runtime fingerprint, `SGLANG_PIC_*` flags
* [docs/VALIDATION.md](docs/VALIDATION.md) — what is verified, by whom, where; what is not
* [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md), [docs/CODE_MAP.md](docs/CODE_MAP.md)

## License

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The runtime patches and the files in `patched_sglang_files/` / `upstream_support_files/`
are derived from [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0, © the SGLang authors).
