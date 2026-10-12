# Quickstart

Three levels, each needing more hardware than the last. Install first ([INSTALLATION.md](INSTALLATION.md)).

## 1. The protocol on a CPU, in-process (no GPU, no network)

```bash
python examples/compile_link_decode.py
```

It drives the real host pipeline for five scripted turns against a strict fake runtime that enforces three rules of the real contract: a Compile registers a cache key,
a Link needs every key it names to be resident (otherwise it is refused with the runtime's residency-recovery detail), and plain generation is refused. It prints, per turn:

* each block's `policy_action` → `layout_adjustment` → `effective_action` and its token count,
* how many Compile requests the host sent, reused vs compiled tokens, the cache keys the Link named,
* `runtime_owned_fields_on_the_wire` — must be empty (the host never sends `imp_indices`, `kvlink`, block ranges, …),

then makes the "runtime" forget every block and shows the host recovering by recompiling under the same identity (`outcome: recovered_after_source_missing`). Exit code 0 means every check held.
Turn 1 compiles the task instruction; turn 2 already **reuses** it (`reused_tokens` > 0, no Compile) and keeps the new conversation turn in the mutable suffix; later turns reuse the settled turns.
This shows the *protocol*; it says nothing about performance or about the real runtime.

`python examples/compile_link_decode.py --fake-http` runs the same session through the real HTTP transport against a fake server on a local port.

## 2. Compile → Link → Decode against the real runtime (GPU)

```bash
bash scripts/serve_sglang.sh patched &                 # wait for "The server is fired up"
python scripts/check_server.py --role patched --api-base http://127.0.0.1:30000/v1
python examples/compile_link_decode.py --api-base http://127.0.0.1:30000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

Here the same five turns are sent to the patched server. The first request of a block is a **Compile** (`max_tokens=1`, `extra_key = cache_key`): SGLang prefills the block alone and registers its KV in the
RadixCache. Later turns send a **Link**: the runtime looks the blocks up, gathers their KV into request-private slots, re-rotates RoPE to the new global positions, recomputes the sparse boundary positions
it chose, and runs native decode — the text of each turn is printed in `output`. `scripts/serve_sglang.sh` also writes a per-request runtime trace (`./run/patched_runtime_trace.jsonl`,
`SGLANG_PIC_TRACE_PATH`) in which you can see `model_link_complete` events with the runtime's own reuse/recompute accounting.

**Status of this level:** the example's real-server mode and the server launch scripts were **not executed** when this repository was published (no GPU was used); the same code paths are
what the maintainers report running on an H200 with their own harness — see [VALIDATION.md](VALIDATION.md) for exactly what is and is not established.

## 3. Put an agent on the LegoLink path (OpenAI-compatible proxy)

```bash
python scripts/run_openai_proxy.py --help
python scripts/run_openai_proxy.py --port 8000 --model Qwen/Qwen2.5-7B-Instruct \
    --patched-api-base http://127.0.0.1:30000/v1 --variant selected_context_radixcache_legolink \
    --tokenizer-mode sglang --segmentation-mode deterministic --artifacts-root ./run/proxy_sessions
```

`--segmentation-mode` defaults to `llm`, which needs a separate segmentation endpoint (`NOC_SEGMENTATION_API_BASE`) and is never inherited from the generation server; `deterministic` segments by semantic field and needs nothing.

Point any OpenAI-style agent at `http://127.0.0.1:8000/v1`. The proxy keeps the agent's session, segments each request into sub-contexts, plans Compile/Link and answers with an ordinary
Chat Completions response. Other variants (`selected_context_original_sglang`, `full_context_original_sglang`, `passthrough_original_sglang`) need `--original-api-base` pointing at a separate, unmodified server
(`scripts/serve_sglang.sh native`). `--terminal-bench-json-guardrail` enables the JSON guardrail written for Terminal-Bench's Terminus agents; leave it off for other agents.
The proxy's HTTP layer, session handling and guardrails are covered by `tests/test_noc_openai_proxy.py` against a fake backend.

## What to look at next

* `docs/ARCHITECTURE.md` — what each stage does and the host/runtime contract.
* `noc_context_manager/pipeline.py` — `NOCRequestPipeline.execute`, the facade over the stages.
* `examples/compile_link_decode.py` — the whole session is ~60 lines of script plus the driver.
* `tests/test_pic_link_contract.py`, `tests/test_residency_singleflight.py`, `tests/pic/` — the contract, recovery and runtime math.
