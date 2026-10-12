# Known limitations

Nothing below is hidden by a fallback. Items marked **open** are not fixed; items marked **by design** are behaviour you should know about.

## Not established by this repository

* **No GPU result is included or was produced when this repository was published.** The patched runtime was never started here. See [VALIDATION.md](VALIDATION.md).
* **Sparse-repair quality and latency benefit are not claimed.** The maintainers' own post-hoc analysis of their H200 K sweep reports exact-output agreement far below the 95 % gate for sparse
  repair windows and a time-to-first-token slower than a cold upstream prefill at every window size on that workload ([VALIDATION.md](VALIDATION.md) §3). Treat the repair path as a mechanism under
  study, not as a result.
* **One model family is instrumented.** The patch series changes exactly one model file, `models/qwen2.py`. Other families (including Qwen3) are unverified on this runtime; a CPU
  arithmetic check of RoPE re-positioning with QK-norm exists in the authors' notes but is not part of this repository and says nothing about the GPU path.
* Tensor parallelism > 1, `--enable-overlap-schedule`, CUDA graphs, and the FlashInfer/Triton repair paths have no evidence here. The launch script disables overlap scheduling and CUDA graphs on purpose.
* The `cost_benefit` compile policy needs a measured `CompileCostProfile` for the exact deployment; none ships, so it refuses to run without one (the profiling scripts that produce it are not part of this repository). The default policy is `legacy_fixed_64` (compile iff ≥ 64 model tokens) — the baseline the cost-benefit policy has to be compared against, deliberately kept.

## Architecture

* **One physical RadixCache per scheduler process.** Namespaces are host-side names over one registry; multiple physical RadixCaches are a design, not code (**open**).
* The link plan's shape is *prefix blocks + one mutable suffix*. Arbitrary compile/mutable interleaving would need a "LinkPlan v2" that does not exist (**open**). Because the layout stage depends on the order of surrounding blocks, say "the compile *economic* policy is block-local", never "final compile behaviour is position-independent".
* `NamespaceMode.GLOBAL` (the adapter default) does not enforce persistence: a `no_cache` block and a settled `user_prompt` can be served from, and compiled into, the catalog. The isolating modes enforce it (**by design**, documented).
* Single-flight residency recovery is **in-process**. Other processes sharing one runtime are not coordinated; cold same-key misses at first Compile are duplicated work, not prevented. Aliases are per catalog instance; a shared catalog file carries primary entries only.
* The LegoLink-only server needs the default `RadixCache` branch of upstream's cache builder (no hierarchical cache, no hybrid SWA/SSM model, no LMCache).
* The host and the runtime define the `pic_link` field lists and the execution-governance rules twice, because the host cannot import the Linux-only runtime. Tests (`tests/test_pic_link_contract.py`, `scripts/verify_runtime.py`) pin them equal; change both sides together.
* The scheduler-side token registry holds at most `SGLANG_PIC_MAX_BLOCKS` (default 256) distinct blocks; reaching it drops the oldest entry.
* The scheduler-side registries (`PICRequestTokenRegistry`, the RadixCache block registry) take no locks: they rely on the scheduler's single event-loop thread.

## Code kept for a reason

* `noc_context_manager/sglang_bridge.py` is a legacy plan-file bridge that builds `imp_indices`/`kvlink`/`block_ranges` on the host. The current runtime rejects those fields and nothing on the serving path imports it; a contract test compares its output with the adapter's. Do not build on it.
* `NOCRequestScheduler` is a deprecated alias of `NOCRequestPlanner`.
* Modules such as `registry.py`, `reuse.py`, `selector.py`, `similarity.py`, `schema.py`, `agents.py`, `agent_loop.py`, `block_kv_cache.py`, `runtime_isolation.py` are earlier prototypes that other modules still import; they are not on the Compile → Link → Decode path.
* `group_repair_profile.py` and patches 0017–0020/0022 (group promotion) are experimental and off by default.

## Tests

* 2 tests need a CUDA device and skip without one (`tests/pic/test_kv_relocation.py`, `tests/pic/test_sparse_attention_reference.py`).
* Tests that load the runtime build a temporary merged source tree from `patched_sglang_files/` and `upstream_support_files/`; they never import SGLang. If you have a patched checkout at `external/sglang/`, `tests/pic/support.py` uses it instead.
* Some test and docstring text refers to research scripts of the original repository (for example `scripts/run_*`); they were deliberately not published and are cited only as the origin of evidence.
