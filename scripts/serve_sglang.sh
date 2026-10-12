#!/usr/bin/env bash
# Launch one SGLang server in the foreground: the patched LegoLink-only runtime, or an unmodified/"native" server for the baseline.
#
#   bash scripts/serve_sglang.sh patched      # LegoLink-only: serves Compile + Link requests, refuses plain generation
#   bash scripts/serve_sglang.sh native       # plain generation with the prefix cache; never serves a Link
#
# Run it inside whatever owns your GPU (a shell on a workstation, tmux, or a scheduler allocation such as SLURM). The script never sets
# CUDA_VISIBLE_DEVICES: pick the GPU the usual way, or with BASE_GPU_ID (an index among the devices that are visible).
#
# Configuration is by environment variable (defaults in brackets):
#   PYTHON_BIN [python3]         interpreter whose environment has the patched SGLang installed (patched) / any SGLang (native)
#   MODEL_PATH [Qwen/Qwen2.5-7B-Instruct]   SERVED_MODEL_NAME [= MODEL_PATH]
#   HOST [127.0.0.1]  PORT [30000 patched / 30001 native]   TP_SIZE [1]   BASE_GPU_ID [0]
#   MEM_FRACTION [0.80]          SGLang --mem-fraction-static, a share of the memory free at launch (the H200 runs used 0.45 / 0.35 with both servers on one GPU)
#   CONTEXT_LENGTH [32768]  MAX_PREFILL_TOKENS [32000]  ATTENTION_BACKEND [triton]  MAX_TOTAL_TOKENS [unset]  MODEL_DTYPE [unset]
#   REPAIR_BACKEND [native]      sparse-repair attention backend of the patched runtime: native | flashinfer | auto | hybrid
#   SGLANG_PIC_MAX_BLOCKS [256]  the patched runtime's block registry cap
#   TRACE_PATH [./run/patched_runtime_trace.jsonl]   the patched runtime's per-request JSONL trace (truncated at start)
#   EXTRA_ARGS [empty]           extra `sglang.launch_server` arguments, one string
#
# After it prints "The server is fired up", check the role with:  python scripts/check_server.py --role patched --api-base http://127.0.0.1:30000/v1
set -euo pipefail

ROLE="${1:-}"
[[ "$ROLE" == "patched" || "$ROLE" == "native" ]] || { echo "usage: serve_sglang.sh patched|native" >&2; exit 64; }

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-1}"
BASE_GPU_ID="${BASE_GPU_ID:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
REPAIR_BACKEND="${REPAIR_BACKEND:-native}"
if [[ "$ROLE" == "patched" ]]; then PORT="${PORT:-30000}"; else PORT="${PORT:-30001}"; fi
[[ "$BASE_GPU_ID" =~ ^[0-9]+$ ]] || { echo "BASE_GPU_ID must be a non-negative integer" >&2; exit 64; }

command -v ninja >/dev/null 2>&1 || { echo "ERROR: ninja is not on PATH; the runtime JIT-compiles kernels at first use. Activate the virtualenv that installed SGLang, or pip install ninja." >&2; exit 1; }

command=("$PYTHON_BIN" -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" --host "$HOST" --port "$PORT" --tp-size "$TP_SIZE"
  --mem-fraction-static "$MEM_FRACTION" --context-length "$CONTEXT_LENGTH" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --chunked-prefill-size -1 --attention-backend "$ATTENTION_BACKEND" --sampling-backend pytorch
  --disable-overlap-schedule --disable-cuda-graph --enable-cache-report --base-gpu-id "$BASE_GPU_ID")
[[ -z "${MAX_TOTAL_TOKENS:-}" ]] || command+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
[[ -z "${MODEL_DTYPE:-}" ]] || command+=(--dtype "$MODEL_DTYPE")

if [[ "$ROLE" == "patched" ]]; then
  TRACE_PATH="${TRACE_PATH:-./run/patched_runtime_trace.jsonl}"
  mkdir -p "$(dirname "$TRACE_PATH")"
  : > "$TRACE_PATH"   # a fresh server has a fresh trace
  # The LegoLink-only deployment refuses SGLang's plain-generation startup warmup, so it is skipped; issue your own first requests (docs/QUICKSTART.md).
  command+=(--skip-server-warmup)
  # shellcheck disable=SC2086
  exec env \
    SGLANG_PIC_MAX_BLOCKS="${SGLANG_PIC_MAX_BLOCKS:-256}" \
    SGLANG_PIC_LEGOLINK_ONLY=1 \
    SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS=1 \
    SGLANG_PIC_BATCH_BLOCK_KV_LINK=1 \
    SGLANG_PIC_EXPERIMENTAL_INDEXED_MIXED_PREFILL=1 \
    SGLANG_PIC_EXPERIMENTAL_GROUP_PROMOTION=0 \
    SGLANG_PIC_SPARSE_REPAIR_ATTENTION_BACKEND="$REPAIR_BACKEND" \
    SGLANG_PIC_TRACE_PATH="$TRACE_PATH" \
    "${command[@]}" ${EXTRA_ARGS:-}
else
  # shellcheck disable=SC2086
  exec env \
    SGLANG_PIC_LEGOLINK_ONLY=0 \
    SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS=0 \
    SGLANG_PIC_BATCH_BLOCK_KV_LINK=0 \
    SGLANG_PIC_EXPERIMENTAL_INDEXED_MIXED_PREFILL=0 \
    SGLANG_PIC_EXPERIMENTAL_GROUP_PROMOTION=0 \
    "${command[@]}" ${EXTRA_ARGS:-}
fi
