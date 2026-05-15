#!/usr/bin/env bash
# What this script does:
#   Starts a vLLM OpenAI-compatible server for baseten/Qwen3-4B-NVFP4-PTQ
#   with either the gemlite NVFP4 path or stock vLLM.
#
# Backend toggled via VLLM_GEMLITE_ENABLE; the autopatch hook installed by
# gemlite/vllm/patch.sh reads it at vLLM import time.
#
# Usage:
#   bash serve_nvfp4.sh --backend gemlite
#   bash serve_nvfp4.sh --backend default
#
# Stop with Ctrl-C. Only run one backend at a time.

set -euo pipefail

BACKEND=""
MODEL="baseten/Qwen3-4B-NVFP4-PTQ"
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=8192
GPU_UTIL=0.85
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)                BACKEND="$2"; shift 2 ;;
        --model)                  MODEL="$2"; shift 2 ;;
        --port)                   PORT="$2"; shift 2 ;;
        --host)                   HOST="$2"; shift 2 ;;
        --max-model-len)          MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_UTIL="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$BACKEND" in
    gemlite) export VLLM_GEMLITE_ENABLE=1 ;;
    default) export VLLM_GEMLITE_ENABLE=0 ;;
    *) echo "error: --backend must be 'gemlite' or 'default' (got: '${BACKEND}')" >&2; exit 2 ;;
esac

export TRITON_PTXAS_BLACKWELL_PATH="${TRITON_PTXAS_BLACKWELL_PATH:-/usr/local/cuda-13.0/bin/ptxas}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

echo "[serve_nvfp4] backend=${BACKEND} VLLM_GEMLITE_ENABLE=${VLLM_GEMLITE_ENABLE}"
echo "[serve_nvfp4] model=${MODEL}"

exec python3 -m vllm.entrypoints.cli.main serve \
    "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --no-enable-prefix-caching \
    --disable-log-requests \
    "${EXTRA_ARGS[@]}"
