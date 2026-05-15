#!/usr/bin/env bash
# What this script does:
#   Serve baseten/Qwen3-4B-NVFP4-PTQ twice on the same GPU (sequentially):
#     1. stock vLLM NVFP4 path
#     2. gemlite NVFP4 path (via VLLM_GEMLITE_ENABLE=1 + autopatch hook)
#   For each run: wait until /v1/models is ready, send one deterministic
#   chat prompt (temperature=0), record the answer, stop the server.
#   Then diff the two answers and print a summary.
#
# Outputs:
#   /tmp/gen_default_nvfp4.txt   - stock vLLM generation
#   /tmp/gen_gemlite_nvfp4.txt   - gemlite-routed generation
#   /tmp/gen_nvfp4_diff.txt      - unified diff of the two
#   /tmp/server_{gemlite,default}_nvfp4.log - full server logs
#
# Prereq: gemlite autopatch installed once with
#   bash /root/data/gemlite/gemlite/vllm/patch.sh install

set -u

PORT=8000
HEALTH_URL="http://127.0.0.1:${PORT}/v1/models"
MAX_WAIT=600
SETTLE_BETWEEN=15
SERVE_SCRIPT="/root/data/gemlite/gemlite/vllm/examples/serve_nvfp4.sh"
CHAT_SCRIPT="/root/data/gemlite/gemlite/vllm/examples/chat_once.py"

run_backend() {
    local backend=$1
    local out_file=$2
    local server_log=$3

    echo "======================================================================"
    echo "[compare] starting backend=${backend}"
    echo "======================================================================"

    setsid bash "${SERVE_SCRIPT}" --backend "${backend}" \
        > "${server_log}" 2>&1 &
    local server_pid=$!
    echo "[compare] server pid=${server_pid}, log=${server_log}"

    local elapsed=0
    local ready=0
    while [[ ${elapsed} -lt ${MAX_WAIT} ]]; do
        if curl -sf "${HEALTH_URL}" > /dev/null; then
            ready=1; break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "[compare] server pid ${server_pid} died; see ${server_log}"
            tail -30 "${server_log}"
            return 1
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done

    if [[ ${ready} -ne 1 ]]; then
        echo "[compare] server never became ready after ${MAX_WAIT}s"
        kill -TERM -"${server_pid}" 2>/dev/null
        return 1
    fi

    echo "[compare] server ready after ~${elapsed}s; running chat_once..."
    python3 "${CHAT_SCRIPT}" > "${out_file}" 2>&1
    local rc=$?
    echo "[compare] chat rc=${rc}, wrote ${out_file}"

    echo "[compare] stopping ${backend} server..."
    kill -TERM -"${server_pid}" 2>/dev/null
    local wait_left=60
    while kill -0 "${server_pid}" 2>/dev/null && [[ ${wait_left} -gt 0 ]]; do
        sleep 2
        wait_left=$((wait_left - 2))
    done
    pkill -TERM -g "${server_pid}" 2>/dev/null
    sleep "${SETTLE_BETWEEN}"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    return 0
}

run_backend default /tmp/gen_default_nvfp4.txt /tmp/server_default_nvfp4.log
run_backend gemlite /tmp/gen_gemlite_nvfp4.txt /tmp/server_gemlite_nvfp4.log

echo ""
echo "======================================================================"
echo "[compare] diffing outputs"
echo "======================================================================"
diff -u /tmp/gen_default_nvfp4.txt /tmp/gen_gemlite_nvfp4.txt > /tmp/gen_nvfp4_diff.txt
if [[ -s /tmp/gen_nvfp4_diff.txt ]]; then
    echo "[compare] outputs differ. First 80 diff lines:"
    head -80 /tmp/gen_nvfp4_diff.txt
else
    echo "[compare] outputs are IDENTICAL."
fi

default_words=$(wc -w < /tmp/gen_default_nvfp4.txt)
gemlite_words=$(wc -w < /tmp/gen_gemlite_nvfp4.txt)
echo ""
echo "[compare] default words: ${default_words}"
echo "[compare] gemlite words: ${gemlite_words}"
