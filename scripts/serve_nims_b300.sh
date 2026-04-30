#!/usr/bin/env bash
# serve_nims_b300.sh — start the NeMo Retriever Embedding NIM (port 8001)
# and the Llama-3.2-NV-RerankQA-1B-v2 NIM (port 8002) on the B300 pod,
# co-tenant with the existing vllm-omni-b300 container at port 8000.
#
# Pre-conditions on the pod:
#   - vllm-omni-b300 running with --gpu-memory-utilization 0.70 (frees ~83 GB)
#   - `docker login nvcr.io` already succeeded with NGC personal API key
#   - NIM images pulled:
#       nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest   (~6 GB)
#       nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:latest  (~6 GB)
#
# Per NVIDIA NIM docs, NIMs cache resolved model weights under NIM_CACHE_PATH.
# We mount the same host cache directory across runs so subsequent restarts
# skip model download (~5–10 min saved per restart).
#
# Usage on the pod:
#   NGC_API_KEY=<key> bash /tmp/serve_nims_b300.sh

set -uo pipefail

NIM_CACHE_HOST="${NIM_CACHE_HOST:-${HOME}/medomni/nim_cache}"
EMBED_PORT="${EMBED_PORT:-8001}"
RERANK_PORT="${RERANK_PORT:-8002}"
EMBED_IMG="nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest"
RERANK_IMG="nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:latest"

: "${NGC_API_KEY:?NGC_API_KEY must be set in this shell (sourced from /tmp/.ngc_env)}"

log() { printf "[serve_nims_b300] %s\n" "$*" >&2; }

mkdir -p "$NIM_CACHE_HOST"
chmod 770 "$NIM_CACHE_HOST" || true

log "stopping any prior NIM containers"
docker rm -f nim-embed nim-rerank 2>/dev/null || true

log "launching nim-embed (llama-nemotron-embed-1b-v2) on :${EMBED_PORT}"
docker run -d --name nim-embed \
  --gpus all \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e NGC_API_KEY \
  -e NIM_CACHE_PATH=/opt/nim/.cache \
  -v "$NIM_CACHE_HOST:/opt/nim/.cache" \
  -p "127.0.0.1:${EMBED_PORT}:8000" \
  "$EMBED_IMG"

log "launching nim-rerank (llama-3.2-nv-rerankqa-1b-v2) on :${RERANK_PORT}"
docker run -d --name nim-rerank \
  --gpus all \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e NGC_API_KEY \
  -e NIM_CACHE_PATH=/opt/nim/.cache \
  -v "$NIM_CACHE_HOST:/opt/nim/.cache" \
  -p "127.0.0.1:${RERANK_PORT}:8000" \
  "$RERANK_IMG"

log "waiting for embed NIM /v1/models (first start downloads model, ~5–10 min)"
for i in $(seq 1 30); do
  sleep 30
  if curl -sf "http://127.0.0.1:${EMBED_PORT}/v1/models" >/dev/null 2>&1; then
    log "embed NIM ready after $((i * 30 / 60)) min $((i * 30 % 60)) sec"
    break
  fi
  echo "  embed waiting... $((i * 30)) sec"
done

log "waiting for rerank NIM /v1/models"
for i in $(seq 1 30); do
  sleep 30
  if curl -sf "http://127.0.0.1:${RERANK_PORT}/v1/models" >/dev/null 2>&1; then
    log "rerank NIM ready after $((i * 30 / 60)) min $((i * 30 % 60)) sec"
    break
  fi
  echo "  rerank waiting... $((i * 30)) sec"
done

log "smoke tests:"
curl -sS "http://127.0.0.1:${EMBED_PORT}/v1/models" | head -c 400; echo
curl -sS "http://127.0.0.1:${RERANK_PORT}/v1/models" | head -c 400; echo

log "GPU mem after both NIMs up:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
