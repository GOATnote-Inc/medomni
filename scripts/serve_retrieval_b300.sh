#!/usr/bin/env bash
# serve_retrieval_b300.sh — serve NeMo Retriever model weights
# (`llama-nemotron-embed-1b-v2` + `llama-3.2-nv-rerankqa-1b-v2`) via vllm on
# the B300, co-tenant with the existing vllm-omni-b300 + voice-gateway-free.
#
# Why vllm and not the canonical NIM container:
#   The NeMo Retriever NIM containers (v1.x at the time of this writing)
#   ship ONNX Runtime CUDA binaries compiled for Hopper sm_90; on Blackwell
#   sm_103 they fail at first inference with `cudaErrorSymbolNotFound` on
#   the ReduceSum / GroupNorm kernels. vllm 0.20.0 has Blackwell support
#   built-in and serves the same HF weights with the same OpenAI-shaped
#   `/v1/embeddings` and `/v1/score` endpoints.
#
# Deviation from NVIDIA-canonical: serving framework (vllm vs Triton-NIM).
# Model weights identical to canonical. v1 swaps to NIM when Blackwell-native
# tags ship.

set -uo pipefail

EMBED_PORT="${EMBED_PORT:-8001}"
RERANK_PORT="${RERANK_PORT:-8002}"
VLLM_TAG="${VLLM_TAG:-v0.20.0}"
HF_CACHE="${HF_CACHE:-${HOME}/medomni/hf_cache}"

EMBED_MODEL="${EMBED_MODEL:-nvidia/llama-nemotron-embed-1b-v2}"
RERANK_MODEL="${RERANK_MODEL:-nvidia/llama-3.2-nv-rerankqa-1b-v2}"

: "${HF_TOKEN:?HF_TOKEN must be set}"

log() { printf "[serve_retrieval_b300] %s\n" "$*" >&2; }

mkdir -p "$HF_CACHE"

log "stopping any prior containers on these ports"
docker rm -f vllm-embed vllm-rerank 2>/dev/null || true

# Embedder: vllm 0.20 uses --runner pooling --convert embed
log "launching vllm-embed (${EMBED_MODEL}) on :${EMBED_PORT}"
docker run -d --name vllm-embed \
  --gpus all --shm-size=4g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -p "127.0.0.1:${EMBED_PORT}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "vllm/vllm-openai:$VLLM_TAG" \
  --model "$EMBED_MODEL" \
  --runner pooling \
  --convert embed \
  --host 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.05 \
  --max-model-len 8192 \
  --no-async-scheduling

# Reranker: cross-encoder via --runner pooling --convert classify (the
# llama-3.2-nv-rerankqa model expects pairwise (query, passage) scoring;
# vllm exposes /v1/score for this)
log "launching vllm-rerank (${RERANK_MODEL}) on :${RERANK_PORT}"
docker run -d --name vllm-rerank \
  --gpus all --shm-size=4g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -p "127.0.0.1:${RERANK_PORT}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "vllm/vllm-openai:$VLLM_TAG" \
  --model "$RERANK_MODEL" \
  --runner pooling \
  --convert classify \
  --host 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.05 \
  --max-model-len 8192 \
  --no-async-scheduling

log "wait for /v1/models on both (cold start ~5–10 min, weight download dominates)"
for kind in embed rerank; do
  port=$([ "$kind" = embed ] && echo "$EMBED_PORT" || echo "$RERANK_PORT")
  for i in $(seq 1 25); do
    sleep 30
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "vllm-${kind} ready on :${port} after $((i * 30)) sec"
      break
    fi
    [ "$i" = "25" ] && { log "FAIL: vllm-${kind} not ready"; docker logs --tail 30 "vllm-${kind}"; exit 1; }
  done
done

log "smoke tests:"
curl -sS "http://127.0.0.1:${EMBED_PORT}/v1/models" | head -c 400; echo
curl -sS "http://127.0.0.1:${RERANK_PORT}/v1/models" | head -c 400; echo

log "GPU mem after embed+rerank up:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
