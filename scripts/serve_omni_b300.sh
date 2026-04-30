#!/usr/bin/env bash
# serve_omni_b300.sh — sovereign serve of Nemotron-3-Nano-Omni on the
# Brev B300 pod (unnecessary-peach-catfish, NVIDIA B300 288 GiB HBM3E,
# Helsinki, $8.88/hr).
#
# B300 is Blackwell Ultra (sm_103a). Critical knowledge baked in:
#   1. CUDA 13 required (B300 needs CUDA 13+; cubins for compute_90a do
#      NOT run on Blackwell — must be native compute_103 or PTX).
#   2. NVFP4 has a vLLM V1-engine bug at batch >1 on Blackwell —
#      workaround: --no-async-scheduling. See
#      https://github.com/NVIDIA-NeMo/Nemotron/issues/125
#   3. TRTLLM attention backend has is_strictly_contiguous bug on
#      Blackwell — workaround: --disable-attention-backend trtllm. See
#      https://github.com/vllm-project/vllm/issues/32353
#   4. Audio + reasoning are MUTUALLY EXCLUSIVE on Omni — set
#      enable_thinking=false for any audio request, temperature=0.2.
#   5. 1M context OOMs — cap --max-model-len at 262144 (256K).
#
# Why B300 vs H200:
#   - B300 has 5th-gen tensor cores → NVFP4 natively at 14 PFLOPS dense.
#   - 288 GiB HBM3E lets us co-tenant Omni + NV-Embed-v2 + Llama-Nemotron-
#     Rerank-VL + Llama-Guard-3-8B + nx-cugraph KG ALL on one pod (~38 GB
#     models + ~100 GB for KV cache at 256K + still ~150 GB headroom).
#   - 8 TB/s HBM3E bandwidth → ~1.67x H200 (4.8 TB/s) on memory-bound
#     decode.
#
# Run on the pod (after scp). Reads HF_TOKEN from env.
#
# Usage on laptop (after `brev login`):
#   brev port-forward unnecessary-peach-catfish -p 8000:8000 &
#   brev shell unnecessary-peach-catfish 'HF_TOKEN=hf_xxx bash serve_omni_b300.sh'

set -uo pipefail

MODEL_ID="${MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
SERVE_PORT="${SERVE_PORT:-8000}"
VLLM_TAG="${VLLM_TAG:-v0.20.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"   # 128K is generous; bump to 262144 if needed
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
HF_CACHE="${HF_CACHE:-${HOME}/medomni/hf_cache}"

: "${HF_TOKEN:?HF_TOKEN must be set}"

log() { printf "[serve_omni_b300] %s\n" "$*" >&2; }

require() { command -v "$1" >/dev/null 2>&1 || { log "missing $1"; exit 2; }; }
require docker
require nvidia-smi

log "B300 pre-flight"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.used --format=csv,noheader
nvcc --version 2>&1 | tail -2 || log "no system nvcc (container will provide CUDA)"

# Sanity: refuse to run on non-Blackwell hardware (this script is B300-specific).
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if ! echo "$GPU_NAME" | grep -qE "B(200|300)"; then
  log "WARN: GPU is $GPU_NAME, not B200/B300. Blackwell workarounds may not apply."
  log "      For Hopper (H100/H200), use scripts/serve_trtllm_h200.sh instead."
fi

mkdir -p "$HF_CACHE"

log "pulling vllm/vllm-openai:$VLLM_TAG (large; first time ~10 min)"
docker pull "vllm/vllm-openai:$VLLM_TAG"

log "stopping any prior serve container"
docker rm -f vllm-omni-b300 2>/dev/null || true

log "launching vllm-omni-b300 with Blackwell workarounds"
# NOTE: avoid the trtllm attention backend on Blackwell (is_strictly_contiguous bug
# https://github.com/vllm-project/vllm/issues/32353). vLLM has no CLI flag to
# disable a backend; instead force a specific backend via VLLM_ATTENTION_BACKEND.
# FLASH_ATTN is the safe default on Blackwell+NVFP4 in v0.20.0.
docker run -d --name vllm-omni-b300 \
  --gpus all --shm-size=32g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -p "127.0.0.1:${SERVE_PORT}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "vllm/vllm-openai:$VLLM_TAG" \
  --model "$MODEL_ID" \
  --host 0.0.0.0 \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-num-seqs 384 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --kv-cache-dtype fp8 \
  --video-pruning-rate 0.5 \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-async-scheduling

log "wait for /v1/models (cold start ~15-25 min on B300; weight download from HF + model load)"
for i in $(seq 1 40); do
  sleep 60
  if curl -sf "http://127.0.0.1:${SERVE_PORT}/v1/models" >/dev/null 2>&1; then
    log "Omni up at :${SERVE_PORT} after ${i} min"
    curl -s "http://127.0.0.1:${SERVE_PORT}/v1/models" | head -50
    log "smoke completion test:"
    curl -sS -X POST "http://127.0.0.1:${SERVE_PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL_ID"'","messages":[{"role":"user","content":"In one sentence: what is tamoxifen?"}],"max_tokens":80,"temperature":0.2}' \
      | head -50
    exit 0
  fi
  echo "  waiting... ${i} min"
done

log "FAIL: Omni did not become ready in 40 min"
log "container logs (last 80 lines):"
docker logs vllm-omni-b300 2>&1 | tail -80
exit 1
