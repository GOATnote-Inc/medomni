#!/usr/bin/env bash
# serve_judge_b300.sh — serve a CROSS-FAMILY judge on B300:8003 to grade
# Nemotron-3-Nano-Omni outputs without same-model self-preference bias.
#
# Why Qwen2.5-7B-Instruct:
#   - True cross-family: Qwen2 transformer base, NOT Nemotron-3 Mamba2.
#   - Sovereign: open-weights (no Meta gating, no cloud API dependency).
#   - NVIDIA's own Nemotron-3 evaluation recipe uses Qwen3-235B as one of
#     its cross-family judges; Qwen2.5-7B is the lighter sibling.
#   - Light: ~14 GB BF16; fits B300 alongside Omni + embed+rerank.
#   - Sufficient for rubric grading: the GRADER_TEMPLATE is a JSON-output
#     classification task per criterion, not open-ended reasoning.
#
# Co-tenant memory budget (B300 275 GB total):
#   vllm-omni-b300:   191 GB  (cap 0.70)
#   vllm-embed:        ~5 GB  (cap 0.05)
#   vllm-rerank:       ~5 GB  (cap 0.05)
#   vllm-judge (this): ~28 GB (cap 0.10, FP16)
#   ──────────────────────────────────
#   total:           ~229 GB / 275 GB = 83% — comfortable

set -uo pipefail

JUDGE_PORT="${JUDGE_PORT:-8003}"
VLLM_TAG="${VLLM_TAG:-v0.20.0}"
HF_CACHE="${HF_CACHE:-${HOME}/medomni/hf_cache}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

: "${HF_TOKEN:?HF_TOKEN must be set}"

log() { printf "[serve_judge_b300] %s\n" "$*" >&2; }

mkdir -p "$HF_CACHE"

log "stopping any prior judge container"
docker rm -f vllm-judge 2>/dev/null || true

log "launching vllm-judge (${JUDGE_MODEL}) on :${JUDGE_PORT}"
docker run -d --name vllm-judge \
  --gpus all --shm-size=4g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -p "127.0.0.1:${JUDGE_PORT}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "vllm/vllm-openai:$VLLM_TAG" \
  --model "$JUDGE_MODEL" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.10 \
  --dtype bfloat16 \
  --seed 42 \
  --no-async-scheduling

log "wait for /v1/models (cold start ~5–10 min)"
for i in $(seq 1 25); do
  sleep 30
  if curl -sf "http://127.0.0.1:${JUDGE_PORT}/v1/models" >/dev/null 2>&1; then
    log "vllm-judge ready on :${JUDGE_PORT} after $((i * 30)) sec"
    curl -s "http://127.0.0.1:${JUDGE_PORT}/v1/models" | head -c 400; echo
    log "smoke completion (cross-family judge identity check):"
    curl -sS -X POST "http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$JUDGE_MODEL"'","messages":[{"role":"user","content":"In one sentence: what model are you?"}],"max_tokens":40,"temperature":0.0}' \
      | head -c 600; echo
    log "GPU mem after judge up:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  [ "$i" = "25" ] && { log "FAIL: vllm-judge not ready"; docker logs --tail 30 vllm-judge; exit 1; }
done
