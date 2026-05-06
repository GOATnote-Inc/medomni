#!/usr/bin/env bash
# launch_b300_prod.sh — instantly-relaunchable B300 prod inference launcher.
#
# Purpose: re-provision Nemotron-3-Nano-Omni NVFP4 on a fresh Brev B300
# pod (e.g. unnecessary-peach-catfish) after a delete-and-recreate. NVIDIA
# warned 2026-05-06 that the current B300 carries a provisioning bug and
# may need to be swapped. This script is the green-pod boot sequence in
# the blue-green replacement plan documented at
# `runbooks/blue-green-pod-replacement.md` (§1 catfish-B300).
#
# Run on the pod itself (after `brev shell unnecessary-peach-catfish`,
# or after scp-ing this file in). Do NOT run this on the laptop.
#
# Idempotent: docker rm -f the prior container before relaunch.
# Verify-then-claim: the script's exit code is the smoke result. A 0
# means /v1/models responded AND a clinical chat completion came back
# with non-empty content.
#
# What's pinned, what's free:
#   - Model ID and HF repo: pinned in MODEL_ID below.
#   - vLLM image tag: VLLM_TAG; SHA digest in B300_VLLM_IMAGE_DIGEST
#     (placeholder — fill in from `docker inspect` on a healthy catfish
#     before the swap window per `runbooks/parity-anchors.md`).
#   - All vLLM flags below are the verified-working set as of 2026-05-03
#     (see memory `nemotron_omni_tool_call_parser.md`).
#
# Critical Blackwell knowledge (do NOT remove these flags):
#   - --no-async-scheduling          — vLLM V1 NVFP4 batch>1 bug
#   - VLLM_ATTENTION_BACKEND=FLASH_ATTN — TRTLLM is_strictly_contiguous bug
#   - --reasoning-parser nemotron_v3 — Nemotron-3 reasoning prefix
#   - --enable-auto-tool-choice + --tool-call-parser qwen3_coder
#   - smoke max_tokens >= 4000      — model reasons before tool call
#   - NEVER pass --allowed-local-media-path unless the path exists inside
#     the container (returns 400 for ALL requests if missing).
#
# ENTRYPOINT note (memory `feedback_check_docker_entrypoint_before_docker_run.md`):
# The vllm/vllm-openai image's ENTRYPOINT is `["vllm","serve"]`. Do NOT
# prepend `vllm serve` in the run command body or you get
# `vllm serve vllm serve --model ...` and a hard crash. The docker run
# below passes ONLY model + flags; vllm + serve come from ENTRYPOINT.
# The script verifies this via `docker inspect` before the run.
#
# Required env:
#   HF_TOKEN          — Hugging Face read-only, gated-model access
#                       (do NOT echo or log this)
#
# Optional env:
#   B300_HOST         — informational, used for logging only
#                       (default: $(hostname))
#   MODEL_ID          — override the model. Default is the prod NVFP4 SKU.
#   VLLM_TAG          — vllm/vllm-openai tag. Default v0.20.0.
#   B300_VLLM_IMAGE_DIGEST — sha256:... digest. If set, pulled by digest;
#                       if "" or unset, pulled by tag with a warning.
#   SERVE_PORT        — vLLM HTTP port. Default 8000.
#   MAX_MODEL_LEN     — context length. Default 131072 (128K).
#   GPU_MEM_UTIL      — vLLM memory frac. Default 0.85.
#   HF_CACHE          — HF cache mount source. Default ~/medomni/hf_cache.
#   READY_TIMEOUT_MIN — wait minutes for /v1/models. Default 40.
#
# Exit codes:
#   0  — /v1/models healthy AND smoke chat completion returned content
#   1  — readiness timeout
#   2  — missing dep (docker, nvidia-smi, curl, jq)
#   3  — HF_TOKEN unset
#   4  — wrong GPU (not Blackwell)
#   5  — entrypoint check failed
#   6  — smoke chat completion empty / non-2xx

set -uo pipefail

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
SERVE_PORT="${SERVE_PORT:-8000}"
VLLM_TAG="${VLLM_TAG:-v0.20.0}"
B300_VLLM_IMAGE_DIGEST="${B300_VLLM_IMAGE_DIGEST:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
HF_CACHE="${HF_CACHE:-${HOME}/medomni/hf_cache}"
READY_TIMEOUT_MIN="${READY_TIMEOUT_MIN:-40}"
B300_HOST="${B300_HOST:-$(hostname)}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-omni-b300}"

log()  { printf "[launch_b300_prod] %s\n" "$*" >&2; }
fail() { log "FAIL: $*"; exit "${2:-1}"; }

# --- Pre-flight ------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
  fail "HF_TOKEN not set; Nemotron-Omni weights are gated. Source from console env-var UI." 3
fi

for bin in docker nvidia-smi curl jq; do
  command -v "$bin" >/dev/null 2>&1 || fail "missing dep: $bin" 2
done

log "B300 pre-flight on host=${B300_HOST}"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if ! echo "$GPU_NAME" | grep -qE "B(200|300)"; then
  fail "GPU is '$GPU_NAME', not B200/B300. NVFP4 is Blackwell-only. For Hopper fp8 fallback, see runbooks/blue-green-pod-replacement.md §3." 4
fi
log "GPU OK: $GPU_NAME"

mkdir -p "$HF_CACHE"

# --- Image pull (digest-pinned if available) -------------------------------

if [[ -n "$B300_VLLM_IMAGE_DIGEST" ]]; then
  IMAGE_REF="vllm/vllm-openai@${B300_VLLM_IMAGE_DIGEST}"
  log "pulling image by digest: $IMAGE_REF"
else
  IMAGE_REF="vllm/vllm-openai:${VLLM_TAG}"
  log "WARN: B300_VLLM_IMAGE_DIGEST not set; pulling by mutable tag $IMAGE_REF"
  log "      Capture digest from a healthy pod and persist in runbooks/parity-anchors.md."
fi
docker pull "$IMAGE_REF" || fail "docker pull failed for $IMAGE_REF"

# --- ENTRYPOINT verification (memory: never prepend vllm serve) ------------

ENTRYPOINT=$(docker inspect --format '{{json .Config.Entrypoint}}' "$IMAGE_REF" 2>/dev/null || echo "null")
log "image ENTRYPOINT: $ENTRYPOINT"
if ! echo "$ENTRYPOINT" | grep -qE '"vllm".*"serve"'; then
  log "ENTRYPOINT does not look like [\"vllm\",\"serve\"]. Refusing to run; review docker_inspect output."
  fail "entrypoint check failed (got: $ENTRYPOINT)" 5
fi
log "ENTRYPOINT OK — passing model+flags only (no leading 'vllm serve')."

# --- Container relaunch ----------------------------------------------------

log "stopping any prior $CONTAINER_NAME container (idempotent)"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

log "launching $CONTAINER_NAME on :${SERVE_PORT}"
# IMPORTANT: NO `--allowed-local-media-path` here. If you add it, ensure
# the path exists inside the container or ALL requests will 400.
docker run -d --name "$CONTAINER_NAME" \
  --gpus all --shm-size=32g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -p "127.0.0.1:${SERVE_PORT}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  --restart unless-stopped \
  "$IMAGE_REF" \
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
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-async-scheduling

# --- Readiness wait --------------------------------------------------------

log "waiting up to ${READY_TIMEOUT_MIN} min for /v1/models (cold start ~15-25 min: HF download + load)"
ready=0
for i in $(seq 1 "$READY_TIMEOUT_MIN"); do
  sleep 60
  if curl -sf "http://127.0.0.1:${SERVE_PORT}/v1/models" >/dev/null 2>&1; then
    log "/v1/models responded after ${i} min"
    ready=1
    break
  fi
  echo "  waiting... ${i}/${READY_TIMEOUT_MIN} min" >&2
done

if [[ "$ready" -ne 1 ]]; then
  log "readiness timeout. Container logs (last 80 lines):"
  docker logs "$CONTAINER_NAME" 2>&1 | tail -80 >&2 || true
  exit 1
fi

# --- Smoke: clinical chat completion (max_tokens >= 4000 per memory) -------

log "smoke: clinical chat completion (max_tokens=4000)"
SMOKE_PROMPT='{"model":"'"$MODEL_ID"'","messages":[{"role":"user","content":"In one sentence: what class of drug is tamoxifen, and name one common indication?"}],"max_tokens":4000,"temperature":0.2}'

SMOKE_RESP=$(curl -sS -X POST "http://127.0.0.1:${SERVE_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$SMOKE_PROMPT" 2>&1) || fail "smoke curl failed: $SMOKE_RESP" 6

# Inspect the artifact content, not just the exit code
# (memory: feedback_eval_preflight_judge_key.md / feedback_pilot_before_full_sweep.md)
SMOKE_CONTENT=$(echo "$SMOKE_RESP" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
if [[ -z "$SMOKE_CONTENT" ]]; then
  log "smoke FAIL: empty content. Raw response (first 500 chars):"
  echo "$SMOKE_RESP" | head -c 500 >&2
  echo >&2
  exit 6
fi

log "smoke OK. First 200 chars of completion:"
printf "%s\n" "$SMOKE_CONTENT" | head -c 200 >&2
echo >&2

log "verified: container=$CONTAINER_NAME port=$SERVE_PORT model=$MODEL_ID host=$B300_HOST"
log "next: capture image digest into runbooks/parity-anchors.md"
log "      docker inspect --format '{{index .RepoDigests 0}}' $IMAGE_REF"
exit 0
