#!/usr/bin/env bash
# launch_h200_factory.sh — instantly-relaunchable H200 factory + imaging-RAG.
#
# Purpose: re-provision warm-lavender-narwhal (Brev H200, Hopper SM 9.0)
# after a delete-and-recreate. Mirrors the verified-running 2026-05-06
# state captured by the user via direct `ps -ef` on the live pod:
#
#   1. vllm serve  Nemotron-3-Nano-30B-A3B-BF16  on :8000  (PID 76635, 9d)
#   2. factory_loop.py  --task reasoning  ...  on :8000-client (PID 1150324, 5d)
#   3. nvembed_server.py  (NV-Embed-v2)         (PID 30554)
#   4. biomedclip_server.py  (BiomedCLIP)       (PID 49483)
#
# Run on the pod itself (after `brev shell warm-lavender-narwhal`).
#
# Idempotent: tmux sessions and docker containers are torn down by name
# before relaunch.
#
# Verify-then-claim: exits 0 only after each of the four services
# responds on its expected port AND a smoke completion comes back from
# vLLM. Each readiness probe inspects content, not just exit code.
#
# Source-of-truth note:
# The factory_loop.py / nvembed_server.py / biomedclip_server.py source
# lives in the *private* sister repo `prism42-nemotron-med` and is cloned
# onto the pod at `/home/ubuntu/medomni-imaging-rag/`. This script does
# not vendor that source — it expects $IMAGING_RAG_DIR to be a working
# clone (see `runbooks/blue-green-pod-replacement.md` §2 for the clone
# step). If the directory is missing, the factory + RAG launches are
# skipped with a loud warning, but vLLM still comes up so prod inference
# is restored ASAP.
#
# Required env:
#   HF_TOKEN          — Hugging Face read-only, gated-model access
#
# Optional env:
#   IMAGING_RAG_DIR   — path to medomni-imaging-rag clone on the pod
#                       (default: /home/ubuntu/medomni-imaging-rag)
#   DATA_QUEUE_DIR    — factory output dir (default: /home/ubuntu/data-queue)
#   SEEDS_FILE        — clinical seeds (default:
#                       $DATA_QUEUE_DIR/seeds/clinical_questions.jsonl)
#   VLLM_MODEL_BF16   — H200-friendly model. Default Nemotron-3-Nano BF16.
#   VLLM_TAG          — vllm/vllm-openai tag. Default v0.20.0.
#   H200_VLLM_IMAGE_DIGEST — sha256 digest. Placeholder if unset.
#   SERVE_PORT        — vLLM HTTP port. Default 8000.
#   NVEMBED_PORT      — NV-Embed-v2 port. Default 8003.
#   BIOMEDCLIP_PORT   — BiomedCLIP port. Default 8004.
#   FACTORY_RATE_CAP  — factory rate-cap-per-task. Default 200.
#   FACTORY_MAX_DISK  — factory --max-disk-gb. Default 5.
#   READY_TIMEOUT_MIN — vLLM readiness wait. Default 30.
#   SKIP_RESTORE_DATA_QUEUE=1 — skip object-store restore step.
#
# Exit codes:
#   0  — vLLM healthy + smoke OK; factory and RAG started (or warned-skipped)
#   1  — vLLM readiness timeout
#   2  — missing dep
#   3  — HF_TOKEN unset
#   4  — wrong GPU (not Hopper)
#   6  — vLLM smoke chat completion empty / non-2xx

set -uo pipefail

VLLM_MODEL_BF16="${VLLM_MODEL_BF16:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
VLLM_TAG="${VLLM_TAG:-v0.20.0}"
H200_VLLM_IMAGE_DIGEST="${H200_VLLM_IMAGE_DIGEST:-}"
SERVE_PORT="${SERVE_PORT:-8000}"
NVEMBED_PORT="${NVEMBED_PORT:-8003}"
BIOMEDCLIP_PORT="${BIOMEDCLIP_PORT:-8004}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
HF_CACHE="${HF_CACHE:-${HOME}/medomni/hf_cache}"
READY_TIMEOUT_MIN="${READY_TIMEOUT_MIN:-30}"

IMAGING_RAG_DIR="${IMAGING_RAG_DIR:-/home/ubuntu/medomni-imaging-rag}"
DATA_QUEUE_DIR="${DATA_QUEUE_DIR:-/home/ubuntu/data-queue}"
SEEDS_FILE="${SEEDS_FILE:-${DATA_QUEUE_DIR}/seeds/clinical_questions.jsonl}"
FACTORY_RATE_CAP="${FACTORY_RATE_CAP:-200}"
FACTORY_MAX_DISK="${FACTORY_MAX_DISK:-5}"

CONTAINER_NAME="${CONTAINER_NAME:-vllm-nemotron-bf16}"

log()  { printf "[launch_h200_factory] %s\n" "$*" >&2; }
warn() { printf "[launch_h200_factory] WARN: %s\n" "$*" >&2; }
fail() { log "FAIL: $*"; exit "${2:-1}"; }

# --- Pre-flight ------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
  fail "HF_TOKEN not set; Nemotron weights are gated. Use console env-var UI." 3
fi

for bin in docker nvidia-smi curl jq tmux; do
  command -v "$bin" >/dev/null 2>&1 || fail "missing dep: $bin" 2
done

log "H200 pre-flight"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if ! echo "$GPU_NAME" | grep -qE "H(100|200)"; then
  fail "GPU is '$GPU_NAME', not H100/H200. Wrong launcher; pick the matching script." 4
fi
log "GPU OK: $GPU_NAME"

mkdir -p "$HF_CACHE"
mkdir -p "$DATA_QUEUE_DIR"

# --- Optional: restore data-queue from object store ------------------------

if [[ "${SKIP_RESTORE_DATA_QUEUE:-0}" != "1" ]]; then
  if [[ -x "$(dirname "$0")/snapshot_h200_factory_state.sh" ]]; then
    log "data-queue restore from object store NOT performed automatically."
    log "  Run the restore command from runbooks/blue-green-pod-replacement.md §2 manually,"
    log "  then re-run this script with SKIP_RESTORE_DATA_QUEUE=1."
  fi
fi

# --- vLLM image pull -------------------------------------------------------

if [[ -n "$H200_VLLM_IMAGE_DIGEST" ]]; then
  IMAGE_REF="vllm/vllm-openai@${H200_VLLM_IMAGE_DIGEST}"
  log "pulling image by digest: $IMAGE_REF"
else
  IMAGE_REF="vllm/vllm-openai:${VLLM_TAG}"
  warn "H200_VLLM_IMAGE_DIGEST not set; pulling mutable tag $IMAGE_REF"
fi
docker pull "$IMAGE_REF" || fail "docker pull failed for $IMAGE_REF"

# ENTRYPOINT verification (do not prepend `vllm serve`)
ENTRYPOINT=$(docker inspect --format '{{json .Config.Entrypoint}}' "$IMAGE_REF" 2>/dev/null || echo "null")
log "image ENTRYPOINT: $ENTRYPOINT"
if ! echo "$ENTRYPOINT" | grep -qE '"vllm".*"serve"'; then
  fail "ENTRYPOINT mismatch (got: $ENTRYPOINT). Refusing to run." 5
fi

# --- 1. vLLM Nemotron-3-Nano BF16 ------------------------------------------

log "stopping prior $CONTAINER_NAME (idempotent)"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

log "launching $CONTAINER_NAME on :${SERVE_PORT}"
docker run -d --name "$CONTAINER_NAME" \
  --gpus all --shm-size=32g \
  -e HF_TOKEN="$HF_TOKEN" \
  -p "127.0.0.1:${SERVE_PORT}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  --restart unless-stopped \
  "$IMAGE_REF" \
  --model "$VLLM_MODEL_BF16" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL"

log "waiting up to ${READY_TIMEOUT_MIN} min for /v1/models"
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
  log "readiness timeout. logs:"
  docker logs "$CONTAINER_NAME" 2>&1 | tail -60 >&2 || true
  exit 1
fi

# Smoke: completion content check (artifact, not exit code)
SMOKE_RESP=$(curl -sS -X POST "http://127.0.0.1:${SERVE_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$VLLM_MODEL_BF16"'","messages":[{"role":"user","content":"List two contraindications to NSAIDs in one sentence."}],"max_tokens":256,"temperature":0.2}' 2>&1) \
  || fail "smoke curl failed: $SMOKE_RESP" 6
SMOKE_CONTENT=$(echo "$SMOKE_RESP" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
[[ -z "$SMOKE_CONTENT" ]] && { log "smoke empty: $SMOKE_RESP"; exit 6; }
log "vLLM smoke OK: $(printf '%.140s' "$SMOKE_CONTENT")"

# --- 2-4. Imaging-RAG sidecars + factory loop ------------------------------

if [[ ! -d "$IMAGING_RAG_DIR" ]]; then
  warn "IMAGING_RAG_DIR=$IMAGING_RAG_DIR not present on this pod."
  warn "Skipping nvembed_server / biomedclip_server / factory_loop launch."
  warn "vLLM is up — prod inference is restored. Clone the imaging-rag repo and re-run with"
  warn "  SKIP_RESTORE_DATA_QUEUE=1 to bring up the factory side."
  log "verified (vLLM-only): port=$SERVE_PORT model=$VLLM_MODEL_BF16"
  exit 0
fi

# 2. NV-Embed-v2 server
log "launching nvembed_server.py on :${NVEMBED_PORT} via tmux"
tmux kill-session -t nvembed 2>/dev/null || true
tmux new-session -d -s nvembed \
  "cd '$IMAGING_RAG_DIR' && HF_TOKEN='$HF_TOKEN' PORT='$NVEMBED_PORT' \
   python3 nvembed_server.py 2>&1 | tee -a /tmp/nvembed.log"

# 3. BiomedCLIP server
log "launching biomedclip_server.py on :${BIOMEDCLIP_PORT} via tmux"
tmux kill-session -t biomedclip 2>/dev/null || true
tmux new-session -d -s biomedclip \
  "cd '$IMAGING_RAG_DIR' && HF_TOKEN='$HF_TOKEN' PORT='$BIOMEDCLIP_PORT' \
   python3 biomedclip_server.py 2>&1 | tee -a /tmp/biomedclip.log"

# Wait for sidecars (best-effort; don't fail vLLM if RAG sidecars stall)
for svc in "nvembed:${NVEMBED_PORT}" "biomedclip:${BIOMEDCLIP_PORT}"; do
  name="${svc%:*}"; port="${svc#*:}"
  log "  waiting up to 5 min for $name on :$port"
  ok=0
  for i in $(seq 1 30); do
    sleep 10
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 \
       || curl -sf "http://127.0.0.1:${port}/" >/dev/null 2>&1; then
      log "  $name up"
      ok=1; break
    fi
  done
  [[ "$ok" -ne 1 ]] && warn "$name did not respond on :$port; check /tmp/${name}.log"
done

# 4. factory_loop.py — exact command line captured 2026-05-06 from live narwhal
if [[ ! -f "$SEEDS_FILE" ]]; then
  warn "seeds file $SEEDS_FILE missing; factory will not start."
  warn "Restore from object store (see runbooks/blue-green-pod-replacement.md §2)."
else
  log "launching factory_loop.py via tmux"
  tmux kill-session -t factory 2>/dev/null || true
  tmux new-session -d -s factory \
    "cd '$IMAGING_RAG_DIR' && python3 factory_loop.py \
       --task reasoning \
       --min-votes 1 \
       --rate-cap-per-task '$FACTORY_RATE_CAP' \
       --output-dir '$DATA_QUEUE_DIR' \
       --seeds-file '$SEEDS_FILE' \
       --max-disk-gb '$FACTORY_MAX_DISK' \
       --vllm-url http://127.0.0.1:${SERVE_PORT}/v1 \
       --vllm-model '$VLLM_MODEL_BF16' \
       2>&1 | tee -a /tmp/factory_loop.log"
  sleep 10
  if pgrep -af "factory_loop.py" >/dev/null 2>&1; then
    log "factory_loop running; tail /tmp/factory_loop.log to monitor"
  else
    warn "factory_loop did not stay up. /tmp/factory_loop.log:"
    tail -40 /tmp/factory_loop.log >&2 2>/dev/null || true
  fi
fi

log "verified: vLLM=$CONTAINER_NAME on :$SERVE_PORT, nvembed on :$NVEMBED_PORT, biomedclip on :$BIOMEDCLIP_PORT, factory_loop in tmux"
log "next: schedule snapshot_h200_factory_state.sh via cron (see that script's header)."
exit 0
