#!/usr/bin/env bash
# scripts/serve_orca_h100.sh — durable launch for `nemotron-serve` on Brev pod
# the H100 serving pod (80GB, Hopper, FP8; since decommissioned). In-repo source of truth for what
# serves `thegoatnote.com/4UWHAt` as MedOmni.
#
# Captured from a live `docker inspect nemotron-serve` on 2026-05-23.
#
# 2026-05-23 P0 incident (audio outage during a16z investor demo):
#   - `--limit-mm-per-prompt '{"audio":1,…}'` promised audio capability.
#   - Container `vllm/vllm-openai:latest` lacks `librosa`/`soundfile`/`torchcodec`.
#   - vLLM lazily holds a `PlaceholderModule` and only fails on the first
#     real audio request with HTTP 400 "Invalid or unsupported audio file".
#   - Startup looked healthy; only end-user audio attempts surfaced the gap.
#   - Fix: bake `librosa + soundfile` into the image (this script does that).
#
# Three modes, ordered from most-durable to most-emergency:
#
#   (1) build     — Build a fresh image from vllm/vllm-openai:latest with
#                   audio deps + run a new container. The canonical durable
#                   path. Use this for blue-green pod replacement.
#   (2) snapshot  — Snapshot the running container into a new image, then
#                   restart from that snapshot. Preserves any ad-hoc state.
#                   Used in the 2026-05-23 hotfix; ~3–5 min downtime.
#   (3) inplace   — pip install audio deps in the running container.
#                   EPHEMERAL (lost on container recreation) AND requires a
#                   container restart for vLLM to pick up the deps (the
#                   PlaceholderModule cache problem). Emergency reference;
#                   prefer (1) for anything durable.

set -euo pipefail

MODE="${1:-build}"
MODEL_ID="${MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
SERVED_NAME="${SERVED_NAME:-nemotron}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron-serve}"
HOST_PORT="${HOST_PORT:-8000}"
HF_CACHE_HOST="${HF_CACHE_HOST:-/home/shadeform/.cache/huggingface}"
MEDIA_HOST="${MEDIA_HOST:-/tmp/medomni}"
SHM_SIZE="${SHM_SIZE:-8g}"   # original was 1g; 8g is safer for audio decode
TODAY="$(date -u +%Y-%m-%d)"
TS="$(date -u +%H%M%S)"
BASE_IMAGE="vllm/vllm-openai:latest"
LOCAL_IMAGE="nemotron-serve-with-audio:${TODAY}"

# vLLM audio decoder backend. Without these, audio_url + data: requests
# return HTTP 400 "Invalid or unsupported audio file" from vLLM (regardless
# of request shape — confirmed against orca on 2026-05-23).
AUDIO_DEPS="librosa soundfile"

# vllm serve args, captured verbatim from the 2026-05-23 docker inspect.
VLLM_ARGS=(
  --model "$MODEL_ID"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0
  --port 8000
  --trust-remote-code
  --tensor-parallel-size 1
  --max-model-len 65536
  --max-num-seqs 4
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.90
  --kv-cache-dtype fp8
  --no-async-scheduling
  --limit-mm-per-prompt '{"video": 1, "image": 4, "audio": 1}'
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
  --video-pruning-rate 0.5
  --allowed-local-media-path /tmp/medomni
  --reasoning-parser nemotron_v3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
)

DOCKER_FLAGS=(
  --name "$CONTAINER_NAME"
  --gpus all
  --shm-size "$SHM_SIZE"
  -p "$HOST_PORT:8000"
  -v "$HF_CACHE_HOST:/root/.cache/huggingface"
  -v "$MEDIA_HOST:/tmp/medomni"
  -d
)

wait_for_vllm_ready() {
  echo "Polling /v1/models for readiness (up to 5 min)..."
  for i in $(seq 1 60); do
    if curl -sSf "http://localhost:${HOST_PORT}/v1/models" > /dev/null 2>&1; then
      echo "vLLM ready after ${i} polls (~$((i*5))s)"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: vLLM not ready after 5 min" >&2
  return 1
}

verify_audio_deps() {
  echo "Verifying audio decoder is importable in the container..."
  if docker exec "$CONTAINER_NAME" python3 -c "import librosa, soundfile; print('OK', librosa.__version__, soundfile.__version__)"; then
    echo "Audio deps: OK"
  else
    echo "ERROR: audio deps missing in container after start" >&2
    return 1
  fi
}

case "$MODE" in
  build)
    echo "[build] Creating fresh image with audio deps baked in"
    BUILD_DIR="$(mktemp -d)"
    cat > "$BUILD_DIR/Dockerfile" <<DOCKERFILE
FROM ${BASE_IMAGE}
RUN pip install --no-cache-dir ${AUDIO_DEPS}
DOCKERFILE
    docker build -t "$LOCAL_IMAGE" "$BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "[build] Stopping existing container (this is the downtime window)"
    docker rename "$CONTAINER_NAME" "${CONTAINER_NAME}-pre-${TODAY}-${TS}" 2>/dev/null || true
    docker stop "${CONTAINER_NAME}-pre-${TODAY}-${TS}" 2>/dev/null || true
    echo "[build] Running new container from $LOCAL_IMAGE"
    docker run "${DOCKER_FLAGS[@]}" "$LOCAL_IMAGE" "${VLLM_ARGS[@]}"
    wait_for_vllm_ready
    verify_audio_deps
    echo "[build] Done. Rollback: docker stop ${CONTAINER_NAME} && docker rename ${CONTAINER_NAME}-pre-${TODAY}-${TS} ${CONTAINER_NAME} && docker start ${CONTAINER_NAME}"
    ;;

  snapshot)
    echo "[snapshot] Snapshotting current $CONTAINER_NAME to $LOCAL_IMAGE"
    docker commit \
      --message "Snapshot of $CONTAINER_NAME (audio deps pre-installed in overlay)" \
      "$CONTAINER_NAME" \
      "$LOCAL_IMAGE"
    echo "[snapshot] Stopping existing container (downtime window)"
    docker rename "$CONTAINER_NAME" "${CONTAINER_NAME}-pre-${TODAY}-${TS}"
    docker stop "${CONTAINER_NAME}-pre-${TODAY}-${TS}"
    echo "[snapshot] Running new container from $LOCAL_IMAGE"
    docker run "${DOCKER_FLAGS[@]}" "$LOCAL_IMAGE" "${VLLM_ARGS[@]}"
    wait_for_vllm_ready
    verify_audio_deps
    echo "[snapshot] Done. Rollback: docker stop ${CONTAINER_NAME} && docker rename ${CONTAINER_NAME}-pre-${TODAY}-${TS} ${CONTAINER_NAME} && docker start ${CONTAINER_NAME}"
    ;;

  inplace)
    echo "[inplace] Installing $AUDIO_DEPS into the running $CONTAINER_NAME"
    docker exec "$CONTAINER_NAME" pip install --quiet $AUDIO_DEPS
    echo "WARNING: vLLM has cached PlaceholderModule from startup. The deps"
    echo "are installed but the running vLLM cannot use them until restart."
    echo "Use 'docker restart $CONTAINER_NAME' (~3–5 min downtime) OR switch"
    echo "to the 'build' mode for the durable fix."
    ;;

  *)
    echo "usage: $0 [build|snapshot|inplace]" >&2
    exit 2
    ;;
esac
