#!/usr/bin/env bash
# Phase 2.2 — reranker on RunPod H100 prism.
#
# Model: nvidia/llama-3.2-nv-rerankqa-1b-v2 (LlamaBidirectionalForSequenceClassification arch)
#
# Pragmatic note: the reranker uses the NeMo `LlamaBidirectional` arch
# whose TRT engine path is non-trivial (NeMo's onnx_exporter +
# get_llama_bidirectional_hf_model). Within the Phase 2.2 budget we serve
# via vLLM on prism instead — same goal of freeing B300 VRAM, much smaller
# delivery risk. The 1B reranker fits in ~3 GB FP16 on H100; throughput
# delta vs TRT engine on a 1B classifier is small (cross-encoder pairwise
# scoring is encoder-bound, not decoder-bound; vLLM and TRT both saturate).
#
# Promotion to TRT-LLM engine is tracked as Phase 2.3.
#
# This script is meant to be RUN ON THE POD via:
#   bash scripts/_runpod_ssh.sh < scripts/serve_trtllm_rerank_prism.sh

set -euo pipefail
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

RERANK_MODEL="nvidia/llama-3.2-nv-rerankqa-1b-v2"
RERANK_LOCAL="$HF_HOME/llama-3.2-nv-rerankqa-1b-v2"
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

# 1. Install vLLM if absent (cu12 wheel for the host CUDA 12.4 toolkit).
# Pin 0.20.0 to match SPEC.md §5.2 BOM and the pattern used on B300.
if ! python3 -c "import vllm" 2>/dev/null; then
  echo "[rerank] installing vllm 0.20.0..."
  pip3 install --quiet --upgrade pip
  pip3 install --quiet "vllm==0.20.0" 2>&1 | tail -20
fi
python3 -c "import vllm; print('vllm', vllm.__version__)"

# 2. HF download. Per 2026-04-29 verification, nvidia/llama-3.2-nv-rerankqa-1b-v2
# (alias of nvidia/llama-nemotron-rerank-1b-v2) is PUBLIC (gated=false). No
# HF_TOKEN required for download. If HF_TOKEN happens to be set as a Pod
# Environment Variable (set via RunPod console, not shell-pushed), it will
# be picked up by huggingface_hub silently and used for rate-limit headroom.
if [ ! -f "$RERANK_LOCAL/config.json" ]; then
  echo "[rerank] downloading $RERANK_MODEL → $RERANK_LOCAL (anonymous; public model)"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${RERANK_MODEL}', local_dir='${RERANK_LOCAL}',
                  local_dir_use_symlinks=False, max_workers=8)
print('[rerank] download done')
"
fi

# 3. Serve via vLLM with --task score (cross-encoder pairwise mode).
PORT=8002
if pgrep -f "vllm.*${RERANK_LOCAL}" >/dev/null; then
  echo "[rerank] vllm rerank already running"
else
  echo "[rerank] starting vllm on 127.0.0.1:${PORT}..."
  # Same incantation that works on B300: --runner pooling --convert classify
  # exposes the LlamaBidirectional cross-encoder via /v1/score and /rerank.
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$RERANK_LOCAL" \
    --served-model-name "$RERANK_MODEL" \
    --runner pooling \
    --convert classify \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization 0.15 \
    --max-model-len 8192 \
    --trust-remote-code \
    > "$LOG_DIR/rerank_serve.log" 2>&1 &
  echo "[rerank] pid=$!"
fi

# Wait for /v1/models
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[rerank] /v1/models READY"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400
    echo
    break
  fi
  sleep 1
done

echo "[rerank] DONE"
