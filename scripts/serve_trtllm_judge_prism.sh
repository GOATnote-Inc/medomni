#!/usr/bin/env bash
# Phase 2.2 — TensorRT-LLM-FP8 cross-family judge on RunPod H100 prism.
#
# Model: Qwen/Qwen2.5-7B-Instruct (sovereignty-mandated cross-family judge,
#        per SPEC.md §5.2 line "Judge (cross-family, sovereign)").
#
# Pipeline:
#   1. HF snapshot_download Qwen2.5-7B-Instruct → /workspace/hf_cache
#   2. ModelOpt PTQ FP8 calibration on cnn_dailymail (TRT-LLM standard recipe)
#   3. trtllm-build → /workspace/trtllm_cache/qwen2.5-7b-fp8/
#   4. trtllm-serve on 127.0.0.1:8003 (OpenAI-compatible)
#
# This script is meant to be RUN ON THE POD via:
#   bash scripts/_runpod_ssh.sh < scripts/serve_trtllm_judge_prism.sh
#
# Pod context:
#   - RunPod H100 80GB (Hopper sm_90, FP8-native)
#   - Host CUDA 12.4, driver 580.126.09
#   - No docker. Native pip install of tensorrt-llm.
#   - /workspace is NFS-shared persistent (mfs#us-mo-1.runpod.net).
#   - System Python is 3.11.10 BUT TRT-LLM has zero cp311 wheels published.
#     We install + run under /usr/bin/python3.10 with site-packages at
#     /workspace/py310-site (persistent, NFS) via PYTHONPATH.
#   - HF_TOKEN is NOT required (Qwen2.5-7B and the reranker are PUBLIC on HF).
#     If HF rate-limits, set HF_TOKEN as a Pod Environment Variable in the
#     RunPod console — the proxy does NOT echo console-set env vars.
#
# Sovereignty: weights resident on /workspace/hf_cache. No cloud LLM keys.
set -euo pipefail

export HF_HOME=/workspace/hf_cache
export TRT_CACHE=/workspace/trtllm_cache
export PYTHONUNBUFFERED=1
export PY310_SITE=/workspace/py310-site
export PYTHONPATH="${PY310_SITE}:${PYTHONPATH:-}"
PY=/usr/bin/python3.10
mkdir -p "$HF_HOME" "$TRT_CACHE" "$PY310_SITE"

JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
JUDGE_LOCAL="$HF_HOME/qwen2.5-7b-instruct"
ENGINE_DIR="$TRT_CACHE/qwen2.5-7b-fp8"
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

# 1. Install TRT-LLM if absent. Pin 0.21.0 (cp310 wheels available; Hopper FP8
#    fully supported; built against CUDA 12.x).
if ! "$PY" -c "import tensorrt_llm" 2>/dev/null; then
  echo "[judge] installing tensorrt-llm 0.21.0 + modelopt 0.21.0 into $PY310_SITE..."
  # Bootstrap pip-for-3.10 if absent.
  if ! "$PY" -c "import pip" 2>/dev/null; then
    pip3 download --no-deps -d /tmp/pipwheel pip setuptools wheel >/dev/null
    PYTHONPATH=/tmp/pipwheel/$(ls /tmp/pipwheel/ | grep '^pip-') "$PY" -m pip \
      install --target="$PY310_SITE" \
      /tmp/pipwheel/pip-*.whl /tmp/pipwheel/setuptools-*.whl /tmp/pipwheel/wheel-*.whl
  fi
  "$PY" -m pip install --target="$PY310_SITE" \
    --extra-index-url https://pypi.nvidia.com \
    "tensorrt-llm==0.21.0" \
    "nvidia-modelopt[torch]==0.21.0" \
    "transformers>=4.45,<4.50" \
    "huggingface-hub>=0.26" \
    "datasets>=3.0" \
    "accelerate>=0.34" 2>&1 | tail -30
fi
"$PY" -c "import tensorrt_llm; print('trtllm', tensorrt_llm.__version__)"

# 2. HF download. Qwen2.5-7B-Instruct is PUBLIC (gated=false); no token needed.
#    If HF rate-limits, the user can set HF_TOKEN via Pod Env Vars in the
#    RunPod console (NEVER pushed via the _runpod_ssh.sh wrapper — PTY-echo
#    leaks the secret to logs/transcripts).
if [ ! -f "$JUDGE_LOCAL/config.json" ]; then
  echo "[judge] downloading $JUDGE_MODEL → $JUDGE_LOCAL (anonymous; public model)"
  "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('${JUDGE_MODEL}', local_dir='${JUDGE_LOCAL}',
                  local_dir_use_symlinks=False, max_workers=8)
print('[judge] download done')
"
fi

# 3. Serve via TRT-LLM 0.21.0 PyTorch backend with FP8 quant.
# This unifies steps 3 + 4 of the original recipe (ModelOpt PTQ + trtllm-build
# + trtllm-serve) into a single command. The PyTorch backend in 0.21.0 supports
# `--backend pytorch --quant_config fp8` which does just-in-time FP8 quantization
# at engine-load time. Skips the 20-30 min static `trtllm-build` step entirely.
#
# Reference: TensorRT-LLM 0.21.0 docs:
#   https://nvidia.github.io/TensorRT-LLM/v0.21.0/commands/trtllm-serve.html
#
# Trade-off: PyTorch-backend FP8 may not match the absolute throughput of a
# fully-compiled TRT engine (GEMM kernel autotuning is skipped) but it
# preserves the FP8 numerics + KV-cache quantization that drive the bulk of
# the H100 throughput win over BF16.

# 4. Serve. trtllm-serve exposes OpenAI-compatible /v1/chat/completions on $PORT.
PORT=8003
TRTLLM_SERVE="${PY310_SITE}/bin/trtllm-serve"

# Build a minimal extra_llm_api_options YAML that pins fp8 KV cache.
LLM_OPTS="$LOG_DIR/judge_llm_opts.yaml"
cat > "$LLM_OPTS" <<'YAMLEOF'
# TRT-LLM 0.21.0 PyTorch backend: enable FP8 weight + KV-cache quantization.
# Reference: https://nvidia.github.io/TensorRT-LLM/v0.21.0/llm-api/reference.html
kv_cache_dtype: fp8
YAMLEOF

if pgrep -f "trtllm-serve.*${JUDGE_LOCAL}" >/dev/null; then
  echo "[judge] trtllm-serve already running"
else
  echo "[judge] starting trtllm-serve on 127.0.0.1:${PORT} (pytorch-backend, fp8)..."
  PYTHONPATH="$PY310_SITE" nohup "$TRTLLM_SERVE" \
    "$JUDGE_LOCAL" \
    --backend pytorch \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max_batch_size 8 \
    --max_seq_len 8192 \
    --max_num_tokens 8192 \
    --kv_cache_free_gpu_memory_fraction 0.6 \
    --extra_llm_api_options "$LLM_OPTS" \
    > "$LOG_DIR/judge_serve.log" 2>&1 &
  echo "[judge] pid=$!"
fi

# Wait up to 600s for /v1/models — first-time pytorch-backend load includes
# JIT FP8 quantization of all linear layers + KV-cache allocation, which can
# take 60-120s on a 7B model.
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[judge] /v1/models READY (after ${i}s)"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400
    echo
    break
  fi
  sleep 1
done

echo "[judge] DONE"
