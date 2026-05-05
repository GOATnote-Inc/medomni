# V_final → HF release runbook

**Date:** 2026-05-05
**Stage:** 4 of the world-class trajectory (after V3.5 ships per PREREG #61).
**Status:** READY-TO-FIRE template. Will be executed when V3.5 paired-bootstrap CI clears the ship rule.
**Predecessor:** [`findings/2026-05-05-world-class-medomni-strategy/SPEC.md`](../2026-05-05-world-class-medomni-strategy/SPEC.md) §3 Stage 4.

---

## What this stage does

Takes the V3.5 LoRA-stacked checkpoint (cumulative on V2.5 reasoning-SFT + V2.7 tool-call-SFT + V3 GRPO + V3.5 DPO refusal) and produces the publish-ready `huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical` Apache-2.0 artifact:

1. Merge V3.5 LoRA adapter into the BF16 base weights
2. Quantize merged-BF16 → NVFP4 via TensorRT Model Optimizer for Blackwell-edge deploy
3. Smoke-test merged-NVFP4 on 30 fixtures
4. Replace catfish vllm-omni-b300's served model with V_final-NVFP4
5. Smoke-test the public `/4UWHAt/api/agent` end-to-end against V_final
6. HF push: bf16 + nvfp4 variants + adapters + model card + safety datasheet

Total wall: ~4-6 hr (laptop assembly + B300 quantization + smoke).
Total cost: ~$50.

---

## Pre-flight (gating before fire)

| Item | State at fire-time | How to verify |
|---|---|---|
| V3.5 LoRA adapter on lobster | `/workspace/ckpt/v3.5-dpo-out/iter_NNNNN/` exists with adapter_config.json + adapter_model.safetensors | `ls -la /workspace/ckpt/v3.5-dpo-out/` |
| V3.5 PREREG ship rule met | per `findings/2026-05-05-v3.5-dpo-refusal/PREREG.yaml` § ship_rule | `cat results/v3.5-paired-eval/CARD.md` shows MedSafetyBench delta lower-CI-bound ≥ +3pp + no capability regression |
| Cumulative V2.5+V2.7+V3+V3.5 LoRA reachable | LoRAs compose; the V3.5 adapter delta is on top of V3 (which is on top of V2.7, etc.) | inspect adapter chain in `/workspace/ckpt/v*-out/` |
| BF16 base cached on lobster | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` BF16 variant in HF cache | `ls /root/.cache/huggingface/hub/ | grep -i Omni` |
| TensorRT Model Optimizer accessible | `modelopt.torch.quantization` importable in lobster's nemo container | `docker run --rm nvcr.io/nvidia/nemo:26.04.00 python -c "from modelopt.torch.quantization import calibrate_loop"` |
| Disk ≥ 80 GB free on lobster | merged-bf16 model is ~60 GB; quantized-nvfp4 is ~22 GB; room for both intermediate | `df -h /` |
| Catfish maintenance window scheduled | swap is ~5 min outage of `/4UWHAt/api/agent` (model reload). Schedule for low-traffic window | user confirms |
| HF write token (NOT same as read-only) on laptop | `huggingface-cli whoami` returns the GOATnote-Inc-authorized account | `HF_TOKEN_WRITE=hf_...` in laptop env |
| Safety datasheet `SAFETY.md` co-signed | counsel + credentialed physician sign-off per HF model card draft #43 § "Pre-release checklist" | manual confirmation |

---

## Stage 4.1 — Merge V3.5 LoRA into BF16 base

Run on lobster inside `nvcr.io/nvidia/nemo:26.04.00`:

```bash
ssh evil-cyan-lobster '
docker run -d --name v-final-merge --gpus all \
  -v /workspace:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -w /workspace \
  nvcr.io/nvidia/nemo:26.04.00 \
  python -c "
from megatron.bridge.peft.lora import LoRA
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(\"nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning\", trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=True)

# Stack the V3.5 adapter (which is itself stacked on V3, V2.7, V2.5)
lora_v_final = LoRA.from_checkpoint(\"/workspace/ckpt/v3.5-dpo-out/iter_NNNNN\")
provider.apply_peft(lora_v_final)

# Merge LoRA delta into base weights (in-place)
provider.merge_lora_into_base()

# Save merged BF16
provider.save_hf(\"/workspace/ckpt/v-final-bf16-merged\", safe_serialization=True)
print(\"merged BF16 saved\")
"
'

# Verify
ssh evil-cyan-lobster 'ls -la /workspace/ckpt/v-final-bf16-merged/ | head -10'
# Expect: config.json, model-00001-of-00007.safetensors, ..., generation_config.json, tokenizer.json
```

Wall: ~10-15 min (mostly weight save).

---

## Stage 4.2 — NVFP4 quantize via TensorRT Model Optimizer

Run on **catfish** (Blackwell B300 — NVFP4 quantization is Blackwell-only). Requires the merged BF16 to be transferred from lobster:

```bash
# Transfer merged BF16 from lobster → catfish (~60 GB, ~10 min on private network)
ssh evil-cyan-lobster 'tar c /workspace/ckpt/v-final-bf16-merged' | \
  ssh unnecessary-peach-catfish 'cd /tmp/medomni && tar x'

# Quantize on catfish
ssh unnecessary-peach-catfish '
docker run -d --name v-final-quantize --gpus all \
  -v /tmp/medomni:/tmp/medomni \
  -e HF_TOKEN=$HF_TOKEN \
  -w /tmp/medomni \
  nvcr.io/nvidia/nemo:26.04.00 \
  python -c "
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
  \"/tmp/medomni/ckpt/v-final-bf16-merged\",
  torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=\"auto\"
)
tokenizer = AutoTokenizer.from_pretrained(\"/tmp/medomni/ckpt/v-final-bf16-merged\")

# Calibrate on a small medical-text batch (uses HealthBench-Hard pin items)
config = mtq.NVFP4_DEFAULT_CFG  # NVFP4 weights + FP4 attention on Blackwell

def calibrate_loop():
    # 64 samples is sufficient for NVFP4 calibration per ModelOpt docs
    import json
    with open(\"/workspace/corpus/pins/healthbench-hard-1000.yaml\") as f:
        items = [...]  # parse YAML, take first 64 items
    for item in items[:64]:
        inputs = tokenizer(item[\"prompt\"], return_tensors=\"pt\").to(model.device)
        model(**inputs)

mtq.quantize(model, config, forward_loop=calibrate_loop)

mtq.export(model, \"/tmp/medomni/ckpt/v-final-nvfp4\")
print(\"NVFP4 export complete\")
"
'

ssh unnecessary-peach-catfish 'ls -la /tmp/medomni/ckpt/v-final-nvfp4/'
# Expect: config.json, model-*.safetensors, quant_config.json
```

Wall: ~30-45 min (calibration is the longest phase).

---

## Stage 4.3 — Smoke-test merged-NVFP4

```bash
# Stand up V_final on catfish second port (DON'T touch prod vllm-omni-b300 yet)
ssh unnecessary-peach-catfish '
docker run -d --name vllm-v-final-smoke --gpus all --network host --ipc=host \
  -v /tmp/medomni:/tmp/medomni \
  --shm-size=32g \
  medomni/omni:v0.20.0-audio \
    --model /tmp/medomni/ckpt/v-final-nvfp4 \
    --served-model-name v-final \
    --host 0.0.0.0 --port 8005 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.15 \
    --kv-cache-dtype fp8 \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
'

# Wait 8-10 min for cold start
ssh unnecessary-peach-catfish 'curl -s http://127.0.0.1:8005/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])"'
# Expect: v-final

# Run 30-fixture smoke
.venv/bin/python scripts/v_final_smoke.py \
  --serve-url http://catfish:8005/v1 \
  --serve-model v-final \
  --fixtures findings/2026-05-05-v-final-hf-release/30-smoke-fixtures.yaml \
  --grader gpt-4.1 \
  --out results/v-final-smoke-2026-05-05/CARD.md
```

Smoke pass criteria: ≥27 of 30 fixtures (90%) score within ±0.05 of the corresponding V3.5 score on the same fixtures (no big regressions from the merge+quantize lossy step).

If smoke fails: investigate whether the issue is merge bug (compare a fwd-pass output between V3.5 LoRA-applied vs merged-BF16) or quantization loss (compare merged-BF16 vs merged-NVFP4 scores).

---

## Stage 4.4 — Catfish swap (production cutover)

Per `feedback_stage_prod_flag_changes_one_at_a_time.md` and the iter-15 lesson: **rollback ready before stop**.

```bash
ssh unnecessary-peach-catfish '
# Save rollback config
docker inspect vllm-omni-b300 > /tmp/vllm-omni-b300.rollback.json
# Stop and remove
docker stop vllm-omni-b300; docker rm vllm-omni-b300
# Start with V_final-NVFP4 as the served model
docker run -d --name vllm-omni-b300 --restart unless-stopped --gpus all --network host --ipc=host \
  -v /tmp/medomni:/tmp/medomni \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=32g \
  medomni/omni:v0.20.0-audio \
    --model /tmp/medomni/ckpt/v-final-nvfp4 \
    --served-model-name nemotron \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.72 \
    --kv-cache-dtype fp8 \
    --no-async-scheduling \
    --limit-mm-per-prompt "{\"video\": 1, \"image\": 4, \"audio\": 1}" \
    --media-io-kwargs "{\"video\": {\"fps\": 2, \"num_frames\": 256}}" \
    --video-pruning-rate 0.5 \
    --allowed-local-media-path /tmp/medomni \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
'

# Poll readiness — DO NOT declare failure before 10 min
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 60
  ssh catfish 'curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/models'
done

# Smoke public URL
curl -sI https://www.thegoatnote.com/4UWHAt/
curl -sI https://www.thegoatnote.com/4UWHAt/receipts
# Both should return 200/308

# End-to-end smoke via /api/agent (1-token fast prompt)
curl -X POST https://www.thegoatnote.com/4UWHAt/api/agent ...

# IF FAILURE — restore V0 base from rollback
# (quoted command depends on the iter-16 saved rollback pattern)
```

If the public smoke fails, immediately restore from rollback. Public outage budget: ~5-10 min worst case.

Stop the smoke endpoint after prod swap succeeds:
```bash
ssh catfish 'docker stop vllm-v-final-smoke; docker rm vllm-v-final-smoke'
```

---

## Stage 4.5 — HF push (Apache-2.0 release)

After production cutover succeeds and is stable for ≥1 hr, push to Hugging Face:

```bash
# Laptop side
cd /Users/kiteboard/medomni
huggingface-cli login --token $HF_TOKEN_WRITE  # write-scoped token

# Pull the merged-BF16 from lobster (~60 GB)
scp -r evil-cyan-lobster:/workspace/ckpt/v-final-bf16-merged /tmp/medomni-hf-bf16
# Pull the merged-NVFP4 from catfish (~22 GB)
scp -r unnecessary-peach-catfish:/tmp/medomni/ckpt/v-final-nvfp4 /tmp/medomni-hf-nvfp4

# Push BF16 variant
huggingface-cli repo create GOATnote-Inc/medomni-nemotron-3-nano-omni-medical --type model
cd /tmp/medomni-hf-bf16
git lfs install
git remote add hf https://huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical
# Copy README, SAFETY.md, model card, eval CARDs
cp /Users/kiteboard/medomni/findings/2026-05-05-hf-model-card-draft/CARD.md README.md
cp /Users/kiteboard/medomni/SAFETY.md SAFETY.md
git add . && git commit -m "Initial release: v3.5 → BF16 merged" && git push hf main

# Push NVFP4 variant as a separate revision
cd /tmp/medomni-hf-nvfp4
git init
git remote add hf https://huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical
git checkout -b nvfp4
git add . && git commit -m "NVFP4 quantization for Blackwell-edge deploy"
git push hf nvfp4

# Optional: separate adapters branch with V2.5 / V2.7 / V3 / V3.5 LoRA artifacts
# (smaller download for users who want to apply on top of base Nemotron-3-Nano-Omni)
```

---

## Stage 4.6 — Eval gauntlet sweep (CARD-bearing)

Per the SPEC §6 12-benchmark gauntlet. Run all 12 against V_final:

| Benchmark | Anchor to beat | V_final target |
|---|---|---|
| HealthBench-Hard | GPT-5 thinking 0.46 (or Muse Spark 42.8) | competitive (5pp gap acceptable for open) |
| MedQA-USMLE | Med-Gemini-L 97% | ≥85% (open SOTA territory) |
| **MedXpertQA-Text** | **AlphaMed-8B (already > DeepSeek-V3-671B + Sonnet)** | **beat AlphaMed-8B by ≥3pp via 30B + GRPO** |
| **MedAgentBench v1** | Claude 3.5 Sonnet 69.7%; Opus 4.x ~75% | **HEADLINE: beat Opus 4.x by ≥5pp paired CI** |
| BFCL v3 | xLAM-2-70b ~0.91 | beat the open 70B floor |
| MedSafetyBench | GPT-5 ~0.94 | ≥0.92 (within range) |
| ... | ... | ... |

CARD outputs `results/v-final-eval-gauntlet-2026-MM-DD/`:
- per-benchmark CARD.md with paired CIs vs V0 baseline
- summary CARD.md with 12-benchmark table
- arXiv preprint draft (separate path, not blocking HF release)

---

## Decision gates

| Gate | Pass criterion | On fail |
|---|---|---|
| Merge | merged-BF16 forward pass agrees with V3.5 LoRA-applied within float-noise | rebuild merge with explicit dtype management |
| NVFP4 quantize | calibrated; no NaN in outputs; perplexity within 5% of merged-BF16 | re-calibrate with larger sample; check Mamba block compatibility |
| Smoke 30-fixture | ≥27/30 within ±0.05 of V3.5 | abort cutover; investigate quantization loss locus |
| Catfish cutover | `/4UWHAt/api/agent` returns 200 within 10 min | rollback to V0 (saved docker inspect) |
| HF push | repo + branches accessible publicly; safety datasheet co-signed | block push until counsel + physician sign-off |
| Eval gauntlet | MedAgentBench V_final > Opus 4.x lower CI bound | retrain or re-PEFT specific axes |

---

## What this runbook does NOT do

- Does not auto-execute. Requires user-action at each `ssh` block. The runbook is the contract; firing is per-step explicit.
- Does not change `/4UWHAt` UI or API contract. The model swap is invisible to the demo's frontend.
- Does not commit catfish to multi-LoRA hot-loading. Stage 4 produces a single merged-and-quantized model; the `--enable-lora` flag is for development iteration (V2.5/V2.7/V3 cycle), not the final release artifact.
- Does not backport changes to medomni public repo's `web/` UI. Public demo continues serving the agent surface via `/api/agent` regardless of which model variant catfish hosts.

---

## Sources

- World-class trajectory SPEC: `findings/2026-05-05-world-class-medomni-strategy/SPEC.md` §3 Stage 4
- V3.5 PREREG: `findings/2026-05-05-v3.5-dpo-refusal/PREREG.yaml`
- Catfish flag-validation runbook: `findings/2026-05-05-catfish-flag-validation/RUNBOOK.md`
- HF model card draft: `findings/2026-05-05-hf-model-card-draft/CARD.md`
- TensorRT Model Optimizer: NVIDIA developer docs, modelopt.torch.quantization API
- Memory: `feedback_stage_prod_flag_changes_one_at_a_time.md`, `feedback_check_docker_entrypoint_before_docker_run.md`
