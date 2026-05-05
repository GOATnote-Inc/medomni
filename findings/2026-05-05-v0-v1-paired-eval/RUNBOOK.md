# V0 → V1 paired-eval runbook

**Date:** 2026-05-05
**Trigger:** ESCALATION block in LOOP-STATUS.md — V1 trained 2026-05-03 but never evaluated against V0. Blocks closure on the V1 disposition decision (per the world-class trajectory SPEC: V1 is salvaged as a research artifact, not deployed).
**Status:** READY-TO-FIRE pending V1 export completion (currently OOM-killed, blocked on HF_TOKEN).

---

## Critical findings before fire

### V1 LoRA target_modules (extracted from `v1_export_2.log` 2026-05-05)

```
peft_cfg = {
  'alpha': 32, 'dim': 32, 'dropout': 0.0, 'dropout_position': 'pre',
  'target_modules': ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2', 'in_proj', 'out_proj']
}
```

`linear_fc1` and `linear_fc2` are the FFN/MoE-expert projection layers. Per Team #2's NVFP4-LoRA-hot-load constraint (`findings/2026-05-05-world-class-medomni-strategy/SPEC.md` §4.2): **MoE-expert weights cannot be LoRA-hot-loaded on NVFP4 base.** V1 deploy to catfish would require merge-then-requantize (a multi-hour pipeline that replaces catfish's served model with a text-only variant — losing multimodal Omni). **V1 is text-only research-artifact-only.** This eval CARD will document that explicitly.

### V1 base mismatch with catfish

V1 trained against `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (text-only). catfish serves `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (multimodal). **Different base models.** Even ignoring the MoE-expert constraint, V1's LoRA weights are not interpretable on a different base model.

### V1 was actually multi-task TEXT SFT, not imaging-PEFT

Per `prod_train.log` on lobster: V1 trained on MedQA + HealthBench + MedMCQA + PubMedQA-L (all answer-only text datasets). Some docs claimed V1 was "imaging-PEFT on PubMedVision" — that was an earlier-planned-but-not-fired stage (or a separate narwhal-side training that didn't produce the V1 in `/workspace/ckpt/v1-pathd-out/`). The V1 we have is the **answer-only-multi-task-SFT** approach — exactly the one Team #1's research synthesis named as **dominated** by reasoning-trace SFT + verifiable-reward RL.

### Predicted result of this eval

Per Team #1: answer-only multi-task SFT plateaus 5-10pp below the GRPO ceiling. Empirical anchors:
- AlphaMed-8B (GRPO) beats DeepSeek-V3-671B + Claude-3.5-Sonnet on MedXpertQA
- MediX-R1-8B (GRPO) at 68.8% beats MedGemma-27B at 68.4%
- DPO-only / SFT-only recipes (Med42-v2, OpenBioLLM, vanilla Aloe-Beta) plateau

**Prediction:** V1 vs V0 paired-bootstrap CI on MedQA + HealthBench-Hard + MedXpertQA-Text shows MARGINAL or zero lift; CIs likely overlap zero. Either result is informative:
- (a) marginal lift confirms the dominated-approach hypothesis → V2.5 reasoning-SFT is the right pivot
- (b) significant lift would surprise; would warrant re-thinking V2.5 priority vs. just continuing answer-only SFT

---

## Pre-flight blockers

| Blocker | State | Resolution |
|---|---|---|
| **V1 in HF format** | Not exported. Container `v1-export-2` Exited (137) ~2 hr ago during base-model download. Output `/workspace/ckpt/v1-pathd-hf` empty. | Set HF_TOKEN via Brev console env-var UI (NEVER via ssh per `feedback_runpod_proxy_pty_echo.md`). Re-fire `export_v1_adapter_strict.py` with HF_TOKEN in env. |
| **sovereign_bench.py available on lobster** | Not found at `/workspace/sovereign_bench.py`. Likely in `prism42-nemotron-med` repo, needs copy onto lobster. | `scp prism42-nemotron-med/scripts/sovereign_bench.py lobster:/workspace/scripts/` from laptop |
| **V0 + V1 vllm endpoints can co-resident** | Lobster H200 144 GB; Qwen judge holds ~14 GB; BF16-30B + V1 LoRA ≈ 60 GB; second BF16-30B (V0) ≈ 60 GB. **Total: 134 GB / 144 GB.** | Tight but fits. Use `--gpu-memory-utilization 0.42` per endpoint (60/144 = 0.42) on each. |
| **HealthBench Hard 1000 manifest accessible from lobster** | `corpus/pins/healthbench-hard-1000.yaml` exists in laptop `prism42-nemotron-med` repo; must be on lobster. | `scp prism42-nemotron-med/corpus/pins/healthbench-hard-1000.yaml lobster:/workspace/corpus/pins/` |

---

## Execution (when blockers clear)

### Phase 1 — V1 export (re-fire after HF_TOKEN set)

```bash
ssh evil-cyan-lobster '
# Confirm HF_TOKEN is set in pod env
[ -n "$HF_TOKEN" ] || { echo "HF_TOKEN missing — set via Brev console"; exit 1; }

# Clean prior failed export
docker rm -f v1-export-2 2>/dev/null

# Re-fire with HF_TOKEN
docker run -d --name v1-export-3 --gpus all \
  -v /workspace:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -w /workspace \
  nvcr.io/nvidia/nemo:26.04.00 \
  bash -c "python /workspace/export_v1_adapter_strict.py 2>&1 | tee /workspace/logs/v1_export_3.log"

# Monitor — should complete in ~10-15 min once base download starts
docker logs -f v1-export-3 2>&1 | head -50
'

# Verify output
ssh evil-cyan-lobster 'ls -la /workspace/ckpt/v1-pathd-hf/'
# Expect: adapter_config.json, adapter_model.safetensors
```

### Phase 2 — Spin up V0 + V1 vllm endpoints

```bash
ssh evil-cyan-lobster '
# V0 — base BF16, port 8001
docker run -d --name vllm-v0 --gpus all --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=16g \
  vllm/vllm-openai:v0.20.0 \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --served-model-name v0-base \
    --host 0.0.0.0 --port 8001 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.42 \
    --kv-cache-dtype fp8

# V1 — same base + V1 LoRA, port 8002
docker run -d --name vllm-v1 --gpus all --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /workspace/ckpt/v1-pathd-hf:/lora/v1 \
  --shm-size=16g \
  vllm/vllm-openai:v0.20.0 \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --served-model-name v1-text-sft \
    --host 0.0.0.0 --port 8002 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.42 \
    --kv-cache-dtype fp8 \
    --enable-lora \
    --lora-modules v1=/lora/v1
'

# Smoke both endpoints (wait 8-10 min for cold start each)
ssh evil-cyan-lobster '
  curl -s http://127.0.0.1:8001/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])"
  curl -s http://127.0.0.1:8002/v1/models | python3 -c "import json,sys; print([m[\"id\"] for m in json.load(sys.stdin)[\"data\"]])"
'
# Expect: v0-base; ['v1-text-sft', 'v1']
```

### Phase 3 — Run paired eval

```bash
# From laptop (sovereign_bench.py invocation; tunnel through ssh -L if needed)
cd /Users/kiteboard/prism42-nemotron-med
ssh -fN -L 8001:127.0.0.1:8001 evil-cyan-lobster
ssh -fN -L 8002:127.0.0.1:8002 evil-cyan-lobster

# Pre-flight judge key check (per feedback_eval_preflight_judge_key.md)
.venv/bin/python -c "
import os
from openai import OpenAI
c = OpenAI()
r = c.chat.completions.create(model='gpt-4.1', messages=[{'role':'user','content':'ping'}], max_completion_tokens=1)
print('judge OK:', r.choices[0].message.content)
"

# V0 run
.venv/bin/python scripts/sovereign_bench.py \
  --manifest corpus/pins/healthbench-hard-1000.yaml \
  --serve-url http://127.0.0.1:8001/v1 \
  --serve-model v0-base \
  --judge-url https://api.openai.com/v1 \
  --judge-model gpt-4.1 \
  --n 1000 --trials 3 \
  --seeds 42,123,7919 \
  --max-tokens 2048 \
  --out results/v0-baseline-2026-05-05/healthbench-hard-n1000.json

# V1 run (same prompts, same seeds, same grader → paired CI)
.venv/bin/python scripts/sovereign_bench.py \
  --manifest corpus/pins/healthbench-hard-1000.yaml \
  --serve-url http://127.0.0.1:8002/v1 \
  --serve-model v1-text-sft \
  --judge-url https://api.openai.com/v1 \
  --judge-model gpt-4.1 \
  --n 1000 --trials 3 \
  --seeds 42,123,7919 \
  --max-tokens 2048 \
  --out results/v1-text-sft-2026-05-05/healthbench-hard-n1000.json
```

### Phase 4 — Compute paired-bootstrap CIs

```bash
.venv/bin/python scripts/compare_cards.py \
  --baseline results/v0-baseline-2026-05-05/healthbench-hard-n1000.json \
  --comparator results/v1-text-sft-2026-05-05/healthbench-hard-n1000.json \
  --bootstrap 10000 \
  --ci 0.95 \
  --out results/v0-vs-v1-paired-card/CARD.md
```

### Phase 5 — Cleanup

```bash
# Free GPU memory after eval completes
ssh evil-cyan-lobster '
docker stop vllm-v0 vllm-v1
docker rm vllm-v0 vllm-v1
'
```

---

## Reproducibility manifest (per SPEC §6.2)

The CARD at `results/v0-vs-v1-paired-card/CARD.md` MUST include:

- `weights_sha256`: V0 base + V1 LoRA adapter
- `base_model_id`: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `eval_script_git_sha` + `eval_script_url`: `prism42-nemotron-med` HEAD when eval ran
- `prompt_template_sha256`: HealthBench Hard 1000 system prompt
- `system_prompt_sha256`: separately for V0 + V1 if differ
- `grader_model_id`: `gpt-4.1` (specific dated variant if known)
- `grader_prompt_sha256`: HealthBench rubric grader template
- `decode_params`: temp=0.0, top_p=1.0, max_new_tokens=2048, no thinking budget
- `seed_list`: [42, 123, 7919]
- `item_manifest_sha256`: corpus/pins/healthbench-hard-1000.yaml
- `hardware`: H200 (lobster), CUDA 13.2.x, vllm v0.20.0
- `runtime_seconds`: per-trial wall clock
- `total_tokens_in/out`: cumulative
- `cost_usd`: gpt-4.1 grader cost (estimated ~$8-15 for 1000 items × 3 trials × 2 models)

---

## Acceptance criteria — what makes the CARD shippable

1. Both V0 and V1 ran 1000 items × 3 seeds × N≥3 trials each on the SAME item set (paired)
2. gpt-4.1 grader returned valid scores for ≥95% of items per trial
3. Paired-bootstrap 10,000 resamples produces 95% CIs
4. V0/V1 delta + CI is the headline number; per-axis breakdowns (responding-under-uncertainty, communication-quality, etc.) are secondary

If V1 vs V0 CI lower bound > 0 → V1 measurably moved the needle on text-only SFT (surprising; would warrant re-thinking V2.5 priority)
If V1 vs V0 CI overlaps zero → V1 is dominated as Team #1 predicted; V2.5 reasoning-SFT is correct pivot
If V1 vs V0 CI upper bound < 0 → V1 REGRESSED; investigate (corpus quality, training instability)

---

## What this runbook does NOT do

- Does not deploy V1 to catfish. V1 LoRA targets `linear_fc1/fc2` (MoE experts) which cannot be hot-loaded on NVFP4 base; V1 is research-artifact-only.
- Does not commit `corpus/pins/healthbench-hard-1000.yaml` to medomni public repo. That manifest is in the private `prism42-nemotron-med` repo per CLAUDE.md §1 (training-loop surface).
- Does not modify catfish. Lobster-only eval. Public demo `/4UWHAt` unaffected.
- Does not iterate V2 if V1 lifts. The V2.5 pivot is justified by the May-2026 SOTA literature regardless of V1 result; the V0→V1 CARD is methodology validation, not a recipe-redirect signal.

---

## Sources

- iter-14 ESCALATION block (`LOOP-STATUS.md`)
- World-class trajectory SPEC (`findings/2026-05-05-world-class-medomni-strategy/SPEC.md` §4.2 NVFP4-LoRA constraint)
- V2.5 PREREG (`findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`)
- Memory: `feedback_eval_preflight_judge_key.md`, `feedback_runpod_proxy_pty_echo.md`
- Team #1 agent transcript: `tasks/ab9620b82dca5231d.output` (predicted V1 = answer-only-SFT plateau)
