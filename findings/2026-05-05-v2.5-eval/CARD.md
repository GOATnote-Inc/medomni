# V2.5 reasoning-SFT eval CARD (scaffold)

**Status:** SCAFFOLD — fill in once V2.5 production training completes.
**Authored:** 2026-05-05 iter-54 (during 5-hour autonomous mission).
**Pre-registration:** `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`.
**Training run:** `/workspace/v2.5-prod` on evil-cyan-lobster (H200).

## Run identifiers (TBD on completion)

| Field | Value |
|---|---|
| run_id | `v2.5-prod-2026-05-05` |
| started | 2026-05-05 21:36 UTC (iter-52 re-fire after iter-52 OOM at seq=2048) |
| training script | `prism42-nemotron-med/scripts/train_peft_reasoning.py` |
| script git sha | TBD |
| base_model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| base_weights_sha256 | snapshot `a5fba9ed825b7a641606d97c8f189ce01e3a7cf0` |
| training corpora | MedReason 32K + medical-o1-reasoning-SFT (en) 25K |
| corpora_sha256 | TBD (pull from train_summary.json) |
| total_steps | TBD (planned 3243 for full epoch; re-fire seq=1536 after OOM at step 45) |
| wallclock | TBD |
| final_train_loss | TBD |
| final_eval_loss | TBD |
| adapter_path | `/workspace/v2.5-prod/adapter_model.safetensors` |
| adapter_sha256 | TBD |
| trainable_params | 36,581,376 (0.116% of 31.6B; smoke confirmed) |

## Hyperparameters (final-config; differs from PREREG due to OOM remediation)

| Field | PREREG | Smoke (50-step) | Production (post-OOM) |
|---|---|---|---|
| precision | bf16-mixed | bf16-mixed | bf16-mixed |
| max_seq_length | 8192 | 1024 | **1536** (after seq=2048 OOM iter-52) |
| per_device_batch_size | 2 | 1 | 1 |
| grad_accum | 16 (eff 32) | 4 (eff 4) | 16 (eff 16) |
| learning_rate | 2e-5 | 2e-5 | 2e-5 |
| warmup_ratio | 0.03 | 0.03 | 0.03 |
| scheduler | cosine | cosine | cosine |
| epochs | 1 | n/a (50 steps) | 1 |
| LoRA rank | 64 | 64 | 64 |
| LoRA alpha | 128 | 128 | 128 |
| LoRA target_modules | `q,k,v,o_proj, in,out_proj, mlp1` (PR #82) | same | same |
| attn_implementation | sdpa or fa2 | eager (FA2 unavailable) | eager |
| PYTORCH_CUDA_ALLOC_CONF | n/a | n/a | `expandable_segments:True` (post-OOM) |
| gradient_checkpointing | optional | rejected by NemotronH | rejected by NemotronH |

## Smoke result (iter-47, anchor for production gate)

```json
{
  "adapter_exists": true,
  "adapter_sha256": "bb6cce472ca5394edf6adf74707e4b64c28522b3d2e984626335d3f0bedbf35a",
  "final_eval_loss": 1.5110784769058228,
  "final_train_loss": 1.356953501701355,
  "n_examples_seen": 1900,
  "walltime_s": 468.4742081165314
}
```

Smoke passed: train loss decreased, eval loss only slightly higher (no overfit), adapter saved with valid sha256. Production fire authorized.

## Production training timeline (live; updated per loop iter)

| Iter | Time UTC | Step | Loss | Notes |
|---|---|---|---|---|
| iter-47 | 21:08 | 50/50 (smoke) | train 1.357 / eval 1.511 | smoke PASS — adapter saved |
| iter-47 | 21:09 | 0/3243 (prod fire) | — | first prod fire seq=2048 |
| iter-52 | 21:29 | 45/3243 | OOM | judge-qwen + training collision; saved durable lesson |
| iter-52 | 21:36 | 0/3243 (re-fire) | — | seq=1536 + expandable_segments |
| iter-54 | 21:46 | 23/3243 | TBD | 24.4s/step stable |
| ... | | | | |
| (final) | TBD | 3243/3243 | TBD | epoch complete; ship-rule eval fires |

## Eval protocol (per PREREG `eval_protocol`)

Run after `adapter_model.safetensors` lands at `/workspace/v2.5-prod/`:

```bash
ssh evil-cyan-lobster '
cd /home/ubuntu/medomni && /home/ubuntu/.venv/bin/python3 scripts/sovereign_bench.py \
  --base nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --lora /workspace/v2.5-prod/adapter_model.safetensors \
  --tasks MedQA-USMLE,MedXpertQA-Text,HealthBench-Hard,PubMedQA-L \
  --paired-bootstrap 95 \
  --n-trials 3 \
  --seeds 42,123,7919 \
  --output-dir /workspace/v2.5-eval \
  > /tmp/v2.5-eval.log 2>&1
'
```

Estimated wall-time: ~2 hr on lobster (assumes judge-qwen still serving; eval reads from same vllm endpoint).

## Ship rule (verbatim from PREREG)

| Metric | Required | Got (TBD) | PASS? |
|---|---|---|---|
| MedQA-USMLE delta lower-CI-bound > 0 | > 0 pp | TBD | TBD |
| MedXpertQA-Text delta lower-CI-bound ≥ +5pp | ≥ +5 pp | TBD | TBD |
| HealthBench-Hard delta point-estimate > 0 | > 0 | TBD | TBD |
| PubMedQA-L no-regression | ≥ -1 pp lower CI | TBD | TBD |
| Manifest sha256 verifiable | yes | TBD | TBD |

**On PASS:** proceed to V2.7 tool-call SFT (no public deploy yet). PREREG amendment (PR #86: PRM channel) applies at V3 stage; V2.5 PREREG unchanged except `mlp1` correction (PR #82).

**On FAIL:** revert; debug data quality (likely insufficient CoT diversity); re-author V2.5b PREREG.

## Differential from V0 (anchor)

V0 baselines (canonical gpt-4.1 graded, per `findings/2026-05-01-medgemma-recipe/MEDOMNI-NEMOTRON-RECIPE.md`):

| Bench | V0 (canonical) | V2.5 target (PREREG) | V2.5 actual (TBD) |
|---|---|---|---|
| MedQA-USMLE | 0.74 (rough; multimodal HB-Hard 0.054 was the eye-opener) | +5 pp delta | TBD |
| MedXpertQA-Text | TBD V0 anchor | +10 pp delta | TBD |
| HealthBench-Hard | 0.054 | +0 pp (not regressing is the gate) | TBD |
| PubMedQA-L | TBD | not regressing (-1pp lower CI) | TBD |

## What this CARD does NOT do

- Does NOT replace the formal V_final HF model card (separate doc at
  `findings/2026-05-05-hf-model-card-draft/CARD.md`).
- Does NOT propagate to README until ship rule is decided.
- Does NOT include audio/vision evals (V2.5 is text-only reasoning SFT;
  vision/audio deltas measured at V_final post-merge per the trajectory).

## Cross-references

- [`findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`](../2026-05-05-v2.5-reasoning-sft/PREREG.yaml) — pre-reg
- [`findings/2026-05-05-fire-v2.5-runbook/RUNBOOK.md`](../2026-05-05-fire-v2.5-runbook/RUNBOOK.md) — Step 0-6 user-action
- [`findings/2026-05-05-v2.5-base-precision-mismatch/CARD.md`](../2026-05-05-v2.5-base-precision-mismatch/CARD.md) — why BF16 not FP8
- [`feedback_lobster_oom_judge_collision.md`](../../../.claude/projects/-Users-kiteboard/memory/feedback_lobster_oom_judge_collision.md) — durable lesson from iter-52 OOM
