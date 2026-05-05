# 5-hour autonomous plan — 2026-05-05 13:50 PT → 18:50 PT

**Trigger:** User stepped away for 5 hours; explicitly authorized autonomous execution. Mission: maximize GPU utility + ship measurable outcomes + research NVIDIA best practices.

**Stop condition:** 18:50 PT (loop self-expires by not calling ScheduleWakeup on the wake fired at or after 18:35 PT).

## Priority list (each wake advances the next pending item)

### Tier 1 — Highest-value GPU outcome (MUST land in 5 hr)

1. **Verify smoke pass** (iter-48, ~14:00 PT) — read `/workspace/v2.5-smoke/train.log` final 50 steps. Pass criteria: loss decreased, 50 steps completed, adapter not necessarily saved (smoke = no save). Ship rule: step time ≤ 12s/step on H200, no NaN losses.

2. **Fire V2.5 production training** (iter-48 if smoke passes, ~14:00 PT). 8-10 hr expected wall-clock; will run past user's 5-hour window. Command:
   ```bash
   ssh evil-cyan-lobster '
   cd /home/ubuntu/medomni && nohup /home/ubuntu/.venv/bin/python3 scripts/train_peft_reasoning.py \
     --output-dir /workspace/v2.5-prod \
     --batch-size 1 --grad-accum 16 --max-seq-len 4096 \
     > /tmp/v2.5-prod.log 2>&1 &
   '
   ```
   (Note: smoke script's effective config; production tunes max-seq-len up if memory permits per iter-49 probe.)

3. **Monitor V2.5 production every 15 min** (iters 49+). Probe pattern:
   ```bash
   ssh evil-cyan-lobster '
   pgrep -af train_peft_reasoning | head -1
   tail -3 /tmp/v2.5-prod.log | tr "\r" "\n" | tail -1
   ls -la /workspace/v2.5-prod/checkpoint-*/adapter_model.safetensors 2>/dev/null | tail -2
   '
   ```
   ESCALATE on terminal text if: (a) process disappears unexpectedly, (b) loss spikes >5× initial, (c) step_time >2× smoke, (d) disk fills <5 GB free, (e) NaN appears in log.

### Tier 2 — Parallel research while training runs (Hours 2-5)

4. **NVIDIA best-practices research** (dispatch agent on iter-49). Cover:
   - NeMo Framework PEFT recipes for 30B-MoE-Mamba models (May 2026)
   - TensorRT Model Optimizer NVFP4 quantization for medical LLMs
   - vLLM 0.20+ flag tuning for B300/Hopper serving
   - Megatron-Bridge vs HF-PEFT-eager comparative speedup data
   - NVIDIA AI Enterprise / NIM medical-model packaging requirements
   - NVIDIA Healthcare Blueprints (BioNeMo, CUDA-Q, AI-on-NIM)
   Output: `findings/2026-05-05-nvidia-best-practices/SPEC.md` — ranked recommendations + cross-reference to existing PREREGs.

5. **V3 PREREG amendment** (iter-50+ once research lands). Bake in PRM channel from `findings/2026-05-05-process-supervision-verifiability/SPEC.md` — composite reward weight re-table to include 4th channel at 0.15.

6. **V3.5 PREREG amendment** (iter-51+). Bake in Cal-DPO + `<abstain/>` channel + Health-ORSC-Bench joint ship rule from `findings/2026-05-05-reliability-calibration/SPEC.md`.

7. **V_final inference scaffolding** (iter-52+). Author `mvp/medomni-inference/`:
   - `system_prompt_v1.md` (MedAgentBench-v2 plan-then-act)
   - `verifier_vote.py` (Best-of-K + PRM-min skeleton)
   - `skills/differential.md`, `skills/calc.md`, `skills/handoff.md`

8. **V2.5 eval CARD scaffold** (iter-53+). Pre-author `findings/2026-05-05-v2.5-eval/CARD.md` with placeholders for ship-rule metrics. Ready to fill once training completes.

9. **README trajectory update** (iter-54+). Surface "V2.5 in flight" + 5-hour autonomous deliverables.

### Tier 3 — Loop hygiene + close-out

10. **Cumulative deliverables CARD** (iter-72, ~18:35 PT). Final summary of every measurable outcome shipped during the 5-hour window. PR + INDEX update.

## Hard constraints (carry-over from harmony contract)

- READ-ONLY ssh probes for monitoring; NO destructive ops on training pod
- NO touching judge-qwen / kokoro / `/var/lib/docker`
- One substantive commit per branch (auto-merger race rule)
- Babysitter cannot kill the V2.5 training process unless catastrophic error
- Stop-condition: NO ScheduleWakeup call on wake fired ≥18:35 PT → loop expires naturally

## Measurable outcomes the user will see at 18:50 PT

| # | Outcome | Verify |
|---|---|---|
| 1 | V2.5 LoRA adapter trained (or substantial training progress, X% complete) | `ls /workspace/v2.5-prod/checkpoint-*/adapter_model.safetensors` |
| 2 | NVIDIA best-practices SPEC (~700 words) | `findings/2026-05-05-nvidia-best-practices/SPEC.md` |
| 3 | V3 PREREG amendment merged | git log `findings/2026-05-05-v3-grpo/PREREG.yaml` |
| 4 | V3.5 PREREG amendment merged | git log `findings/2026-05-05-v3.5-dpo-refusal/PREREG.yaml` |
| 5 | V_final inference scaffolding committed | `mvp/medomni-inference/` |
| 6 | V2.5 eval CARD scaffold | `findings/2026-05-05-v2.5-eval/CARD.md` |
| 7 | Cumulative summary CARD | `findings/2026-05-05-five-hour-autonomous-plan/RESULTS.md` |

## Iteration cadence

- **15-min cadence** for first hour (smoke verify + production fire + first monitor)
- **20-min cadence** for hours 2-4 (training is autonomous; mostly research + doc work)
- **15-min cadence** for last hour (verify training progress + write summary CARD)
