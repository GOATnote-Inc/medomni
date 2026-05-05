# Findings index

Ordered top-to-bottom by relevance for someone landing on the repo cold and asking "what's the path from here to the world-class medical-LLM HF release?"

## The trajectory chain (read in this order)

1. **[`2026-05-05-world-class-medomni-strategy/SPEC.md`](2026-05-05-world-class-medomni-strategy/SPEC.md)** — strategy SPEC. 4-agent research synthesis (training SOTA / B300 serving / multimodal / eval gauntlet). Replaces the prior 2024-vintage V0→V3 plan with the May-2026-current recipe: V2.5 reasoning-SFT → V2.7 tool-call-SFT → V3 GRPO → V3.5 DPO refusal → V_final HF release. Headline target: **open 30B beats Claude Opus on Stanford MedAgentBench by ≥5pp paired CI**.

2. **[`2026-05-05-v2.5-reasoning-sft/PREREG.yaml`](2026-05-05-v2.5-reasoning-sft/PREREG.yaml)** — V2.5 pre-registration. Reasoning-trace SFT on MedReason 32K + medical-o1-reasoning-SFT 25K + 5K R1-distilled USMLE. 8-10 hr H200, ~$80. Ship rule: MedQA-USMLE +5pp, MedXpertQA-Text +10pp paired-CI vs V0.

3. **[`2026-05-05-v2.7-tool-call-sft/PREREG.yaml`](2026-05-05-v2.7-tool-call-sft/PREREG.yaml)** — V2.7. Loss-masked tool-call SFT (Hermes-fc-v1 + ToolACE filtered + 5K synthetic FHIR/MedAgentBench). 4-6 hr, ~$40. Ship rule: BFCL-v3 +10pp, MedAgentBench-v1 +15pp.

4. **[`2026-05-05-v3-grpo/PREREG.yaml`](2026-05-05-v3-grpo/PREREG.yaml)** — V3 GRPO (replaces planned DPO). HuatuoGPT-o1 40K verifiable + Clinical-R1 multi-objective. Composite reward (correctness 0.5 + format 0.2 + sovereign Qwen judge 0.3). 12-18 hr, ~$130. Ship rule: MedXpertQA-Text +5pp, HealthBench-Hard +3pp.

5. **[`2026-05-05-v3.5-dpo-refusal/PREREG.yaml`](2026-05-05-v3.5-dpo-refusal/PREREG.yaml)** — V3.5 DPO refusal calibration. MedSafetyBench + synthetic HealthBench-uncertainty pairs. 4 hr, ~$40. Safety-only — no capability lift target.

6. **[`2026-05-05-v-final-hf-release/RUNBOOK.md`](2026-05-05-v-final-hf-release/RUNBOOK.md)** — V_final → HF release. Merge V3.5 LoRA → BF16 → NVFP4 quantize via TensorRT Model Optimizer → catfish swap → HF push as Apache-2.0. 4-6 hr, ~$50.

## Supporting documents

7. **[`2026-05-05-corpora-license-confirmation/CARD.md`](2026-05-05-corpora-license-confirmation/CARD.md)** — verified 5 of 6 training corpora are Apache-2.0 (V2.5 pre-flight CLEARED). MedSafetyBench needs upstream LICENSE check at V3.5 fire-time; substitution path documented.

8. **[`2026-05-05-v0-v1-paired-eval/RUNBOOK.md`](2026-05-05-v0-v1-paired-eval/RUNBOOK.md)** — close the V1 disposition question. V1 (text multi-task SFT, NOT imaging-PEFT as some docs claim) cannot be deployed to catfish (NVFP4 base + LoRA targets MoE experts). Eval as research artifact only.

9. **[`2026-05-05-catfish-flag-validation/RUNBOOK.md`](2026-05-05-catfish-flag-validation/RUNBOOK.md)** — gate before any future catfish vllm flag-change attempt. After iter-16's ~10 min outage from 3-flag bump, validate each flag individually on a non-prod endpoint first.

10. **[`2026-05-05-hf-model-card-draft/CARD.md`](2026-05-05-hf-model-card-draft/CARD.md)** — draft model card for the planned `huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical` Apache-2.0 release. Apache-2.0 vs MedGemma's HAI-DEF gating is the distribution differentiator.

## Auxiliary

- **[`2026-05-04-pattern-b-spike/RESULTS.md`](2026-05-04-pattern-b-spike/RESULTS.md)** — Pattern B (dual lookup) FHIR-fetch latency benchmark: p95=11ms across 12 patients × 60 samples. Verdict: ships.
- **[`2026-05-05-secret-scanner-swap/CARD.md`](2026-05-05-secret-scanner-swap/CARD.md)** — gitleaks → TruffleHog OSS swap research (NVIDIA bionemo-framework convention).
- **[`2026-05-05-up037-safety-plan/SPEC.md`](2026-05-05-up037-safety-plan/SPEC.md) + [CARD.md](2026-05-05-up037-safety-plan/CARD.md)** — Generator-Validator-Attacker pattern applied to lint cleanup; surfaced 1 latent F821 bug.

## How to fire the chain

The trajectory is **gated on three user-action items** per the LOOP-STATUS ESCALATION block:

1. Set HF_TOKEN on lobster via Brev console env-var UI (never via ssh per `feedback_runpod_proxy_pty_echo.md`)
2. Free ≥30 GB on lobster (currently 94% full); see ESCALATION block for 4-step remediation
3. Download Omni multimodal base (auto once #1 + #2 done)

Once cleared, V2.5 fires per its PREREG. Each subsequent stage's ship rule gates the next. ~32-41 H200-hrs over 5-6 days end-to-end.

## Related private-repo content

The actual training scripts + factory_loop.py + heartbeat_eval_loop.sh + corpus pins live in the **private** sister repo `github.com/GOATnote-Inc/prism42-nemotron-med`. This medomni public repo holds the demo at `https://www.thegoatnote.com/4UWHAt`, the trajectory documents above, and release artifacts.
