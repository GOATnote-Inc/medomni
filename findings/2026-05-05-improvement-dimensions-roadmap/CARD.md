# Improvement-dimensions roadmap — synthesis CARD

**Date:** 2026-05-05 iter-38
**Trigger:** User asked: "consider variants of prompt engineering, RAG and finetuning. consider parallel agent teams to accomplish and consider how this might improve our clinical reasoning and verifiability and reliability."
**Method:** 4-agent parallel research synthesis (read-only, WebFetch). Returns to this CARD.

## TL;DR — what to add at which stage

The existing 5-stage trajectory (V2.5 → V2.7 → V3 → V3.5 → V_final) is correct in shape. The 4-agent synthesis identifies **6 surgical additions** that compound clinical reasoning + verifiability + reliability without restructuring the chain:

| Stage | Add (cost) | Why | Source |
|---|---|---|---|
| V3 GRPO | 4th reward channel: **PRM score** at 0.15 weight (re-weight: 0.45/0.15/0.25/0.15) | Med-PRM EMNLP 2025 +13.5 absolute on MedQA; MedS3 +6.45 across 11 benches | [process-supervision-verifiability/SPEC.md](../2026-05-05-process-supervision-verifiability/SPEC.md) |
| V3.5 DPO refusal | **Cal-DPO** + `<abstain/>` token + Health-ORSC-Bench negatives | MedHallu +38% F1 from explicit abstain; vanilla DPO drifts implicit reward off scale | [reliability-calibration/SPEC.md](../2026-05-05-reliability-calibration/SPEC.md) |
| V_final inference | **MedAgentBench-v2 plan-then-act + tool-call exemplars** system prompt | +20-28 pts on MedAgentBench; aligns with V2.7 tool-call training | [clinical-prompt-engineering/SPEC.md](../2026-05-05-clinical-prompt-engineering/SPEC.md) |
| V_final inference | **Best-of-K + PRM-min + claim-audit** wrapper | MedS3-style trim, FactScore-style claim verification | [process-supervision-verifiability/SPEC.md](../2026-05-05-process-supervision-verifiability/SPEC.md) |
| V_final inference | **Semantic Entropy Probes** (linear probe, ~1× decode cost) | ECE 2-4pp vs verbalized's 28.6pp; cheapest reliable uncertainty | [reliability-calibration/SPEC.md](../2026-05-05-reliability-calibration/SPEC.md) |
| V_final inference | **Conformal prediction sets** for DDx turns | Distribution-free coverage; regulator-legible | [reliability-calibration/SPEC.md](../2026-05-05-reliability-calibration/SPEC.md) |
| RAG infra (parallel) | **Hybrid sparse+dense+ColBERT rerank + MedScore + MedCAT bridge** | Closes alias gaps (ICD/drug/rare-disease); claim-level verifiability; FHIR↔PrimeKG↔retrieval | [clinical-rag-architecture/SPEC.md](../2026-05-05-clinical-rag-architecture/SPEC.md) |

## Three load-bearing caveats from the synthesis

1. **CoT is not an audit trail.** Anthropic 2505.05410 shows reasoner CoT verbalizes <20% of decision-influencing computation. Train V3 PRM to make chains *verifiable against guidelines*, but gate refusals + safety on CLAIM AUDIT (output), not chain inspection. (Process supervision SPEC §"Faithfulness".)
2. **Verbalized confidence is dangerously overconfident.** ECE blows 3.5pp → 28.6pp under verbalized vs logits (ABC paper). Never log only verbalized. SEP + conformal beat it at near-zero extra cost. (Reliability SPEC §"Black-box vs white-box".)
3. **Don't ship long-context-only RAG.** Cost is ~1250× retrieval; loses citation traceability that clinicians legally need. Use long-context to *reason over* the retrieved set, not to *replace* it. (RAG SPEC §"Long-context-only fallback".)

## Three "don't do this" findings

- **Don't add a second graph** for personalization. MedCAT linker over Pattern B FHIR + existing PrimeKG covers it. (RAG Agent B's "load-bearing correction".)
- **Don't keep few-shot exemplars for o-style reasoners.** Microsoft's 2411.03590 shows reasoners *lose* points from few-shot. Ensembling still gains. Drop k-NN exemplars from any V_final wrapper. (Prompt-engineering SPEC §"Medprompt-style ensembling without few-shot".)
- **Don't gate refusals on a CoT-monitor.** Per-turn output check (claim audit + outcome RM), never inspection of the chain. (Process-supervision SPEC §"Faithfulness".)

## Compound impact estimate (rough)

Stacking the 6 stage additions:

| Metric | V0 baseline | V_final (current PREREG) | V_final + 6 additions | Δ |
|---|---|---|---|---|
| MedAgentBench | 69.7% (Claude Sonnet) | ~78% | ~92-95% | +6 stages = +17-20 pts |
| MedQA (ECE) | n/a | ~8pp | ~3pp | -5pp |
| Worst-at-k (HealthBench) | ~22% (o3 = 60%) | ~50% | ~58% | +8 pts |
| Hallucination rate | ~15-20% | ~10% | ~3-5% | -7-15 pts |
| Refusal F1 (MedSafety × ORSC joint) | n/a | training-only | calibrated | dual-axis ship rule |

Numbers are research-paper estimates; medomni-specific ones land in PREREG ship rules.

## Concrete next-PR sequence (after disk + Omni unblock)

Per harmony contract, babysitter cannot fire training. User-action sequence:

1. **PREREG amendments** (one PR each, low-risk doc edits):
   - V3 PREREG → add PRM channel; revise composite reward weights.
   - V3.5 PREREG → swap DPO → Cal-DPO; add `<abstain/>`; add Health-ORSC-Bench joint ship rule.
2. **MVP scaffolding** (when V_final approaches):
   - `mvp/medomni-inference/system_prompt_v1.md` (MedAgentBench-v2 header).
   - `mvp/medomni-inference/verifier_vote.py` (Best-of-K + PRM-min).
   - `mvp/medomni-inference/skills/` (`/differential`, `/calc`, `/handoff`).
3. **RAG infra additions** (parallel track, doesn't block training):
   - BM25 sidecar + ColBERTv2 rerank.
   - MedScore + RAGAS faithfulness gate.
   - MedCAT linker bridging FHIR + PrimeKG.

These compound. Order is decoupled — RAG can start now; PREREG amendments wait for V2.5 fire; MVP wrappers wait for V_final.

## Relation to existing trajectory

This CARD does NOT replace `/findings/2026-05-05-world-class-medomni-strategy/SPEC.md`. It augments it by:
- Naming concrete recipes per stage (was "GRPO" — now "GRPO + PRM channel").
- Adding inference-time wrappers (V_final was "merge → quantize → push" — now also "system prompt + verifier vote + SEP + conformal").
- Surfacing 3 don't-do-this errors that would have wasted iterations.

## What this CARD does NOT do

- Does not fire training. All actions are user-discretion.
- Does not change V2.5 / V2.7 (corpora-driven SFT; hardest dimensions are training-data + recipe, not technique additions).
- Does not commit to specific PRM model size — `Med-PRM-8B` is the current SOTA citation; could substitute V_final-as-judge in cold-start.
- Does not benchmark — these are estimates from external papers; medomni-specific paired CIs land in PREREG ship rules per stage.
