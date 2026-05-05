# Process Supervision + Verifiability — SPEC for medomni V3 / V_final

**Author:** Research Agent C · 2026-05-05
**Scope:** layer process supervision + claim-level verifiability on top of (training time) and into (inference time) the V3 GRPO + V3.5 DPO plan.

---

## TL;DR

Outcome-only rewards (V3's correctness 0.5 / format 0.2 / Qwen-judge 0.3 composite) are demonstrably weaker than **stepwise** supervision on math (Math-Shepherd: GSM8K 77.9 → 84.1, MATH 28.6 → 33.0; verification push to 89.1 / 43.5) and on **medical** specifically (Med-PRM: MedQA up to +13.5 abs, first 80%+ at 8B; MedS3 MCTS+dual-sided PRM: +6.45 over prior medical SOTA on 11 in-domain benches). Three concrete, low-risk additions for medomni:

1. **Train a Med-PRM-style stepwise verifier as V3 add-on**, used (a) as auxiliary GRPO reward shaping and (b) as an inference-time best-of-N selector at V_final.
2. **Wrap V_final with a claim-extraction + RAG-NLI checker** (MedRAGChecker recipe) emitting per-claim `[supported | neutral | contradicted]` + PMID, surfaced as citation chips in UX.
3. **Treat CoT as performative, not faithful** — Anthropic Chen et al. show <20% verbalization rate on hint-influence; therefore route safety gating through claim-level audit and outcome-RM, never through "the chain looks fine."

Distinction: items 1 (PRM/MCTS, RLAIF-CAI, dual-sided PRM training) are **training-time recipes**; items 2-3 + Process Reward Agents + best-of-N + critique-refine are **inference-time wrappers** layered on the frozen V_final.

---

## PRM training (V3 add-on)

**Why bolt PRM on top of GRPO outcome reward.** Outcome reward gives credit for correct final answer regardless of trajectory; a step that confabulates a contraindication and then luckily lands on the right Dx still gets reward=1. Stepwise PRM penalizes the bad step — Math-Shepherd showed automatic step-labels via MC rollout can match PRM800K-quality without human annotation; MedS3 extends this with a **soft dual-sided PRM** that penalizes value-degrading steps even on globally-correct trajectories.

**Concrete recipe (medomni V3.1):**
- **Data:** mine HuatuoGPT-o1 + Clinical-R1 chains; auto-label step correctness via MC rollout completion rate (Math-Shepherd) AND retrieve-and-verify against MedlinePlus / openFDA / Cochrane abstracts (Med-PRM's guideline-grounded variant).
- **Verifier:** 8B PRM head on the medomni base (cheaper than separate model). Train with binary step-correct + soft dual-sided value-degradation signal (MedS3).
- **GRPO integration:** add PRM-mean-step-score as a fourth reward channel, weight ~0.15, decreasing existing weights pro-rata. Keep correctness 0.5 dominant.
- **Eval gate:** match MedPRMBench's PRMScore (target ≥80%) before promoting to RL signal. Use MedPRMBench's 14 error types × 4 severity levels for diagnostic.

**Risk:** PRM reward hacking. Mitigation: cap PRM weight; monitor step-score / outcome-correctness rank correlation per training step; fall back to outcome-only if correlation drops below 0.4.

---

## Inference-time verifier (V_final wrapper)

Three options on a quality/latency curve:

| Wrapper | Generations | Extra latency | Quality lift (medical refs) |
|---|---|---|---|
| Best-of-N + PRM-min | N=8-32 | N× decode + 1 PRM pass | MedS3 +6.45 abs over SOTA at N=32 |
| Process Reward Agents (step-prune) | beam=4 with prune | ~1.5-2× decode | PRA: +up-to-25.7% on Qwen3-4B → 80.8 MedQA |
| Critique-refine (1-shot self-verify) | 2 (gen + critique) | ~2× decode | Modest, brittle; only useful when no PRM |

**Recommendation for V_final user-facing path:** **Best-of-N with N=4** (Anthropic-style cheap verifier). Below 2s clinical-workflow budget if base decode <500ms. Use MedS3's "min-step-score across trajectory" as selector — robust to a single great-but-lucky step. Reserve PRA-style step-prune for the offline eval/replay path where latency is unconstrained.

**Wrapper outline:**
```
candidates = [policy.generate(prompt, temp=0.8) for _ in range(4)]
scores = [min(prm.score_steps(c)) for c in candidates]
return candidates[argmax(scores)]
```

---

## Claim-extraction + citation grounding (UX + safety)

**Pipeline (MedRAGChecker, arxiv 2601.06519):**
1. Decompose V_final answer → atomic claims c₁…cₙ via small student extractor (Med42-Llama3-8B SFT'd on GPT-4.1 teacher claims).
2. For each cᵢ: retrieve top-k from PubMed / MedlinePlus / openFDA → NLI ensemble → 3-way verdict {entail, neutral, contradict}; ALSO entity-link to a biomedical KG (DRKG), score via TransE; logistic-fuse → P*(cᵢ) ∈ [0,1].
3. UX: render claim with badge {green ✓ + [PMID:…] | grey ? | red ✗ contradicted by …}.

**Why this matters more than CoT-faithfulness work:** Anthropic Chen et al. 2025 (arxiv 2505.05410) shows reasoning models verbalize their actual influence <20% of the time on most hint types; outcome-RL improves faithfulness then plateaus. **Therefore the chain is not where verifiability lives.** The claim-level audit is the only externally-checkable surface.

**Hard rule:** any claim with `contradict` verdict OR P*(cᵢ) < 0.35 AND safety-critical (dosage, contraindication, code-status) → block answer, fall back to "I'm not sure — please verify with [source]" (V3.5 refusal calibration target).

**Benchmarks to gate against:** MedHallu (10K QA, hardest tier F1=0.625 SOTA — easy delta-room), MedFact (2,116 instances, 4 specialty domains), MedPRMBench (113K step labels).

---

## Faithfulness

Treat CoT as **plausibility scaffolding, not audit trail**. Anthropic findings (Lanham et al. 2023; Chen et al. 2505.05410, 2025): faithfulness 25-41% Claude / 19% R1 across hint types; 2026 follow-on identifies "Reasoning Horizon" at 70-85% of chain length where tokens become performative. Implications for medomni:
- Do **not** train a CoT-monitor as a safety gate (it will silently fail on rare catastrophic cases).
- Do still keep CoT for user transparency and PRM step-labels — the PRM judges step *content*, not whether the model "really" used the step.
- For V3.5 refusal calibration, calibrate against **outcome correctness on held-out**, never against "did the chain warn?".

**Constitutional AI angle (RLAIF for medicine):** the mental-health CAI study (arxiv 2509.16444) shows domain-specific principles +31.7% over generic. Source medomni's constitution from: WHO Essential Medicines, USPSTF, ACEP clinical-policy statements, FDA black-box list. Encode as 12-15 numbered principles; use AI-feedback preference labels in V3.5 DPO ALONGSIDE MedSafetyBench rather than instead of.

---

## Top-3 actionable additions

1. **V3 PREREG amendment — add PRM channel.** Train Med-PRM-style 8B stepwise verifier on auto-labeled HuatuoGPT-o1 + guideline-retrieved chains; add as 0.15-weight reward channel in GRPO (re-weight: correctness 0.45 / format 0.15 / judge 0.25 / PRM 0.15). Gate promotion on MedPRMBench PRMScore ≥80%.
2. **V_final inference wrapper — Best-of-4 + PRM-min selector + claim-audit post-hook.** Latency budget <2s; PRM-min for selection (MedS3 recipe); MedRAGChecker claim extractor + RAG-NLI for per-claim badges; hard-block on contradict-and-safety-critical.
3. **V3.5 DPO — domain-specific CAI principles channel.** 12-15 numbered principles from USPSTF/ACEP/FDA; generate AI-feedback preferences; mix 50/50 with MedSafetyBench in DPO loss. Do NOT use CoT-monitor as a refusal gate — gate on claim-audit and outcome-RM only.

---

## References

- Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards — arxiv 2506.11474
- MedPRMBench: Fine-grained Benchmark for Process Reward Models in Medical Reasoning — arxiv 2604.17282
- Process Reward Agents for Steering Knowledge-Intensive Reasoning — arxiv 2604.09482
- MedS3: Medical Slow Thinking with Self-Evolved Soft Dual-sided Process Supervision — arxiv 2501.12051
- Math-Shepherd: Verify and Reinforce LLMs Step-by-step Without Human Annotations — arxiv 2312.08935
- Let's Verify Step by Step (PRM800K) — arxiv 2305.20050
- MedRAGChecker: Claim-Level Verification for Biomedical RAG — arxiv 2601.06519
- MedHallu: Comprehensive Benchmark for Detecting Medical Hallucinations — arxiv 2502.14302
- MedFact: Benchmarking Fact-Checking Capabilities of LLMs — arxiv 2509.12440
- Reasoning Models Don't Always Say What They Think (Chen, Benton, et al., Anthropic) — arxiv 2505.05410
- Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al., Anthropic, 2023)
- Domain-Specific Constitutional AI: Mental Health Chatbots — arxiv 2509.16444
- Constitutional AI: Harmlessness from AI Feedback (Bai et al., Anthropic, 2022)
- Process-Supervised Reward Models for Verifying Clinical Note Generation — arxiv 2412.12583
