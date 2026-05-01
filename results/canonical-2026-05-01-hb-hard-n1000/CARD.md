# CARD — HealthBench Hard, canonical gpt-4.1 grader, N=1000 paired

**Generated:** 2026-05-01
**Run ID:** hb-hard-canonical-n1000-v0v1
**Bench:** Tonic/Health-Bench-Eval-OSS-2025-07 split=hard, N=1000 paired
**Model:** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
**Hardware:** Brev B300 SXM6 (Verda Helsinki, NVFP4)
**Grader:** OpenAI **gpt-4.1-2025-04-14** (canonical HealthBench protocol per arXiv:2505.08775)
**Inference:** thinking=on, temperature=0.0, seed=42, item-conc=12, rubric-conc=8
**Bootstrap:** 5000 paired resamples on shared ids (n_shared=1000)
**Pre-registration:** [PREREG.yaml](PREREG.yaml)

---

## TL;DR

| Arm | n | mean score (gpt-4.1) | wall time |
|---|---|---|---|
| V0 (no RAG) | 1000 | **+0.0537** | 13.5 min B300 + 13 min laptop grade |
| V1 (+PrimeKG nx-cugraph k-2 slice) | 1000 | **−0.0005** | 13 min B300 + 13 min laptop grade |
| **Δ V1−V0 (paired bootstrap)** | 1000 | **−0.0542** | 95% CI **[−0.0731, −0.0357]** |

**The 95% CI does NOT cross 0.** Statistically significant regression of **−5.4 percentage points** under canonical gpt-4.1 grading.

**Pre-registered ship rule applied literally**: Δ V1−V0 was required to be ≥ +0.020 with 95% CI excluding 0 to ship V1 to live `/api/ask`. Observed Δ = −0.054 with CI strictly negative. **Decision: DROP PrimeKG-as-prepended-slice from any plan to wire into live serving.** This decision is unchanged from the prior Qwen-graded run; the canonical grader makes the regression more severe.

---

## The calibration finding (Qwen vs gpt-4.1 grader, same items)

Same N=1000 responses, two graders:

| | Qwen2.5-7B-Instruct (prior) | gpt-4.1-2025-04-14 (canonical) | grader-Δ |
|---|---|---|---|
| V0 mean | 0.1733 | **0.0537** | **+0.1196** |
| V1 mean | 0.1460 | −0.0005 | +0.1465 |
| Δ V1−V0 | −0.0273 [−0.0435, −0.0112] | **−0.0542** [−0.0731, −0.0357] | Δ-of-deltas −0.0269 |

**Calibration verdict per PREREG decision_rule** (|grader-Δ| > 0.030 → "Qwen judge retired from any published number; gpt-4.1 mandatory for canonical numbers"):

> **Observed |grader-Δ| on V0 = 0.120, on V1 = 0.147 — both ≫ 0.030.**
>
> **Qwen2.5-7B-Instruct is retired from any published medomni number.** All V0/V1/V2/V3 measurements going forward are gpt-4.1-graded under canonical protocol. Qwen judge may continue as a fast-iteration internal proxy but its absolute numbers do not appear in CARDs, model cards, or preprints.

The Qwen judge was systematically lenient on HB Hard rubrics by ~12pp absolute. The relative direction (V1 < V0) was preserved, but the magnitude was understated by ~2× and the absolute floor (V0 was at 0.05, not 0.17) was missed.

---

## Comparison vs frontier (OpenAI HealthBench paper, arXiv:2505.08775)

| Stack | HealthBench Hard | Grader | Sampling | N |
|---|---|---|---|---|
| **Nemotron-3-Nano-Omni-30B-NVFP4 (sovereign, this run, V0)** | **0.054** | gpt-4.1-2025-04-14 | T=0, thinking=on | 1000 |
| **Nemotron-Omni V1 (+PrimeKG slice)** | −0.001 | gpt-4.1 | T=0, thinking=on | 1000 |
| OpenAI top published model on HB Hard (o3) | ≤ 0.32 | gpt-4.1 | T=1.0 | 1000 |
| GPT-4.1 (paper Fig. 5, HB **overall**, not Hard) | 0.48 | gpt-4.1 | T=1.0 | 5000 |
| Claude 3.7 Sonnet ext. thinking (paper Fig. 5, HB overall) | 0.35 | gpt-4.1 | per provider | 5000 |
| Claude Opus 4.7 (prism42 prior baseline, **non-canonical**) | 0.196 ± 0.068 | Opus self-judge (Opus 4.7 graded itself) | thinking=off | 30 |

**Honest framing**: under canonical gpt-4.1 grading, Nemotron-Omni-30B-Reasoning vanilla on HB Hard (text-only) sits at **0.054**, against OpenAI's published frontier of ≤ 0.32 on the same benchmark. **Gap to frontier: ~26.5 pp absolute, ~83% relative.**

This is a much weaker text-modality position than the Qwen-graded baseline implied. The PEFT recipe ([MEDOMNI-NEMOTRON-RECIPE.md](../2026-05-01-medgemma-recipe/MEDOMNI-NEMOTRON-RECIPE.md)) targets V2 multi-task SFT (HealthBench-train + MedQA-train + MedMCQA + PubMedQA-Labeled) and V3 DPO refusal calibration — both stages are sized to close more than 5pp on HB Hard, but the absolute frontier remains far.

The narrower comparison band against Opus 4.7's prism42 baseline (0.196 ± 0.068, n=30) is **not apples-to-apples**: Opus 4.7 self-judged its own responses (severe optimistic bias), n=30 vs n=1000, thinking=off, and a stratified clinical subset rather than the full Hard split. The canonical Nemotron-Omni 0.054 (gpt-4.1, n=1000) is a much harder number than 0.196 (Opus self-judge, n=30); they should not be directly compared.

---

## Per-arm response shape (qualitative inspection)

V0 mean is small-positive (0.054); V1 mean is essentially zero with negative tail (−0.001). HealthBench rubrics include negative-points criteria (penalties for unsafe / wrong / disclaimer-laden responses). The arm-mean signs tell us:

- **V0** produces responses that earn slightly more positive than negative rubric points on average — barely net-positive.
- **V1** (with PrimeKG slice prepended) produces responses where the negative penalties dominate — model is being penalized for content the slice introduces (off-topic graph nodes, distracted answers, omitted safety-critical content).

This is consistent with the prior diagnosis (`findings/2026-04-30-imaging-rag/cards/hb-hard-1000-CARD.md`) that prepended-retrieval distracts the model. The canonical gpt-4.1 grader makes the distraction effect ~2× more visible than Qwen.

---

## Sovereignty proof

- B300 pod (`unnecessary-peach-catfish`): NEVER had `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or any cloud LLM key in env during response generation. Verified via `for k in OPENAI_API_KEY ANTHROPIC_API_KEY ...; do refuse if set; done` in `scripts/run_phase0_v0_baseline.sh` analog.
- Stage 1 (response generation): localhost-only `vllm` + `primekg` services, sovereign by construction.
- Stage 2 (gpt-4.1 grading): laptop only; key sourced via `set -a && source ~/lostbench/.env && set +a` and never persisted to repo or pod.
- No shared state between pod and grader except the response JSON files.

---

## What this CARD does NOT say

1. **NOT a verdict on graph-RAG generally.** Measures *PrimeKG-as-prepended-slice with string-match seeding*. NV-Embed-seeded retrieval, tool-call retrieval (model decides when to invoke), and different data shapes (Cochrane / JAMA / NEJM full-text vs 4M-edge entity graph) are different experiments.
2. **NOT generalizable across model families.** This is on Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4. Larger models with longer effective context may integrate the slice productively where this one does not.
3. **NOT a clinical-correctness audit.** HealthBench rubrics are physician-written but the grader is gpt-4.1, not a physician panel. Per arXiv:2505.08775 §8 the gpt-4.1 grader is validated against physician agreement; we adopt that validation.
4. **NOT a complete V0 baseline for medomni.** This measures HB Hard text-only. The full medomni V0 baseline includes imaging (VQA-RAD 0.643/0.481, SLAKE-en 0.744/0.689 — see [imaging V0-CARD.md](../2026-05-01-imaging-peft-v1/V0-CARD.md)). Imaging V0 is **MedGemma-4B-class**; text V0 is much further from the frontier. This asymmetry shapes the PEFT recipe (V2 multi-task SFT prioritizes text-modality medical instruction data).

---

## Reproducibility manifest

| | value |
|---|---|
| Bench script | scripts/bench_hb_local_async.py |
| Bench script sha256 | e75891c096ffcd405cca8ea1d1a11376d3294a052b98fe510ac4d2b754b18a9c (pre-patch) |
| Grader script | scripts/grade_saved.py |
| Grader script sha256 (post per-item rubric_sem fix, commit 8ba4752) | TBD (recompute) |
| simple-evals upstream pin | ee3b0318d8d1d9d72755a4120879be65f7c07e9e |
| Bootstrap seed | 42 |
| Wall time total | ~52 min (13 min B300 V0 + 13 min B300 V1 + 13 min grade V0 + 13 min grade V1) |
| Cost total | ~$10 (B300 ~$4 + OpenAI grader ~$6) — **plus ~$30 lost to the buggy grader's first pass** |

---

## Decision applied (literal pre-registered rule)

| pre-reg rule | observed | decision |
|---|---|---|
| SHIP V1 if Δ ≥ +0.020 AND 95% CI excludes 0 | Δ = −0.054, CI [−0.073, −0.036] | DROP — fails magnitude AND direction |
| Qwen judge calibration: |grader-Δ| ≤ 0.030 | observed +0.120 (V0) / +0.147 (V1) | RETIRE Qwen from canonical numbers |

**Both decisions applied. Recorded in MODEL_SPEC.md and (forthcoming) issue #2 closing comment.**

---

## Supersedes

This CARD supersedes:
- `results/ci-medomni-heldout-consolidated-20260429-173557/CARD.md` (N=6×3 Nemotron self-judged with the wildly-misleading 0.385±0.000 claim — retired due to small-sample artifact + same-family judge bias + grader-stricter true number)
- `findings/2026-04-30-imaging-rag/cards/hb-hard-1000-CARD.md` (Qwen-graded V0=0.173/V1=0.146 — retained as the *Qwen calibration reference* but no longer the published number)

The public-repo `GOATnote-Inc/medomni` README link to the 0.385 claim is now actively misleading vs canonical 0.054. **Issue #2 (`supersede stale public CARD`) blocks-by this CARD landing — now unblocked.**

---

## Ablation: thinking=on vs thinking=off (added 2026-05-01T13:47Z)

Re-ran the full N=1000 V0+V1 sweep on B300 with `chat_template_kwargs.enable_thinking=False` (no reasoning-budget tokens) and graded with the same canonical gpt-4.1 grader. Same seed (42), same dataset, same items — only the inference-time thinking flag changes.

| | V0 (no RAG) | V1 (+PrimeKG) | Δ V1−V0 |
|---|---|---|---|
| **thinking=on** (canonical) | **+0.0537** | −0.0005 | −0.0542 [−0.0731, −0.0357] |
| **thinking=off** | −0.0713 | **−0.1003** | −0.0290 [−0.0456, −0.0126] |
| **Δ-thinking** (on − off) | **+0.1250** | **+0.0998** | — |

### Two findings

1. **Thinking-mode is load-bearing for HB Hard quality (~+12.5pp on V0 absolute).** Disabling reasoning budget produces responses that earn substantially more rubric penalties than positive points (V0 net-negative at −0.071). The reasoning step is what lets Nemotron-Omni-30B-Reasoning produce HB-rubric-passing responses; without it the model is below floor.

2. **PrimeKG regression holds in both modes** (Δ excludes 0 in both rows), but is smaller in thinking=off (−0.029 vs −0.054). Plausible mechanism: thinking=off responses are already so degraded that the prepended-slice distractor has less room to drag them further down. Does NOT change the literal pre-registered DROP decision on V1.

### Recipe implication

**Keep `enable_thinking=True` in V1/V2/V3 PEFT serve config.** The thinking head IS what makes Nemotron-Omni-Reasoning hit double-digit rubric scores at all on HB Hard. PEFT training must preserve (and ideally strengthen) reasoning behavior — corpus mix should include chain-of-thought examples, training prompts should expose the thinking-token format, and post-PEFT serve must default `enable_thinking=True`.

This is the first concrete recipe knob the V0 ablation set has informed. More ablations (seed sensitivity, sampling-temp, max-tokens) can follow; thinking-mode was the biggest expected lever and the data confirms.

### Cost

Total ablation: ~$5 (B300 ~13 min thinking=off bench + ~13 min gpt-4.1 grading at 1.57 items/s with patched per-item rubric_sem).
