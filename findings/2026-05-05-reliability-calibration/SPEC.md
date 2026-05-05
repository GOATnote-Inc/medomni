# Reliability, Calibration & Abstention SPEC for medomni

**Agent D / 4** · 2026-05-05 · scope: V3.5 PREREG add-on + V_final inference wrappers

## TL;DR

medomni V3.5 (DPO refusal calibration) is correctly aimed but under-instrumented. The 2026 literature converges on three load-bearing claims:

1. **Verbalized confidence alone is overconfident and degrades under CoT** ([Wang et al. 2026](https://arxiv.org/pdf/2506.00072)). Logit-based and sampling-based estimators dominate it on free-form medical QA, though verbalized wins on MMLU-style multiple choice ([Vashurin et al. 2025](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737/128713/)).
2. **Explicit abstention options matter more than scale or prompting**. MedAbstain ([2601.12471](https://arxiv.org/abs/2601.12471)) shows abstain-channel access yields larger safety gains than perturbations or larger models — a direct training-signal argument for V3.5.
3. **Conformal prediction gives distribution-free coverage guarantees** on medical MCQA at usable set sizes ([Wang & Lin 2026, MDPI Mathematics 13(9):1538](https://arxiv.org/html/2503.05505); [ConU, Wang et al. 2025](https://aclanthology.org/2024.findings-emnlp.404.pdf)). At α=0.1, ConU achieves near-exact correctness coverage on MedQA. This is an inference-time wrapper, not a training change.

## Calibration metrics — how we'd measure V3.5 and V_final

| Metric | What it answers | medomni use |
|---|---|---|
| **ECE** (Expected Calibration Error) | Are 70%-confident answers right 70% of the time? | Primary scalar; report on HealthBench-Hard, MedSafetyBench, MedHallu-hard |
| **Brier score** | Joint accuracy + calibration | Secondary, robust to bin choice |
| **AUROC over P(correct)** | Can confidence rank right vs wrong answers? | Report alongside ECE — ECE alone is gameable by constant outputs |
| **AURC** (Risk-Coverage) | At what coverage % does selective accuracy = X? | Headline number for selective-answering (the "abstain" curve) |
| **Conformal coverage @ α** | Empirical correctness inside prediction set | α∈{0.05, 0.10}; track APSS (avg set size) |
| **Refusal rate / over-refusal** | False refusals on benign clinical queries | From [Health-ORSC-Bench, 2601.17642](https://arxiv.org/abs/2601.17642) |

The ABC paper ([Yan et al. 2025, 2509.19375](https://arxiv.org/pdf/2509.19375)) shows verbalized probability can blow ECE from 3.5pp → 28.6pp; logits + post-hoc calibration dominates. **Recommendation: never log only verbalized confidence; always emit logit-based P(true) alongside.**

## Training-time additions to V3.5 PREREG (the one add-on)

The current preference-pair recipe (MedSafetyBench + synthetic uncertainty) optimizes preference order but not calibration scale. **Cal-DPO** ([Xiao et al. 2025, NeurIPS, OpenReview 57OQXxbTbY](https://openreview.net/forum?id=57OQXxbTbY)) adds an MSE term that anchors the implicit reward to ground-truth scale, materially reducing post-DPO miscalibration vs vanilla DPO. **Uncertainty-penalized DPO** down-weights gradient on low-confidence pairs, mitigating noisy-pair gradient blow-up — relevant because synthetic HealthBench-uncertainty pairs *will* have ambiguous winners.

**Concrete PREREG add-on:** swap vanilla DPO loss for Cal-DPO with β tuned on a 200-pair holdout from MedAbstain-style splits. Add a mandatory **abstain channel** in the SFT/DPO chat template (`<abstain reason="…"/>` token) — MedHallu shows precision/F1 +38% from a "not sure" option ([Pandit et al. 2025, ACL](https://aclanthology.org/2025.emnlp-main.143/)). The abstain channel is the lever DPO actually trains; without it preference pairs can only reorder confident answers.

## Inference-time uncertainty wrappers (the two for V_final)

These are *post-training* shells, evaluated on V_final after V3.5 is frozen. They do not change weights.

**(a) Semantic-entropy gate** — [Farquhar, Kossen, Kuhn, Gal 2024 *Nature*](https://www.nature.com/articles/s41586-024-07421-0). Sample N=10 generations at temp=0.7, cluster by NLI-equivalence, compute entropy over clusters; route high-SE turns to abstain or to a specialist tool. Cost: 10× decode but parallelizable. Cheaper variant: **Semantic Entropy Probes** ([Kossen et al. 2024, 2406.15927](https://arxiv.org/abs/2406.15927)) — linear probe on hidden states approximates SE from a single generation (≈1× cost), with modest AUROC degradation.

**(b) Conformal differential-diagnosis sets** — for top-k diagnosis tasks, calibrate on a holdout of 500–1000 medomni outputs, output a set guaranteed to contain ground truth at 1−α=0.9. Use ConU/enhanced-CP ([2503.05505](https://arxiv.org/html/2503.05505)). UI surfaces the set, not the singleton. This is the only technique with a *finite-sample* guarantee — defensible to clinicians and regulators in a way logit-thresholds are not.

Hallucination-detection wrapper option (defer to V_final+1): SelfCheckGPT-NLI ([Manakul et al. 2023](https://github.com/potsawee/selfcheckgpt)) for free-text claims; MedHallu hard-set as eval target.

## Refusal evaluation

Stack three benchmarks, report all three:

- **MedSafetyBench** ([Han et al. 2024, 2403.03744](https://arxiv.org/abs/2403.03744)) — 450 AMA-ethics-grounded harmful prompts; under-refusal harm score
- **Health-ORSC-Bench** ([2601.17642](https://arxiv.org/abs/2601.17642)) — over-refusal on benign health queries (the failure mode V3.5 will *create*)
- **HealthBench worst-at-k** ([OpenAI 2025, 2505.08775](https://arxiv.org/abs/2505.08775)) — physician-rubric reliability on the hardest 5%; o3 only reaches 60%

Joint plot: under-refusal % (MedSafetyBench) on x-axis, over-refusal % (Health-ORSC-Bench) on y-axis. V3.5 must move toward origin vs V3 baseline; a Pareto regression on either axis is a stop-ship.

## Top-3 actionable items

1. **V3.5 PREREG add-on:** Replace DPO with **Cal-DPO** + add explicit `<abstain/>` channel in chat template + train on Health-ORSC-Bench negatives to bound over-refusal. Single change, addresses the strongest 2025-26 finding.
2. **V_final wrapper #1:** **Semantic Entropy Probes** for per-turn hallucination/uncertainty score; threshold-tuned on MedHallu-hard.
3. **V_final wrapper #2:** **Conformal prediction sets** on differential-diagnosis turns at α=0.10, with APSS reported per condition class. Distribution-free, regulator-legible.

## References

- Farquhar, Kossen, Kuhn, Gal. *Detecting hallucinations in LLMs using semantic entropy.* Nature 2024. https://www.nature.com/articles/s41586-024-07421-0
- Kossen et al. *Semantic Entropy Probes.* arXiv 2406.15927.
- Pandit et al. *MedHallu.* ACL 2025. https://aclanthology.org/2025.emnlp-main.143/
- Han et al. *MedSafetyBench.* arXiv 2403.03744.
- *Health-ORSC-Bench.* arXiv 2601.17642.
- *MedAbstain: Knowing When to Abstain.* arXiv 2601.12471.
- OpenAI. *HealthBench.* arXiv 2505.08775.
- Wang & Lin. *Enhanced Conformal Prediction for Medical MCQA.* arXiv 2503.05505 / MDPI Math 13(9):1538.
- Wang et al. *ConU: Conformal Uncertainty for LLMs.* EMNLP Findings 2024.
- Xiao et al. *Cal-DPO: Calibrated DPO.* NeurIPS 2024, OpenReview 57OQXxbTbY.
- Vashurin et al. *LM-Polygraph UQ benchmark.* TACL 2025.
- Yan et al. *ABC: Approximate Bayesian Computation for LLM UQ.* arXiv 2509.19375.
- Manakul et al. *SelfCheckGPT.* EMNLP 2023.
