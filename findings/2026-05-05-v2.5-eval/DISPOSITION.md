# V2.5 Reasoning-SFT — Final Disposition

**Date:** 2026-05-07. **Status:** Closed (FAIL ship-rule); cost-aware path: ship V0+thinking=True, defer retraining.

## Verdict (verbatim from `findings/2026-05-05-v2.5-eval-thinking/SHIP-RULE-RESULTS.md`)

A5 ablation closed. Both arms FAIL.

| | thinking=False (deniable) | thinking=True (canonical) |
|---|---|---|
| Verdict | 1/4 PASS | 0/4 PASS |
| MedQA-USMLE | -1.17pp | +1.17pp [-1.00, +3.50] |
| PubMedQA-L | -0.67pp | -2.50pp [-5.00, -0.17] |
| MedXpertQA-Text | -1.33pp | -1.33pp [-4.83, +2.17] |
| HealthBench-Hard | +1.48pp | -1.31pp [-3.71, +1.13] |

## Why this is the right place to stop iterating on V2.5

Three converging facts:

1. **The base model is the medical reasoner, not the SFT.** With thinking=True the V0 baseline scores **MedQA 83.50%** (≥ many SOTA claims), **HB-Hard 12.52%**, **PubMedQA 67.33%**. The 30B Nemotron-Omni-Reasoning has strong latent reasoning the prior eval was suppressing.

2. **V2.5's training actively narrowed the reasoning distribution.** Both arms run with thinking ON; V0 wins HB-Hard (-1.31pp Δ), MedXpertQA (-1.33pp Δ), PubMedQA (-2.50pp Δ — significantly so). Only MedQA shows a small positive Δ (+1.17pp, lower-CI crosses 0). This is consistent with reasoning-SFT corpus narrowness (MedReason + medical-o1 + R1-distill share idiomatic patterns) and/or LoRA aggressiveness (r=64, α=128 on 31.6B base = α/r=2.0).

3. **The cost of another training pass on a single H100 is high relative to expected return.** V2.5 was a 23-hour H200 train. H100 has 80 GB; a 30B PEFT trainer needs ~80-90 GB (base + adapter + grad + optim) — barely fits, requires fp8 base + smaller batch + gradient checkpointing. The trainer ergonomics are an order of magnitude worse than V2.5's H200 setup, and we have no evidence a corpus tweak inverts the result.

## Disposition

**Production:** ship `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (BF16 or fp8) **with `enable_thinking=True` by default** at the public demo `/4UWHAt`. The V2.5 LoRA does not deploy.

**Audit trail:** PR #122 + this directory + Issue #130 + `feedback_two_stage_grading_pattern_is_canonical.md` constitute the rigor backbone. Negative result is the publication-grade story; reproducible by anyone via `REPRO.sh`.

**README baselines:** update V0 anchors to thinking=True numbers. The pre-thinking baselines understate the model by an order of magnitude on rubric tasks.

**Backlog (Issue #130, deferred until training budget returns):** V2.7 (PRM channel + tool-call SFT) preferred over V2.5b (corpus tweak retrain). Justification: V2.5 evidence shows reasoning-SFT did not add value on top of an already-reasoning base; tool-call SFT is a fundamentally different objective with a cleaner training signal (correctness verifier exists; medical reasoning verifier is rubric-noisy).

## What this DOES NOT close

- The publication of the negative result + the methodology paper. Worth its own RFC/PR.
- The `enable_thinking` PREREG lock. Future PREREGs must explicitly set this. Driver default is now True (PR #122 commit `6c14c26`).
- The decision on whether to run an A5-style ablation by default for every future SFT eval. Recommendation: yes — the cost (2× compute) is bounded and the signal (deniability check) is high-value.

## Artifacts (frozen)

- `findings/2026-05-05-v2.5-eval/SHIP-RULE-RESULTS.md` — Run 1 (thinking=False)
- `findings/2026-05-05-v2.5-eval/MANIFEST.sha256` — Run 1 86-file hash
- `findings/2026-05-05-v2.5-eval/REPRO.sh` — Run 1 reproducer
- `findings/2026-05-05-v2.5-eval/LEAKAGE-AUDIT.md` — 0/800 prompts hit V1 train
- `findings/2026-05-05-v2.5-eval/METHODOLOGY-PIVOT.md` — why Run 2
- `findings/2026-05-05-v2.5-eval-thinking/SHIP-RULE-RESULTS.md` — Run 2 (thinking=True)
- `findings/2026-05-05-v2.5-eval-thinking/MANIFEST.sha256` — Run 2 hash
- `findings/2026-05-05-v2.5-eval-thinking/REPRO.sh` — Run 2 reproducer
- `findings/2026-05-05-v2.5-eval-thinking/LEAKAGE-AUDIT.md` — Run 2 leakage check
- Adapter sha256 (preserved at `/workspace/v2.5-prod/adapter_model.safetensors` on lobster until lobster teardown): `94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c`

When lobster is torn down (per GPU shutdown plan), the V2.5 adapter weights vanish. The artifacts in PR #122 are the durable record. If V2.5b is ever attempted, retraining from the BF16 base is the path; the V2.5 adapter is not preserved.
