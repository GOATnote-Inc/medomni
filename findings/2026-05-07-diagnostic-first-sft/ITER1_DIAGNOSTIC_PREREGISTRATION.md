# iter=1 collapse-diagnostic — pre-registration

**Date:** 2026-05-22
**Authors:** Brandon Dent + Claude Opus 4.7
**Status:** Pre-registered — falsifiability criteria locked in **before** any per-item judge score is read.
**Lock-in mechanism:** the durable lock is the git commit hash + push timestamp; the file's Write timestamp is fungible.

## Context

`EXPERIMENT_LOG.jsonl` iter=1 (2026-05-09T22:35:00Z) records B5/B6/B7 judge means jumping from `{B5: 0.26, B6: 0.48, B7: 0.51}` to `{B5: 1.00, B6: 0.98, B7: 1.00}` after a single intervention (pattern-aware sharpened prompts added to `corpus_generator.py`), with 13 of 15 patterns at mean 1.00 and `corpus_pass_rate = 1.000`. The recorded decision was `KEEP`; the merged artifact is `v25b_corpus_FINAL.jsonl` (2,464 examples).

The shape — one prompt-engineering intervention → 13/15 patterns at perfect mean — is consistent with **both** hypotheses:

- **H1 (real lift):** the pattern-aware prompts surfaced previously-latent corpus quality; the judge correctly recognises it.
- **H2 (judge collapse):** the judge folded under the new prompt shape (pattern-matched on cosmetic structure rather than evaluating content); lost top-end discrimination.

The aggregate `corpus_pass_rate ≥ 0.6` metric cannot distinguish H1 from H2 — both produce pass-rate → 1.0. Distinguishing them requires *distributional* analysis, not aggregate analysis. This is exactly the "presence vs weight" failure pattern named in issue #396 / m13v's critique, recurring at the metric layer: PROGRAM.md defined a canonical metric (item 11 of the audit grades PASS for "single canonical metric defined") but that PASS measures presence, not whether the metric still discriminates. The discipline we shipped in PR #404 for §0 ("≤15 lines, justify-or-evict") needs the same operational answer here — and *this* pre-registration is that answer.

## Falsifiability criteria

iter=1 is NOT judge collapse if and only if **all four** criteria below hold. The rule is conjunctive: 3-of-4 partial pass is FAIL. Thresholds are stated ex ante and will not be relaxed post-observation.

### C1 — Distribution shape on B5 / B6 / B7

For each of B5, B6, B7, the per-item `judge_score` distribution post-intervention must satisfy **all three** sub-conditions:

| Sub-condition | Threshold | Failure shape |
| --- | --- | --- |
| Density at the top end | ≤ 70% of items in `[0.95, 1.00]` | > 70% is the delta-function saturation signature |
| Per-pattern stdev | σ ≥ 0.10 | < 0.10 means the judge has near-zero spread left |
| Lower-end presence | ≥ 10% of items < 0.90 | < 10% means the judge can no longer find imperfect items |

Failure of any sub-condition on any of B5 / B6 / B7 → **C1 fails**.

### C2 — Negative control on iter=0's 20 worst items

Select the 20 items from the iter=0 corpus that scored lowest under the iter=0 judge (means ≤ 0.30 on their assigned B-pattern). Re-score these *same items* under the current `v25b_judge_filter` configuration — **no regeneration**, only the judge changes.

- **No more than 1 of the 20** items may score ≥ 0.6 under `v25b_judge_filter`. If ≥ 2 do, the judge has gotten more lenient on items the prior judge correctly rejected — meaning the iter=1 "improvement" includes a judge-side component, not just a corpus-side one. → **C2 fails**.
- **Mean shift across the 20 items must be ≤ +0.20.** Belt-and-suspenders on the per-item threshold; catches systematic upward bias even if no single item crosses 0.6.

### C3 — Structural distinctness of the 2-of-15 non-saturated patterns

The 2 patterns whose post-intervention mean is < 1.00 (the ≈ 0.88-mean patterns — exact pattern-IDs to be identified from `v25b_corpus_FINAL.jsonl`'s per-pattern aggregates as the first step of the diagnostic) must be in a **different B-tier or reasoning-axis** from the 13 saturated patterns.

- **PASS:** the 2 are in a distinct B-tier or address a reasoning-axis the prompt-engineering intervention was not designed to lift (e.g., higher-Bloom items, items requiring numeric reasoning the SHARPENED_PATTERN_PROMPTS did not cover). This locates the only signal-bearing dimension iter=1 still produces and supports H1.
- **FAIL:** the 2 are in the same B-tier and same reasoning-axis as the 13, just stochastically lower. The 2 are noise; the 13 are noise-with-ceiling-effect.

### C4 — Corpus-wide variance retention

Across the **entire post-intervention corpus** (all 15 patterns, all items), the standard deviation of per-item `judge_score` must satisfy:

```
σ_post  ≥  0.6 × σ_pre
```

A drop of more than 40% in corpus-wide judge-score variance indicates judge-side discrimination collapse irrespective of mean changes. The pre-intervention σ is computed from the iter=0 corpus under the iter=0 judge; the post-intervention σ from `v25b_corpus_FINAL.jsonl` under `v25b_judge_filter`.

## Decision rule

| Outcome | iter=1 KEEP decision | iter=2 readiness |
| --- | --- | --- |
| C1 ∧ C2 ∧ C3 ∧ C4 all PASS | Stands | Gate can be retained; raising it is still recommended to avoid hitting the ceiling on every iter |
| Any single FAIL | **Provisionally INVALID** | iter=2 blocked until the judge is recalibrated and a new, separately pre-registered, raised gate is in place |

"Provisionally INVALID" does **not** mean `v25b_corpus_FINAL.jsonl` is poisoned for downstream use — the items themselves haven't changed. It means the `corpus_pass_rate ≥ 0.6` metric is provably unreliable on this judge and must be replaced (not just supplemented) before further iteration.

## Anti-cherry-picking commitment

- **No new criteria** will be invented after data is observed.
- **No threshold movement** — the numbers (70%, 0.10, 10%, ≥0.6 per-item, +0.20 mean, 0.6× variance) are fixed by this commit's hash.
- **No partial-pass interpretation** — the rule is conjunctive (∧, not majority vote).
- **Hard separation of concerns** — the raised iter=2 gate gets its own pre-registration after this diagnostic returns, informed by which specific criterion failed.

## Out of scope

- Whether to keep `v25b_corpus_FINAL.jsonl` for downstream SFT — separate decision after diagnostic.
- The raised-gate design for iter=2 (held-out human spot-check, adversarial judge ensemble, harder rubric on the same judge) — separate pre-reg, conditional on this one.
- Replacement metric design — separate work; criterion-failure mode determines what's needed.

---

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
