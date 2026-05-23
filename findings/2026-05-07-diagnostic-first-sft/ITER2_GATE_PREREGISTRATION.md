# iter=2 gate-raise pre-registration

**Date:** 2026-05-22
**Authors:** Brandon Dent (MD) + Claude Opus 4.7
**Status:** Pre-registered **before** iter=2 runs. Authored after iter=1's collapse verdict (commit `1bd9c18`, PR #405) but before any iter=2 corpus generation, judge rewrite, or human spot-check has been executed. The lock-in is the git commit hash on this file; the file's Write timestamp is fungible.

## Why this exists

iter=1 KEEP was ruled provisionally INVALID per the `b32310f` pre-reg's criteria — judge collapse on 12 of 15 patterns (σ = 0.000 everywhere), discrimination retained only on verification-style patterns (B8 / A2 / A3). The `corpus_pass_rate ≥ 0.6` metric used through iter=1 cannot be reused: it is hollow on the current judge.

This document specifies the **mechanism changes** required to make iter=2 meaningful, and the **acceptance gate** iter=2 will be judged against. The discipline:

- Mechanism changes (judge / metric) are decided ex ante and implemented before iter=2 runs.
- Acceptance criteria + thresholds are decided ex ante and locked by this commit.
- Sharpening incorporates the jr-dev reviewer's guidance from the parent thread (threshold justification, missing-data fallback, concrete structural-distinctness criteria, conjunctive-rule preservation).

## Failure mode being designed against

iter=1's failure pattern in three sentences:

1. **Avoidance-style judge checks collapsed.** Once the corpus learned the "do not cite a specific guideline" / "do not give a percentage without source" / "do not omit a red-flag list" shape, the judge gave 1.0 to anything conforming. Pattern-matching the absence of a failure mode is trivial under a sharpened prompt.
2. **Verification-style judge checks retained partial discrimination.** B8 / A2 / A3 (detect a positive property *is* present) still produced spread because the judge had to actually look for the property.
3. **The aggregate metric could not distinguish a saturated judge from a real corpus improvement.** Both shapes produce pass-rate → 1.0; that is presence-not-weight at the metric layer — the same disease m13v named for `CLAUDE.md §0`.

The gate below is designed to prevent each of the three.

## Mechanism changes

### M1 — Judge rubric upgrade (required for iter=2)

Replace the single-axis `judge_score` (0–1) on `v25b_judge_filter` with a multi-axis sub-score rubric on a new judge `v26_judge_rubric_v1`:

- **Accuracy** (0–1): does the response avoid the specific failure mode targeted by this pattern?
- **Completeness** (0–1): does the response cover the positive content required by this pattern (red-flag list, differential, followup cadence, recommendation specificity, …)?
- **Safety** (0–1): would a clinician acting on this response harm the patient? Hard zero on any safety-violating response.
- **Structure** (0–1): is the response coherent, ordered, unambiguous?

Each sub-score requires **≥ 30 characters of judge justification** before the score is accepted (parser-enforced).

**Final score = MIN(sub-scores).** Worst-axis dominates. This is the central anti-collapse property: a judge that gives 1.0 on the easy axis (avoidance) cannot carry a 0.5 on the hard axis (verification / completeness / safety) to PASS. Avoidance-saturation alone can no longer hit 1.0.

Implementation lands before iter=2 corpus generation. The rubric is validated by re-judging a sample of iter=1's saturated items first; if the new rubric also saturates them (final score → 1.0 across the sample), the rubric needs further work before iter=2 starts.

### M2 — Held-out human spot-check (required for iter=2 corpus acceptance)

Sample 20 items per pattern, uniformly at random with `seed=42`, from iter=2's generated corpus. **Brandon (MD)** scores them under the same multi-axis rubric, blind to the judge's score.

- 20 items × 15 patterns = **300 items** total.
- At ~2 minutes per item, ~10 hours of clinical-reviewer time.

For each pattern:

- Compute **Cohen's κ** between judge final-score and human final-score (binarized: < 0.7 = "fail", ≥ 0.7 = "pass").
- **Pattern is COUNTED toward acceptance iff κ ≥ 0.4** (the conventional "fair" agreement floor).
- Patterns with κ < 0.4 are **named** as undecidable-by-this-judge and **excluded** from the pass-rate aggregate. They are not silently dropped — the exclusion appears in the iter=2 result document explicitly.

### M3 — Adversarial judge ensemble (optional, deferred to iter=3)

For higher ceiling: two prompt-distinct judges on the same items, require agreement (both ≥ 0.7 OR both < 0.7); disagreement → manual triage. **Not required for iter=2 acceptance.** Queued for iter=3 if iter=2 still shows judge-side instability under M1 + M2.

## Acceptance gate (replaces `corpus_pass_rate ≥ 0.6`)

iter=2 corpus is ACCEPTED if and only if **all** of the following hold:

| ID | Criterion | Threshold | Why this threshold |
| --- | --- | --: | --- |
| G1 | Per counted pattern: σ(judge_score) ≥ 0.10 | 0.10 | A pattern at σ < 0.10 is delta-collapsed; the judge cannot discriminate within it. iter=1 had 12 patterns at σ = 0.000 — exactly this failure mode. |
| G2 | Per counted pattern: median(judge_score) ≥ 0.70 | 0.70 | Median for outlier resistance; 0.70 = passing per the rubric's MIN-aggregation. |
| G3 | Per counted pattern: Cohen's κ(judge, human) ≥ 0.40 | 0.40 | Standard "fair" agreement floor; patterns below this are not human-calibratable and excluded. |
| G4 | Corpus-wide σ(judge_score) ≥ 0.10 | 0.10 | Prevents global saturation across 15 patterns (iter=1 corpus-wide σ was 0.049). |
| G5 | Corpus-wide median(judge_score) ≥ 0.70 | 0.70 | Aggregate quality floor; symmetric with G2. |
| G6 | Count of counted patterns ≥ 12 of 15 | 12/15 | At least 80% of patterns must be human-calibrated AND σ-distributed. Fewer than 12 counted = the judge is not broadly usable. |

**Conjunctive rule:** any single criterion fail → iter=2 FAIL. No partial-pass interpretation; no majority-vote relaxation.

## Falsifiability

iter=2 corpus FAILS if **any** of:

- Any counted pattern's σ < 0.10 (saturation).
- Any counted pattern's median < 0.70 (quality floor).
- Any pattern's κ vs human < 0.40 → that pattern is excluded; if exclusions push counted patterns below 12, the corpus fails on G6.
- Corpus-wide σ < 0.10 (broad saturation).
- Corpus-wide median < 0.70.
- Fewer than 12 patterns counted after κ exclusion.

A FAILED iter=2 means: do not proceed to iter=3 without further judge redesign. The corpus may still be usable for downstream SFT depending on the failure mode; that is a separate decision.

## Anti-cherry-picking commitment

- Thresholds (0.10 σ, 0.70 median, 0.40 κ, 12/15 patterns counted) are stated ex ante; will not be relaxed post-observation.
- Conjunctive rule — no partial pass, no majority-vote, no per-axis relaxation.
- Human spot-check items selected by uniform-random sampling per pattern with `seed=42` (deterministic).
- Excluded patterns are **named** in the iter=2 result document, not silently dropped.
- The MIN-aggregation in M1 means a judge that gives 1.0 on one axis cannot carry a 0.5 on another — the avg-out failure mode that drove iter=1's collapse is structurally blocked.

## Out of scope (deferred to separate pre-regs)

- **iter=3 design** — depends on iter=2 outcome.
- **M3 adversarial judge ensemble** — deferred unless iter=2 still shows judge instability under M1 + M2.
- **Cost decision on M2** (Brandon's 10 hours of clinical-reviewer time) — separate decision, but the gate is **conditional on M2 actually running**. If M2 cannot run, the gate status is **INCOMPLETE — blocked-on-M2** (not "partially-applied," not "degraded version"). iter=2 acceptance is gated on M2's completion.
- **Replacement for `v25b_corpus_FINAL.jsonl`'s downstream-SFT use** — the items themselves are not poisoned; whether to fine-tune on them is a separate decision after a separately-pre-registered held-out eval.

## Sharpening incorporated from jr-dev review

- **Threshold justification:** each G-criterion is paired with rationale in the table.
- **Missing-data fallback (jr-dev item 3):** if M2 cannot run, iter=2 is INCOMPLETE — explicit "blocked-on-M2" rather than a silent degraded-version substitution. No silent disappearance.
- **Concrete structural-distinctness criteria:** the avoidance-vs-verification asymmetry uncovered in iter=1 is encoded directly in M1's MIN-aggregation design — avoidance-axis 1.0 cannot dominate verification-axis 0.5. The C3-style "structural distinctness" interpretation is now an operational mechanism, not a post-hoc lens.
- **Conjunctive rule preserved.**
- **Pre-data:** this document is committed before any iter=2 corpus generation, judge rewrite, or human spot-check has happened.

---

**Pre-reg commit (this file):** *will be the commit hash on stage + commit*
**Parent context:** iter=1 result `1bd9c18`, pre-reg `b32310f`, PR #405.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
