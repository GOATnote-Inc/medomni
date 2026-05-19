# κ-shadow report — Claude opus 4.7 vs gpt-4.1 (NOT physician adjudication)

> **Methodological caveat:** This is a CROSS-MODEL agreement metric. Claude opus
> 4.7 was given the same kappa_blind_review.md document as a physician adjudicator
> would see, applied the same locked taxonomy + tiebreakers, and labeled
> `kappa_claude_shadow_labels.csv`. Cohen's κ is computed against the gpt-4.1
> labels in `kappa_answer_key.jsonl`. **This is NOT a physician κ and does not
> validate clinical correctness.** It tests prompt-engineering robustness across
> two model families with similar training-data exposures and shared "AI assistant"
> priors. Treat as a screening signal.

---

- Paired items: **30** (used in κ: 29; no-fit: 1)
- Cohen's κ: **0.054**
- Interpretation: **SLIGHT — classifier basically not discriminating**
- Raw agreement (diagonal): **11/29 = 38%**

## Disagreement matrix (rows = gpt-4.1, cols = Claude shadow)

| gpt-4.1 \ Claude | cat 1 | cat 2 | cat 3 | cat 4 | cat 5 | row total |
|---|---:|---:|---:|---:|---:|---:|
| **cat 1** Knowledge Gap | **9** | 2 | 3 | 1 | 2 | 17 |
| **cat 2** Reasoning Collapse | 0 | **0** | 0 | 0 | 0 | 0 |
| **cat 3** Calibration Misalignment | 1 | 0 | **1** | 0 | 1 | 3 |
| **cat 4** Context Misapplication | 1 | 0 | 1 | **0** | 1 | 3 |
| **cat 5** Hallucinated Safeguards | 3 | 1 | 1 | 0 | **1** | 6 |
| **col total** | 14 | 3 | 6 | 1 | 5 | **29** |

## What this means for V2.5b corpus design

**The 75/22/2/2/0 distribution from the gpt-4.1 classifier is unreliable.** Two
models with similar capabilities and the same prompt agreed on only 38% of
cases. The implications:

1. **Calibration Misalignment (#3) is severely under-detected by gpt-4.1.**
   Claude saw 6 cases (21%); gpt-4.1 saw 3 (10%) and only 1 of those 3 matched.
   The disagreement pattern (gpt-4.1 cat 1 → Claude cat 3, n=3) suggests
   gpt-4.1 routes "V2.5 over-hedged with disclaimers" cases to Knowledge Gap
   when they're actually Calibration miscalibrations.

2. **Reasoning Collapse (#2) is invisible to gpt-4.1, but Claude saw 3 cases.**
   The smoking-gun pattern: V2.5 shows visible "Wait... Actually... Let's recall"
   self-questioning mid-chain (items 13/14/24). gpt-4.1's prompt doesn't surface
   this signal effectively.

3. **Hallucinated Safeguards (#5) detection diverges in BOTH directions.**
   gpt-4.1 found 6 cases; Claude found 5; only 1 was the same item. They each
   flagged DIFFERENT items as fabrication. The "rigid threshold" detector
   activates differently per model.

4. **Knowledge Gap (#1) is the catch-all both models reach for.** gpt-4.1
   put 17/29 (59%) in #1; Claude put 14/29 (48%). Per Landis-Koch, κ=0.054
   means both models are basically guessing within the per-marginal
   distribution.

## Implications

The original corpus split (70% Knowledge Gap breadth + 25% anti-hallucination)
**probably under-allocates to calibration training.** Claude's distribution
suggests ~20% of V2.5 regressions are urgency-miscalibration / over-hedging
patterns that won't be fixed by either broadening facts or adding
anti-fabrication training.

Three honest interpretations:

1. **Both models are right on different items** — the failure modes overlap
   genuinely (e.g., V2.5 over-hedges = both Knowledge Gap missing the
   escalation cue AND Calibration Misalignment dialing down urgency).
2. **The 5-category taxonomy is too fine-grained** for either model to apply
   reliably. A 2-3 category split (fabrication / silent omission / other)
   would yield higher κ.
3. **The classifier prompt is the load-bearing variable** — both models would
   converge with a sharper prompt. We've done 4 rounds of prompt iteration; the
   marginal return is diminishing.

## Recommendation (solo-dev velocity preserving)

**Collapse the taxonomy** to two corpus sections that BOTH models agree exist:

| Section | Approx share (taking max of both classifiers) | Corpus weight |
|---|---:|---:|
| **A: Active fabrication / over-specification** (was #5, partially #4) | 25% | 30% |
| **B: Silent omission + over-hedging** (was #1, partially #3) | 70% | 65% |
| **C: Reserved/probe** (was #2, kept as held-out) | 5% | 5% |

This honors the distribution where models agree, drops the disputed slices, and
ships V2.5b with a corpus that targets the load-bearing patterns. The eval
re-run is the validation, not the corpus split.

Alternative: refine the classifier with a sharper prompt that surfaces
calibration shifts and reasoning-collapse signals more clearly, re-run N=230,
re-derive distribution. ~$1, ~12 min.

For solo-dev: the collapsed-taxonomy path ships faster.
