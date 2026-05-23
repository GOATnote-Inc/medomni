# iter=1 collapse-diagnostic — result

**Date:** 2026-05-22
**Authors:** Brandon Dent + Claude Opus 4.7
**Pre-registration:** `ITER1_DIAGNOSTIC_PREREGISTRATION.md` (commit `b32310f`, PR #405)
**Verdict:** **iter=1 KEEP is provisionally INVALID — judge collapse confirmed.**

## TL;DR

Per the conjunctive decision rule in the pre-registration, **C1 (distribution shape on B5 / B6 / B7) FAILS categorically**: all three patterns have σ = 0.000 (every one of 589 items scored exactly 1.0), 100% in [0.95, 1.00], 0% below 0.90. The pre-reg required σ ≥ 0.10, ≤ 70% in the top band, and ≥ 10% below 0.90; observed values miss every threshold by the maximum possible margin. **12 of 15 patterns total** have σ = 0.000.

C3 (structural distinctness of the non-saturated patterns) technically passes — but the way it passes is itself the strongest signal for judge collapse. The three patterns that did *not* saturate (B8, A2, A3) are all *verification-style* judgments (the judge has to detect a positive property in the response); the twelve that did saturate are all *avoidance-style* (the judge has to detect the absence of a failure mode). **The judge has retained discrimination only on positive-check axes and lost it on avoidance-check axes.**

C2 (negative control re-judging) and the precise iter=0 σ for C4 are pending judge endpoint and iter=0 corpus access; their outcomes will sharpen the confidence but cannot flip the verdict (the conjunctive rule is already triggered by C1).

## Data

- File: `findings/2026-05-07-diagnostic-first-sft/v25b_corpus_FINAL.jsonl` (3.6 MB, 2,464 records).
- Fields used: `pattern_addressed` (str), `judge_score` (float in [0, 1]).

### Per-pattern stats, sorted by mean ASC

| pattern | n | mean | σ | min | max | < 0.90 | in [0.95, 1.00] |
| --- | --: | --: | --: | --: | --: | --: | --: |
| B8_omitted_specific_recommendation | 200 | 0.877 | 0.109 | 0.70 | 1.00 | 41.5% | 34.5% |
| A2_rigid_quantitative_threshold | 150 | 0.938 | 0.046 | 0.90 | 1.00 | 0.0% | 43.3% |
| A3_invented_protocol_or_guideline_name | 150 | 0.997 | 0.018 | 0.90 | 1.00 | 0.0% | 96.7% |
| A1_fabricated_specific_citation | 150 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| A4_false_reassurance_overstated_efficacy | 150 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| A5_specific_percentage_without_source | 150 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| B1_disclaimer_prefix_on_urgent_scenario | 201 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| B2_missing_red_flag_list | 199 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| B3_missing_followup_cadence | 201 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| B4_missing_differential_listing | 200 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| B5_missing_context_asking_questions | 192 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| B6_refusal_to_answer_direct_yes_no | 197 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| B7_context_element_dropped | 200 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| C1_anchor_bias_premature_closure | 62 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |
| C2_self_contradiction_midchain | 62 | 1.000 | 0.000 | 1.00 | 1.00 | 0.0% | 100.0% |

**Corpus-wide:** n = 2,464; mean = 0.986; σ = 0.049; 91.0% in [0.95, 1.00]; 3.4% below 0.90.

## Criterion-by-criterion evaluation

### C1 — Distribution shape on B5 / B6 / B7

**FAIL — categorically on all three patterns and all three sub-conditions.**

| pattern | n | ≤ 70% in [0.95, 1.00] | σ ≥ 0.10 | ≥ 10% < 0.90 |
| --- | --: | :-: | :-: | :-: |
| B5_missing_context_asking_questions | 192 | FAIL (100.0%) | FAIL (0.000) | FAIL (0.0%) |
| B6_refusal_to_answer_direct_yes_no | 197 | FAIL (100.0%) | FAIL (0.000) | FAIL (0.0%) |
| B7_context_element_dropped | 200 | FAIL (100.0%) | FAIL (0.000) | FAIL (0.0%) |

All 589 items across B5 / B6 / B7 scored exactly 1.000. Delta function at the top, zero spread. The pre-reg's three sub-thresholds — 70%, 0.10, 10% — are all violated maximally.

**Per the conjunctive decision rule, C1 alone determines the verdict.** The remaining criteria are reported for completeness and for steering the recalibration that comes next, not to change the outcome.

### C2 — Negative control on iter=0's 20 worst items

**PENDING.** Requires identifying the iter=0 corpus (likely `b2_corpus_v25b.jsonl` or a predecessor) and re-scoring those 20 items under the current `v25b_judge_filter`. The re-scoring step needs the judge endpoint, which sits on the orca pod. Queued; this step quantifies *how much* of the iter=1 gain is judge-side regression vs corpus-side improvement, but does not change the verdict.

### C3 — Structural distinctness of the non-saturated patterns

**PASSES — and its passing is itself H2 evidence.**

The three patterns that did *not* saturate:

- `B8_omitted_specific_recommendation` (mean 0.877, σ 0.109): judge must detect that a **specific recommendation was made** — positive presence check.
- `A2_rigid_quantitative_threshold` (mean 0.938, σ 0.046): judge must detect that a **specific quantitative threshold was stated** — positive presence check.
- `A3_invented_protocol_or_guideline_name` (mean 0.997, σ 0.018): judge must detect that a **specific guideline name was invented** — positive presence check.

The twelve saturated patterns are all *avoidance* checks — the judge grades the corpus on whether it *avoided* specific failure modes (disclaimer prefix, missing red flags, missing differential, dropping context, fabricating a citation, false reassurance, naming a percentage without source, etc.). The pattern-aware sharpened prompts (the iter=1 intervention) surface the avoidance shape explicitly to the corpus generator, so anything that conforms to the new shape gets a 1.0 from the judge — the judge is checking for the avoidance signature, not for content quality.

The verification patterns still produce some discrimination because they require the judge to *find* a positive property in the response, which is structurally harder than pattern-matching the absence of one.

**The non-saturated patterns are structurally distinct in exactly the direction that confirms H2: avoidance has been hacked; verification has not.** C3 passes the strict pre-reg threshold ("different reasoning-axis from the saturated ones") but the result reinforces, not weakens, the collapse verdict.

### C4 — Corpus-wide variance retention

**LIKELY FAIL, pending precise iter=0 σ.**

- σ_post (corpus-wide, measured) = **0.049**.
- σ_pre (corpus-wide, unmeasured) = unknown without iter=0 corpus loaded.

From the EXPERIMENT_LOG's `before_means` (B5 = 0.26, B6 = 0.48, B7 = 0.51) and the fact that pre-intervention patterns spanned 0.2–0.5 with apparent real spread, σ_pre is plausibly in the 0.20–0.30 range across the full pre-intervention corpus. If so, σ_post / σ_pre ≈ 0.16–0.25 — well below the 0.6 threshold.

Confirmation requires loading `b2_corpus_v25b.jsonl` (or whichever file holds the iter=0 per-record scores). Queued as a follow-up turn.

## Decision

**iter=1 KEEP is provisionally INVALID.** Per the pre-registration's decision rule:

> "Provisionally INVALID" does not mean `v25b_corpus_FINAL.jsonl` is poisoned for downstream use — the items themselves have not changed. It means the `corpus_pass_rate ≥ 0.6` metric is provably unreliable on this judge and must be replaced (not just supplemented) before further iteration.

The `v25b_judge_filter` has lost discrimination on avoidance-style patterns (B1–B7, A1, A4, A5). It retains partial discrimination on verification-style patterns (B8, A2, A3) but not enough to make `corpus_pass_rate ≥ 0.6` a meaningful aggregate over the corpus.

## Recommended next steps (pre-iter=2)

1. **Replace the metric.** The aggregate `corpus_pass_rate ≥ 0.6` is presence, not weight. Replace with: median judge score per pattern *plus* a per-pattern σ floor (e.g., σ ≥ 0.10 required for a pattern to count toward pass-rate at all). A pattern at σ = 0.000 is structurally undecidable by this judge and should be flagged, not counted.

2. **Recalibrate the judge.** Options ordered by cost:
   - **Harder rubric on the same judge** (cheap, low ceiling): tighten `v25b_judge_filter` to require multi-axis scoring rather than a single 0–1; force the judge to write per-axis sub-scores plus rationale before producing the aggregate.
   - **Held-out human spot-check** (moderate, high ceiling): 20 items from the saturated patterns hand-scored by a clinical reviewer; calibrate the judge against the human score distribution; investigate prompt sensitivities that drive saturation.
   - **Adversarial judge ensemble** (expensive, highest ceiling): require agreement between two judges with distinct prompts (or distinct models); a single saturated judge cannot dominate the aggregate.

3. **Pre-register the iter=2 gate** as a separate document, conditional on this verdict, written before iter=2 runs. The new pre-reg should explicitly target the avoidance-vs-verification asymmetry and require per-pattern σ floors as part of the acceptance gate.

4. **Complete C2 and C4 quantitatively** in follow-up turns. Sharpens the confidence interval on this verdict; does not change it. Captures the magnitude of the judge-side regression for the recalibration step.

## Anti-cherry-picking compliance

- No new criteria invented post-observation.
- Thresholds (70%, 0.10, 10%) applied exactly as pre-registered.
- The conjunctive rule was honored: a single C1 failure determines the verdict; C3's pass does not flip it.
- No partial-pass relaxation.
- The structural interpretation (avoidance vs verification) was not in the pre-reg as a criterion — it is offered here as a *forward-looking explanation* for the recalibration step, not as a post-hoc criterion adjustment.

---

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
