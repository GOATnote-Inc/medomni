# iter=1 collapse-diagnostic — C2 / C4 addendum

**Date:** 2026-05-22
**Authors:** Brandon Dent + Claude Opus 4.7
**Status:** Addendum to `ITER1_DIAGNOSTIC_RESULT.md` (commit `1bd9c18`, PR #405). Resolves C4 and reframes C2 from the new local-data evidence. **The iter=1 KEEP verdict (provisionally INVALID) is unchanged.** The mechanism understanding is sharpened: this is generator-side Goodhart, not judge collapse.

## Context

The original result deferred C2 ("re-judge iter=0's 20 worst items under `v25b_judge_filter`") and C4 ("corpus-wide σ_post ≥ 0.6 × σ_pre") as *"pending judge endpoint / iter=0 corpus access."* Post-merge exploration of `findings/2026-05-07-diagnostic-first-sft/` identified the iter=0 judged baseline on disk: `v25b_judged_full.jsonl` (4,950 records, judged under `v25b_judge_filter`).

Identity confirmation — per-pattern means match EXPERIMENT_LOG's `before_means` to ±0.003:

| pattern | EXPERIMENT_LOG `before_mean` | `v25b_judged_full` mean | delta |
| --- | --: | --: | --: |
| B5_missing_context_asking_questions | 0.26 | 0.260 | −0.000 |
| B6_refusal_to_answer_direct_yes_no | 0.48 | 0.481 | +0.001 |
| B7_context_element_dropped | 0.51 | 0.513 | +0.003 |

This identifies `v25b_judged_full.jsonl` as the iter=0 corpus scored under the **same** judge (`v25b_judge_filter`) later used to score `v25b_corpus_FINAL.jsonl`. The judge's identity across iter=0 and iter=1 is the central new fact and forces a reframing of the diagnostic mechanism — see *Reframing* below.

## C4 — resolved: FAIL (decisive, ~76% below threshold)

| quantity | value | source |
| --- | --: | --- |
| σ_pre | **0.3471** | `v25b_judged_full.jsonl`, corpus-wide, 4,950 records |
| σ_post | 0.0490 | `v25b_corpus_FINAL.jsonl`, corpus-wide, 2,464 records (computed in `ITER1_DIAGNOSTIC_RESULT.md`) |
| ratio σ_post / σ_pre | **0.141** | |
| pre-reg threshold | ≥ 0.60 | from `b32310f` |
| **C4 verdict** | **FAIL** | the ratio is ~24% of the required floor — corpus-wide variance collapsed to 14% of its prior value |

The pre-intervention corpus-wide σ was 0.347 — a wide distribution running from 0.00 to 1.00 (13.2% of items at exactly 0.0; 57.5% in [0.95, 1.00]; 36.5% below 0.90). The post-intervention σ collapsed to 0.049 with 91.0% of items in [0.95, 1.00]. Variance retention fell ~4× below the pre-registered threshold.

C4 provides a second independent pre-registered criterion (after C1) statistically confirming the verdict.

## C2 — reframed: N/A as pre-registered; underlying question answered

The pre-reg specified: *"Re-score the 20 worst iter=0 items under `v25b_judge_filter`; no more than 1 of 20 may score ≥ 0.6; mean shift ≤ +0.20."* The implicit assumption was a judge change between iter=0 and iter=1 — re-judging the same items under the new judge would reveal regression.

**The data refutes the assumption.** `v25b_judged_full.jsonl` *is* the iter=0 corpus scored under `v25b_judge_filter` (the same judge used in iter=1). There is no older judge to compare against; re-judging the same items under the same judge is a determinism check, not a regression check.

Two operational findings:

1. **The 20 worst iter=0 items are all at `judge_score = 0.00`, concentrated in section-A patterns** (*Active fabrication / over-specification*). The specific pattern attribution among A1 / A2 / A3 depends on the sort tiebreaker — multiple section-A patterns have items at exactly 0.0, and ordering by `(score, pattern, id)` vs `(score, id, pattern)` selects different subsets. The substantive fact (all 0.00, all section-A failure modes the judge correctly flagged in iter=0) is tiebreaker-independent.
2. **These items are not present in `v25b_corpus_FINAL.jsonl`** (ID overlap check: 0 of 20 under either tiebreaker). The post-intervention corpus was *regenerated* under the sharpened prompts (different items, same pattern coverage), not rescored. So C2 cannot be executed verbatim — the items don't exist in the post corpus.

**C2 status: N/A as pre-registered.** The underlying question C2 was designed to answer (*has the judge gotten more lenient?*) is **answered NO** by the per-pattern means equivalence: same judge in both iters, iter=0 B5/B6/B7 means match EXPERIMENT_LOG's stated values exactly. The judge is stable; whatever drove iter=1's saturation is not judge-side regression.

## Reframing: generator Goodhart, not judge collapse

The original pre-reg framed iter=1's saturation as H2 *(judge collapse)*. The new data refines the mechanism:

- The judge (`v25b_judge_filter`) is **stable**: same model, same prompt, same criteria across both iters. It gave 0.0 to fabricated-citation items in iter=0 and continues to grade against the same axes.
- The corpus generator was **sharpened against the judge's criteria** via the new `_pattern_aware_prompt` + `SHARPENED_PATTERN_PROMPTS`. The intervention rewrote the corpus to avoid the failure modes the judge checks for.
- Saturation isn't the judge giving 1.0 because it lost discrimination — it's the judge correctly giving 1.0 because the corpus no longer contains items that fail its criteria.

This is **Goodhart's law on the generator side**: when a measure becomes the target of the optimization, the optimization achieves the measure without necessarily improving what the measure was supposed to track. The judge's criteria are narrower than "good clinical reasoning"; by sharpening the generator against those narrow criteria, iter=1 produced a corpus that scores perfectly on them without producing demonstrably better clinical content along axes the judge does not check.

**The verdict is unchanged.** iter=1's KEEP remains provisionally INVALID; the `corpus_pass_rate ≥ 0.6` metric is hollow under generator-Goodhart pressure just as it would have been under judge collapse.

**The corrective direction is sharpened.**

| failure mode | implied fix | aligned with |
| --- | --- | --- |
| Judge collapse (original framing) | Recalibrate the judge | (not the actual mechanism here) |
| Generator Goodhart (reframed mechanism) | **Broaden the metric so it captures dimensions the generator can't trivially satisfy by avoidance-shape sharpening alone** | iter=2 gate-raise **M1** (multi-axis rubric: accuracy + completeness + safety + structure with MIN-aggregation) and **M2** (held-out human spot-check) |

The iter=2 gate-raise pre-registration (`0e008a6`) is correctly aligned with the generator-Goodhart fix without amendment — MIN-aggregation across four axes structurally blocks single-axis sharpening, and the held-out human spot-check uses an oracle the generator cannot have been tuned against.

## Updated diagnostic summary

| criterion | original status | resolution |
| --- | --- | --- |
| **C1** distribution shape on B5 / B6 / B7 | FAIL (categorical, all 3 sub-conditions × all 3 patterns) | unchanged — primary determinant |
| **C2** per-item negative control on iter=0 worst-20 | PENDING | **N/A as pre-registered** (judge didn't change; items didn't survive); underlying question answered NO |
| **C3** structural distinctness of non-saturated patterns | PASSES in the H2 direction | unchanged; refined as Goodhart-symptom — non-saturated patterns are verification-style which the generator cannot satisfy by avoidance alone |
| **C4** corpus-wide σ retention | PENDING | **FAIL** decisively (ratio 0.141 vs threshold 0.60, ~76% below) |

The conjunctive decision rule is satisfied with a second independent failure (C4) confirming what C1 already determined.

## Anti-cherry-picking compliance

- C4's threshold (σ_post ≥ 0.6 × σ_pre) was stated ex ante in `b32310f` and applied exactly as written.
- C2's reframing is a *coverage* observation (the criterion's implicit premise — a judge change — did not occur), not a threshold relaxation. The criterion is marked **N/A** and the underlying question is answered with the new lens; no new threshold is introduced.
- The mechanism reframing (judge collapse → generator Goodhart) does not modify any criterion or move any threshold. It refines the corrective direction; the verdict is unchanged.

---

**Original pre-reg:** `b32310f` (PR #405).
**Original result:** `1bd9c18` (PR #405).
**iter=2 gate-raise pre-reg:** `0e008a6` (PR #405) — already aligned with the generator-Goodhart fix via M1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
