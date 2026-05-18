# Diagnostic-first failure-mode probe — V2.5-thinking HB-Hard regressions

**Date:** 2026-05-07
**Step 1 of:** Karpathy-style diagnostic-first SFT loop (V2.5b corpus → train → re-eval)
**Source:** `findings/2026-05-05-v2.5-eval-thinking/graded/healthbench-hard__*` (PR #122 FAIL data)
**Compute:** laptop-only (gpt-4.1 via canonical .env), zero orca usage this turn

## Method

1. **Pair** all HB-Hard records by `(item_id, seed)` across V0 vs V2.5-thinking arms.
2. **Select** items where `v25.score < v0.score` → 230 V2.5 regressions.
3. **Surface** all rubric-events where `points * (v25_met - v0_met) < 0` (positive credit lost OR penalty triggered) with full per-arm judge explanations.
4. **Pre-rewrite** "Fails to <X>" rubric criterion text to positive form before showing the classifier (Round 4 patch — removed polarity confusion).
5. **Classify** each regression into one of 5 mutually-exclusive failure modes via gpt-4.1 (`temperature=0.0`):

| # | Category | Definition |
|---|---|---|
| 1 | Knowledge Gap | V2.5 silently omits factual info V0 included |
| 2 | Reasoning Collapse | V2.5 sets up correct chain then contradicts itself |
| 3 | Calibration Misalignment | Confidence/urgency shifts without info change |
| 4 | Context Misapplication | V2.5 loses an explicit prompt element (timeline/PMH/acuity) |
| 5 | Hallucinated Safeguards | V2.5 actively fabricates thresholds, citations, statistics, false reassurance |

Tiebreakers + rubric-event hints + `confident=false` escape hatch all in `_build_prompt` (commit-style on disk, untracked worktree).

## Results — N=230 V2.5-thinking regressions vs V0 on HB-Hard

| # | Category | Count | Share (of confident) |
|---|---|---:|---:|
| 1 | Knowledge Gap | **171** | **75.7%** |
| 2 | Reasoning Collapse | 0 | 0.0% |
| 3 | Calibration Misalignment | 3 | 1.3% |
| 4 | Context Misapplication | 3 | 1.3% |
| 5 | **Hallucinated Safeguards** | **49** | **21.7%** |
| | Errors (classifier returned cat=0 etc) | 3 | (excluded) |
| | Low-confidence | 1 | (excluded) |

**Spend:** ~$1.25 in gpt-4.1 across pilot v1-v4 (N=10×4) + cohort N=30 + full N=230.

## Key findings

1. **75.7% Knowledge Gap dominance** — V2.5-thinking SFT teaches new patterns at the cost of breadth. Many regressions are silent omissions of information V0 had.

2. **21.7% Hallucinated Safeguards (Bug-6 fix paid off)** — at the N=10 pilot this looked like 10%; at N=230 it's actually ~1-in-5. V2.5 introduces fabricated rigid thresholds, invented statistics, false reassurance, and made-up guideline citations. **This is the safety-critical pattern.**

3. **Reasoning Collapse (#2) is unobserved** — 0/230. Either truly absent in this corpus (V2.5 doesn't contradict itself; it just omits or fabricates) OR the classifier can't discriminate it. The N=10 pilot, N=30 cohort, and full N=230 all agree.

4. **Categories #3 (Calibration) and #4 (Context Misapp) at ~1% each** — barely above noise floor. May warrant collapsing to a single "other" category in V2.5b.

5. **Smoking-gun item 06942620** (-10 penalty for "Fails to advise emergency visit") classified as KG even after the Round 4 polarity-fix patch. Inspection confirms this is a TRUE silent omission on a safety-critical rubric — the 5-category taxonomy lacks a "safety-critical" tag orthogonal to mode. Future taxonomies should either:
   - (a) add a `safety_critical: bool` flag derivable from `points <= some_threshold`, OR
   - (b) define a new category #6 specifically for safety-critical omissions

## Implications for V2.5b corpus design

The dominant remediation targets, in proportion:

- **75% of corpus → broaden Knowledge Gap coverage**: more diverse clinical scenarios, fuller answer templates, explicit "include X" patterns. Prefer breadth over depth on the topics V2.5 silently omitted.
- **22% of corpus → anti-hallucination training**: explicit refusal patterns for "rigid threshold" and "specific percentage" claims unless backed by rubric-grade source. Teach the model to say "varies by guideline" instead of "<2 cm".
- **3% of corpus → mixed Calibration + Context Misapp**: too rare to design dedicated training; fold into general clinical-reasoning examples.
- **0% of corpus → Reasoning Collapse**: not a present pattern; skip.

## Reproducibility

```bash
# Tests (laptop, ~0.03s)
.venv/bin/python -m pytest tests/test_failure_cluster.py -v   # 16 passed
.venv/bin/ruff check scripts/ship_rule_lib/failure_cluster.py  # clean

# Re-run full classification (~12 min, ~$1)
set -a && source /Users/kiteboard/lostbench/.env && set +a
N_PILOT=230 RUN_TAG=full_n230 .venv/bin/python findings/2026-05-07-diagnostic-first-sft/pilot.py
```

Inputs frozen on the `feat/ship-rule-eval-driver-clean` worktree at
`.claude/worktrees/ship-rule-eval/findings/2026-05-05-v2.5-eval-thinking/graded/`.

## Next: C1 — V2.5b corpus SPEC

Corpus design SPEC follows in `V2.5B-CORPUS-SPEC.md` (next iter) with one section per category in proportion to the distribution above.
