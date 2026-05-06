# E6 — Statistical correction protocol for V2.5 ship-rule + extended evals

**Date:** 2026-05-06
**Authority:** Pre-registered (this commit timestamp). Any eval CARD that
references this protocol must use these exact procedures or document the
divergence.

## 1. Multiple-comparison correction

The V2.5 ship-rule (per `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`
ship_rule block) tests **k = 4** primary hypotheses simultaneously:

1. MedQA-USMLE delta lower-CI-bound > 0
2. MedXpertQA-Text delta lower-CI-bound >= +5pp
3. HealthBench-Hard delta point-estimate > 0
4. PubMedQA-L delta lower-CI-bound >= -1pp (no-regression)

**Correction:** **Holm-Bonferroni**, alpha_family = 0.05.

Procedure:
1. Compute per-benchmark p_i (from paired-bootstrap; see §2 for the
   definition of p in the bootstrap framework).
2. Sort p_(1) <= p_(2) <= ... <= p_(k).
3. Reject H_(i) iff p_(i) <= alpha / (k - i + 1) for all i' <= i.
4. STOP rejecting at the first i where the inequality fails.

Equivalent SciPy:
```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='holm')
```

**For ship-rule we use the more stringent Holm — not Bonferroni** because
Bonferroni is too conservative for a 4-test family (lose 50% of
detectable effect size at k=4).

For A4 (extended evals) k = 4 again (MedMCQA, MMLU-Med, CareQA, MMLU-STEM).
Same Holm.

For A5 k = 2 (the two primary thinking-on-vs-off criteria). Same Holm.

For B4 k = 3 (VQA-RAD, SLAKE, ASR). Same Holm.

## 2. Paired-bootstrap CI (per benchmark)

The PREREG mandates paired-bootstrap with 10000 resamples and 95% CI on
the SAME item set comparing V2.5 vs V0.

Procedure for each benchmark:
1. Let `s_v2.5_i` = score on item i with V2.5 serve.
2. Let `s_v0_i`   = score on item i with V0 serve (same item, same seed,
   same prompt, same judge).
3. Let `d_i = s_v2.5_i - s_v0_i`.
4. Resample (with replacement) from `{d_i}` 10000 times; compute the mean
   of each resample.
5. CI_95 = (2.5th percentile of resample means, 97.5th percentile).
6. Reject H_0_i (delta == 0) iff 0 is outside the CI_95.

**Bootstrap p-value (for Holm correction):**
```
p_i = 2 * min(
  fraction of resample means <= 0,
  fraction of resample means >= 0
)
```
Two-tailed; clamp at 1e-4 (the resolution of 10000 resamples).

## 3. Effect size — Cohen's d (paired)

For each benchmark report alongside the CI:

```
d_paired = mean(d_i) / sd(d_i)
```

Interpretation guide (Cohen 1988, applied to paired delta):
- d < 0.2: trivial
- 0.2 <= d < 0.5: small
- 0.5 <= d < 0.8: medium
- 0.8 <= d: large

**Required reporting cell** in each CARD.md:
| Benchmark | Delta (pp) | 95% CI | Cohen's d | p (raw) | p (Holm) | Pass? |
|---|---|---|---|---|---|---|

## 4. Post-hoc power calculation

For each benchmark with a published `success_criteria` margin (e.g.
MedXpertQA-Text >= +5pp), report:

- **observed effect size** d_obs
- **power achieved** at alpha = 0.05 two-tailed, n = paired-sample-size

Formula (paired t-power):
```python
from statsmodels.stats.power import TTestPower
power = TTestPower().power(
    effect_size=d_obs,    # observed (NOT pre-specified) Cohen's d
    nobs=n_pairs,
    alpha=0.05,
    alternative='two-sided',
)
```

A power < 0.80 paired with a non-significant result is a **null-result
caveat** — the experiment was underpowered to detect the pre-specified
margin. Document, do not auto-fail.

## 5. Equivalence testing (B3 NVFP4 vs BF16)

For B3 the null is "EQUIVALENT" not "no difference". Use **Two One-Sided
Tests (TOST)** at margin = 1pp, alpha = 0.025 each side.

```python
from statsmodels.stats.weightstats import ttost_paired
p1, _, _ = ttost_paired(scores_nvfp4, scores_bf16, low=-0.01, upp=0.01)
# Reject non-equivalence iff p1 < 0.05 AND CI_95 lies entirely within [-0.01, +0.01]
```

This is the ONLY test in the E-track family that uses TOST; all others
use the standard "is delta > 0 (or > margin)" framing.

## 6. Required reporting block (paste into CARD.md per benchmark)

```markdown
### <Benchmark>

| Field | Value |
|---|---|
| n_pairs | XXX |
| V2.5 mean (pp) | XX.X |
| V0 mean (pp) | XX.X |
| Delta (pp) | +X.X |
| 95% paired-bootstrap CI | [XX.X, XX.X] |
| Cohen's d (paired) | X.XX |
| p (raw, two-tailed) | X.XXX |
| p (Holm, k=4) | X.XXX |
| Power at observed d (alpha=0.05) | X.XX |
| Pass? | Y/N |
| Pass criterion (from PREREG) | <verbatim> |
| Notes | <e.g. judge_incomplete count, contamination flag> |
```

## 7. Hard rules

- **Do not change k after seeing the data.** k is fixed by the PREREG.
- **Do not selectively drop a benchmark** to lower k. If a benchmark
  fails its PREREG'd inclusion criteria (data leakage E4 verdict, judge
  failure, etc.), DROP IT and STATE the drop in the CARD; the family
  size k REMAINS the original count for Holm purposes (this prevents
  garden-of-forking-paths bias).
- **Report negative results** with the same rigor. A "didn't pass"
  benchmark gets the same table as a "passed" one.
- **Cross-reference E4 contamination verdict** in every CARD that uses
  any of the four ship-rule benchmarks.

## 8. Tooling check

The lobster `python3` env (per `manifest-lobster.json`) is expected to
have `statsmodels`, `scipy`, and `numpy`. Confirmation step queued for
the eval-runner agent before computing — fail-loud if missing.

The laptop `python3` may not have `statsmodels` (the laptop is a thin
client; heavy stats run on lobster). If needed laptop-side, install
into a scratch venv at `/tmp/stats-venv` to avoid polluting system
Python.
