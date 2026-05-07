"""Paired-bootstrap stats engine for V0 vs V2.5 ship-rule decisions.

Per PREREG `eval_protocol`:
    * paired bootstrap, 10000 resamples, 95% CI
    * `same_item_set: true` — every item is graded under both V0 and V2.5
      so we resample item-pairs (not unpaired arms).
    * Holm-Bonferroni across 4 benchmarks at family-wise alpha=0.05.
    * Cohen's d effect size (Cohen 1988); on paired binary scores we use
      d_z = mean(diff) / sd(diff).
    * Post-hoc power: parametric power for paired t under observed d_z and
      n_pairs at the (already corrected) per-benchmark alpha. Reported for
      transparency only; not gating.

Stdlib-only — no scipy/numpy hard dep, since this script must run on the
laptop side without the GPU stack. We use Python's `statistics` for stdev
and a hand-rolled Mersenne-twister-seeded resample loop.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass
class PairedResult:
    """Per-benchmark paired-bootstrap outcome."""

    n_pairs: int
    v0_mean: float
    v25_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    cohen_d: float
    power_post_hoc: float | None
    n_resamples: int
    seed: int

    def passes(self, *, lower_required: float) -> bool:
        return self.ci_low >= lower_required


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_inv_cdf(p: float) -> float:
    """Beasley-Springer-Moro approximation to the standard normal quantile.

    Good to ~1e-8 over (1e-9, 1-1e-9). Sufficient for power calcs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def paired_bootstrap(
    v0_scores: list[float],
    v25_scores: list[float],
    *,
    n_resamples: int = 10_000,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> PairedResult:
    """Compute paired bootstrap CI on the delta v25 - v0.

    Inputs must be aligned per-item: v0_scores[i] and v25_scores[i] are the
    same evaluation item. Items where either side is None must be filtered
    out by the caller before this function — paired bootstrap loses its
    validity if items drop on one side only.

    Returns PairedResult with delta = mean(v25) - mean(v0), CI on delta,
    two-sided bootstrap p-value (fraction of resampled deltas <=0 doubled,
    clipped to [0,1]), Cohen's d_z, and parametric post-hoc power.
    """
    if len(v0_scores) != len(v25_scores):
        raise ValueError(
            f"paired arms must have equal length; got {len(v0_scores)} vs {len(v25_scores)}"
        )
    n = len(v0_scores)
    if n < 2:
        return PairedResult(
            n_pairs=n,
            v0_mean=float("nan"),
            v25_mean=float("nan"),
            delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_two_sided=float("nan"),
            cohen_d=float("nan"),
            power_post_hoc=None,
            n_resamples=n_resamples,
            seed=seed,
        )

    diffs = [v25_scores[i] - v0_scores[i] for i in range(n)]
    v0_mean = statistics.fmean(v0_scores)
    v25_mean = statistics.fmean(v25_scores)
    delta = v25_mean - v0_mean

    rng = random.Random(seed)
    resampled_deltas: list[float] = []
    indices = list(range(n))
    for _ in range(n_resamples):
        # Resample paired indices with replacement.
        sample_idx = [rng.choice(indices) for _ in range(n)]
        resample_diffs = [diffs[i] for i in sample_idx]
        resampled_deltas.append(statistics.fmean(resample_diffs))
    resampled_deltas.sort()
    lo_idx = max(0, int(math.floor((ci_alpha / 2.0) * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1.0 - ci_alpha / 2.0) * n_resamples)) - 1)
    ci_low = resampled_deltas[lo_idx]
    ci_high = resampled_deltas[hi_idx]

    # Two-sided bootstrap p: fraction of resampled deltas <= 0 (centered),
    # doubled. We center the bootstrap distribution at zero per Efron-Tibshirani 1993.
    centered = [d - delta for d in resampled_deltas]
    n_extreme = sum(1 for d in centered if abs(d) >= abs(delta))
    p_two_sided = min(1.0, max(1.0 / n_resamples, n_extreme / n_resamples))

    # Cohen's d_z = mean(diff) / sd(diff). Returns NaN if sd == 0.
    if n >= 2:
        try:
            sd_diff = statistics.stdev(diffs)
        except statistics.StatisticsError:
            sd_diff = 0.0
        cohen_d = delta / sd_diff if sd_diff > 0 else float("inf") if delta != 0 else 0.0
    else:
        cohen_d = float("nan")

    power = post_hoc_power(d_z=cohen_d, n=n, alpha=ci_alpha) if math.isfinite(cohen_d) else None

    return PairedResult(
        n_pairs=n,
        v0_mean=v0_mean,
        v25_mean=v25_mean,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_two_sided=p_two_sided,
        cohen_d=cohen_d,
        power_post_hoc=power,
        n_resamples=n_resamples,
        seed=seed,
    )


def post_hoc_power(*, d_z: float, n: int, alpha: float) -> float:
    """Parametric power for a paired t-test (one-tailed positive direction).

    Non-central t approximated by shifted-normal: power ≈ Φ(d_z * sqrt(n) - z_{1-α}).
    Conservative for small n; informational only.
    """
    if n < 2 or not math.isfinite(d_z):
        return float("nan")
    z_crit = _normal_inv_cdf(1.0 - alpha)
    ncp = d_z * math.sqrt(n)
    return _normal_cdf(ncp - z_crit)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down at family-wise alpha. Returns reject[] aligned to input."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    reject = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        thresh = alpha / (n - rank)
        if p <= thresh:
            reject[orig_idx] = True
        else:
            # All remaining (higher p) also fail to reject — Holm step-down.
            break
    return reject


def align_paired(
    v0_records: list[dict], v25_records: list[dict], *, key: str = "item_id"
) -> tuple[list[dict], list[dict], list[str]]:
    """Filter both arms to the intersection of `key` values, in sorted-key order.

    Returns (v0_aligned, v25_aligned, dropped_ids). Items missing on either
    side are dropped — paired bootstrap requires same-item-set.
    """
    by_v0 = {r[key]: r for r in v0_records}
    by_v25 = {r[key]: r for r in v25_records}
    common = sorted(set(by_v0) & set(by_v25))
    dropped = sorted(set(by_v0) ^ set(by_v25))
    v0_a = [by_v0[k] for k in common]
    v25_a = [by_v25[k] for k in common]
    return v0_a, v25_a, [str(d) for d in dropped]
