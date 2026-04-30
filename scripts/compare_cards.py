#!/usr/bin/env python3
"""compare_cards — diff two sovereign_bench artifact JSONs and emit a
Markdown delta table. Use to compare Phase 2.1 vs Phase 1.5, B300-judge
vs prism-judge, or any two CARD-producing runs.

Usage:
    python scripts/compare_cards.py BASELINE.json CANDIDATE.json
    python scripts/compare_cards.py BASELINE.json CANDIDATE.json --out delta.md

Prints a table comparing per-fixture scores, per-axis means, run config
deltas (retrieval mode, judge model, embed model, temperature, etc.),
and a single-line "verdict": "PASS" if mean improved by >= 0.05 and no
fixture regressed > 0.10, "MIXED" otherwise. Exits 0 always; the verdict
is informational, not a CI gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _per_fixture_scores(art: dict) -> dict[str, float]:
    """Map fixture_id -> score (mean across trials)."""
    out: dict[str, list[float]] = {}
    for trial in art.get("trial_results", []):
        for ex in trial.get("per_example", []):
            fid = ex.get("example_id", "?")
            score = ex.get("score")
            if score is not None:
                out.setdefault(fid, []).append(float(score))
    return {fid: statistics.fmean(s) for fid, s in out.items()}


def _per_axis_means(art: dict) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for trial in art.get("trial_results", []):
        for ex in trial.get("per_example", []):
            for ax, v in (ex.get("per_axis") or {}).items():
                if v is not None:
                    out.setdefault(ax, []).append(float(v))
    return {ax: statistics.fmean(vs) for ax, vs in out.items()}


def _config_summary(art: dict) -> dict[str, object]:
    return {
        "serve_model": art.get("serve_model"),
        "judge_model": art.get("judge_model"),
        "embed_model": art.get("embed_model"),
        "rerank_model": art.get("rerank_model"),
        "retrieval_mode": art.get("retrieval_mode"),
        "n_corpus_chunks": art.get("n_corpus_chunks"),
        "retrieval_top_n": art.get("retrieval_top_n"),
        "temperature": art.get("temperature"),
        "clinical_system_prompt": art.get("clinical_system_prompt"),
        "n_per_trial": art.get("n_per_trial"),
        "trials": art.get("trials"),
    }


def _delta_arrow(a: float, b: float, threshold: float = 0.005) -> str:
    d = b - a
    if d > threshold:
        return f"+{d:.3f}"
    if d < -threshold:
        return f"{d:.3f}"
    return "±0.000"


def _verdict(base_mean: float, cand_mean: float, fixture_deltas: dict[str, float]) -> str:
    mean_lift = cand_mean - base_mean
    worst_regression = min(fixture_deltas.values()) if fixture_deltas else 0.0
    if mean_lift >= 0.05 and worst_regression > -0.10:
        return "PASS — significant lift, no major regression"
    if mean_lift >= 0.02 and worst_regression > -0.05:
        return "MARGINAL — small lift, no meaningful regression"
    if mean_lift > 0:
        return f"MIXED — lift {mean_lift:+.3f} but worst regression {worst_regression:+.3f}"
    if abs(mean_lift) < 0.005:
        return "FLAT — no measurable change"
    return f"REGRESSION — mean dropped {mean_lift:+.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="path to baseline artifact JSON")
    parser.add_argument("candidate", type=Path, help="path to candidate artifact JSON")
    parser.add_argument("--out", type=Path, default=None, help="write Markdown to this path (default: stdout)")
    parser.add_argument("--baseline-label", default=None, help="display label for baseline")
    parser.add_argument("--candidate-label", default=None, help="display label for candidate")
    args = parser.parse_args()

    base = _load(args.baseline)
    cand = _load(args.candidate)

    base_label = args.baseline_label or args.baseline.parent.name
    cand_label = args.candidate_label or args.candidate.parent.name

    base_fix = _per_fixture_scores(base)
    cand_fix = _per_fixture_scores(cand)
    all_fix = sorted(set(base_fix) | set(cand_fix))

    base_axes = _per_axis_means(base)
    cand_axes = _per_axis_means(cand)
    all_axes = sorted(set(base_axes) | set(cand_axes))

    base_cfg = _config_summary(base)
    cand_cfg = _config_summary(cand)

    fix_deltas = {
        fid: cand_fix.get(fid, 0.0) - base_fix.get(fid, 0.0)
        for fid in all_fix
        if fid in base_fix and fid in cand_fix
    }
    base_mean = statistics.fmean(base_fix.values()) if base_fix else 0.0
    cand_mean = statistics.fmean(cand_fix.values()) if cand_fix else 0.0

    out_lines: list[str] = []
    a = out_lines.append
    a(f"# CARD delta — {base_label} → {cand_label}")
    a("")
    a(f"- **Baseline**: `{args.baseline}`")
    a(f"- **Candidate**: `{args.candidate}`")
    a(f"- **Mean score**: `{base_mean:.3f}` → `{cand_mean:.3f}` ({_delta_arrow(base_mean, cand_mean)})")
    a(f"- **Verdict**: {_verdict(base_mean, cand_mean, fix_deltas)}")
    a("")
    a("## Per-fixture deltas")
    a("")
    a("| Fixture | Baseline | Candidate | Delta |")
    a("|---|---|---|---|")
    for fid in all_fix:
        b = base_fix.get(fid)
        c = cand_fix.get(fid)
        b_str = f"{b:.3f}" if b is not None else "(missing)"
        c_str = f"{c:.3f}" if c is not None else "(missing)"
        if b is None or c is None:
            d_str = "n/a"
        else:
            d_str = _delta_arrow(b, c)
        a(f"| {fid} | {b_str} | {c_str} | {d_str} |")
    a("")
    a("## Per-axis means")
    a("")
    a("| Axis | Baseline | Candidate | Delta |")
    a("|---|---|---|---|")
    for ax in all_axes:
        b = base_axes.get(ax)
        c = cand_axes.get(ax)
        b_str = f"{b:+.3f}" if b is not None else "(missing)"
        c_str = f"{c:+.3f}" if c is not None else "(missing)"
        if b is None or c is None:
            d_str = "n/a"
        else:
            d_str = _delta_arrow(b, c)
        a(f"| {ax} | {b_str} | {c_str} | {d_str} |")
    a("")
    a("## Run config diff")
    a("")
    a("| Field | Baseline | Candidate |")
    a("|---|---|---|")
    for k in sorted(set(base_cfg) | set(cand_cfg)):
        bv = base_cfg.get(k)
        cv = cand_cfg.get(k)
        if bv == cv:
            a(f"| {k} | `{bv}` | (same) |")
        else:
            a(f"| **{k}** | `{bv}` | `{cv}` |")
    a("")

    rendered = "\n".join(out_lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
