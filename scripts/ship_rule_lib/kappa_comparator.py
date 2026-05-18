"""κ-calibration comparator for the V2.5b failure-mode probe.

Reads physician-adjudicated CSV (`kappa_user_labels.csv`) and the gpt-4.1
answer key (`kappa_answer_key.jsonl`), pairs by `item_id`, computes Cohen's
κ and a per-category disagreement matrix.

Stdlib only — no scikit-learn, no numpy. Cohen's formula and matrix counts
are self-contained.

Usage:
    .venv/bin/python -c "from scripts.ship_rule_lib.kappa_comparator import run_report; \
        run_report(user_csv='findings/2026-05-07-diagnostic-first-sft/kappa_user_labels.csv', \
                   key_jsonl='findings/2026-05-07-diagnostic-first-sft/kappa_answer_key.jsonl')"
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_CATEGORIES: tuple[int, ...] = (1, 2, 3, 4, 5)


def cohen_kappa(
    rater_a: list[int],
    rater_b: list[int],
    *,
    categories: Sequence[int] = DEFAULT_CATEGORIES,
) -> float:
    """Cohen's κ for two equal-length integer label streams.

    κ = (po - pe) / (1 - pe)
    where po = observed agreement, pe = expected agreement by chance.
    """
    n = len(rater_a)
    if n == 0:
        raise ValueError("non-empty raters required")
    if n != len(rater_b):
        raise ValueError(
            f"equal-length raters required (got {len(rater_a)} vs {len(rater_b)})"
        )
    po = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    a_counts = Counter(rater_a)
    b_counts = Counter(rater_b)
    pe = sum((a_counts[c] / n) * (b_counts[c] / n) for c in categories)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def disagreement_matrix(
    *,
    gpt: list[int],
    user: list[int],
    categories: Sequence[int] = DEFAULT_CATEGORIES,
) -> dict[int, dict[int, int]]:
    """Confusion-style matrix: mat[gpt_cat][user_cat] = count."""
    mat: dict[int, dict[int, int]] = {
        c: {c2: 0 for c2 in categories} for c in categories
    }
    for g, u in zip(gpt, user, strict=True):
        if g in mat and u in mat[g]:
            mat[g][u] += 1
    return mat


def load_user_labels(csv_path: Path) -> list[dict[str, Any]]:
    """Read the physician-filled CSV. Skips rows where category is blank.
    Treats `0` as 'no category fits' and surfaces it via `no_category_fits=True`."""
    csv_path = Path(csv_path)
    rows: list[dict[str, Any]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            cat_raw = (r.get("category") or "").strip()
            if cat_raw == "":
                continue
            try:
                cat = int(cat_raw)
            except ValueError:
                continue
            confident_raw = (r.get("confident") or "").strip().lower()
            confident = confident_raw in ("true", "t", "yes", "y", "1")
            rows.append(
                {
                    "item_id": r["item_id"],
                    "seed": int(r.get("seed", "0") or 0),
                    "category": cat,
                    "confident": confident,
                    "no_category_fits": cat == 0,
                    "notes": r.get("notes", ""),
                }
            )
    return rows


def load_answer_key(key_jsonl: Path) -> dict[str, dict[str, Any]]:
    """Read gpt-4.1 ground-truth answer key. Returned dict keyed by item_id."""
    out: dict[str, dict[str, Any]] = {}
    with Path(key_jsonl).open() as f:
        for line in f:
            row = json.loads(line)
            out[row["item_id"]] = row
    return out


def pair_user_to_key(
    *,
    user_csv: Path,
    key_jsonl: Path,
) -> list[dict[str, Any]]:
    """Inner-join physician labels with gpt-4.1 labels by item_id."""
    user_rows = load_user_labels(user_csv)
    key = load_answer_key(key_jsonl)
    paired: list[dict[str, Any]] = []
    for u in user_rows:
        k = key.get(u["item_id"])
        if k is None:
            continue
        paired.append(
            {
                "item_id": u["item_id"],
                "seed": u["seed"],
                "user_category": u["category"],
                "user_confident": u["confident"],
                "user_notes": u.get("notes", ""),
                "gpt41_category": k.get("gpt41_category"),
                "gpt41_category_name": k.get("gpt41_category_name"),
                "gpt41_justification": k.get("gpt41_justification"),
            }
        )
    return paired


def run_report(
    *,
    user_csv: str | Path,
    key_jsonl: str | Path,
    out_md: str | Path | None = None,
) -> str:
    """End-to-end: pair, compute κ + matrix, render markdown report.

    The report is also returned as a string. If `out_md` is given, written there.
    Excludes user rows with category=0 (no-category-fits) from the κ computation
    but counts them in the disagreement matrix as a separate observation.
    """
    paired = pair_user_to_key(user_csv=Path(user_csv), key_jsonl=Path(key_jsonl))
    if not paired:
        return "# κ report\n\nNo paired rows found. Has the physician filled in any labels yet?\n"

    # Exclude no-category-fits items from κ calc; they're informational.
    used = [p for p in paired if p["user_category"] != 0]
    no_fit = [p for p in paired if p["user_category"] == 0]

    if not used:
        kappa = float("nan")
    else:
        gpt_labels = [p["gpt41_category"] for p in used]
        usr_labels = [p["user_category"] for p in used]
        kappa = cohen_kappa(gpt_labels, usr_labels)

    if used:
        mat = disagreement_matrix(
            gpt=[p["gpt41_category"] for p in used],
            user=[p["user_category"] for p in used],
        )
    else:
        mat = {c: {c2: 0 for c2 in DEFAULT_CATEGORIES} for c in DEFAULT_CATEGORIES}

    lines: list[str] = []
    lines.append("# κ-calibration report — physician vs gpt-4.1\n")
    lines.append(f"- Paired items: **{len(paired)}** (used in κ: {len(used)}; "
                 f"no-fit: {len(no_fit)})")
    lines.append(f"- Cohen's κ: **{kappa:.3f}**")
    if kappa >= 0.81:
        verdict = "ALMOST PERFECT (Landis & Koch 1977)"
    elif kappa >= 0.61:
        verdict = "SUBSTANTIAL (Landis & Koch) — green-light V2.5b generation"
    elif kappa >= 0.41:
        verdict = "MODERATE — refine classifier prompt before generation"
    elif kappa >= 0.21:
        verdict = "FAIR — classifier needs material refinement"
    elif kappa >= 0.0:
        verdict = "SLIGHT — classifier basically not discriminating"
    else:
        verdict = "NEGATIVE — classifier worse than chance; likely bug"
    lines.append(f"- Interpretation: **{verdict}**")
    lines.append("")
    lines.append("## Disagreement matrix (rows = gpt-4.1, cols = physician)\n")
    header = ["gpt-4.1\\physician"] + [f"cat {c}" for c in DEFAULT_CATEGORIES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for c in DEFAULT_CATEGORIES:
        row = [f"cat {c}"] + [str(mat[c][c2]) for c2 in DEFAULT_CATEGORIES]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if no_fit:
        lines.append("## No-category-fits (physician marked 0)\n")
        for p in no_fit:
            lines.append(
                f"- `{p['item_id'][:8]}@seed{p['seed']}` "
                f"(gpt-4.1 said {p['gpt41_category']}: {p['gpt41_category_name']})"
            )
        lines.append("")
    # Disagreements (off-diagonal, top by count)
    disagreements = [
        (g, u, mat[g][u]) for g in DEFAULT_CATEGORIES for u in DEFAULT_CATEGORIES
        if g != u and mat[g][u] > 0
    ]
    disagreements.sort(key=lambda x: -x[2])
    if disagreements:
        lines.append("## Top off-diagonal cells (gpt-4.1 → physician)\n")
        for g, u, n in disagreements[:8]:
            lines.append(f"- gpt-4.1 cat {g} → physician cat {u}: **{n}** items")
        lines.append("")

    md = "\n".join(lines)
    if out_md is not None:
        Path(out_md).write_text(md)
    return md
