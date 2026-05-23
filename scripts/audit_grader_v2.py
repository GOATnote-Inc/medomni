#!/usr/bin/env python3
"""audit_grader_v2 — weight-aware dashboard for medomni.

The structural replacement for the v1 hourly bot that updates issue #396.
v1 grades presence; v2 grades load-bearing properties. See
`findings/research/2026-05-23-audit-grader-v2/SPEC.md` for the design
rationale and the v1 retirement plan.

Three load-bearing categories:
  A. Hot-path discipline (CLAUDE.md §0)
  B. Metric integrity (canonical metrics)
  C. Product progress (research-loop liveness)

One decorative-legacy category:
  D. Presence-only checks reproduced from v1, clearly marked as decorative.

Each load-bearing check returns:
  - status: PASS / FAIL / DEFERRED / N/A
  - reason: one sentence explaining *why* this status
  - evidence: file paths + line numbers + actual values

Output: markdown dashboard to stdout. The CI workflow pipes that into
  `gh issue edit <N> --repo <repo> --body-file -`

Exit code is always 0 — this script is a status reporter, not a CI gate.
"""

from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc)
DIAG_DIR = REPO / "findings/2026-05-07-diagnostic-first-sft"


# ----------------------------------------------------------------------
# Check primitive
# ----------------------------------------------------------------------


class Check:
    __slots__ = ("id", "title", "status", "reason", "evidence")

    def __init__(self, axis_id: str, title: str) -> None:
        self.id = axis_id
        self.title = title
        self.status: str = "DEFERRED"
        self.reason: str = "(no check ran)"
        self.evidence: list[str] = []

    def _set(self, status: str, reason: str, evidence: tuple) -> "Check":
        self.status = status
        self.reason = reason
        self.evidence = [str(e) for e in evidence]
        return self

    def passed(self, reason: str, *evidence) -> "Check":
        return self._set("PASS", reason, evidence)

    def failed(self, reason: str, *evidence) -> "Check":
        return self._set("FAIL", reason, evidence)

    def deferred(self, reason: str, *evidence) -> "Check":
        return self._set("DEFERRED", reason, evidence)

    def na(self, reason: str) -> "Check":
        return self._set("N/A", reason, ())


# ----------------------------------------------------------------------
# §0 extraction
# ----------------------------------------------------------------------


def _extract_s0(claude_md_text: str) -> str:
    """Extract the `## §0 — ...` block from CLAUDE.md.

    Returns the block text without the heading line, or '' if not found.
    """
    m = re.search(
        r"^##\s*§0[^\n]*\n(.+?)(?=^##\s|\Z)",
        claude_md_text,
        re.MULTILINE | re.DOTALL,
    )
    return (m.group(1) if m else "").strip()


# ----------------------------------------------------------------------
# Category A — Hot-path discipline (CLAUDE.md §0)
# ----------------------------------------------------------------------


def check_a1_line_count(s0: str) -> Check:
    c = Check("A1", "§0 effective line count ≤ 15")
    effective = [ln for ln in s0.splitlines() if ln.strip()]
    n = len(effective)
    ev = (
        f"effective lines: {n}",
        f"first line: {effective[0][:80] if effective else '(empty)'}",
    )
    if n == 0:
        return c.failed("no §0 found in CLAUDE.md", *ev)
    if n <= 15:
        return c.passed(f"§0 is {n} effective lines (≤ 15 budget)", *ev)
    return c.failed(f"§0 is {n} effective lines, over the 15-line budget", *ev)


def check_a2_last_audited(s0: str) -> Check:
    c = Check("A2", "§0 `Last audited:` timestamp present and ≤ 90 days old")
    m = re.search(r"[Ll]ast audited:?\s*(\d{4}-\d{2}-\d{2})", s0)
    if not m:
        return c.failed("no `Last audited:` timestamp in §0")
    audited = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    age_days = (NOW - audited).days
    ev = (f"Last audited: {m.group(1)}", f"age: {age_days} days")
    if age_days <= 90:
        return c.passed(f"audited {age_days} days ago (≤ 90)", *ev)
    return c.failed(f"audited {age_days} days ago (> 90 stale threshold)", *ev)


def check_a3_skill_or_hook_pointer(s0: str) -> Check:
    c = Check("A3", "§0 points at `.claude/skills/` or `.claude/hooks/`")
    skill_ref = ".claude/skills" in s0
    hook_ref = ".claude/hooks" in s0 or ".claude/settings.json" in s0
    ev = (
        f".claude/skills/ referenced: {skill_ref}",
        f".claude/hooks/ or settings.json referenced: {hook_ref}",
    )
    if skill_ref or hook_ref:
        return c.passed("§0 points at load-on-demand alternatives", *ev)
    return c.failed("§0 has no pointer to skills/ or hooks/", *ev)


# ----------------------------------------------------------------------
# Category B — Metric integrity
# ----------------------------------------------------------------------


def _load_judged_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("judge_score") is not None:
                out.append(r)
    except OSError:
        return []
    return out


def _per_pattern_stats(records: list[dict]) -> dict[str, tuple[int, float, float]]:
    by: dict[str, list[float]] = collections.defaultdict(list)
    for r in records:
        by[r.get("pattern_addressed", "?")].append(r["judge_score"])
    out = {}
    for p, scores in by.items():
        n = len(scores)
        out[p] = (
            n,
            statistics.mean(scores),
            statistics.stdev(scores) if n > 1 else 0.0,
        )
    return out


def check_b1_per_pattern_sigma() -> Check:
    c = Check("B1", "per-pattern σ ≥ 0.10 on current iter corpus")
    post = _load_judged_records(DIAG_DIR / "v25b_corpus_FINAL.jsonl")
    if not post:
        return c.deferred("v25b_corpus_FINAL.jsonl absent or has no judged records")
    stats = _per_pattern_stats(post)
    bad = [(p, s) for p, (_n, _m, s) in stats.items() if s < 0.10]
    worst = sorted(bad, key=lambda x: x[1])[:3]
    ev = (
        f"patterns total: {len(stats)}",
        f"patterns with σ < 0.10: {len(bad)}",
        "lowest σ: " + ", ".join(f"{p}={s:.3f}" for p, s in worst) if worst else "lowest σ: (none below 0.10)",
    )
    if not bad:
        return c.passed(f"all {len(stats)} patterns have σ ≥ 0.10", *ev)
    return c.failed(
        f"{len(bad)} of {len(stats)} patterns have σ < 0.10 — judge does not discriminate on those patterns",
        *ev,
    )


def check_b2_variance_retention() -> Check:
    c = Check("B2", "corpus σ_post ≥ 0.6 × σ_pre")
    pre = _load_judged_records(DIAG_DIR / "v25b_judged_full.jsonl")
    post = _load_judged_records(DIAG_DIR / "v25b_corpus_FINAL.jsonl")
    if not pre or not post:
        return c.deferred("missing iter=0 (v25b_judged_full) or iter=1 (v25b_corpus_FINAL) corpus")
    pre_scores = [r["judge_score"] for r in pre]
    post_scores = [r["judge_score"] for r in post]
    s_pre = statistics.stdev(pre_scores)
    s_post = statistics.stdev(post_scores)
    ratio = s_post / s_pre if s_pre > 0 else float("inf")
    ev = (
        f"σ_pre = {s_pre:.4f} (n={len(pre)})",
        f"σ_post = {s_post:.4f} (n={len(post)})",
        f"ratio σ_post / σ_pre = {ratio:.3f}",
    )
    if ratio >= 0.60:
        return c.passed(f"σ_post / σ_pre = {ratio:.3f} ≥ 0.60", *ev)
    return c.failed(
        f"σ_post / σ_pre = {ratio:.3f} < 0.60 — corpus-wide variance collapsed",
        *ev,
    )


def check_b3_prereg_result_chain() -> Check:
    c = Check("B3", "pre-reg / result / gate-raise pre-reg chain is complete")
    p_iter1 = DIAG_DIR / "ITER1_DIAGNOSTIC_PREREGISTRATION.md"
    r_iter1 = DIAG_DIR / "ITER1_DIAGNOSTIC_RESULT.md"
    g_iter2 = DIAG_DIR / "ITER2_GATE_PREREGISTRATION.md"
    has_p1, has_r1, has_g2 = p_iter1.exists(), r_iter1.exists(), g_iter2.exists()
    ev = (
        f"iter=1 pre-reg present: {has_p1}",
        f"iter=1 result present: {has_r1}",
        f"iter=2 gate-raise pre-reg present: {has_g2}",
    )
    if has_p1 and has_r1 and has_g2:
        return c.passed("iter=1 pre-reg + result + iter=2 gate-raise pre-reg all present", *ev)
    missing = []
    if not has_p1:
        missing.append("iter=1 pre-reg")
    if not has_r1:
        missing.append("iter=1 result")
    if not has_g2:
        missing.append("iter=2 gate-raise pre-reg")
    return c.failed(f"missing: {', '.join(missing)}", *ev)


def check_b4_keep_decision_honest() -> Check:
    c = Check("B4", "iter=1 KEEP decision is honored by the σ-checks")
    result = DIAG_DIR / "ITER1_DIAGNOSTIC_RESULT.md"
    if not result.exists():
        return c.deferred("no iter=1 result file")
    text = result.read_text()
    invalid = "provisionally INVALID" in text
    sigma_fail = check_b1_per_pattern_sigma().status == "FAIL" or check_b2_variance_retention().status == "FAIL"
    ev = (
        f"'provisionally INVALID' phrase present in result: {invalid}",
        f"any σ-criterion currently failing: {sigma_fail}",
    )
    if sigma_fail and invalid:
        return c.passed("σ-checks fail AND result already marks KEEP as INVALID — conjunctive rule honored", *ev)
    if not sigma_fail and not invalid:
        return c.passed("σ-checks pass AND result does not invalidate KEEP — consistent", *ev)
    return c.failed(
        "σ-checks and result document disagree on the KEEP verdict — conjunctive rule not honored",
        *ev,
    )


# ----------------------------------------------------------------------
# Category C — Product progress
# ----------------------------------------------------------------------


def check_c1_experiment_log_recency() -> Check:
    c = Check("C1", "`EXPERIMENT_LOG.jsonl` latest entry ≤ 30 days old")
    path = DIAG_DIR / "EXPERIMENT_LOG.jsonl"
    if not path.exists():
        return c.failed("no EXPERIMENT_LOG.jsonl in diagnostic-first-sft/")
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not entries:
        return c.failed("EXPERIMENT_LOG.jsonl is empty")
    last = entries[-1]
    ts = last.get("timestamp_utc")
    if not ts:
        return c.failed("latest entry has no `timestamp_utc` field")
    try:
        last_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return c.failed(f"latest entry's timestamp is unparseable: {ts!r}")
    age_days = (NOW - last_dt).days
    iter_n = last.get("iter", "?")
    ev = (f"latest entry: iter={iter_n}, timestamp={ts}", f"age: {age_days} days")
    if age_days <= 30:
        return c.passed(f"latest iter={iter_n} entry from {age_days} days ago", *ev)
    return c.failed(
        f"latest iter={iter_n} entry is {age_days} days old (> 30 stagnation threshold)",
        *ev,
    )


def check_c2_open_loops() -> Check:
    c = Check("C2", "pre-reg → result pairing (open loops named, not lost)")
    if not DIAG_DIR.exists():
        return c.deferred("no diagnostic-first-sft dir")
    prereg_files = sorted(DIAG_DIR.glob("*PREREGISTRATION*.md"))
    result_files = sorted(DIAG_DIR.glob("*RESULT*.md"))
    addendum_files = sorted(DIAG_DIR.glob("*ADDENDUM*.md"))
    iter1_pre = any("ITER1" in f.name for f in prereg_files)
    iter1_res = any("ITER1" in f.name and "RESULT" in f.name for f in result_files)
    iter2_pre = any("ITER2" in f.name for f in prereg_files)
    iter2_res = any("ITER2" in f.name and "RESULT" in f.name for f in result_files)
    ev = (
        f"pre-reg files: {len(prereg_files)}",
        f"result files: {len(result_files)}",
        f"addendum files: {len(addendum_files)}",
        f"iter=1: pre-reg={iter1_pre}, result={iter1_res}",
        f"iter=2: pre-reg={iter2_pre}, result={iter2_res}",
    )
    open_loops = []
    if iter1_pre and not iter1_res:
        open_loops.append("iter=1 (pre-reg present, no result)")
    if iter2_pre and not iter2_res:
        open_loops.append("iter=2 gate-raise (expected open until iter=2 actually runs)")
    if not open_loops:
        return c.passed("no orphaned pre-regs", *ev)
    return c.deferred(f"open loops named (not lost): {'; '.join(open_loops)}", *ev)


# ----------------------------------------------------------------------
# Category D — Decorative-legacy (v1 presence checks)
# ----------------------------------------------------------------------


def _scripts_no_cat_env() -> bool:
    scripts_dir = REPO / "scripts"
    if not scripts_dir.exists():
        return True
    pat = re.compile(r"\bcat\s+\.env\b")
    for f in scripts_dir.rglob("*.sh"):
        try:
            if pat.search(f.read_text(errors="ignore")):
                return False
        except OSError:
            continue
    return True


def check_d_legacy() -> list[Check]:
    legacy: list[Check] = []

    def add(label: str, exists: bool, evidence: str = "") -> None:
        c = Check(f"D[{label}]", f"{label}")
        if exists:
            c.passed("present (decorative — presence-only; does not load-bear)", evidence)
        else:
            c.failed("absent", evidence)
        legacy.append(c)

    add("CLAUDE.md", (REPO / "CLAUDE.md").exists())
    add(".claude/skills/", (REPO / ".claude/skills").exists())
    add(".claude/agents/", (REPO / ".claude/agents").exists())
    add(".claude/settings.json", (REPO / ".claude/settings.json").exists())
    add(".claude/hooks/", (REPO / ".claude/hooks").exists())
    add(".mcp.json", (REPO / ".mcp.json").exists())
    add("tests/", (REPO / "tests").exists() or (REPO / "web/tests").exists())
    add("scripts/", (REPO / "scripts").exists())
    add("no `cat .env` in scripts/", _scripts_no_cat_env())
    add(".pre-commit-config.yaml", (REPO / ".pre-commit-config.yaml").exists())
    add(
        "PROGRAM.md with allow/lock paths",
        (DIAG_DIR / "PROGRAM.md").exists(),
    )
    add("EXPERIMENT_LOG.jsonl", (DIAG_DIR / "EXPERIMENT_LOG.jsonl").exists())
    return legacy


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------


def _render_check_table(checks: list[Check]) -> str:
    rows = ["| ID | Check | Status | Reason |", "| --- | --- | --- | --- |"]
    for c in checks:
        # Escape pipe chars inside reason
        reason = c.reason.replace("|", "\\|")
        rows.append(f"| `{c.id}` | {c.title} | **{c.status}** | {reason} |")
    return "\n".join(rows)


def _render_evidence(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        if c.evidence:
            lines.append(f"- **`{c.id}` — {c.title}** ({c.status}):")
            for e in c.evidence:
                lines.append(f"  - {e}")
    return "\n".join(lines) if lines else "(no evidence collected)"


def main() -> int:
    claude_md_path = REPO / "CLAUDE.md"
    if not claude_md_path.exists():
        # Hard fail with markdown stub
        print("# audit (v2) — CLAUDE.md not found\n\nCannot grade without `CLAUDE.md`. This is structural, not a v2-check failure.")
        return 0

    claude_md = claude_md_path.read_text()
    s0 = _extract_s0(claude_md)

    a = [
        check_a1_line_count(s0),
        check_a2_last_audited(s0),
        check_a3_skill_or_hook_pointer(s0),
    ]
    b = [
        check_b1_per_pattern_sigma(),
        check_b2_variance_retention(),
        check_b3_prereg_result_chain(),
        check_b4_keep_decision_honest(),
    ]
    c = [
        check_c1_experiment_log_recency(),
        check_c2_open_loops(),
    ]
    d = check_d_legacy()
    load_bearing = a + b + c

    def cnt(xs: list[Check], s: str) -> int:
        return sum(1 for x in xs if x.status == s)

    pass_ = cnt(load_bearing, "PASS")
    fail_ = cnt(load_bearing, "FAIL")
    deferred_ = cnt(load_bearing, "DEFERRED")
    na_ = cnt(load_bearing, "N/A")

    out: list[str] = []
    out.append("# audit (v2): weight-aware best-practices dashboard")
    out.append("")
    out.append(f"**Last run:** {NOW.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    out.append("**Source:** `scripts/audit_grader_v2.py`")
    out.append("**Spec:** `findings/research/2026-05-23-audit-grader-v2/SPEC.md`")
    out.append(
        "**Cadence:** daily — the v1 hourly grader at issue #396 had 10 consecutive identical updates; hourly was theater."
    )
    out.append("")
    out.append("## Summary — load-bearing axes (A + B + C)")
    out.append("")
    out.append(f"- **PASS:** {pass_}")
    out.append(f"- **FAIL:** {fail_}")
    out.append(f"- **DEFERRED:** {deferred_}")
    out.append(f"- **N/A:** {na_}")
    out.append("")
    out.append("Category D (decorative-legacy) is reported separately and does **not** count toward the headline.")
    out.append("")
    out.append("## A. Hot-path discipline (`CLAUDE.md` §0)")
    out.append("")
    out.append(_render_check_table(a))
    out.append("")
    out.append("## B. Metric integrity (canonical metrics)")
    out.append("")
    out.append(_render_check_table(b))
    out.append("")
    out.append("## C. Product progress (research-loop liveness)")
    out.append("")
    out.append(_render_check_table(c))
    out.append("")
    out.append("## D. Decorative-legacy (v1 presence-only)")
    out.append("")
    out.append(
        "These are the v1 grader's 13 binary presence checks, reproduced for continuity. **They do not load-bear.** A `PASS` here means the artifact exists; it does NOT mean the artifact is correct or useful. Headline axes are A / B / C above."
    )
    out.append("")
    out.append(_render_check_table(d))
    out.append("")
    out.append("## Evidence")
    out.append("")
    out.append(_render_evidence(load_bearing))
    out.append("")
    out.append("---")
    out.append("")
    out.append(
        "*v2 grader is the structural response to @m13v's critique on issue #396 — presence is not weight. See `findings/research/2026-05-23-audit-grader-v2/SPEC.md` for design rationale + v1 retirement plan.*"
    )

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
