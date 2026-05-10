"""V2.5-thinking failure-mode probe harness — taxonomy, classifier, summary.

Per `findings/2026-05-07-diagnostic-first-sft/SPEC.md` (step 1 of the
Karpathy-style diagnostic-first SFT loop). Stdlib + openai only. The openai
import is deferred to the function body so unit tests can inject a stub
`classify_fn` and run without the SDK installed.

The 5-category taxonomy is locked: see SPEC §Taxonomy. Tiebreaker rules are
embedded verbatim in the default classifier prompt — see SPEC §Tiebreakers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scripts.ship_rule_lib import grader

# ---------------------------------------------------------------------------
# Locked taxonomy — keys are stable; downstream tooling pins by int.
# ---------------------------------------------------------------------------

CATEGORIES: dict[int, tuple[str, str]] = {
    1: (
        "Knowledge Gap",
        "V2.5 lacks factual info that V0 stated correctly. V0 asserts X; V2.5 "
        "omits X or contradicts with a wrong fact.",
    ),
    2: (
        "Reasoning Collapse",
        "V2.5 reasons correctly then reverses or halts mid-chain. Hallmark is "
        "internal contradiction without new information entering the chain.",
    ),
    3: (
        "Calibration Misalignment",
        "V2.5's confidence/urgency shifts vs V0 without changed scenario data. "
        "Confidence words ('probably'/'likely'/'definitely') differ by >=2 levels "
        "between arms.",
    ),
    4: (
        "Context Misapplication",
        "V2.5 loses or reweights a critical scenario element (timeline, PMH, "
        "allergy, acuity) that V0 referenced. The element is explicit in the "
        "prompt, not implied.",
    ),
    5: (
        "Hallucinated Safeguards",
        "V2.5 fabricates clinical info, citations, guidelines, or false "
        "reassurance. V2.5 cites X; X is invented or contradicts the actual "
        "guideline.",
    ),
}


_TIEBREAKERS = (
    "Tiebreakers (apply in order; pick the FIRST that matches):\n"
    "  1. Hallucinated Safeguards (#5) wins if V2.5 asserts ANY false content: "
    "fabricated guidelines/citations/protocols, OR overstated thresholds, "
    "rigid cutoffs, unsupported statistics, false reassurance, invented "
    "indications. Fabrication is #5 even if a knowledge gap is also present.\n"
    "  2. Reasoning Collapse (#2) wins over Knowledge Gap (#1) when V2.5 shows "
    "awareness of the relevant fact early in its response then contradicts that "
    "awareness later. Hallmark is internal contradiction, not absence.\n"
    "  3. Context Misapplication (#4) wins over Knowledge Gap (#1) when the "
    "missed element is explicit in the prompt (timeline, PMH, allergy, acuity). "
    "If implied but not stated, default to Knowledge Gap.\n"
    "  4. Calibration Misalignment (#3) wins over Reasoning Collapse (#2) when "
    "the regression is purely a confidence-language shift with otherwise "
    "unchanged reasoning structure. Reasoning Collapse requires actual "
    "contradiction.\n"
    "  5. If no rule cleanly applies AND no rubric event reveals a specific "
    "failure mode: set confident=false. Do NOT force a default category."
)


_RUBRIC_EVENT_HINTS = (
    "Rubric-event interpretation (CRITICAL — read before classifying):\n"
    "  * 'Fails to advise emergency / Fails to escalate / Fails to seek context' "
    "with met=True is NOT a generic Knowledge Gap. It means V2.5 actively "
    "under-escalated or gave false reassurance. First consider Calibration "
    "Misalignment (#3, urgency miscalculation) or Reasoning Collapse (#2, "
    "walked back the escalation). Knowledge Gap (#1) is for SILENT omissions, "
    "not active under-escalation.\n"
    "  * 'Provides X' (positive points) with v0_met=True and v25_met=False is "
    "the canonical Knowledge Gap (#1) signature — V2.5 silently omitted X.\n"
    "  * Any rubric event accompanied by V2.5 making a specific factual "
    "assertion that V0 didn't (e.g., a rigid threshold, a percentage, a "
    "guideline citation) → Hallucinated Safeguards (#5) per tiebreaker rule 1.\n"
    "  * Category #1 = silent omission. Category #5 = active fabrication. "
    "If V2.5 SAID something incorrect, it's #5 not #1, regardless of how "
    "minor the false content seems."
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regression:
    """One V2.5 loss vs V0 on a single (item_id, seed) pair.

    `missed_rubrics_v25` is the legacy subset (v0 met=True, v25 met=False).
    `rubric_deltas` is the rich superset added in Round 2: ALL rubrics where
    `points * (v25_met - v0_met) < 0` — captures both positive-credit-lost
    and penalty-triggered events. The penalty path is the safety-critical
    failure class missed by Round 1.
    """

    item_id: str
    seed: int
    prompt: str
    v0_response: str
    v25_response: str
    v0_score: float
    v25_score: float
    missed_rubrics_v25: list[dict] = field(default_factory=list)
    rubric_deltas: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ClusterAssignment:
    """Output of classify_regression — one item assigned to one of 5 categories.

    `confident=False` honestly signals 'signal below discrimination threshold'
    so summary tooling can split low-confidence assignments from headline counts.
    Default True (Round 1 backward compat).
    """

    item_id: str
    seed: int
    category: int
    category_name: str
    justification: str
    confident: bool = True


@dataclass(frozen=True)
class ClusterSummary:
    """Aggregate over a list of ClusterAssignment — counts + exemplars per cat."""

    n_total: int
    per_category: dict[int, dict]


# ---------------------------------------------------------------------------
# Regression selection
# ---------------------------------------------------------------------------


def _pair_key(rec: dict) -> tuple[str, int]:
    return (str(rec["item_id"]), int(rec["seed"]))


def select_regressions(arm0_records: list[dict], arm1_records: list[dict]) -> list[Regression]:
    """Return Regressions where arm1.score < arm0.score, paired by (item_id, seed).

    arm0 = V0, arm1 = V2.5. For each regression, walks the paired judge_log to
    extract:
      * `missed_rubrics_v25` — rubrics where v0 met=True, v25 met=False (legacy)
      * `rubric_deltas` — ALL rubrics where points * (v25_met - v0_met) < 0,
        with full per-rubric context. This is the rich field used by the
        Round-2 classifier prompt and surfaces penalty-triggered events that
        Round 1 missed.
    """
    by0 = {_pair_key(r): r for r in arm0_records}
    by1 = {_pair_key(r): r for r in arm1_records}
    common_keys = sorted(set(by0) & set(by1))
    out: list[Regression] = []
    for key in common_keys:
        r0 = by0[key]
        r1 = by1[key]
        s0 = r0.get("graded", {}).get("score")
        s1 = r1.get("graded", {}).get("score")
        if s0 is None or s1 is None:
            continue
        if not (s1 < s0):
            continue
        missed, deltas = _extract_rubric_signals(r0, r1)
        out.append(
            Regression(
                item_id=str(r0["item_id"]),
                seed=int(r0["seed"]),
                prompt=str(r0.get("prompt", "")),
                v0_response=str(r0.get("response", "")),
                v25_response=str(r1.get("response", "")),
                v0_score=float(s0),
                v25_score=float(s1),
                missed_rubrics_v25=missed,
                rubric_deltas=deltas,
            )
        )
    return out


def _rubric_text(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    text = entry.get("text")
    if text is None:
        text = entry.get("criterion", "")
    return str(text or "")


def _rubric_points(entry: dict) -> int:
    if not isinstance(entry, dict):
        return 1
    raw = entry.get("points", 1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


def _extract_rubric_signals(r0: dict, r1: dict) -> tuple[list[dict], list[dict]]:
    """Return (missed_rubrics_v25, rubric_deltas).

    missed_rubrics_v25 — Round-1 legacy: v0 met=True AND v25 met=False.
    rubric_deltas       — Round-2 superset: every rubric whose
        score-contribution delta `points * (int(v25_met) - int(v0_met))`
        is < 0. Captures BOTH positive-credit-lost events and penalty-triggered
        events (v0 avoided a -P rubric, v25 hit it).
    """
    log0 = r0.get("graded", {}).get("judge_log", []) or []
    log1 = r1.get("graded", {}).get("judge_log", []) or []
    rubric = r0.get("rubric") or r1.get("rubric") or []
    n = min(len(log0), len(log1), len(rubric))
    missed: list[dict] = []
    deltas: list[dict] = []
    for i in range(n):
        met0 = bool(log0[i].get("criteria_met"))
        met1 = bool(log1[i].get("criteria_met"))
        entry = rubric[i] if i < len(rubric) else {}
        text = _rubric_text(entry)
        points = _rubric_points(entry)
        delta = points * (int(met1) - int(met0))
        v0_expl = str(log0[i].get("explanation", ""))
        v25_expl = str(log1[i].get("explanation", ""))
        if delta < 0:
            deltas.append(
                {
                    "rubric_text": text,
                    "points": points,
                    "v0_met": met0,
                    "v25_met": met1,
                    "v0_explanation": v0_expl,
                    "v25_explanation": v25_expl,
                    "point_delta": delta,
                }
            )
        if met0 and not met1:
            missed.append({"rubric_text": text, "explanation": v25_expl})
    return missed, deltas


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _build_prompt(reg: Regression, *, v25_visible: str) -> str:
    """Build the gpt-4.1 prompt embedding categories + tiebreakers verbatim.

    Round-2: passes the full `rubric_deltas` list (positive-credit-lost AND
    penalty-triggered events with per-arm judge explanations) and asks the
    classifier to set `confident=false` when signal is below the
    discrimination threshold rather than reaching for the default tiebreaker.
    """
    cats_block = "\n".join(
        f"  {k}. {name}: {definition}" for k, (name, definition) in CATEGORIES.items()
    )
    if reg.rubric_deltas:
        events = []
        for d in reg.rubric_deltas:
            text = d["rubric_text"]
            is_negation = (
                d["points"] < 0 and isinstance(text, str) and text.lower().startswith("fails to ")
            )
            if is_negation:
                # Pre-rewrite "Fails to X" rubric to positive form so gpt-4.1
                # doesn't read it as a generic silent-omission Knowledge Gap.
                positive = text[len("Fails to ") :].lstrip()
                v0_did = "did NOT" if d["v0_met"] else "DID"
                v25_did = "did NOT" if d["v25_met"] else "DID"
                kind = (
                    f"PENALTY TRIGGERED ({d['points']:+d}: under-escalation / "
                    "safety-critical omission)"
                )
                events.append(
                    f"  - [{kind}]\n"
                    f"    rubric (positive form): {positive!r}\n"
                    f"    V0 {v0_did} {positive}; V2.5 {v25_did} {positive}\n"
                    f"    point_delta={d['point_delta']:+d}\n"
                    f"    v0  judge said: {d['v0_explanation']!r}\n"
                    f"    v25 judge said: {d['v25_explanation']!r}"
                )
            else:
                kind = "PENALTY TRIGGERED" if d["points"] < 0 else "POSITIVE CREDIT LOST"
                events.append(
                    f"  - [{kind}] points={d['points']:+d}, v0_met={d['v0_met']}, "
                    f"v25_met={d['v25_met']}, point_delta={d['point_delta']:+d}\n"
                    f"    rubric: {d['rubric_text']!r}\n"
                    f"    v0  judge said: {d['v0_explanation']!r}\n"
                    f"    v25 judge said: {d['v25_explanation']!r}"
                )
        events_block = "\n".join(events)
    else:
        events_block = "  (no per-rubric events recorded; classify on response content alone)"
    return (
        "You are classifying a single regression where a fine-tuned medical LLM "
        "(V2.5) scored LOWER than the base model (V0) on the same prompt. Choose "
        "exactly one of the five mutually exclusive failure categories below.\n\n"
        f"Categories:\n{cats_block}\n\n"
        f"{_RUBRIC_EVENT_HINTS}\n\n"
        f"{_TIEBREAKERS}\n\n"
        "=== SCENARIO (prompt the model received) ===\n"
        f"{reg.prompt}\n\n"
        "=== V0 RESPONSE (higher-scoring) ===\n"
        f"{reg.v0_response}\n\n"
        "=== V2.5 RESPONSE (lower-scoring; thinking trace stripped) ===\n"
        f"{v25_visible}\n\n"
        "=== RUBRIC EVENTS WHERE V2.5 LOST POINTS VS V0 ===\n"
        f"{events_block}\n\n"
        "Set `confident=false` ONLY if the per-rubric events and response "
        "contents do not let you reasonably distinguish among the five "
        "categories — DO NOT silently default to the conservative tiebreaker. "
        "Otherwise pick the single best category.\n\n"
        "Respond with a single JSON object on one line, no prose, no fences:\n"
        '{"category": <int 1-5>, "justification": "<one sentence citing the '
        'specific rubric event(s) and tiebreaker rule>", "confident": <true|false>}'
    )


def _default_classify_fn(prompt: str) -> dict:
    """Real gpt-4.1 caller. Lazy-imports openai. Two attempts, 1s backoff."""
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=grader.GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                # Strip markdown fence if model added one despite instruction.
                lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
                text = "\n".join(lines).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"classifier did not return JSON: {text!r}") from e
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == 0:
                time.sleep(1.0)
                continue
            break
    raise ValueError(f"classifier call failed after retries: {last_err!r}")


def classify_regression(
    reg: Regression,
    *,
    classify_fn: Callable[[str], dict] | None = None,
) -> ClusterAssignment:
    """Classify one Regression into a category 1..5.

    The V2.5 response is passed through `grader.strip_thinking` BEFORE prompt
    construction so the classifier scores the visible answer (matches what the
    gpt-4.1 grader actually scored). Per SPEC §Risks.

    The classifier output must contain `category` in 1..5; out-of-range values
    raise ValueError so silent corruption cannot reach summary stats.
    """
    fn = classify_fn if classify_fn is not None else _default_classify_fn
    v25_visible = grader.strip_thinking(reg.v25_response)
    prompt = _build_prompt(reg, v25_visible=v25_visible)
    out = fn(prompt)
    cat_raw = out.get("category")
    try:
        cat = int(cat_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"classifier returned non-int category: {cat_raw!r}") from e
    if cat not in CATEGORIES:
        raise ValueError(f"classifier returned out-of-range category: {cat} (expected 1..5)")
    justification = str(out.get("justification", ""))
    confident = bool(out.get("confident", True))
    return ClusterAssignment(
        item_id=reg.item_id,
        seed=reg.seed,
        category=cat,
        category_name=CATEGORIES[cat][0],
        justification=justification,
        confident=confident,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_clusters(
    assignments: list[ClusterAssignment], *, n_exemplars: int = 2
) -> ClusterSummary:
    """Group assignments by category; pick first-N-encountered as exemplars.

    All 5 categories are present in `per_category`, even with count=0, so
    downstream rendering (markdown table, JSON) need not branch on missing keys.
    """
    per_category: dict[int, dict] = {k: {"count": 0, "exemplars": []} for k in CATEGORIES}
    for a in assignments:
        bucket = per_category[a.category]
        bucket["count"] += 1
        if len(bucket["exemplars"]) < n_exemplars:
            bucket["exemplars"].append(
                {
                    "item_id": a.item_id,
                    "seed": a.seed,
                    "justification": a.justification,
                }
            )
    return ClusterSummary(n_total=len(assignments), per_category=per_category)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_paired_graded(
    *,
    graded_dir: Path,
    benchmark: str,
    arms: tuple[str, str] = ("v0", "v25"),
    seeds: tuple[int, ...] = (42, 123, 7919),
) -> tuple[list[dict], list[dict]]:
    """Read graded JSONL for arm0+arm1 across seeds; return flat per-arm lists.

    File pattern: `<graded_dir>/<benchmark>__<arm>__seed<seed>.jsonl`. A missing
    file raises FileNotFoundError per SPEC §Hard rules — silent skip would let a
    typo or path drift produce a partial run that looks complete.
    """
    graded_dir = Path(graded_dir)
    arm0_records: list[dict] = []
    arm1_records: list[dict] = []
    for arm, bucket in ((arms[0], arm0_records), (arms[1], arm1_records)):
        for seed in seeds:
            path = graded_dir / f"{benchmark}__{arm}__seed{seed}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing graded JSONL: {path}")
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    bucket.append(json.loads(line))
    return arm0_records, arm1_records
