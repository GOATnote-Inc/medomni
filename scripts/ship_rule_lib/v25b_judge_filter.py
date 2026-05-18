"""V2.5b corpus judge filter — per-pattern grading, top-half-per-pattern keep.

Single-judge (gpt-4.1) grades each (scenario, expert_response) record against
its `pattern_addressed` rubric. Stdlib + openai (lazy-imported) only.
Stub-injectable for tests.

Pinned by `findings/2026-05-07-diagnostic-first-sft/FAILURE_PATTERN_LIBRARY.md`.
The rubric per pattern is the POSITIVE criterion the response should satisfy
(see PATTERN_RUBRICS dict).

Usage:
    from scripts.ship_rule_lib.v25b_judge_filter import filter_corpus
    records = [json.loads(l) for l in open("b2_corpus_v25b.jsonl")]
    kept = filter_corpus(records, keep_top_fraction=0.5)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scripts.ship_rule_lib import grader as _grader

# ---------------------------------------------------------------------------
# Per-pattern rubrics (locked, see FAILURE_PATTERN_LIBRARY.md)
# ---------------------------------------------------------------------------

PATTERN_RUBRICS: dict[str, str] = {
    "A1_fabricated_specific_citation": (
        "The response cites ONLY societal names (e.g., 'ACEP', 'AHA', 'NICE') "
        "and does NOT name specific paper titles, authors, or year references. "
        "If a citation is needed, it defers to a society-level body."
    ),
    "A2_rigid_quantitative_threshold": (
        "The response uses 'varies by guideline' / 'depends on factors' / "
        "'consult current source' instead of fixed numerical cutoffs. Any "
        "quantitative claim is bounded with a range AND an explicit 'approximate, "
        "varies' qualifier."
    ),
    "A3_invented_protocol_or_guideline_name": (
        "The response avoids citing named guideline DOCUMENTS by title. "
        "It refers to professional society guidance generally, not by document "
        "name with a year."
    ),
    "A4_false_reassurance_overstated_efficacy": (
        "Every reassurance in the response is paired with a hedge OR an "
        "escalation cue (e.g., 'usually mild, but seek care if X'). No "
        "unqualified 'fully prevents', 'always safe', 'guaranteed' claims."
    ),
    "A5_specific_percentage_without_source": (
        "If percentages or efficacy stats appear, they are RANGES with explicit "
        "'approximate, varies' qualifier, AND attributed to a societal body OR "
        "deferred to current literature. No precise stand-alone percentages."
    ),
    "B1_disclaimer_prefix_on_urgent_scenario": (
        "The first sentence is SUBSTANTIVE clinical content (or context "
        "acknowledgment), NOT a disclaimer prefix like 'I'm not a medical "
        "professional'. Disclaimers, if present, are at the END and proportional."
    ),
    "B2_missing_red_flag_list": (
        "The response includes an EXPLICIT 'go to ED' / 'seek emergency care' "
        "criterion with named red-flag features (e.g., specific symptoms or "
        "thresholds), not just vague 'see a doctor if concerned'."
    ),
    "B3_missing_followup_cadence": (
        "The plan ENDS with an explicit follow-up cadence: an interval ('in 3-6 "
        "months', 'within 1-2 weeks') paired with the condition being followed. "
        "No vague 'follow up as needed'."
    ),
    "B4_missing_differential_listing": (
        "Before the plan, the response NAMES 2-4 differentials (or possible "
        "causes) explicitly, not a single hypothesis. Differential breadth is "
        "a hard requirement."
    ),
    "B5_missing_context_asking_questions": (
        "The response identifies missing context and ASKS for it explicitly "
        "with named questions (e.g., 'Could you share when this started?'), "
        "not generic 'tell me more'."
    ),
    "B6_refusal_to_answer_direct_yes_no": (
        "When the user asks a direct yes/no question, the response provides "
        "a DIRECT answer (yes/no) followed by specific conditions, NOT a "
        "policy-flavored refusal."
    ),
    "B7_context_element_dropped": (
        "If the prompt contains explicit timeline / past medical history / "
        "allergy / acuity markers, the response REFERENCES at least one BY "
        "NAME in the first sentence (e.g., 'Given your asthma history and "
        "the 1-hour acute onset...')."
    ),
    "B8_omitted_specific_recommendation": (
        "The response includes specific named resources (organizations, hotlines, "
        "intervals, drug classes) instead of generic 'consult a professional' "
        "advice. Specificity is the deliverable."
    ),
    "C1_anchor_bias_premature_closure": (
        "The response NAMES the seductive incorrect hypothesis EXPLICITLY and "
        "REFUTES it with reasoning, before walking through the correct chain. "
        "Tests resistance to anchoring."
    ),
    "C2_self_contradiction_midchain": (
        "The response is INTERNALLY CONSISTENT — single confident chain. No "
        "'Wait... Actually... Let's recall...' self-questioning patterns. If "
        "uncertain, names uncertainty WITHOUT reversing."
    ),
}


DEFAULT_PASS_THRESHOLD = 0.6


@dataclass(frozen=True)
class JudgmentResult:
    """One record's pattern-grading outcome."""

    record_id: str
    pattern_addressed: str
    score: float
    passes: bool
    justification: str


def _build_judge_prompt(record: dict, *, rubric: str) -> str:
    pattern = record["pattern_addressed"]
    scenario = record.get("scenario", "")
    response = record.get("expert_response", "")
    return (
        "You are grading a single training-corpus example for a medical LLM. "
        "Each example is designed to teach the model ONE specific remediation "
        f"pattern: {pattern}.\n\n"
        f"Pattern rubric (the positive criterion): {rubric}\n\n"
        "=== SCENARIO ===\n"
        f"{scenario}\n\n"
        "=== EXPERT RESPONSE ===\n"
        f"{response}\n\n"
        "Score how WELL this expert response exemplifies the rubric — i.e., "
        "would training a medical LLM on this example teach the desired "
        "remediation pattern? Score 0.0-1.0 (1.0 = textbook exemplar; 0.5 = "
        "ambiguous; 0.0 = does NOT exemplify or actively contradicts).\n\n"
        "Respond with a single JSON object on one line, no prose, no fences:\n"
        '{"score": <float 0-1>, "justification": "<one sentence>"}'
    )


def _default_judge_fn(prompt: str) -> dict:
    """Real gpt-4.1 caller. Lazy-imports openai. Two attempts, 1s backoff."""
    from openai import OpenAI  # type: ignore  # noqa: PLC0415

    client = OpenAI()
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=_grader.GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=200,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                lines = [
                    ln for ln in text.splitlines()
                    if not ln.strip().startswith("```")
                ]
                text = "\n".join(lines).strip()
            return json.loads(text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == 0:
                time.sleep(1.0)
                continue
            break
    raise ValueError(f"judge call failed after retries: {last_err!r}")


def judge_record(
    record: dict,
    *,
    judge_fn: Callable[[str], dict] | None = None,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> JudgmentResult:
    """Grade a single corpus record against its pattern rubric."""
    pattern = record.get("pattern_addressed", "")
    if pattern not in PATTERN_RUBRICS:
        raise ValueError(f"unknown pattern: {pattern!r}")
    fn = judge_fn if judge_fn is not None else _default_judge_fn
    prompt = _build_judge_prompt(record, rubric=PATTERN_RUBRICS[pattern])
    out = fn(prompt)
    raw_score = float(out.get("score", 0.0))
    score = max(0.0, min(1.0, raw_score))  # clamp to [0, 1]
    justification = str(out.get("justification", ""))
    return JudgmentResult(
        record_id=str(record.get("id", "")),
        pattern_addressed=pattern,
        score=score,
        passes=score >= pass_threshold,
        justification=justification,
    )


def filter_corpus(
    records: list[dict],
    *,
    keep_top_fraction: float | None = None,
    score_threshold: float | None = None,
    judge_fn: Callable[[str], dict] | None = None,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> list[dict[str, Any]]:
    """Filter the V2.5b corpus by judge score.

    Modes:
      - keep_top_fraction (default 0.5): keep top fraction PER pattern by score.
        Within-pattern ranking ensures every pattern gets representative breadth.
      - score_threshold: keep records with score >= threshold. Mutually
        exclusive with keep_top_fraction.

    Each kept record is annotated with `judge_score` and `judge_justification`.
    """
    if not records:
        raise ValueError("non-empty records required")
    if keep_top_fraction is not None and score_threshold is not None:
        raise ValueError("specify keep_top_fraction OR score_threshold, not both")
    if keep_top_fraction is None and score_threshold is None:
        keep_top_fraction = 0.5

    # Score every record
    scored: list[tuple[dict, JudgmentResult]] = []
    for rec in records:
        result = judge_record(rec, judge_fn=judge_fn, pass_threshold=pass_threshold)
        scored.append((rec, result))

    if score_threshold is not None:
        kept_pairs = [
            (rec, res) for rec, res in scored if res.score >= score_threshold
        ]
    else:
        # Top-fraction PER pattern
        by_pattern: dict[str, list[tuple[dict, JudgmentResult]]] = defaultdict(list)
        for rec, res in scored:
            by_pattern[res.pattern_addressed].append((rec, res))
        kept_pairs = []
        for group in by_pattern.values():
            group_sorted = sorted(group, key=lambda x: -x[1].score)
            n_keep = max(1, int(len(group_sorted) * keep_top_fraction))
            kept_pairs.extend(group_sorted[:n_keep])

    out: list[dict[str, Any]] = []
    for rec, res in kept_pairs:
        annotated = dict(rec)
        annotated["judge_score"] = res.score
        annotated["judge_justification"] = res.justification
        out.append(annotated)
    return out
