"""TDD red phase for V2.5b corpus judge filter.

Pinned by `findings/2026-05-07-diagnostic-first-sft/FAILURE_PATTERN_LIBRARY.md`
+ `V2.5B-CORPUS-SPEC.md` §Judge filter. Stdlib only — openai lazy-imported.
Single-judge (gpt-4.1) per-pattern grading: does the (scenario, expert_response)
example exemplify its `pattern_addressed` remediation?

Per-pattern rubrics live in `_PATTERN_RUBRICS` and pin the positive criterion
per pattern. Judge returns score 0-1 + pass bool + justification. Filter keeps
top-half by score within each pattern (default), OR threshold-based.
"""

from __future__ import annotations

import pytest

from scripts.ship_rule_lib import v25b_judge_filter as jf


# ---------------------------------------------------------------------------
# Per-pattern rubrics
# ---------------------------------------------------------------------------


def test_pattern_rubrics_cover_all_15_patterns() -> None:
    """All A1-A5, B1-B8, C1-C2 must have a rubric prompt."""
    expected = {
        "A1_fabricated_specific_citation",
        "A2_rigid_quantitative_threshold",
        "A3_invented_protocol_or_guideline_name",
        "A4_false_reassurance_overstated_efficacy",
        "A5_specific_percentage_without_source",
        "B1_disclaimer_prefix_on_urgent_scenario",
        "B2_missing_red_flag_list",
        "B3_missing_followup_cadence",
        "B4_missing_differential_listing",
        "B5_missing_context_asking_questions",
        "B6_refusal_to_answer_direct_yes_no",
        "B7_context_element_dropped",
        "B8_omitted_specific_recommendation",
        "C1_anchor_bias_premature_closure",
        "C2_self_contradiction_midchain",
    }
    assert set(jf.PATTERN_RUBRICS.keys()) == expected
    for pattern, rubric in jf.PATTERN_RUBRICS.items():
        assert isinstance(rubric, str)
        assert len(rubric) > 30, f"rubric for {pattern} too short"


# ---------------------------------------------------------------------------
# Single-record judging
# ---------------------------------------------------------------------------


def _mk_record(pattern: str = "A1_fabricated_specific_citation") -> dict:
    return {
        "id": "v25b_secA_00000",
        "section": "A",
        "section_name": "Active fabrication / over-specification",
        "pattern_addressed": pattern,
        "scenario": "What's the exact dose for X?",
        "expert_response": "Dosing varies by guideline; consult ACEP for current recommendations.",
    }


def test_judge_record_uses_injected_stub() -> None:
    captured: list[str] = []

    def stub(prompt: str) -> dict:
        captured.append(prompt)
        return {"score": 0.8, "justification": "exemplifies anti-fabrication"}

    record = _mk_record()
    result = jf.judge_record(record, judge_fn=stub)
    assert result.record_id == "v25b_secA_00000"
    assert result.pattern_addressed == "A1_fabricated_specific_citation"
    assert result.score == pytest.approx(0.8)
    assert result.passes is True  # default threshold 0.6
    assert "exemplifies" in result.justification
    assert len(captured) == 1
    # Rubric for the pattern must be embedded in the prompt
    assert "ACEP" in captured[0] or "societal" in captured[0]


def test_judge_record_clamps_score_to_0_1() -> None:
    def stub_high(prompt: str) -> dict:
        return {"score": 1.5, "justification": "j"}

    def stub_neg(prompt: str) -> dict:
        return {"score": -0.3, "justification": "j"}

    rec = _mk_record()
    assert jf.judge_record(rec, judge_fn=stub_high).score == pytest.approx(1.0)
    assert jf.judge_record(rec, judge_fn=stub_neg).score == pytest.approx(0.0)


def test_judge_record_passes_below_threshold_returns_false() -> None:
    def stub_low(prompt: str) -> dict:
        return {"score": 0.3, "justification": "weak"}

    rec = _mk_record()
    res = jf.judge_record(rec, judge_fn=stub_low)
    assert res.passes is False


def test_judge_record_rejects_unknown_pattern() -> None:
    rec = _mk_record(pattern="X9_unknown_pattern")
    def stub(prompt: str) -> dict:
        return {"score": 1.0, "justification": "j"}
    with pytest.raises(ValueError, match="pattern"):
        jf.judge_record(rec, judge_fn=stub)


# ---------------------------------------------------------------------------
# Filter — top-half-per-pattern
# ---------------------------------------------------------------------------


def test_filter_corpus_top_half_per_pattern() -> None:
    """4 records / pattern → keep top 2 by score (50%)."""
    def stub(prompt: str) -> dict:
        # Encode score in the prompt via a marker (test fixture quirk)
        if "high" in prompt:
            return {"score": 0.9, "justification": "good"}
        if "low" in prompt:
            return {"score": 0.4, "justification": "weak"}
        return {"score": 0.7, "justification": "ok"}

    records = []
    for i, marker in enumerate(["high", "high", "low", "low"]):
        records.append({
            "id": f"v25b_secA_{i:05d}",
            "section": "A",
            "section_name": "x",
            "pattern_addressed": "A1_fabricated_specific_citation",
            "scenario": f"{marker} scenario",
            "expert_response": "r",
        })
    kept = jf.filter_corpus(records, keep_top_fraction=0.5, judge_fn=stub)
    assert len(kept) == 2
    # Both kept records have score 0.9 (the "high" ones)
    assert all(r["judge_score"] == pytest.approx(0.9) for r in kept)


def test_filter_corpus_threshold_mode() -> None:
    def stub(prompt: str) -> dict:
        if "above" in prompt:
            return {"score": 0.7, "justification": "good"}
        return {"score": 0.5, "justification": "below"}

    records = [
        {
            "id": f"v25b_secA_{i:05d}",
            "section": "A",
            "section_name": "x",
            "pattern_addressed": "A1_fabricated_specific_citation",
            "scenario": s,
            "expert_response": "r",
        }
        for i, s in enumerate(["above", "above", "below", "below"])
    ]
    kept = jf.filter_corpus(records, score_threshold=0.6, judge_fn=stub)
    assert len(kept) == 2
    assert all(r["judge_score"] >= 0.6 for r in kept)


def test_filter_corpus_balances_across_patterns() -> None:
    """Mixed-pattern corpus — top-half applied PER pattern, not globally."""
    def stub(prompt: str) -> dict:
        if "A1_high" in prompt:
            return {"score": 0.9, "justification": "j"}
        if "A1_low" in prompt:
            return {"score": 0.3, "justification": "j"}
        if "B1_high" in prompt:
            return {"score": 0.85, "justification": "j"}
        return {"score": 0.4, "justification": "j"}  # B1_low

    records = []
    for pattern, marker in [("A1_fabricated_specific_citation", "A1_high"),
                            ("A1_fabricated_specific_citation", "A1_high"),
                            ("A1_fabricated_specific_citation", "A1_low"),
                            ("A1_fabricated_specific_citation", "A1_low"),
                            ("B1_disclaimer_prefix_on_urgent_scenario", "B1_high"),
                            ("B1_disclaimer_prefix_on_urgent_scenario", "B1_high"),
                            ("B1_disclaimer_prefix_on_urgent_scenario", "B1_low"),
                            ("B1_disclaimer_prefix_on_urgent_scenario", "B1_low")]:
        records.append({
            "id": f"v25b_{marker}",
            "section": pattern[0],
            "section_name": "x",
            "pattern_addressed": pattern,
            "scenario": marker,
            "expert_response": "r",
        })
    kept = jf.filter_corpus(records, keep_top_fraction=0.5, judge_fn=stub)
    # 4 kept total: 2 from A1, 2 from B1 (top-half PER pattern)
    assert len(kept) == 4
    by_pattern: dict[str, list] = {}
    for r in kept:
        by_pattern.setdefault(r["pattern_addressed"], []).append(r)
    assert len(by_pattern["A1_fabricated_specific_citation"]) == 2
    assert len(by_pattern["B1_disclaimer_prefix_on_urgent_scenario"]) == 2


def test_filter_corpus_writes_judge_metadata() -> None:
    """Each kept record gets judge_score + judge_justification fields."""
    def stub(prompt: str) -> dict:
        return {"score": 0.8, "justification": "good fit"}

    records = [_mk_record() for _ in range(2)]
    kept = jf.filter_corpus(records, keep_top_fraction=1.0, judge_fn=stub)
    for r in kept:
        assert "judge_score" in r and r["judge_score"] == pytest.approx(0.8)
        assert "judge_justification" in r
        assert r["judge_justification"] == "good fit"


def test_filter_corpus_rejects_zero_records() -> None:
    def stub(prompt: str) -> dict:
        return {"score": 1.0, "justification": "j"}
    with pytest.raises(ValueError):
        jf.filter_corpus([], keep_top_fraction=0.5, judge_fn=stub)
