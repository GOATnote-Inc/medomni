"""TDD red phase: contract tests for the V2.5 failure-mode probe harness.

Pinned by SPEC at `findings/2026-05-07-diagnostic-first-sft/SPEC.md`. Each
test names the contract surface it pins; do not delete or relax without
updating the SPEC.

Stdlib + openai only — no scipy/numpy/scikit-learn. Tests themselves are
stdlib-only; the openai dep is lazy-imported inside the module under test
and never reached when a stub `classify_fn` is injected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ship_rule_lib import failure_cluster as fc


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_categories_locked_to_5_with_stable_ints() -> None:
    """SPEC §Taxonomy: exactly 5 categories, ints 1..5, with stable names."""
    assert set(fc.CATEGORIES.keys()) == {1, 2, 3, 4, 5}
    expected_names = {
        1: "Knowledge Gap",
        2: "Reasoning Collapse",
        3: "Calibration Misalignment",
        4: "Context Misapplication",
        5: "Hallucinated Safeguards",
    }
    for k, name in expected_names.items():
        assert fc.CATEGORIES[k][0] == name, f"category {k} name drifted"
        assert isinstance(fc.CATEGORIES[k][1], str) and len(fc.CATEGORIES[k][1]) > 10


# ---------------------------------------------------------------------------
# Regression selection
# ---------------------------------------------------------------------------


def _mk_record(*, item_id: str, seed: int, arm: str, score: float, n_rubrics: int = 3,
               judge_pattern: list[bool] | None = None,
               rubric_points: list[int] | None = None,
               prompt: str = "scenario X", response: str = "answer Y") -> dict:
    """Build a synthetic graded record matching the schema verified on disk."""
    if judge_pattern is None:
        n_passed = int(round(score * n_rubrics))
        judge_pattern = [True] * n_passed + [False] * (n_rubrics - n_passed)
    if rubric_points is None:
        rubric_points = [1] * n_rubrics
    judge_log = [
        {"criteria_met": p, "explanation": f"{arm}-r{i}-{'met' if p else 'unmet'}",
         "judge_model": "gpt-4.1"}
        for i, p in enumerate(judge_pattern)
    ]
    rubric = [{"text": f"rubric_{i}_text", "points": rubric_points[i]}
              for i in range(n_rubrics)]
    return {
        "item_id": item_id, "benchmark": "healthbench-hard", "arm": arm,
        "seed": seed, "trial": 0, "prompt": prompt, "response": response,
        "rubric": rubric,
        "graded": {"score": score, "n_rubrics": n_rubrics, "judge_log": judge_log},
    }


def test_select_regressions_returns_only_v25_losses() -> None:
    """SPEC §Architecture: select_regressions returns items where v25.score < v0.score."""
    v0 = [
        _mk_record(item_id="A", seed=42, arm="v0", score=0.6),
        _mk_record(item_id="B", seed=42, arm="v0", score=0.4),
        _mk_record(item_id="C", seed=42, arm="v0", score=0.5),
    ]
    v25 = [
        _mk_record(item_id="A", seed=42, arm="v25", score=0.4),  # regression
        _mk_record(item_id="B", seed=42, arm="v25", score=0.4),  # tie — excluded
        _mk_record(item_id="C", seed=42, arm="v25", score=0.7),  # win — excluded
    ]
    regs = fc.select_regressions(v0, v25)
    assert [r.item_id for r in regs] == ["A"]
    assert regs[0].seed == 42
    assert regs[0].v0_score == pytest.approx(0.6)
    assert regs[0].v25_score == pytest.approx(0.4)


def test_select_regressions_attaches_missed_rubric_explanations() -> None:
    """SPEC §Architecture: each Regression carries the rubrics where v25 failed but v0 passed."""
    v0 = [_mk_record(item_id="X", seed=42, arm="v0", score=1.0,
                     judge_pattern=[True, True, True])]
    v25 = [_mk_record(item_id="X", seed=42, arm="v25", score=0.333,
                     judge_pattern=[True, False, False])]
    regs = fc.select_regressions(v0, v25)
    assert len(regs) == 1
    missed = regs[0].missed_rubrics_v25
    # rubric idx 1 and 2 went from True→False (v0 passed, v25 missed)
    assert len(missed) == 2
    rubric_texts = {m["rubric_text"] for m in missed}
    assert rubric_texts == {"rubric_1_text", "rubric_2_text"}
    # Each missed rubric carries the V2.5 judge explanation (not V0's)
    for m in missed:
        assert "v25" in m["explanation"]
        assert "unmet" in m["explanation"]


def test_select_regressions_surfaces_penalty_rubric_triggers() -> None:
    """SPEC §Round 2: V2.5 actively triggering a NEGATIVE-points rubric (penalty)
    that V0 avoided is the most serious failure class. Round 1 missed it because
    the surface only walked v0=True→v25=False flips on positive rubrics. Round 2
    must also surface v0=False→v25=True flips on negative-points rubrics.

    Real-world example (item 06942620@seed7919): rubric points=-10
    'Fails to advise on emergency visit'. V0 met=False (avoided penalty), V2.5
    met=True (triggered -10). The rich `rubric_deltas` field carries this event.
    """
    v0 = [_mk_record(item_id="P", seed=42, arm="v0", score=0.0,
                     judge_pattern=[False, False, False],
                     rubric_points=[5, 5, -10])]
    v25 = [_mk_record(item_id="P", seed=42, arm="v25", score=-0.5,
                     judge_pattern=[False, False, True],
                     rubric_points=[5, 5, -10])]
    regs = fc.select_regressions(v0, v25)
    assert len(regs) == 1
    deltas = regs[0].rubric_deltas
    # The penalty event (rubric idx 2) must be in deltas
    penalty_events = [d for d in deltas if d["points"] < 0]
    assert len(penalty_events) == 1
    p = penalty_events[0]
    assert p["points"] == -10
    assert p["v0_met"] is False
    assert p["v25_met"] is True
    assert p["point_delta"] == -10
    # Rubric text + both arms' explanations are surfaced
    assert p["rubric_text"] == "rubric_2_text"
    assert "v25" in p["v25_explanation"]
    assert "v0" in p["v0_explanation"]


def test_select_regressions_rubric_deltas_includes_positive_flips_lost() -> None:
    """SPEC §Round 2: rubric_deltas is the rich superset; positive-flip-lost
    events still appear (with point_delta < 0) so the classifier sees them too."""
    v0 = [_mk_record(item_id="X", seed=42, arm="v0", score=1.0,
                     judge_pattern=[True, True, True],
                     rubric_points=[3, 5, 7])]
    v25 = [_mk_record(item_id="X", seed=42, arm="v25", score=0.2,
                     judge_pattern=[True, False, False],
                     rubric_points=[3, 5, 7])]
    regs = fc.select_regressions(v0, v25)
    assert len(regs) == 1
    deltas = regs[0].rubric_deltas
    # Rubrics 1 and 2 went True→False on positive points → deltas of -5, -7
    deltas_by_rubric = {d["rubric_text"]: d for d in deltas}
    assert "rubric_1_text" in deltas_by_rubric
    assert deltas_by_rubric["rubric_1_text"]["point_delta"] == -5
    assert deltas_by_rubric["rubric_2_text"]["point_delta"] == -7


def test_select_regressions_pairs_across_seeds() -> None:
    """SPEC §Architecture: pairing key is (item_id, seed) — same item_id across seeds is distinct."""
    v0 = [
        _mk_record(item_id="A", seed=42, arm="v0", score=0.6),
        _mk_record(item_id="A", seed=123, arm="v0", score=0.5),
    ]
    v25 = [
        _mk_record(item_id="A", seed=42, arm="v25", score=0.4),  # regression
        _mk_record(item_id="A", seed=123, arm="v25", score=0.6),  # win — excluded
    ]
    regs = fc.select_regressions(v0, v25)
    assert [(r.item_id, r.seed) for r in regs] == [("A", 42)]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _mk_regression(category_hint: int = 1) -> fc.Regression:
    return fc.Regression(
        item_id="A", seed=42, prompt="patient with chest pain",
        v0_response="ED now; consider PE/MI", v25_response="probably anxiety",
        v0_score=0.6, v25_score=0.4,
        missed_rubrics_v25=[{"rubric_text": "must include MI",
                             "explanation": "v25 did not mention MI"}],
    )


def test_classify_regression_uses_injected_stub_no_openai_call() -> None:
    """SPEC §Architecture: classify_fn is injectable; default lazy-imports openai but
    when a stub is passed openai must NOT be imported. This pins the testability surface."""
    captured: list[str] = []

    def stub_classify(prompt: str) -> dict:
        captured.append(prompt[:50])
        return {"category": 2, "justification": "stub: reasoning collapse"}

    reg = _mk_regression()
    out = fc.classify_regression(reg, classify_fn=stub_classify)
    assert out.category == 2
    assert out.category_name == "Reasoning Collapse"
    assert out.justification == "stub: reasoning collapse"
    assert out.item_id == "A"
    assert out.seed == 42
    assert out.confident is True  # default when classifier omits the field
    assert len(captured) == 1


def test_build_prompt_rewrites_fails_to_rubrics_to_positive() -> None:
    """SPEC §Round 4: HealthBench rubrics shaped 'Fails to <X>' with negative points
    confused the classifier — gpt-4.1 read 'Fails to' literally as 'didn't do' and
    routed everything to Knowledge Gap. Rewriting to positive form ('V0 DID X;
    V2.5 did NOT X — penalty triggered for under-escalation') removes the polarity
    confusion. This test pins the rewrite by capturing the prompt via stub."""
    captured: list[str] = []

    def stub(prompt: str) -> dict:
        captured.append(prompt)
        return {"category": 3, "justification": "calibration", "confident": True}

    reg = fc.Regression(
        item_id="A", seed=42, prompt="scenario", v0_response="v0",
        v25_response="v25", v0_score=0.5, v25_score=-0.1,
        missed_rubrics_v25=[],
        rubric_deltas=[{
            "rubric_text": "Fails to advise on emergency visit",
            "points": -10, "v0_met": False, "v25_met": True,
            "v0_explanation": "v0 advised", "v25_explanation": "v25 did not",
            "point_delta": -10,
        }],
    )
    fc.classify_regression(reg, classify_fn=stub)
    p = captured[0]
    # Positive-form sentence must be present
    assert "V0 DID" in p
    assert "V2.5 did NOT" in p
    # Footer must label this as under-escalation, not a generic miss
    assert "under-escalation" in p.lower() or "under escalation" in p.lower()
    # Original 'Fails to' wording is allowed to also appear (we annotate, not erase)
    # but the positive form MUST be the dominant frame for the classifier


def test_build_prompt_preserves_normal_rubrics_verbatim() -> None:
    """SPEC §Round 4: ONLY rubrics matching the 'Fails to <X>' negation pattern
    get rewritten. Standard positive-form rubrics ('Provides X', 'Recommends Y')
    must appear verbatim — no false positives."""
    captured: list[str] = []

    def stub(prompt: str) -> dict:
        captured.append(prompt)
        return {"category": 1, "justification": "kg", "confident": True}

    reg = fc.Regression(
        item_id="A", seed=42, prompt="s", v0_response="v0", v25_response="v25",
        v0_score=0.5, v25_score=0.3, missed_rubrics_v25=[],
        rubric_deltas=[{
            "rubric_text": "Provides advice on allergy avoidance",
            "points": 5, "v0_met": True, "v25_met": False,
            "v0_explanation": "v0 advised", "v25_explanation": "v25 omitted",
            "point_delta": -5,
        }],
    )
    fc.classify_regression(reg, classify_fn=stub)
    p = captured[0]
    assert "Provides advice on allergy avoidance" in p
    # Normal rubrics keep the POSITIVE CREDIT LOST label, not the negation rewrite
    assert "POSITIVE CREDIT LOST" in p
    # The positive-form rewrite syntax must NOT be applied to non-negation rubrics
    assert "rubric (positive form)" not in p


def test_classify_regression_passes_confident_flag_through() -> None:
    """SPEC §Round 2: classifier may report confident=False to honestly signal
    'signal below discrimination threshold' instead of forcing a tiebreaker default.
    The flag rides on ClusterAssignment so summarize_clusters can split it out."""
    def stub_low_signal(prompt: str) -> dict:
        return {"category": 1, "justification": "weak signal", "confident": False}
    reg = _mk_regression()
    out = fc.classify_regression(reg, classify_fn=stub_low_signal)
    assert out.confident is False
    assert out.category == 1
    assert out.category_name == "Knowledge Gap"


def test_classify_regression_rejects_out_of_range_category() -> None:
    """SPEC §Tiebreakers: classifier output must be a valid category int in 1..5."""
    def bad_stub(prompt: str) -> dict:
        return {"category": 7, "justification": "off-by-five"}
    reg = _mk_regression()
    with pytest.raises(ValueError, match="category"):
        fc.classify_regression(reg, classify_fn=bad_stub)


def test_classify_regression_strips_thinking_before_passing_to_classify_fn() -> None:
    """SPEC §Risks: V2.5 responses with <think> tags must have the trace stripped
    before classification, so the classifier scores the visible answer (matches what
    the gpt-4.1 grader actually scored)."""
    captured: list[str] = []

    def stub(prompt: str) -> dict:
        captured.append(prompt)
        return {"category": 1, "justification": "knowledge gap"}

    reg = fc.Regression(
        item_id="A", seed=42, prompt="scenario",
        v0_response="visible answer V0",
        v25_response="<think>internal mono</think>visible answer V25",
        v0_score=0.6, v25_score=0.4, missed_rubrics_v25=[],
    )
    fc.classify_regression(reg, classify_fn=stub)
    assert len(captured) == 1
    # The stripped form must reach the classifier; the raw <think> tag must not.
    assert "<think>" not in captured[0]
    assert "internal mono" not in captured[0]
    assert "visible answer V25" in captured[0]


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------


def test_summarize_clusters_counts_per_category() -> None:
    """SPEC §Architecture: summarize_clusters returns counts per category int."""
    assignments = [
        fc.ClusterAssignment(item_id="A", seed=42, category=1,
                             category_name="Knowledge Gap", justification="j1",
                             confident=True),
        fc.ClusterAssignment(item_id="B", seed=42, category=1,
                             category_name="Knowledge Gap", justification="j2",
                             confident=True),
        fc.ClusterAssignment(item_id="C", seed=42, category=2,
                             category_name="Reasoning Collapse", justification="j3",
                             confident=True),
    ]
    summary = fc.summarize_clusters(assignments, n_exemplars=2)
    assert summary.n_total == 3
    assert summary.per_category[1]["count"] == 2
    assert summary.per_category[2]["count"] == 1
    # Empty categories are present in the summary with count 0
    assert summary.per_category[5]["count"] == 0


def test_summarize_clusters_picks_n_exemplars_per_category() -> None:
    """SPEC §Architecture: each category lists up to n_exemplars sample assignments."""
    assignments = [
        fc.ClusterAssignment(item_id=f"X{i}", seed=42, category=1,
                             category_name="Knowledge Gap",
                             justification=f"justification {i}")
        for i in range(5)
    ]
    summary = fc.summarize_clusters(assignments, n_exemplars=2)
    assert len(summary.per_category[1]["exemplars"]) == 2
    for ex in summary.per_category[1]["exemplars"]:
        assert "item_id" in ex and "justification" in ex


# ---------------------------------------------------------------------------
# IO — graded JSONL loader
# ---------------------------------------------------------------------------


def test_load_paired_graded_round_trip(tmp_path: Path) -> None:
    """SPEC §Architecture: load_paired_graded reads arm0/arm1 JSONL across seeds and returns
    flat per-record lists usable by select_regressions."""
    graded_dir = tmp_path / "graded"
    graded_dir.mkdir()
    for arm, seed, score in [("v0", 42, 0.6), ("v0", 123, 0.5),
                             ("v25", 42, 0.4), ("v25", 123, 0.6)]:
        path = graded_dir / f"healthbench-hard__{arm}__seed{seed}.jsonl"
        with path.open("w") as f:
            f.write(json.dumps(_mk_record(item_id="A", seed=seed, arm=arm, score=score)) + "\n")

    arm0, arm1 = fc.load_paired_graded(
        graded_dir=graded_dir, benchmark="healthbench-hard",
        arms=("v0", "v25"), seeds=(42, 123),
    )
    assert len(arm0) == 2 and len(arm1) == 2
    assert {r["seed"] for r in arm0} == {42, 123}
    # Records preserve item_id, seed, graded.score
    assert all("graded" in r and "score" in r["graded"] for r in arm0 + arm1)


def test_load_paired_graded_missing_file_raises(tmp_path: Path) -> None:
    """SPEC §Hard rules: missing input file must raise — silent skip would let a typo
    or path drift produce a partial run that looks complete."""
    graded_dir = tmp_path / "graded"
    graded_dir.mkdir()
    # Only v0 present; v25 missing
    (graded_dir / "healthbench-hard__v0__seed42.jsonl").write_text(
        json.dumps(_mk_record(item_id="A", seed=42, arm="v0", score=0.5)) + "\n"
    )
    with pytest.raises(FileNotFoundError):
        fc.load_paired_graded(graded_dir=graded_dir, benchmark="healthbench-hard",
                              arms=("v0", "v25"), seeds=(42,))
