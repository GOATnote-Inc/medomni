"""Judge failures must RECUSE records, never score them (readiness audit P1-5).

The old grader coerced a failed judge call (`criteria_met: None` after 3
retries) into `criteria_met: False` and scored the record anyway. On rubrics
with negative-point criteria that silently inflates scores during a judge
outage; on positive criteria it deflates them. The HealthBench runner
already recuses in this situation — the ship-rule grader must match it.

These tests fail on the pre-fix grader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from ship_rule_lib import grader  # noqa: E402

RECORD = {
    "item_id": "hb-recuse-1",
    "arm": "v0",
    "response": "Recommend aspirin and outpatient follow-up.",
    "rubric": [
        {"criterion": "mentions aspirin", "points": 5.0},
        {"criterion": "does not recommend unnecessary surgery", "points": -6.0},
    ],
}

CLEAN_RECORD = {
    "item_id": "hb-clean-1",
    "arm": "v0",
    "response": "Recommend aspirin.",
    "rubric": [
        {"criterion": "mentions aspirin", "points": 5.0},
    ],
}


def fake_judge_ok(response: str, criterion: str, conversation: str = "") -> dict:
    return {"criteria_met": True, "explanation": "ok", "judge_model": "fake"}


def fake_judge_one_failure(response: str, criterion: str, conversation: str = "") -> dict:
    if "surgery" in criterion:
        # What _gpt_judge returns after exhausting retries.
        return {
            "criteria_met": None,
            "explanation": "judge_error: APIConnectionError: boom",
            "judge_model": "fake",
        }
    return {"criteria_met": True, "explanation": "ok", "judge_model": "fake"}


def test_judge_failure_recuses_record() -> None:
    graded = grader.grade_healthbench_record(record=RECORD, grader_fn=fake_judge_one_failure)
    assert graded["score"] is None, (
        "a judge failure must recuse the record, not produce a score "
        f"(got score={graded['score']!r})"
    )
    assert graded.get("recused") is True
    # The judge log still records what happened for auditability.
    assert any(j.get("criteria_met") is None for j in graded["judge_log"])


def test_all_judgments_present_scores_record() -> None:
    graded = grader.grade_healthbench_record(record=RECORD, grader_fn=fake_judge_ok)
    assert graded["score"] is not None
    assert not graded.get("recused")
    assert graded["n_rubrics"] == 2


def test_grade_jsonl_counts_n_recused(tmp_path: Path, monkeypatch) -> None:
    # The failing judge only errors on the surgery criterion, so RECORD is
    # recused and CLEAN_RECORD grades normally.
    monkeypatch.setattr(grader, "_gpt_judge", lambda model: fake_judge_one_failure)

    in_p = tmp_path / "gen.jsonl"
    out_p = tmp_path / "graded.jsonl"
    with in_p.open("w") as fh:
        fh.write(json.dumps(RECORD) + "\n")
        fh.write(json.dumps(CLEAN_RECORD) + "\n")

    aggregate = grader.grade_jsonl(
        in_jsonl=in_p,
        out_jsonl=out_p,
        benchmark="healthbench-hard",
    )

    assert aggregate["n_recused"] == 1
    assert aggregate["n_graded"] == 1
    # Recusal is not the same bucket as "missing rubric".
    assert aggregate["n_missing"] == 0

    graded_rows = [json.loads(line) for line in out_p.read_text().splitlines()]
    by_id = {r["item_id"]: r["graded"] for r in graded_rows}
    assert by_id["hb-recuse-1"]["score"] is None
    assert by_id["hb-recuse-1"]["recused"] is True
    assert by_id["hb-clean-1"]["score"] is not None


def test_empty_rubric_is_missing_not_recused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(grader, "_gpt_judge", lambda model: fake_judge_ok)
    in_p = tmp_path / "gen.jsonl"
    out_p = tmp_path / "graded.jsonl"
    with in_p.open("w") as fh:
        fh.write(json.dumps({"item_id": "hb-norubric", "arm": "v0", "response": "x"}) + "\n")
    aggregate = grader.grade_jsonl(in_jsonl=in_p, out_jsonl=out_p, benchmark="healthbench-hard")
    assert aggregate["n_missing"] == 1
    assert aggregate["n_recused"] == 0
