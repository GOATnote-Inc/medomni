"""TDD red phase for the κ comparator.

Pinned by `findings/2026-05-07-diagnostic-first-sft/STATE.md` K2.0. Stdlib only.
The comparator computes Cohen's κ + a per-category disagreement matrix from
two equal-length integer label streams, with explicit handling of the user's
`0` ("no category fits") sentinel.
"""

from __future__ import annotations

import csv

import pytest

from scripts.ship_rule_lib import kappa_comparator as kc


CATS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Cohen's κ
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement_is_one() -> None:
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    b = list(a)
    assert kc.cohen_kappa(a, b, categories=CATS) == pytest.approx(1.0)


def test_kappa_zero_agreement_is_negative() -> None:
    a = [1, 1, 1, 1, 1]
    b = [2, 2, 2, 2, 2]
    # po=0; pe>0 since both raters use single distinct cats → κ negative
    k = kc.cohen_kappa(a, b, categories=CATS)
    assert k <= 0


def test_kappa_partial_known_value() -> None:
    """Worked example: 10 items, 8 agree, both raters flat over {1,5}.
    a = [1]*5 + [5]*5;  b = [1]*4 + [5]*1 + [1]*1 + [5]*4
    po = 8/10 = 0.8
    a_count = {1:5, 5:5} → both 0.5
    b_count = {1:5, 5:5} → both 0.5
    pe = 0.5*0.5 + 0.5*0.5 = 0.5
    κ = (0.8 - 0.5) / (1 - 0.5) = 0.6"""
    a = [1, 1, 1, 1, 1, 5, 5, 5, 5, 5]
    b = [1, 1, 1, 1, 5, 1, 5, 5, 5, 5]
    assert kc.cohen_kappa(a, b, categories=CATS) == pytest.approx(0.6)


def test_kappa_rejects_unequal_length() -> None:
    with pytest.raises(ValueError):
        kc.cohen_kappa([1, 2], [1, 2, 3], categories=CATS)


def test_kappa_rejects_empty() -> None:
    with pytest.raises(ValueError):
        kc.cohen_kappa([], [], categories=CATS)


# ---------------------------------------------------------------------------
# Disagreement matrix
# ---------------------------------------------------------------------------


def test_disagreement_matrix_counts_correctly() -> None:
    gpt = [1, 1, 1, 5, 5, 4]
    user = [1, 5, 1, 5, 1, 4]
    mat = kc.disagreement_matrix(gpt=gpt, user=user, categories=CATS)
    # mat[gpt_cat][user_cat] = count
    assert mat[1][1] == 2  # gpt=1, user=1 (items 0, 2)
    assert mat[1][5] == 1  # gpt=1, user=5 (item 1)
    assert mat[5][5] == 1  # gpt=5, user=5 (item 3)
    assert mat[5][1] == 1  # gpt=5, user=1 (item 4)
    assert mat[4][4] == 1  # gpt=4, user=4 (item 5)


def test_disagreement_matrix_includes_all_categories_even_when_zero() -> None:
    gpt = [1, 1]
    user = [1, 1]
    mat = kc.disagreement_matrix(gpt=gpt, user=user, categories=CATS)
    # All 5 categories must appear as rows AND columns even when count=0
    for c in CATS:
        assert c in mat
        for c2 in CATS:
            assert c2 in mat[c]


# ---------------------------------------------------------------------------
# CSV / answer-key loading
# ---------------------------------------------------------------------------


def test_load_user_labels_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "seed", "category", "confident", "notes"])
        w.writerow(["A", "42", "1", "True", "easy"])
        w.writerow(["B", "123", "5", "False", "hard"])
    rows = kc.load_user_labels(csv_path)
    assert len(rows) == 2
    assert rows[0]["item_id"] == "A" and rows[0]["category"] == 1
    assert rows[1]["category"] == 5
    assert rows[1]["confident"] is False


def test_load_user_labels_skips_blank_rows(tmp_path) -> None:
    """Partially-filled CSV is normal during physician review — skip rows
    where the user hasn't entered a category yet."""
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "seed", "category", "confident", "notes"])
        w.writerow(["A", "42", "1", "True", "filled"])
        w.writerow(["B", "123", "", "", "blank"])
        w.writerow(["C", "7919", "5", "True", "filled"])
    rows = kc.load_user_labels(csv_path)
    assert [r["item_id"] for r in rows] == ["A", "C"]


def test_load_user_labels_handles_zero_as_no_category_fits(tmp_path) -> None:
    """0 means 'no category fits' (per kappa_blind_review.md §Your call). The
    comparator must surface these explicitly rather than treating as cat 0."""
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "seed", "category", "confident", "notes"])
        w.writerow(["A", "42", "0", "False", "doesn't fit"])
    rows = kc.load_user_labels(csv_path)
    assert rows[0]["category"] == 0
    assert rows[0]["no_category_fits"] is True


# ---------------------------------------------------------------------------
# End-to-end pair from CSV + answer key
# ---------------------------------------------------------------------------


def test_pair_user_to_answer_key_aligns_by_item_id(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    key_path = tmp_path / "key.jsonl"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "seed", "category", "confident", "notes"])
        w.writerow(["A", "42", "1", "True", ""])
        w.writerow(["B", "123", "5", "True", ""])
    import json
    with key_path.open("w") as f:
        f.write(json.dumps({"item_id": "A", "seed": 42, "gpt41_category": 1}) + "\n")
        f.write(json.dumps({"item_id": "B", "seed": 123, "gpt41_category": 1}) + "\n")
    paired = kc.pair_user_to_key(user_csv=csv_path, key_jsonl=key_path)
    assert len(paired) == 2
    assert paired[0]["item_id"] == "A"
    assert paired[0]["user_category"] == 1
    assert paired[0]["gpt41_category"] == 1
    assert paired[1]["user_category"] == 5 and paired[1]["gpt41_category"] == 1
