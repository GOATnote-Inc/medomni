"""Sanity tests for the unified ship-rule eval driver — stdlib-only, no GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from ship_rule_lib import leakage, manifest as manifest_lib, stats  # noqa: E402


def test_paired_bootstrap_known_delta() -> None:
    v0 = [0.5, 0.6, 0.4, 0.5, 0.7, 0.6, 0.5, 0.4, 0.5, 0.6]
    v25 = [0.7, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6, 0.7, 0.8]
    pr = stats.paired_bootstrap(v0, v25, n_resamples=2000, seed=42)
    assert 0.18 < pr.delta < 0.22
    assert pr.ci_low <= pr.delta <= pr.ci_high
    assert pr.n_pairs == 10
    assert pr.cohen_d > 0  # large positive effect


def test_paired_bootstrap_zero_delta() -> None:
    v0 = [0.5] * 10
    v25 = [0.5] * 10
    pr = stats.paired_bootstrap(v0, v25, n_resamples=500, seed=1)
    assert pr.delta == 0.0
    assert pr.ci_low == 0.0 and pr.ci_high == 0.0


def test_holm_bonferroni() -> None:
    rejects = stats.holm_bonferroni([0.001, 0.01, 0.04, 0.20])
    # Holm step-down at α=0.05 over 4 hypotheses:
    #   smallest p=0.001 vs 0.05/4=0.0125 → reject
    #   next p=0.01 vs 0.05/3≈0.0167 → reject
    #   next p=0.04 vs 0.05/2=0.025 → fail to reject → stop
    assert rejects == [True, True, False, False]


def test_holm_all_pass() -> None:
    rejects = stats.holm_bonferroni([0.5, 0.6, 0.7])
    assert rejects == [False, False, False]


def test_align_paired_drops_unpaired() -> None:
    v0 = [{"item_id": "a", "score": 1}, {"item_id": "b", "score": 2}, {"item_id": "c", "score": 3}]
    v25 = [{"item_id": "a", "score": 4}, {"item_id": "c", "score": 5}, {"item_id": "d", "score": 6}]
    a, b, dropped = stats.align_paired(v0, v25, key="item_id")
    assert [r["item_id"] for r in a] == ["a", "c"]
    assert [r["item_id"] for r in b] == ["a", "c"]
    assert sorted(dropped) == ["b", "d"]


def test_sha256_known_value() -> None:
    assert manifest_lib.sha256_str("hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_manifest_roundtrip(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("alpha\n")
    f2 = tmp_path / "sub" / "b.txt"
    f2.parent.mkdir()
    f2.write_text("beta\n")
    out = tmp_path / "MANIFEST.sha256"
    manifest_lib.write_manifest(
        out_path=out,
        repo_root=tmp_path,
        files=[f1, f2],
        metadata={"k": "v"},
    )
    text = out.read_text()
    assert "# k=v" in text
    assert "a.txt" in text and "sub/b.txt" in text
    ok, fails = manifest_lib.verify_manifest(out, tmp_path)
    assert ok, fails

    # Mutate one file → verify must catch.
    f1.write_text("ALPHA-DIFFERENT\n")
    ok, fails = manifest_lib.verify_manifest(out, tmp_path)
    assert not ok and any("a.txt" in f for f in fails)


def test_minhash_jaccard_self_one() -> None:
    sh = leakage.shingles(leakage.tokenize("the quick brown fox jumps over the lazy dog"))
    sig = leakage.minhash(sh, n_perm=64)
    assert leakage.jaccard_estimate(sig, sig) == 1.0


def test_minhash_disjoint_low() -> None:
    a = leakage.minhash(
        leakage.shingles(leakage.tokenize("alpha bravo charlie delta echo")), n_perm=64
    )
    b = leakage.minhash(
        leakage.shingles(leakage.tokenize("foxtrot golf hotel india juliet")), n_perm=64
    )
    assert leakage.jaccard_estimate(a, b) < 0.1


def test_levenshtein_ratio_extremes() -> None:
    assert leakage.levenshtein_ratio("abc", "abc") == 1.0
    assert leakage.levenshtein_ratio("", "") == 1.0
    assert leakage.levenshtein_ratio("abc", "xyz") < 0.5


def test_scan_test_prompts_finds_overlap(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        json.dumps({"text": "the patient presents with chest pain and shortness of breath"}) + "\n"
    )
    sketches = leakage.build_train_sketches([train_jsonl], n_perm=64, n=5)
    assert sketches
    test_items = [
        {"item_id": "t1", "prompt": "the patient presents with chest pain and shortness of breath"},
        {"item_id": "t2", "prompt": "completely unrelated computer science exam answer key"},
    ]
    hits = leakage.scan_test_prompts(
        test_items=test_items,
        train_sketches=sketches,
        benchmark="medqa",
        threshold=0.5,
        n_perm=64,
        n=5,
    )
    flagged = {h.item_id for h in hits}
    assert "t1" in flagged
    assert "t2" not in flagged


def test_post_hoc_power_monotone_in_n() -> None:
    p_small = stats.post_hoc_power(d_z=0.5, n=10, alpha=0.05)
    p_large = stats.post_hoc_power(d_z=0.5, n=200, alpha=0.05)
    assert p_small < p_large
    assert 0.0 <= p_small <= 1.0 and 0.0 <= p_large <= 1.0
