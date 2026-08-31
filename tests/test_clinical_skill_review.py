"""Fail-closed contract of the clinical-safety gate's verdict parser.

The clinical-skill-review workflow merges on FLAG and blocks on BLOCK, so
any model output the parser cannot understand MUST come back BLOCK: a
downgrade-to-FLAG default would turn a judge outage or a formatting change
into an automatic merge pass for clinical-content PRs.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from clinical_skill_review import parse_verdict  # noqa: E402


def test_no_json_at_all_blocks() -> None:
    verdict, blob, _ = parse_verdict("The model rambled with no JSON block.")
    assert verdict == "BLOCK"
    assert "fail-closed" in blob["summary"]


def test_empty_output_blocks() -> None:
    verdict, _, _ = parse_verdict("")
    assert verdict == "BLOCK"


def test_invalid_json_blocks() -> None:
    text = '```json\n{"verdict": "PASS", trailing garbage}\n```'
    verdict, blob, _ = parse_verdict(text)
    assert verdict == "BLOCK"
    assert "invalid JSON" in blob["summary"]


def test_unknown_verdict_value_blocks() -> None:
    text = '```json\n{"verdict": "APPROVE", "summary": "looks fine"}\n```'
    verdict, blob, _ = parse_verdict(text)
    assert verdict == "BLOCK"
    assert "unknown verdict" in blob["summary"]


def test_missing_verdict_key_blocks() -> None:
    text = '```json\n{"verdict": "", "summary": "no verdict present"}\n```'
    verdict, _, _ = parse_verdict(text)
    assert verdict == "BLOCK"


def test_valid_pass_still_passes() -> None:
    text = '```json\n{"verdict": "PASS", "summary": "clean", "findings": []}\n```\nAll good.'
    verdict, blob, rationale = parse_verdict(text)
    assert verdict == "PASS"
    assert blob["summary"] == "clean"
    assert rationale == "All good."


def test_valid_flag_still_flags() -> None:
    text = '```json\n{"verdict": "FLAG", "summary": "cite needed", "findings": []}\n```'
    verdict, _, _ = parse_verdict(text)
    assert verdict == "FLAG"


def test_valid_block_still_blocks() -> None:
    text = '```json\n{"verdict": "BLOCK", "summary": "phi leak", "findings": []}\n```'
    verdict, _, _ = parse_verdict(text)
    assert verdict == "BLOCK"


def test_bare_json_without_fence_parses() -> None:
    text = 'reasoning first\n{"verdict": "PASS", "summary": "ok"}'
    verdict, _, _ = parse_verdict(text)
    assert verdict == "PASS"
