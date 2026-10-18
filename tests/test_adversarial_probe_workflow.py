"""Execute the probe workflow's reporting steps without a live endpoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load((REPO / ".github/workflows/adversarial-probe.yml").read_text())
STEPS = WORKFLOW["jobs"]["probe"]["steps"]


@pytest.mark.parametrize(
    ("rc", "expected_status", "message"),
    [
        ("0", 0, "passed"),
        ("2", 2, "clinical cases were not run"),
        ("3", 3, "pass rate was below threshold"),
        ("", 2, "infrastructure failure"),
        ("1", 2, "infrastructure failure"),
    ],
)
def test_gate_preserves_failure_kind(rc: str, expected_status: int, message: str) -> None:
    gate = next(step for step in STEPS if step.get("name", "").startswith("Gate"))
    result = subprocess.run(
        ["bash", "-e", "-c", gate["run"]],
        env={**os.environ, "PROBE_RC": rc},
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_status
    assert message in result.stdout


@pytest.mark.parametrize("has_current_card", [False, True])
def test_locator_ignores_historical_cards(tmp_path: Path, has_current_card: bool) -> None:
    historical = tmp_path / "findings" / "2026-05-05-adversarial-probe-spec"
    historical.mkdir(parents=True)
    (historical / "CARD.md").write_text("This is a specification, not a result.")
    current_root = tmp_path / "current-run"
    current_root.mkdir()
    current = current_root / "2026-05-05-adversarial-probe-2026101801"
    if has_current_card:
        current.mkdir()
        (current / "CARD.md").write_text("Current result.")
    output = tmp_path / "github-output"
    output.touch()
    locator = next(step for step in STEPS if step.get("id") == "card")
    subprocess.run(
        ["bash", "-e", "-c", locator["run"]],
        cwd=tmp_path,
        env={**os.environ, "PROBE_FINDINGS_DIR": str(current_root), "GITHUB_OUTPUT": str(output)},
        check=True,
    )
    if has_current_card:
        assert f"card_path={current}/CARD.md" in output.read_text()
    else:
        assert output.read_text() == ""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js required for workflow script")
def test_preflight_alarm_without_card(tmp_path: Path) -> None:
    alarm = next(step for step in STEPS if step.get("name") == "Open or update alarm issue")
    harness = (
        """
const context = {serverUrl: 'https://github.com', repo: {owner: 'test', repo: 'test'}, runId: 1};
const core = {info: () => {}};
const github = {
  paginate: async () => [],
  rest: {issues: {listForRepo: () => {}, create: async (issue) => console.log(JSON.stringify(issue))}}
};
(async () => {
"""
        + alarm["with"]["script"]
        + "\n})().catch(e => { console.error(e); process.exit(1); });"
    )
    result = subprocess.run(
        ["node", "-e", harness],
        cwd=tmp_path,
        env={**os.environ, "PROBE_RC": "2", "CARD_PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    issue = json.loads(result.stdout)
    assert issue["labels"] == ["adversarial-probe", "infrastructure"]
    assert "clinical cases were not run" in issue["body"]
    assert "No result artifact was produced" in issue["body"]
    assert "regression detected" not in issue["title"]
