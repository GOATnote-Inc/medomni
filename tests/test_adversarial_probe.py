"""Tests for scripts/adversarial_probe.py.

Covers the preflight smoke-test, exit-code contract, and stream parsing
edge cases. No real network — `requests.post` is monkey-patched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
PROBE_PATH = REPO / "scripts" / "adversarial_probe.py"

# Load the script as a module without installing it. The script is a CLI
# that runs `_validate_cases()` at import — that side effect is desired
# (catches malformed case definitions before any test runs).
spec = importlib.util.spec_from_file_location("adversarial_probe", PROBE_PATH)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
sys.modules["adversarial_probe"] = probe
spec.loader.exec_module(probe)


class _FakeResp:
    def __init__(self, status_code: int, content: bytes = b"", text: str = "") -> None:
        self.status_code = status_code
        self.content = content
        self.text = text or content.decode("utf-8", errors="replace")


def _stream(*deltas: str) -> bytes:
    lines = [f'data: {{"type":"text-delta","id":"text_0","delta":"{d}"}}' for d in deltas]
    lines.append("data: [DONE]")
    return ("\n".join(lines)).encode("utf-8")


def _error_stream(msg: str) -> bytes:
    return f'data: {{"type":"error","errorText":"{msg}"}}\n'.encode()


def _patch_post(
    monkeypatch: pytest.MonkeyPatch, resp: _FakeResp | Exception
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict, timeout: float, headers: dict) -> _FakeResp:  # noqa: A002
        calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(probe.requests, "post", fake_post)
    return calls


# ---------------------------------------------------------------------------
# preflight()
# ---------------------------------------------------------------------------


def test_preflight_ok_on_200_with_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResp(200, content=_stream("Paris", ".")))
    ok, reason = probe.preflight("https://example.test/api/agent")
    assert ok is True
    assert "preflight OK" in reason


def test_preflight_fails_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResp(404, text="The page could not be found"))
    ok, reason = probe.preflight("https://example.test/api/agent")
    assert ok is False
    assert "HTTP 404" in reason


def test_preflight_fails_on_empty_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResp(200, content=b"data: [DONE]\n"))
    ok, reason = probe.preflight("https://example.test/api/agent")
    assert ok is False
    assert "empty text stream" in reason


def test_preflight_fails_on_server_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResp(200, content=_error_stream("rate limited")))
    ok, reason = probe.preflight("https://example.test/api/agent")
    assert ok is False
    assert "stream errors" in reason and "rate limited" in reason


def test_preflight_fails_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, probe.requests.ConnectionError("dns failure"))
    ok, reason = probe.preflight("https://example.test/api/agent")
    assert ok is False
    assert "network error" in reason and "ConnectionError" in reason


# ---------------------------------------------------------------------------
# main() exit-code contract
# ---------------------------------------------------------------------------


def test_main_returns_infrastructure_on_preflight_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_post(monkeypatch, _FakeResp(404, text="Not Found"))
    rc = probe.main(
        [
            "--endpoint",
            "https://example.test/api/agent",
            "--findings-dir",
            str(tmp_path),
            "--limit",
            "1",
            "--timeout",
            "5",
            "--retries",
            "1",
        ]
    )
    assert rc == probe.EXIT_INFRASTRUCTURE
    # Confirm cases were NOT run — no CARD written.
    assert not list(tmp_path.iterdir())


def test_main_skip_preflight_then_dry_run_returns_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # --dry-run records every case as failed but still writes a CARD; with
    # threshold 0.70 and 0 passes it exits EXIT_REGRESSION.
    rc = probe.main(
        [
            "--endpoint",
            "https://example.test/api/agent",
            "--findings-dir",
            str(tmp_path),
            "--skip-preflight",
            "--dry-run",
            "--limit",
            "1",
            "--threshold",
            "0.70",
        ]
    )
    assert rc == probe.EXIT_REGRESSION
    cards = list(tmp_path.glob("2026-05-05-adversarial-probe-*/CARD.md"))
    assert len(cards) == 1


def test_main_skip_preflight_with_high_pass_returns_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Synthesize a response that satisfies RD-001's rubric.
    _patch_post(monkeypatch, _FakeResp(200, content=_stream("Carcinoid", " syndrome")))
    rc = probe.main(
        [
            "--endpoint",
            "https://example.test/api/agent",
            "--findings-dir",
            str(tmp_path),
            "--skip-preflight",
            "--limit",
            "1",
            "--threshold",
            "0.70",
            "--timeout",
            "5",
            "--retries",
            "1",
        ]
    )
    assert rc == probe.EXIT_OK


# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------


def test_parse_uimessage_stream_accumulates_text_deltas() -> None:
    body = _stream("Hello", " ", "world")
    text, errors = probe.parse_uimessage_stream(body)
    assert text == "Hello world"
    assert errors == []


def test_parse_uimessage_stream_captures_error_events() -> None:
    body = _error_stream("model timeout")
    text, errors = probe.parse_uimessage_stream(body)
    assert text == ""
    assert errors == ["model timeout"]
