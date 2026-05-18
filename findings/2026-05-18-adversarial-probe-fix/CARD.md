# Adversarial probe — endpoint fix + alarm de-noise

**Status:** CARD (fix landed)
**Date:** 2026-05-18
**Author:** Brandon Dent, MD (b@thegoatnote.com)

---

## 1. TL;DR

The hourly `adversarial-probe` workflow has fired rc=3 (regression) every run since 2026-05-07 and auto-filed **200+ open issues** in the `adversarial-probe` label. Cause was not a model regression — it was an infrastructure bug: the probe's `DEFAULT_ENDPOINT` was `https://medomni.vercel.app/api/agent`, which returns HTTP 404 because the live demo moved behind a Vercel apex rewrite at `https://www.thegoatnote.com/4UWHAt/` and the bare `medomni.vercel.app` host no longer exposes API routes outside that path. Every probe response was empty, every case failed on "missing required substring", and the gate fired correctly — against a dead URL.

This commit ships four changes together: (a) point the probe at the real endpoint, (b) add a 1-query preflight smoke that fails fast with a distinct exit code so we never burn 5 minutes producing 20 identical empty-response failures again, (c) distinguish infrastructure failures from clinical-regression failures in the workflow's alarm path, (d) dedup issue creation so a sustained outage produces ONE issue with appended comments instead of an issue every hour. Includes 10-test coverage of the preflight contract.

Live validation post-fix: preflight OK on the corrected endpoint, RD-001 (carcinoid) returns 100% pass on a real `--limit 1` run.

---

## 2. How the failure looked from the outside

| Timeline | Event |
|---|---|
| 2026-05-07 | Production demo migrated from 3-GPU B300 + H200×2 fleet to a single H100 serving V0 FP8 (README §"Live status", ~80% cost cut). Routing changed to put the public API only behind `https://www.thegoatnote.com/4UWHAt/`. |
| 2026-05-08 17:35 UTC | First adversarial-probe issue auto-filed. |
| 2026-05-07 → 2026-05-18 | Probe fires hourly, hits 404, scores 0/20, opens an issue. **200 issues** accumulate in the same label, none ever triaged because the title looked like clinical regression. |
| 2026-05-18 (this commit) | Audit catches the runaway pattern; root cause identified as URL drift; fix shipped. |

The probe was correct: hitting `medomni.vercel.app/api/agent` does return 404, every response is empty, every case fails. The probe was *honest*. The probe was wrong *about the URL*.

---

## 3. Root cause + the four-part fix

### 3.1 Endpoint drift (root cause)

The probe hardcoded `https://medomni.vercel.app/api/agent` as `DEFAULT_ENDPOINT`. After the 2026-05-07 migration, the only public surface for the agent is `https://www.thegoatnote.com/4UWHAt/api/agent` (Vercel edge rewrite from the apex). The probe never tracked that move.

**Fix:** `DEFAULT_ENDPOINT` updated in `scripts/adversarial_probe.py` and the workflow `ENDPOINT` env default updated to match. Inline comment explaining the rewrite path so the next person reading the file does not regress this.

### 3.2 Preflight smoke (so this exact failure mode never recurs)

Probing a dead URL with 20 cases produces 20 identical "empty response" failures over 5 minutes wall-clock. That is wasted CI time and indistinguishable from a real regression to anyone reading the CARD. Worse, the workflow files an issue based on the result — generating noise instead of signal.

**Fix:** new `preflight(endpoint)` function that posts a trivial query (`"What is the capital of France? One word answer."`) and returns `(ok, reason)`. Preflight asserts:
- HTTP 200,
- no `error` events in the SSE stream,
- non-empty assembled text-delta.

`main()` calls preflight before running the case suite. On failure, it prints the reason and exits with `EXIT_INFRASTRUCTURE = 2` instead of `EXIT_REGRESSION = 3`. The case suite is skipped — no wasted compute, no 20 empty CARD entries.

A `--skip-preflight` flag exists for tests.

### 3.3 Distinct exit codes → distinct alarm channels

Before: any non-zero rc opened a `regression` issue. After:

| Exit code | Meaning | Issue label | On-call runbook |
|---:|---|---|---|
| 0 | Pass rate ≥ threshold | none | nothing to do |
| 2 | Endpoint unreachable / preflight failed | `infrastructure` | check the URL, wake the platform team, look at Vercel routing |
| 3 | Cases ran, pass rate < threshold | `regression` | look at the model's actual answers in the CARD |

The workflow gate still fails on rc=2 and rc=3 (red X either way), but the issue label tells the responder *what kind of red X* without reading the CARD.

### 3.4 Issue dedup (24-hour window)

Before: every probe failure unconditionally opened a new issue. A 10-day outage produced 240 issues.

After: the workflow searches open issues in the `adversarial-probe` + kind-label intersection created in the last 24h. If one exists, it appends a comment to the most-recent open match with the new CARD body, run URL, and timestamp. If none exists, it opens a fresh one.

Result: a sustained outage produces **one issue per day per kind**, growing with appended comments — not 24 issues per day per kind. The responder gets a single thread to read and a clear close-this-to-restart-the-counter affordance.

---

## 4. Tests

`tests/test_adversarial_probe.py` — 10 tests, all passing locally (Python 3.14, pytest 9.0.3, <100 ms wall):

- `preflight` returns `True` on 200 + non-empty stream
- `preflight` returns `False` on HTTP 404, empty stream, server `error` events, network exceptions
- `main` returns `EXIT_INFRASTRUCTURE` (2) when preflight fails — confirms no CARD is written
- `main` returns `EXIT_REGRESSION` (3) when cases run but all fail (dry-run)
- `main` returns `EXIT_OK` (0) when a stub response satisfies the rubric
- `parse_uimessage_stream` correctly accumulates text-delta and captures `error` events

`requests.post` is monkey-patched throughout — no real network in the suite.

---

## 5. Operational follow-up (this commit + immediate post-push)

After this commit lands, the 200+ stale open issues should be **bulk-closed** with a comment pointing to this CARD. The next probe run will then either:
- (a) preflight OK + cases pass → no issue, green workflow, problem solved, or
- (b) preflight OK + cases fail → ONE new `regression` issue with the actual CARD content for triage, or
- (c) preflight fail → ONE new `infrastructure` issue with the actual cause string.

Bulk-close command (run from this terminal after push):

```
gh issue list -R GOATnote-Inc/medomni --label adversarial-probe --state open --limit 500 \
  --json number --jq '.[].number' \
  | xargs -I{} gh issue close {} -R GOATnote-Inc/medomni \
      -c "Superseded by 2026-05-18 endpoint fix (findings/2026-05-18-adversarial-probe-fix/CARD.md). Probe was hitting a 404 URL, not a model regression. Closing to clear the queue; the next genuine failure will open a fresh issue with dedup behavior."
```

---

## 6. Deferred (separate PRs)

These were intentionally NOT shipped in this commit to keep the diff scoped and reviewable. Each warrants its own PR with its own design discussion:

1. **K-of-N consecutive-failure gating.** Substring rubric has known false-positive noise — a single missing synonym ("neuroendocrine tumor" instead of "carcinoid") fails a case the clinician would accept. Only fire the gate / file the issue if K of last N runs failed (e.g., 3 of last 5). Requires calling `gh run list` from the gate step or persisting state in a status repo.

2. **Per-category trend tracking.** A 6/20 fail is very different if all 6 are `pediatric_dosing` vs spread across 5 categories. The CARD already breaks out per-category — add a 7-day rolling pass-rate per category to the issue body to make trend visible.

3. **Rubric loosening with V0-calibrated synonyms.** Run the probe against V0 with `--threshold 0.0` to baseline the actual answer space, then expand each case's `expected_must_contain` to include the synonyms V0 actually produces (when clinically equivalent). This is the principled alternative to lowering the threshold globally.

4. **Smoke-on-PR.** Add a `--limit 3` preflight + 3-case run to the existing `test.yml` workflow so a PR that breaks the probe surface fails CI before merging, rather than waiting for the next hourly schedule firing.

---

## 7. Provenance + reading order

Identified during the 2026-05-18 cross-repo CI/CD audit
(`findings/2026-05-18-ci-cd-audit/CARD.md` §4.1, recommendation (a) "calibration update"). That CARD's recommendation was wrong — the regression was not a calibration issue, it was a URL bug. This CARD supersedes that recommendation at the diagnostic level.
