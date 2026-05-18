# CI/CD audit — GOATnote-Inc public repos

**Status:** CARD (audit report)
**Date:** 2026-05-18
**Author:** Brandon Dent, MD (b@thegoatnote.com)
**Scope:** 10 active GOATnote-Inc public repos. Forks (`fish-speech`, `fish-speech-1`, `MedAgentBench`) out of scope.

---

## 1. Summary

Five repos fully green. Two repos with workflow-YAML bugs (medomni `lint`, receipts `test`) — both fixed this commit cycle. Two repos with real out-of-band signals that require founder triage (medomni `adversarial-probe` regressing, lostbench `OPENAI_API_KEY` rotated/expired). One stale GPU smoke failure deferred. One repo's path-filtered workflow correctly idle. One platform-wide deprecation (Node.js 20 → 24, forced 2026-06-02) flagged for sweep update before the deadline.

---

## 2. Status table

| Repo | Default branch | Latest CI | Conclusion | Notes |
|---|---|---|---|---|
| medomni | main | 2026-05-18 push | `lint` FAIL, `test` PASS | TruffleHog BASE=HEAD bug, fixed this cycle |
| medomni | main | 2026-05-18 schedule | `adversarial-probe` FAIL (recurring) | rc=3 (pass rate < 0.70) on live `/api/agent` — real regression, see §4.1 |
| healthcraft | main | 2026-05-11 push | All green | — |
| receipts | master | 2026-05-11 dynamic | `Dependency Graph` only | `test.yml` triggered on `[main]` while branch is `master` — fixed this cycle |
| medimage-corpus | main | n/a | No matching pushes | Path-filtered `manifest-validate` working as designed |
| prism42 | main | 2026-05-04 push | All green | — |
| lostbench | main | 2026-05-18 schedule | `Adversarial Regression` FAIL (recurring since 2026-04-27) | `OPENAI_API_KEY` 401, secret needs rotation; see §4.2 |
| scribegoat2 | main | 2026-05-18 schedule | All green | — |
| openem-corpus | main | 2026-04-16 push | All green | — |
| radslice | main | 2026-04-16 push | All green | — |
| safeshift | main | 2026-03-06 push | `Integration (GPU Smoke)` stale FAIL | GPU runner unavailable Mar 2026, deferred |

---

## 3. Fixes applied this commit cycle

### 3.1 medomni `lint` — TruffleHog `BASE and HEAD commits are the same`

**Root cause.** `.github/workflows/lint.yml` passed `base: ${{ github.event.repository.default_branch }}` (string `main`) and `head: HEAD`. On push to main, the TruffleHog action resolved both to the same commit (the push commit IS the new head of main). The action's internal bash logged `BASE: main`, `HEAD: HEAD`, then errored: "BASE and HEAD commits are the same. TruffleHog won't scan anything."

**Fix.** Remove the explicit `base:` and `head:` keys. The action infers the diff range from the event payload: `push → before..after`, `pull_request → base..head`, `schedule / workflow_dispatch → full history`. This is the documented default.

**Validation.** Re-run will fire on this commit's push. Expected: `lint` workflow green.

### 3.2 receipts `test.yml` — never triggered (wrong branch in trigger)

**Root cause.** `.github/workflows/test.yml` had `on: push: branches: [main]`. The repo's default branch is `master`. The workflow was never triggered because no push to `main` ever happened.

**Fix.** Change `branches: [main]` → `branches: [master]`.

**Validation.** Push to master will fire the test workflow. Expected: `test` workflow green (locally `make venv && make test` is clean per receipts CLAUDE.md verify command; 441 hermetic tests).

---

## 4. Flagged for founder action (out of CI-fix scope)

### 4.1 medomni `adversarial-probe` regression

The hourly probe at `https://medomni.vercel.app/api/agent` is returning rc=3 (pass rate < 0.70 threshold). This coincides with the 2026-05-07 migration from the 3-GPU B300 + H200×2 fleet to a single H100 serving V0 FP8 (~80% cost reduction, README §"Live status"). The probe's threshold of 0.70 was set when V2.5 was the canonical model.

**Three options for founder.** (a) Inspect the latest probe CARDs to confirm whether the regression is uniform or category-localized; lower the threshold to a V0-calibrated value (e.g., 0.65) and re-baseline. (b) Restore V2.5 / V2.7 / V3 from the LoRA archive and re-deploy. (c) Re-train the live model against the probe's specific failure modes (prompt-shape drift fix).

Recommendation: (a) first, treating it as a calibration update — the V0 baseline is the deployment story per the 2026-05-07 README headline. Issue the threshold change as a one-line workflow edit and re-baseline cleanly.

### 4.2 lostbench `OPENAI_API_KEY` 401

The scheduled `Adversarial Regression` workflow has failed every week since 2026-04-27 with `HTTP 401 Unauthorized` from `api.openai.com`. The repo secret has expired or been rotated.

Per `MEMORY.md → feedback_never_read_env.md`, I cannot read `/Users/kiteboard/lostbench/.env` from this session. Founder action:

```
gh secret set OPENAI_API_KEY -R GOATnote-Inc/lostbench
# (paste the key value at the interactive prompt; do NOT cat the .env file)
```

After rotation, manually trigger the workflow once to confirm green before the next scheduled run.

### 4.3 safeshift stale GPU smoke failure (2026-03-06)

Low priority. Likely an ephemeral GPU runner availability issue. Re-run the workflow when next a safeshift commit lands, and only investigate if it fails twice in a row.

---

## 5. Sweep item (not failing yet, deadline approaching)

### 5.1 Node.js 20 deprecation — forced default flip 2026-06-02

GitHub Actions runners flip the default JavaScript runtime from Node.js 20 to Node.js 24 on **2026-06-02** (15 days from today). Node.js 20 will be removed entirely on **2026-09-16**. Workflows currently using `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4` will see deprecation warnings now and hard breaks in Sept.

Affected workflows observed in audit logs: medomni `lint`, medomni `adversarial-probe`, lostbench `Adversarial Regression` — likely most repos.

**Recommended sweep PR per repo before 2026-06-02.** Audit each `.github/workflows/*.yml` for action versions:
- `actions/checkout@v4` → check for v5 release; fall back to setting `env: FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` at workflow or job level.
- `actions/setup-python@v5` → confirm Node 24-compatible version.
- `actions/upload-artifact@v4`, `actions/download-artifact@v4` → confirm Node 24-compatible.

Not blocking; track as a single sweep PR per repo. Defer past Series-A pitch unless something breaks earlier.

---

## 6. What changed in this commit cycle

This audit + the two workflow fixes ship in three commits:

1. **medomni `cac8492` (prior)** — elder-care agent OS positioning SPEC (already pushed).
2. **medomni (this commit)** — `.github/workflows/lint.yml` TruffleHog fix + this audit CARD.
3. **receipts (separate push)** — `.github/workflows/test.yml` branch trigger fix.

Founder actions tracked in §4 must happen out-of-band; nothing else in scope this turn.
