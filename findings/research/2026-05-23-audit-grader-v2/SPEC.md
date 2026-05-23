# Audit grader v2 — weight-aware dashboard for medomni

**Date:** 2026-05-23
**Authors:** Brandon Dent + Claude Opus 4.7
**Status:** SPEC. v1 grader (the hourly bot updating issue #396) is recommended for **retirement**; v2 is the structural replacement that ships in this PR.

---

## Why v1 fails

The v1 grader posts a 13-item binary presence checklist to issue #396 every hour. Two failure modes, both observed in the dashboard's own output:

1. **It grades presence, not weight (m13v's structural critique).** The dashboard cannot distinguish *"scaffold exists and load-bears"* from *"scaffold exists and is dead weight"*. Concrete: after the sprint that trimmed `CLAUDE.md` §0 from 22 lines of inherited RunPod content to 5 lines of current-medomni hot-path (PR #404), the grader cannot see that improvement. It can only see whether §0 still exists. The recommended Top-3 next actions on the most recent dashboard — *create `agents/`, add a `UserPromptSubmit` hook, populate more skills* — are the **opposite** of what the sprint concluded. The bot did not learn from the response. It cannot.

2. **It lost the product story.** The original audit's Part 3 #3 was *"EXPERIMENT_LOG stalled at iter=1; confirm loop complete or trigger next cycle."* The current dashboard has nothing about iter=2 or the research loop — replaced with more scaffolding suggestions. The bot is grading scaffolding because that is what it can see; the actual product work (collapse-diagnostic, gate-raise pre-reg, Goodhart reframing across PRs #405 / #406 / #409) does not register on its surface. Same disease, different layer.

A second-order failure that surfaced during the sprint: the bot's source is **untraceable from any maintainer machine** (no `crontab`, no `launchd`, no `~/.claude/scheduled_tasks.json`, no in-repo workflow, no audit-named `GOATnote-Inc` repo). An automated process posting authoritative status to the main issue tracker with no in-repo source is a governance problem independent of m13v's critique.

The conclusion: do not iterate v1. Replace it.

## v2 design — what gets graded

v2 grades **load-bearing properties**, not existence. Three categories.

### A. Hot-path discipline (CLAUDE.md §0)

| Check | Threshold | Why this is load-bearing |
| --- | --- | --- |
| §0 line count | ≤ 15 effective lines | Above 15, the block stops behaving like a hot-path marker (per the PR #404 budget rule) |
| `Last audited:` timestamp present | required | Audit drift is invisible without it |
| `Last audited:` recency | ≤ 90 days | Stale §0 entries fire every turn at the cost of context budget |
| §0 references at least 1 skill or hook | required | A §0 block that does not point at load-on-demand alternatives is suspected drift |
| Pre-commit string blocklist matches reality | required | If §0 names blocked literals that pre-commit-config no longer blocks, the marker is decorative |

### B. Metric integrity (canonical metrics)

For each canonical-metric definition in the repo (currently: `corpus_pass_rate ≥ 0.6` in `findings/2026-05-07-diagnostic-first-sft/PROGRAM.md`):

| Check | Threshold | Why this is load-bearing |
| --- | --- | --- |
| Most-recent iteration's per-pattern σ | ≥ 0.10 (each pattern) | A pattern at σ < 0.10 is delta-collapsed; the judge can no longer discriminate; the metric is hollow on that pattern (per `ITER1_DIAGNOSTIC_RESULT.md`) |
| Corpus-wide σ retention vs baseline | σ_post ≥ 0.6 × σ_pre | Same property as PR #405's C4 criterion; transferred from one-off diagnostic to standing metric |
| Pre-registration exists for current iter | required | Iteration without a pre-reg is uncalibrated; the metric is unverified |
| Most-recent iteration's `decision` field | not `KEEP` if any σ-check fails | Match the conjunctive rule from the pre-reg discipline |

### C. Product progress (research loop liveness)

| Check | Threshold | Why this is load-bearing |
| --- | --- | --- |
| `EXPERIMENT_LOG.jsonl` latest entry recency | ≤ 30 days | Stagnant research loops are the failure-of-product the bot must surface (Part 3 of the v1 dashboard lost this exact bullet) |
| Latest iteration has a corresponding `*_RESULT.md` | required if pre-reg exists | A pre-reg without a result is an open loop |
| Open follow-ups named in result docs | counted, not failed | Visibility into deferred work |

### D. Decorative-legacy (presence-only, marked as such)

v1's 13 presence-checks are reproduced verbatim, **clearly marked as "presence-only / decorative"** so readers know which axes load-bear (A / B / C) and which are checkbox-shaped (D). v1's checks remain useful as smoke alarms (did the `tests/` dir vanish?) but should never be the headline.

Each axis in A / B / C returns:

- `status`: `PASS` / `FAIL` / `DEFERRED` / `N/A`.
- `reason`: one sentence explaining *why* this status, not just *what*.
- `evidence`: file paths + line numbers + actual values.

## v2 implementation

### Files in this PR

- `scripts/audit_grader_v2.py` — the grader. Pure Python stdlib (no external deps required to run in CI). Reads the repo, computes the axes above, emits a markdown dashboard to stdout.
- `.github/workflows/audit-grader-v2.yml` — daily cron at 14:07 UTC (off-minute per Claude Code's CronCreate advisory; once a day rather than hourly per the *"10 consecutive identical updates"* observation that hourly was theater). Runs the script; uses `gh issue edit` to update the dashboard issue's body.
- This SPEC (`findings/research/2026-05-23-audit-grader-v2/SPEC.md`).

### New issue (v2 dashboard surface)

A fresh issue titled *"audit (v2): weight-aware best-practices dashboard"* is opened as part of this PR's lifecycle. The workflow's first run populates it. The issue number is hard-coded in the workflow once created.

This is deliberately **not** issue #396. Reasons:

- #396 is the v1 surface; closing it cleanly requires the v1 source to be located and stopped, which is a maintainer-action item independent of v2.
- m13v's critique thread lives on #396; appending v2's auto-updates would muddy that thread.
- Per the jr-dev review: *"Closing this issue doesn't stop the bot — it'll keep posting hourly to a closed issue."* A fresh issue avoids that braid.

### Migration / retirement of v1

The v1 bot is recommended for retirement. Three steps, in order:

1. **Locate the v1 source.** Maintainer-action. The earlier hunt (per `gh issue comment` on PR #405) failed to find it on this mac; the bot may be on a server, a separate dev machine, or a GitHub App not yet checked. The audit log at `https://github.com/organizations/GOATnote-Inc/settings/audit-log` shows the source IP + credential per posting on #396.
2. **Stop the v1 bot.** Once located, kill the cron / disable the workflow / revoke the credential.
3. **Final comment on #396** announcing v1 retired + pointing at v2's dashboard issue, then close. Per the jr-dev: *"The clean close-of-loop in 72 hours, if m13v doesn't reply: one final comment that says 'v2 grader shipped / bot retired' — whichever you chose — then close."*

This PR ships v2 (steps 0). Steps 1–3 are queued for the maintainer once v2 is producing satisfactory output and the m13v reply window has elapsed.

## Known limitations (recorded up-front)

Per the discipline-floor pattern: name what v2 doesn't catch, before someone else does.

1. **The `git add -A` hook has an argument-expansion blindspot.** Both the user-global `~/.claude/hooks/block-bash-unsafe.sh` and this repo's `.claude/hooks/pretool-git-add-guard.py` regex against the Bash command text. A script with `arg="-A"; git add $arg` does not match the regex — the literal `git add -A` never appears in the command source. PR #404's smoke test covered the literal cases, not the expanded ones. The Anthropic-team-named anti-pattern *"prevent `git add -A` via hook (race condition; use permissions instead)"* may be pointing at exactly this blindspot, not at a concurrent-shell race. Worth investigating before defending the hook against m13v if he engages on that thread.

2. **v2 cannot grade what the model ignores.** m13v's sharper question — *"which §0 lines fire every turn vs which Claude consistently ignores"* — requires instrumented sessions, not a static repo scan. v2 grades the *structural* properties (line budget, recency, presence-of-pointer-to-skill) that approximate "this line earns its slot" but does not measure attention directly. The honest framing is that v2 makes weight measurable along axes structural-discipline can capture; the deeper measurement is still future work.

3. **v2's metric-integrity check is currently scoped to one canonical metric.** `corpus_pass_rate` in `findings/2026-05-07-diagnostic-first-sft/PROGRAM.md`. Other canonical metrics elsewhere in the repo will need their own integrity checks added. v2 fails gracefully on unrecognized metric files (logs `DEFERRED — no integrity check defined for this metric`).

4. **Daily cadence is itself a choice with trade-offs.** A regression detected in the morning will not surface on the dashboard until the following day's run. The adversarial-probe workflow (hourly, fires issues on threshold-cross) is the right tool for sub-day regressions; the audit dashboard is for slow-moving structural drift. The two tools cover different failure-mode time-scales.

## What v2 is NOT

- Not a unit-test runner; the existing CI workflows cover that.
- Not a security scanner; `secrets-scan.yml` covers that.
- Not a deploy-status surface; Vercel + the `bwnfkqe3q`-style manual-deploy outputs cover that.
- Not a per-PR review surface; the safety-engineer-review.yml + agent-pr-review.yml workflows cover that.

v2's specific job: surface structural / metric / product-progress *drift* that none of those other tools see.

---

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
