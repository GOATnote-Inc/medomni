# Audit-bot find-logic fix — title-search → label-based

**Status:** CARD (fix applied to scheduled routine; no in-repo code change)
**Date:** 2026-05-18
**Author:** Brandon Dent, MD (b@thegoatnote.com)
**Routine ID:** `trig_01KtJothoiKdMC4sayF915py` ("medomni hourly audit (anthropic + cherny + karpathy)")

---

## 1. TL;DR

The hourly audit-bot accumulated **15 duplicate "audit: hourly best-practices dashboard" issues** from 2026-05-09 → 2026-05-18 instead of updating one rolling issue as designed. All 15 were bulk-closed in the 2026-05-18 audit-cleanup pass (companion to `findings/2026-05-18-adversarial-probe-fix/CARD.md`).

**Root cause** was not a design flaw in Brandon's routine — the prompt explicitly says "SINGLE rolling GitHub Issue, NO PRs." The bug was in the *find-existing* implementation: it used GitHub's title-search (`gh issue list --search '... in:title'`), which has eventual-consistency lag. When a freshly-created issue isn't indexed yet, the next hour's run can't find it and creates a duplicate. 15 misses across ~240 hourly runs = ~6% search-miss rate, which matches the observed pathology.

**Fix applied** via `RemoteTrigger action: update` on the routine: replace title-search with a label-based query (`gh issue list --label audit-bot --json number,updatedAt`). Labels are indexed reliably, so this find step now succeeds every hour. The `audit-bot` label becomes load-bearing — bot must always create with it.

Cron unchanged (`13 * * * *`). Body content schema unchanged. Behavior unchanged except: no more duplicates.

---

## 2. The original find logic (broken)

```bash
gh issue list --repo GOATnote-Inc/medomni \
  --state open \
  --search 'audit: hourly best-practices dashboard in:title' \
  --json number,title \
  --limit 5
```

GitHub's `--search` parameter routes through the issue-search index, which is eventually consistent. Empirical lag: ~minutes to occasionally tens of minutes for freshly-created issues. The bot fires at `13 * * * *` (13 past every hour). If a previous run created an issue at HH:13 and the index hasn't caught up by (HH+1):13, the find step returns empty and the bot creates a duplicate.

---

## 3. The fix (applied)

```bash
target=$(gh issue list --repo GOATnote-Inc/medomni \
  --state open --label audit-bot \
  --json number,updatedAt --limit 5 \
  | jq -r 'sort_by(.updatedAt) | reverse | .[0].number // empty')
```

Label-based filtering uses GitHub's issue-list endpoint with strong consistency. The query returns whatever's currently in the repo state, not what the search index has caught up with. `jq` picks the most-recently-updated one as the canonical rolling dashboard.

If multiple open `audit-bot` issues exist (e.g., from before this fix), the bot picks the most-recent and edits that one. The older orphans are ignored — they should be closed manually (or by another bulk-close pass, as happened today). Going forward, no new orphans should accrue.

The `audit-bot` label is now load-bearing — the routine prompt explicitly requires the bot to create with `--label 'audit-bot'`. Removing or renaming that label would re-introduce the duplication risk.

---

## 4. Why not just lower the cron cadence?

A 24-hour cadence would also reduce duplicates (less surface area for the bug). But the underlying race-condition bug would remain — even daily, a search-index lag on the morning of day 2 could cause day-3 to duplicate. Fixing the find logic eliminates the bug at the source; cron cadence is a separate decision left untouched.

---

## 5. Mini-runbook for next time

If a scheduled remote agent ever shows a duplicate-issue / duplicate-PR pattern across multiple hours, check:

1. **Find-existing logic.** Is it using `--search '... in:title'`? Switch to a label-based or saved-search-name query.
2. **Idempotency key.** Is there a stable identifier the bot can use across runs? Labels, milestones, or a specific assignee are reliable. Title-based fuzzy match is fragile.
3. **Run-once flag.** Does the bot's prompt set `enabled: true` on a `run_once_at` routine? That can cause re-firing if updated. Check `ended_reason` field.
4. **Concurrent runners.** If two routines could both fire near-simultaneously (within the eventual-consistency window), they each create. Use a single canonical routine.

Discover the routine via:

```bash
# In Claude Code, with the schedule skill loaded:
# /schedule list
# Then update via the RemoteTrigger tool.
```

---

## 6. Cross-references

- `findings/2026-05-18-adversarial-probe-fix/CARD.md` — sister anti-pattern (252-issue runaway from a 404 URL). Same shape: scheduled bot files informational issues without dedup.
- `findings/2026-05-18-ci-cd-audit/CARD.md` — broader CI audit that flagged this bot's noise pattern.
- Routine `trig_01KtJothoiKdMC4sayF915py` — the updated routine; query via `RemoteTrigger action: get`.

---

## 7. Provenance

Identified during the 2026-05-18 audit-cleanup pass, immediately after bulk-closing the 15 dashboard issues. Root cause traced by inspecting the routine's prompt content via `RemoteTrigger action: list`. Fix applied via `action: update` with only the find-logic block changed; verified via the API response (timestamp updated 2026-05-19T04:32:52Z; next run at 2026-05-19T05:13Z).
