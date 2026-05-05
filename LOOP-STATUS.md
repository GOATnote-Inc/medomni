# Loop Status — medomni

Persistent loop agent status board. The loop fires every ~15 min and appends an iteration entry below. Read top-down; oldest at the bottom.

## Charter (do not skip)

Per user directive 2026-05-04: persistent loop, 15 min cadence, runs as full-time project babysitter.

Every iteration:
1. **Check state** — `git status`, `git log -5`, `gh pr list`, `gh run list -L 5`, scan for failing CI, untriaged PRs, regressed deploys.
2. **Act** — fix flaky tests, rebase open PRs, address review comments, update docs, ship small wins. Verify before pushing (build, tsc, smoke).
3. **Report** — append a one-paragraph entry to this file with what you did. Only escalate to user via terminal/Slack if blocked or ambiguous.
4. **Self-improve** — when you spot a recurring mistake, write a feedback memory and update MEMORY.md, then carry on.

Cache cost note: 15 min cadence forces a cache miss every wake (5 min TTL). User chose this; respect it, but be efficient — don't fan out into expensive multi-agent searches each iteration.

## Hard rules (from CLAUDE.md and user directive)

- Prefer simple, reliable fixes. No speculative refactors.
- Never break existing functionality. Verify end-to-end before pushing.
- Honor the §1 isolation contract: never touch prism42 prod surface, ElevenLabs, LiveKit, DNS, or `.vercel/` config.
- No cloud LLM keys. Sovereignty by construction.
- Stage by name. Never `git add -A` or `.`.
- Author email: `b@thegoatnote.com`. One Co-Authored-By per commit.

## Iteration log (newest first)

### iter-0 · 2026-05-04 22:18 PT — bootstrap
State: branch `spike/personalized-records-pattern-b` at `08b3a32`, pushed clean to origin, no open PRs, no failing CI. Just shipped real CC0 chest-X-ray imagery + ImagingPanel B-agent files (which had been sitting uncommitted in the B worktree — root cause now in `feedback_check_worktree_status_before_cherrypick.md`).

Actions this iteration: created this status board, saved two memory files (loop protocol + worktree lesson), scheduled next wake at +900s.

Next: iter-1 will check for upstream changes, scan worktrees for any other uncommitted agent work, and validate the live deploy still serves the imaging panel.
