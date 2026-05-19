# Bounty-bot spam incident — `arthurus36-alt` + branch-protection gap

**Status:** CARD (incident closed, defenses applied)
**Date:** 2026-05-18
**Author:** Brandon Dent, MD (b@thegoatnote.com)

---

## 1. TL;DR

While we were bulk-closing 252 stale `adversarial-probe` issues (see `findings/2026-05-18-adversarial-probe-fix/CARD.md`), an external GitHub account `arthurus36-alt` commented on the just-closed issue #394 offering to "pick this one up". Investigation confirmed it as a bounty-farming / spam-comment bot, not a targeted supply-chain attacker. The bigger finding surfaced during the investigation: medomni's `main` branch had **zero required status checks and zero required PR reviews** — the spam was the noise that made us look, the missing branch protection was the actual risk.

Three defensive actions taken (all reversible, none affecting Brandon's solo-dev workflow):
- Spam comment minimized as SPAM via the GraphQL `minimizeComment` mutation.
- Issue #394 locked with `active_lock_reason: spam` so further bot comments can't accumulate.
- Branch protection applied to `main` mirroring scribegoat2's solo-dev pattern: 4 required status checks (`lint`, `secrets-scan`, `unit`, `manifest-determinism`), `strict: true`, no PR-review requirement, `enforce_admins: false` (admin bypass preserved), no force-push, no deletions.

Bonus prerequisite fix: `tests/test_adversarial_probe.py` (shipped in d5b996b) needs `requests` at import time; the `unit` job's pip install line didn't include it, so the job has been red on every push since. One-word fix in `test.yml`, validated green post-push.

---

## 2. Threat profile — `arthurus36-alt` and the bot family

### 2.1 Account fingerprint

```
login:        arthurus36-alt
created_at:   2025-05-08T17:35:43Z  (~1 yr old)
id:           210898718
followers:    1
following:    0
public_repos: 29
name:         arthurus36
bio:          null
email:        null
company:      null
```

Sister account `arthurus36` (no -alt) also exists, created **2023-05-22**, 20 public repos, 1 follower, no metadata. Two-account `-alt` suffix pattern is a deliberately segregated alt — used when the main got rate-limited / flagged / banned, or to launder activity across accounts. The exact-two-years-apart creation dates suggest scheduled rotation, not coincidence.

All 29 of `arthurus36-alt`'s public repos were created on the day of the comment (**2026-05-18**) as forks of various bounty programs (`screeps-bounty-arena`, `rustchain-bounties`, `FreeCAD-Documentation-Project`, `Bounty-Hunters`, `e-skimming-labs`, `mergefund-hackathon-kit`, `claude-builders-bounty`, ...). Pure cargo-culting to look legitimate to autoreviewers.

### 2.2 Activity pattern

30+ issue comments today across topically unrelated repos in a single burst:

```
bounty programs:    Rustchain, rustchain-bounties, claude-builders-bounty,
                    Bounty-Hunters, daydreamsai/agent-bounties, INDIGOAZUL/la-tanda-web
agent / AI repos:   GOATnote-Inc/medomni, moorcheh-ai/memanto, deep-foundation/deep-packages,
                    baserow/baserow, Glavin001/PeakProgrammer
crypto / trading:   singlesly/bingx-api, jiezishu000/empire-digital-shop,
                    jiezishu000/Jiezishu000, tari-project/wallet-benchmarks
Minecraft mods:     GlowstoneMC/1.13-board, WeaponMechanics/WeaponMechanics
docs / misc:        Reqrefusion/FreeCAD-Documentation-Project (8 separate issues),
                    SecureBananaLabs/bug-bounty (3 separate issues)
```

Hallmark: generic helpful-sounding comments with no domain-specific content. The comment on #394 — "check the latest prompt changes or tool wiring first" — could be pasted into any agent-regression issue verbatim.

### 2.3 Attack model (Hormozi/Munger-style inversion: what's the bot's actual ROI?)

The play is the **bounty / merge-credit farm**:

1. Bot crawls public GitHub event feed for repos with high-velocity activity (the 252-issue bulk close generated a measurable signature).
2. Comments on a recent, ideally closed, issue offering to "pick this up".
3. If the maintainer engages ("yes please"), submits a low-quality / LLM-generated / copy-paste PR.
4. If merged: collects bounty credit (where applicable), commit author credit (resume padding), or — worst case — drops a supply-chain payload into the repo (the xz-utils playbook).
5. If ignored: zero cost, moves on.

The bot does not need to succeed often. Maintainers occasionally say yes out of guilt or speed pressure. The aggregate yield is non-zero across thousands of repos.

Engagement is the only way to lose. **Silence is the correct reply.** Even "thanks but no" confirms a human is watching the repo, which improves the bot's targeting score.

### 2.4 Cross-org scope

```
gh search issues "commenter:arthurus36-alt org:GOATnote-Inc"
→ medomni#394 only
```

Confined to the one repo with high-visibility recent activity. Not a coordinated campaign against the org. Other 9 GOATnote-Inc public repos are clean.

---

## 3. The bigger finding — `main` was under-protected vs the rest of the org

Pre-incident state of `repos/GOATnote-Inc/medomni/branches/main/protection`:

```json
{
  "allow_deletions": false,
  "allow_force_pushes": false,
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "required_status_checks": {"contexts": [], "strict": null}
}
```

Compared to the 2026-03-06 audit recorded in `~/.claude/projects/-Users-kiteboard/memory/MEMORY.md → Branch Protection`:

> "All 5 repos: main branch protected. Required status checks + no force push + no deletions.
> SG2: 20 required checks. LB: 5. OpenEM: 2. SS: 3. RS: 4."

medomni was created **after** that audit (per CLAUDE.md provenance line: squash-import from prism42 on 2026-04-28). The protection rules from the original audit never got applied to it. Result: every CI workflow (lint, test, adversarial-probe, agent-pr-review, clinical-skill-review) was *running* but *advisory only*. A PR with red CI could be merged. A spam-bot PR with red CI could be merged.

This is the real attack surface the bot was probing for, even if accidentally.

### 3.1 Defense applied

`gh api -X PUT repos/GOATnote-Inc/medomni/branches/main/protection` with body:

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["lint", "secrets-scan", "unit", "manifest-determinism"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false,
  "required_conversation_resolution": false,
  "lock_branch": false,
  "allow_fork_syncing": true
}
```

Design choices, in order of priority:

- **No PR review requirement** (`required_pull_request_reviews: null`). Brandon is the only maintainer; requiring a second reviewer would block his own work. Mirrors scribegoat2's setting.
- **Strict status checks** (`strict: true`). PR branch must be up-to-date with main before merge — closes the "merge-conflict window" where an attacker submits a PR against an old base that doesn't contain a recently-added safety check.
- **Four required contexts** (`lint`, `secrets-scan`, `unit`, `manifest-determinism`). These are the names GitHub actually sees per `repos/.../commits/main/check-runs`. Not included: `probe` (schedule-only, not PR/push triggered).
- **Admin bypass preserved** (`enforce_admins: false`). Brandon can still push directly to main as admin. This is the documented solo-dev pattern; it sacrifices belt-and-suspenders for not blocking the founder's own velocity.

### 3.2 Prerequisite fix shipped before applying protection

`tests/test_adversarial_probe.py` (in commit d5b996b) imports the probe module, which imports `requests`. The `unit` job's `pip install` line did not include `requests`, so collection failed with `ModuleNotFoundError`. The `unit` check has been red on every push since d5b996b. Applying branch protection that required `unit` to pass would have blocked all future PRs until that fix landed.

Fix: commit f329803 — one word added (`requests`) to `.github/workflows/test.yml`'s install line. Validated green: unit + manifest-determinism both pass on f329803.

---

## 4. Reproducible defense recipe (for the next time)

Save this as a runbook. The full sequence took ~20 min including investigation.

```bash
# 1. Fingerprint the account
gh api users/<login> --jq '{login, created_at, followers, following, public_repos, name, bio, email}'

# 2. Find sister accounts (common pattern: login + "-alt" or "-2")
for suf in "" "-alt" "-alt2" "-2" "-bot"; do
  gh api "users/<basename>${suf}" --jq '{login, created_at}' 2>/dev/null
done

# 3. Activity burst check (today's events, all repos)
gh api "users/<login>/events/public?per_page=30" \
  --jq '.[] | [.created_at[:10], .type, .repo.name] | @tsv'

# 4. Cross-org search for this commenter
gh search issues "commenter:<login> org:<org>" --json repository,number --jq 'length'

# 5. Minimize the comment as spam (GraphQL)
COMMENT_NODE_ID=$(gh api repos/OWNER/REPO/issues/N/comments \
  --jq '.[] | select(.user.login == "<login>") | .node_id')
gh api graphql \
  -f query='mutation($id:ID!){minimizeComment(input:{subjectId:$id, classifier:SPAM}){minimizedComment{isMinimized minimizedReason}}}' \
  -f id="$COMMENT_NODE_ID"

# 6. Lock the issue thread
gh api -X PUT repos/OWNER/REPO/issues/N/lock -f lock_reason=spam

# 7. Verify
gh api repos/OWNER/REPO/issues/N --jq '{number, state, locked, active_lock_reason}'
```

For branch protection sweep:

```bash
# Diff each public repo's main protection against a template
for r in $(gh repo list <org> --visibility=public --limit 30 --json name --jq '.[].name'); do
  echo "=== $r ==="
  gh api "repos/<org>/$r/branches/main/protection" --jq \
    '{enforce_admins: .enforce_admins.enabled, contexts: (.required_status_checks.contexts | length), strict: .required_status_checks.strict, pr_reviews: (.required_pull_request_reviews | if . then .required_approving_review_count else null end)}' 2>/dev/null || echo "no protection"
done
```

---

## 5. Items NOT taken (and why)

- **Did not respond to the comment.** Engaging confirms a human watches the repo and raises the bot's targeting score.
- **Did not report the user to GitHub Trust & Safety.** Threshold for action is unclear and the cross-repo pattern, while visible, is on the spammy-but-not-malicious side of the line. Reserve T&S reports for credential phishing, malicious PRs with payloads, or coordinated cross-account abuse.
- **Did not block the user at repo or personal level.** They have no commit access to revoke. Personal block hides their content from Brandon's UI but doesn't prevent them from commenting on the repo — minimize + lock is the actual mitigation.
- **Did not require PR review approvals.** Solo-dev pattern; would block Brandon's own work.
- **Did not require signed commits.** Would block agent-driven commits in this session and others. Defer until commit signing is configured globally.

---

## 6. Sweep follow-up (separate ticket, low priority)

Compare branch protection settings across all 10 GOATnote-Inc public repos against this medomni baseline. Per `MEMORY.md → Branch Protection (2026-03-06)`, 5 of them already had protection — re-verify they still do, and that the contexts list is up to date with current workflows. Quick diff loop in §4 above.

---

## 7. Provenance

Triggered by Brandon's manual screenshot review of the `arthurus36-alt` comment on issue #394 immediately after the 252-issue bulk close. Investigation surfaced the bot pattern (low risk) and the branch-protection gap (medium risk). All three defensive actions (minimize, lock, protection apply) applied within ~20 min of detection. Prerequisite `unit` fix shipped at commit f329803 before applying the `unit` status-check requirement.
