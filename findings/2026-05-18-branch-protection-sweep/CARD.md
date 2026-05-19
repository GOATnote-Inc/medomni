# Branch-protection sweep — 9 GOATnote-Inc public repos

**Status:** CARD (sweep complete; 3 repos protected, 1 deferred to founder, 5 audited clean, 1 flagged for closer review)
**Date:** 2026-05-18
**Author:** Brandon Dent, MD (b@thegoatnote.com)

---

## 1. TL;DR

Follow-on to `findings/2026-05-18-bounty-bot-spam-incident/CARD.md` §6. That CARD discovered medomni's `main` had zero required status checks (fixed in-line during the bot-spam incident). This sweep audited the other 9 active GOATnote-Inc public repos. Three more were entirely unprotected and got the standard solo-dev protection applied this cycle (healthcraft, receipts, medimage-corpus). One was intentionally lifted by Brandon for a 48h sprint on 2026-04-23 and never restored — left for explicit founder decision rather than auto-restored (prism42). Five repos audited as already-clean (lostbench, openem-corpus, radslice, safeshift, scribegoat2). One has a complex 20-context list that warrants closer human review before any change (scribegoat2).

Net result: of 10 active first-party public repos, **9 now have at least floor protection** (no force-push + no deletions). 8 of 10 have required status checks. 1 (prism42) flagged for founder action.

---

## 2. Pre-sweep state

```
repo                default   status                                 contexts
medomni             main      NONE → 4 contexts (fixed in bot CARD)  lint,secrets-scan,unit,manifest-determinism
healthcraft         main      NONE → 4 contexts (this CARD)          lint,test (3.10),test (3.12),docker-build
receipts            master    NONE → 1 context (this CARD)           test
medimage-corpus     main      NONE → floor only (this CARD)          (none — manifest-validate is path-filtered)
prism42             main      NONE → deferred for founder            —
lostbench           main      OK (5 contexts, all current)           lint,test (3.10),test (3.12),test (3.13),package
scribegoat2         main      OK by header; 20 contexts deserve a    (see §6)
                              closer look before any change
openem-corpus       main      OK (2 contexts, all current)           lint,Run test suite
radslice            main      OK (4 contexts, all current)           test (3.10),test (3.12),test (3.13),smoke
safeshift           main      OK (3 contexts, all current)           test (3.10),test (3.12),test (3.13)
```

---

## 3. Standard solo-dev protection shape (applied to all three this cycle)

Mirrors scribegoat2 / lostbench / openem-corpus / radslice / safeshift — the pattern Brandon already chose in the 2026-03-06 audit. JSON body, with `contexts` substituted per repo:

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["..."]
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

Design choices (unchanged from the medomni precedent):

- **No PR review requirement** (`required_pull_request_reviews: null`). Solo dev — a second-reviewer requirement would block Brandon's own merges.
- **Strict status checks** (`strict: true`). PR branch must be up-to-date with main — closes the "merge-conflict window" attack.
- **Admin bypass preserved** (`enforce_admins: false`). Brandon can still push directly as admin. Sacrifices belt-and-suspenders for founder velocity.
- **No force-push, no deletions.** History integrity floor. Cheap; no workflow impact.

For repos with no required status checks (medimage-corpus), `required_status_checks` is `null` — the floor (no force-push + no deletions) still applies, just no CI gate.

---

## 4. Changes applied this cycle

### 4.1 healthcraft (main)

- **Pre-sweep**: no protection.
- **Required contexts applied**: `lint`, `test (3.10)`, `test (3.12)`, `docker-build`.
- Source: actual check names returned by `repos/GOATnote-Inc/healthcraft/commits/main/check-runs` — all four were green on HEAD.
- Workflows present (`integration.yml`, `whitepaper.yml`) did not run on HEAD; not added as required because they're conditional. They'll run on relevant PRs and report status normally; just won't gate merge.

### 4.2 receipts (master)

- **Pre-sweep**: no protection. Default branch is `master`, not `main` — verified before applying.
- **Required contexts applied**: `test`.
- Single workflow, single job. Same `test.yml` that this morning's fix wired up (changed `branches: [main]` → `branches: [master]` in commit 2a4e0fb).

### 4.3 medimage-corpus (main)

- **Pre-sweep**: no protection.
- **Required contexts applied**: NONE (`required_status_checks: null`).
- Reason: the only workflow (`manifest-validate.yml`) is path-filtered to `manifests/**`, `schemas/**`, `scripts/manifest/**`, `pyproject.toml`. No recent push matched, so there's no current check name to require. Adding any context that does not run on a PR would block that PR forever.
- The floor still applies: `allow_force_pushes: false`, `allow_deletions: false`, `enforce_admins: false`. This is enough hygiene for a data-registry repo with no test surface. Add required contexts later if a non-path-filtered workflow is added.

---

## 5. Deferred for founder action — prism42 (main)

Per `~/.claude/projects/-Users-kiteboard/memory/MEMORY.md`:

> **TEMPORARY (2026-04-23 to 2026-04-25):** Prism protection lifted for 48h solo-dev sprint — see `project_prism_branch_protection_lifted.md`. Other 4 repos unchanged.

The 48h window closed on 2026-04-25. Today is 2026-05-18 — 23 days past the intended restoration date — and protection is still NONE on `main`. This sweep does **not** auto-restore because:

1. The memory note records an intentional lift, not an accident. Restoring without checking might break an in-flight pattern Brandon has been relying on (e.g., scripted automation that pushes directly to main as a non-admin token).
2. prism42 has stricter file-edit isolation rules in medomni's CLAUDE.md §1 — touching its workflows or settings without explicit sign-off carries higher cost than the other repos.
3. The audited HEAD on prism42 shows 4 check names — `prism42 cleanliness check`, `Tests (pytest)`, `unit (parser sanity)` all green, plus one skipped `integration (pod SSH + bench)`. Those would be the natural required contexts when restoring.

**Action for founder.** Run, when ready:

```bash
cat > /tmp/prism_prot.json <<'JSON'
{"required_status_checks":{"strict":true,"contexts":["prism42 cleanliness check","Tests (pytest)","unit (parser sanity)"]},"enforce_admins":false,"required_pull_request_reviews":null,"restrictions":null,"allow_force_pushes":false,"allow_deletions":false,"required_linear_history":false,"required_conversation_resolution":false,"lock_branch":false,"allow_fork_syncing":true}
JSON
gh api -X PUT repos/GOATnote-Inc/prism42/branches/main/protection --input /tmp/prism_prot.json
```

If sprint mode is still active, leave it.

---

## 6. Flagged for closer review — scribegoat2 20-context list

scribegoat2's current required contexts list has **20 entries**, of which only 6 produced check runs on the current main HEAD. The other 14 are likely path-filtered to specific paths (per `MEMORY.md → ScribeGoat2`: *"Path-filtered checks (evaluation-safety.yml, evaluation-integrity.yml on SG2; quality-gate/audit/review-gate/validate on OpenEM) NOT required — they'd block unrelated PRs."*).

But the actual `contexts` list **does** include several names that look path-filtered (`Evaluation Integrity Gates`, `Epistemic Integrity Gates`, `Safety Integrity Gates`, `Operational Integrity Gates`, `Review Readiness Gates`, `PHI Detection (Healthcare Data Safety)`, `validate-readme-links`, `validate-schemas`, `validate-skill-contracts`, etc.). If any of these are actually path-filtered AND required, a PR that doesn't touch the relevant paths would block indefinitely.

The 20 contexts also include `lint` (verified to exist), `Detect Secrets` (verified), `Gitleaks Secret Scan` (verified), `TruffleHog Deep Scan` (verified), `Custom Pattern Check` (verified), plus `test (3.11)` and `test (3.13)` (which may or may not exist on a given PR depending on matrix config drift).

**Recommendation.** Founder runs:

```bash
# 1. Open a no-op PR to scribegoat2 main and observe which checks fire.
# 2. For each required context that doesn't fire, decide:
#    - Workflow exists but path-filtered? Remove from required, set up trigger that runs always (or accept the block-on-trivial-PR cost).
#    - Workflow deleted/renamed? Remove from required.
# 3. PUT updated contexts list.
```

**Not auto-modified.** Removing required contexts is a security relaxation. Want explicit founder review per context before changes.

---

## 7. Audited clean — no change needed (4 repos)

| repo | default | contexts (all verified present in recent runs) |
|---|---|---|
| lostbench | main | `lint`, `test (3.10)`, `test (3.12)`, `test (3.13)`, `package` |
| openem-corpus | main | `lint`, `Run test suite` |
| radslice | main | `test (3.10)`, `test (3.12)`, `test (3.13)`, `smoke` |
| safeshift | main | `test (3.10)`, `test (3.12)`, `test (3.13)` |

All four have `strict: true`, `enforce_admins: false`, `pr_reviews: null`, `force_push: false`, `deletions: false`. Matches the standard solo-dev shape exactly. No drift since the 2026-03-06 audit.

---

## 8. Post-sweep state (full org)

```
repo                default   status
medomni             main      OK (4 contexts) — fixed earlier today in bot CARD
healthcraft         main      OK (4 contexts) — this CARD
receipts            master    OK (1 context) — this CARD
medimage-corpus     main      OK (floor only) — this CARD
prism42             main      DEFERRED for founder
lostbench           main      OK (5 contexts) — audited clean
scribegoat2         main      OK (20 contexts) — flagged for closer review
openem-corpus       main      OK (2 contexts) — audited clean
radslice            main      OK (4 contexts) — audited clean
safeshift           main      OK (3 contexts) — audited clean
```

8 of 10 fully protected with required status checks. 1 (medimage-corpus) on floor protection. 1 (prism42) awaiting founder decision.

---

## 9. Reproducible audit recipe

```bash
# 1. Enumerate active public repos
gh repo list GOATnote-Inc --visibility=public --limit 30 --json name,defaultBranchRef,isArchived,isFork \
  --jq '.[] | select(.isArchived == false and .isFork == false) | "\(.name)\t\(.defaultBranchRef.name)"'

# 2. Per repo, fetch protection + actual recent check-runs side by side
for r in <list>; do
  default=$(gh repo view GOATnote-Inc/$r --json defaultBranchRef --jq .defaultBranchRef.name)
  echo "=== $r ==="
  gh api "repos/GOATnote-Inc/$r/branches/$default/protection" \
    --jq '{contexts: .required_status_checks.contexts, strict: .required_status_checks.strict}' 2>/dev/null
  gh api "repos/GOATnote-Inc/$r/commits/$default/check-runs" \
    --jq '.check_runs[] | .name' 2>/dev/null | sort -u
done

# 3. For each gap: PUT protection with the standard JSON body in §3, substituting `contexts` to match
#    the currently-passing check names from step 2.
```

Re-run quarterly. Any repo where the `contexts` list and the actual check-run names diverge is a drift event worth investigating.

---

## 10. Provenance

Triggered by §6 of `findings/2026-05-18-bounty-bot-spam-incident/CARD.md`. Builds on the prior medomni protection apply (in that same CARD) plus the 2026-03-06 audit recorded in `MEMORY.md → Branch Protection`. Verified by re-querying each repo's `branches/.../protection` post-apply; all three new applies returned the expected JSON shape.
