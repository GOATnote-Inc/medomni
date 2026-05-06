# Clinical-skill-review CI gate — auto-Claude review on clinical-content PRs

**Status:** SHIPPED 2026-05-05. Track #3 of the 5-track Cherny-pattern initiative.
**Trigger:** Iter request — "build a GitHub Action that runs an automated clinical-safety review by Claude on every PR that touches clinical-content paths."
**Boris Cherny lens:** [Building Claude Code with Boris Cherny](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny). Cherny reports 100% of Anthropic PRs are reviewed by Claude before human review. We adopt that discipline scoped to clinical-content paths in medomni.

## What this PR ships

- **`.github/workflows/clinical-skill-review.yml`** (69 lines). Triggers on `pull_request` (opened/reopened/synchronize) when any of the trigger paths change. Runs on `ubuntu-latest`, 6-min timeout, installs `anthropic` SDK only.
- **`scripts/clinical_skill_review.py`** (205 lines). Pulls the clinical-only diff via `git diff origin/<base>...HEAD`, sends it to `claude-opus-4-7` with the rubric, parses a fenced JSON verdict, posts a review comment via `gh pr comment`, exits 1 on BLOCK.

## Trigger paths (clinical-content only)

The workflow ONLY fires on PRs that change one of:

- `web/lib/agent/skills/**/*.md` — runtime skill markdown (PR #95)
- `mvp/medomni-inference/skills/**/*.md` — canonical skill authoring location
- `mvp/medomni-inference/system_prompt_v1.md` — V_final base prompt
- `corpus/pins/**` — pinned eval fixtures (treated as clinical content)

Routine code/docs PRs are NOT touched. The existing `agent-pr-review.yml` (safety-engineer) continues to handle the broader sovereignty/secrets review on every PR.

## Rubric

Five clinical-safety dimensions, applied in order:

1. **Hallucination risk** — confident clinical claim (dose, threshold, criteria) not verifiable from a source visible in the diff
2. **Tool-call ambiguity** — model-facing instructions about when/how to call a tool that are not unambiguous
3. **Safety-rule contradiction** — diff weakens or contradicts an existing safety rule (refusal, abstain, escalation)
4. **PHI guardrails** — flow added where user PHI could plausibly be logged/exfiltrated/persisted without an explicit guard
5. **Unverified citations** — guideline citation (ACC/AHA, NICE, UpToDate, etc.) without a verifiable identifier

Verdicts: `PASS` (merge OK), `FLAG` (merge OK + follow-up), `BLOCK` (exit 1 — branch protection blocks merge).

## BLOCK gate

When the model returns `BLOCK`:

1. Review comment is posted with the verdict, summary, findings table, and rationale.
2. Script exits non-zero. The workflow run is marked failed.
3. If the workflow is added to the required-status-checks set in branch protection (TODO post-merge), GitHub blocks the merge button until a fixed commit makes the check go green.

PASS and FLAG exit 0 — merge is unaffected; the comment is the audit trail.

## Secrets

Reuses existing repo secret `ANTHROPIC_API_KEY_PR_REVIEW` (same key the `safety-engineer-review` workflow uses). No new secrets added. If the secret is unset, the step emits a `::warning::` and exits 0 — degraded-mode rather than wedged.

## Cost estimate

~$0.10 per review (clinical-only diff truncated at 60k chars; ~2k output tokens). Triggers fire only on clinical-content PR opens/syncs, so volume is low — typically <10/week.

## Follow-ups (not in this PR)

- **Add to required-status-checks.** Once the gate has run green for ~10 PRs, add `clinical-skill-review` to branch protection so BLOCK actually blocks the merge UI (not just turns the check red). Until then, BLOCK is advisory but visible.
- **Tighten rubric per false-positive review.** First few BLOCKs will inform whether the rubric needs softening (e.g. allow internal pin descriptions to ship without a guideline citation).
- **Track agreement.** Compare verdicts vs eventual physician adjudication on the same diff (the same pattern that exposed v8 judge-hallucination κ=0.402 in healthcraft).
