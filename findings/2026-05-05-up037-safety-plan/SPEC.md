# UP037 safety plan — agent-team review for forward-reference quote-strip

## Why this needs its own SPEC

ruff's `UP037` rule rewrites `def f(x: "Foo") -> "Bar":` to `def f(x: Foo) -> Bar:`. Under `from __future__ import annotations` (PEP 563), the original quoted form was deferred-evaluated as a string at runtime — meaning even if `Foo` was never imported or defined in the module, runtime never tripped on it. After UP037 strips the quotes, ruff's static analysis catches the now-bare reference as `F821 Undefined name`. The quoted form was sometimes hiding a latent bug.

**Verified case (iter-2 of the babysitter loop, 2026-05-05):**
`mla/prism/validator.py:70` had `invariants: "list[InvariantCheck] | None" = None`. `InvariantCheck` is referenced exactly once in the file and never imported or defined anywhere. UP037 stripped the quotes; F821 fired. Auto-fix was reverted. Saved as memory `feedback_up037_unmasks_f821.md`.

**Scope:** 20 UP037 sites across the medomni codebase. Verified by `ruff check . --select UP037` on commit `2afc291` (post-#52 main). The 1 known F821-bait site is among them; the other 19 are unknown — could all be safe, could include more latent bugs.

## Why agents (not just `--fix`)

- **Generator-Validator-Attacker pattern** (Anthropic agentic-engineering guidance, Jan-May 2026): high-stakes mechanical changes need a non-applicator to confirm correctness. Same pattern used for kernel correctness research, eval harness changes, anything where a single-pass auto-fix can hide a regression.
- **Verify-then-claim discipline** (medomni CLAUDE.md §4): "Every change ends with a verifying command. 'verified:' not 'done.'" Auto-fix in isolation has no per-site verifier.
- **Pilot before full sweep** (memory `feedback_pilot_before_full_sweep.md`): never ship a full sweep without sampling first. UP037 sampling means: per-site reasoning, not bulk apply.

## Agent-team architecture

Three roles, sequenced. Spawn each as a `general-purpose` subagent.

### Role 1 — Scout (`up037-scout`)

**Goal:** enumerate the 20 UP037 sites, capture context. Single agent, foreground.

**Inputs:** repo at HEAD; CI-pinned ruff version (`~/Library/Python/3.14/bin/ruff` 0.6.9).

**Output:** a JSON file at `findings/2026-05-05-up037-safety-plan/sites.json`:

```json
[
  {
    "id": 1,
    "file": "mla/prism/validator.py",
    "line": 70,
    "type_annotation_text": "list[InvariantCheck] | None",
    "referenced_names": ["InvariantCheck"],
    "function_or_class": "validate",
    "context_snippet": "def validate(...) -> ValidationResult: ..."
  },
  ...
]
```

For each site: parse the type annotation string, extract every ALL-CAPS-or-CamelCase identifier (heuristic for type names), record the surrounding function/class scope.

**Verification:** count of sites in `sites.json` matches `ruff check . --select UP037 | grep -c UP037`.

### Role 2 — Validator (`up037-validator`, parallel-eligible)

**Goal:** for each scout-recorded `referenced_name`, prove it IS defined or imported in the file's scope. If not — flag as latent bug.

**Inputs:** `sites.json`, the file at HEAD.

**Per-site procedure:**
1. `grep -n '\bSCOUT_NAME\b' <file>` — collect all occurrences in the same file
2. Filter occurrences inside the type annotation itself (line `<scout_line>`)
3. Of remaining occurrences: check whether any is an `import`, `from … import`, `class SCOUT_NAME`, or `SCOUT_NAME = …` definition
4. If found → site is **SAFE for UP037 fix**
5. If not found in this file → check the file's `__all__` re-export, then upstream imports the file does (e.g. `from .invariants import InvariantCheck` would put it in scope)
6. If STILL not found → site is **LATENT BUG**: do not fix UP037, instead either (a) add the missing import, (b) remove the bogus annotation, or (c) leave as-is with `# noqa: UP037` + a TODO comment

**Output:** `findings/2026-05-05-up037-safety-plan/validation.json` — per-site `{ id, status: "safe" | "latent_bug" | "ambiguous", evidence: [...], recommended_action: "fix" | "import" | "remove_annotation" | "noqa_with_todo" }`.

**Parallelism:** sites are independent. Spawn N validators where N = min(20, available agent budget). 4 parallel keeps cache-cost reasonable.

### Role 3 — Attacker (`up037-attacker`, foreground)

**Goal:** for each validator-marked "safe" site, attempt to construct a counter-example: a runtime path that would fail if the unquoted form is bad.

**Inputs:** `validation.json`, repo HEAD.

**Per-site procedure:**
1. Apply the UP037 fix to the single site (in-memory, not committed)
2. Run `python -c 'import <module>'` for the modified file
3. Run any relevant test that imports the modified file (`pytest --collect-only -q tests/<related> mla/tests/<related>`)
4. If either fails → flip status from "safe" to "ambiguous"; record the error
5. Run `~/.../ruff check <file> --select F` (look for new F-class errors after the fix)
6. If new F821/F722 → flip status to "latent_bug"

**Output:** `findings/2026-05-05-up037-safety-plan/attacker.json` — per-site verified status.

### Role 4 — Scribe (foreground, no agent)

I (the babysitter) write the PR. Per-site decision per the validator+attacker output:
- "safe" + attacker-passes → apply UP037 fix
- "latent_bug" → do NOT apply UP037; open a follow-up issue tracking the missing import or bogus annotation; add `# noqa: UP037` to keep ruff quiet
- "ambiguous" → defer; open a follow-up issue, add `# noqa: UP037`

Result: zero F821 surprises, every fix justified, every defer documented.

## Acceptance criteria

1. `findings/2026-05-05-up037-safety-plan/{sites,validation,attacker}.json` exist and reconcile (same site count, same IDs).
2. PR diff applies UP037 to ALL "safe + attacker-pass" sites and ONLY those sites.
3. Each non-fixed site has `# noqa: UP037` and a one-line comment naming the issue.
4. `ruff check . --select UP037` count strictly decreases (some non-zero count of `noqa`-protected sites is acceptable).
5. `ruff check . --select F` count does NOT increase post-fix.
6. `pytest --collect-only` collects same set of tests pre/post-fix.
7. CARD `findings/2026-05-05-up037-safety-plan/CARD.md` archives the per-site decisions.

## Cost estimate

- 1 scout (~3 min)
- 4 parallel validators (~5 min wall, 20 site-min total)
- 1 attacker (~10 min — runs Python imports per site)
- 1 scribe pass (~10 min)

Total: ~30 min wall. Cache cost: 4 parallel agents = 4 cache misses + scout/attacker fresh = ~6 misses for the campaign. Acceptable for a 20-site debt that includes at least 1 known latent bug.

## When to fire

Defer until F401 wedge (~59 sites, low-risk case-by-case) lands. F401 is the next safe-ish item; it might also expose latent F-class issues that the validators of this campaign would benefit from seeing first. Estimated firing: iter-13 or iter-14.

## Anthropic best-practice references applied

- **Generator-Validator-Attacker** pattern — explicit roles for proposer / checker / breaker.
- **Pilot before full sweep** — scout enumerates before any fix, validator confirms before any fix, attacker verifies before any fix.
- **Verify-then-claim** — every "safe" decision has evidence; every "latent_bug" defers to human follow-up.
- **Preserve work product** — outputs as JSON CARDs, not in-conversation; audit trail survives the loop wake-cycle cache miss.
- **One-substantive-commit-per-branch** (medomni-specific lesson `feedback_auto_merger_squash_race`) — UP037 fix lands in ONE PR with all evidence files; no follow-up commits to the same branch.

## What this plan explicitly does NOT do

- It does not auto-fix. The whole point is to NOT auto-fix without per-site evidence.
- It does not silence F821 latent bugs by adding `# noqa: F821`. Those become open issues for human review.
- It does not block on Anthropic API access. All four roles are general-purpose subagents; no special-tier access required.
- It does not modify any non-UP037 lint rule. Scope is exactly UP037.
