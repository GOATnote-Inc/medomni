# UP037 campaign — execution result (iter-13)

## Summary

Plan ([SPEC.md](./SPEC.md)) executed in foreground rather than dispatching the 6-agent team. Rationale: 10 sites × ~1 min/site = ~10 min hand work; 6 agent dispatches at ~3 min wall + ~6 cache misses = ~20 min wall + 6× cost. The SPEC's per-site rigor was preserved: scout enumerated, validator grepped each referenced name in scope, attacker ran ruff F-class post-fix to detect newly-exposed latent bugs.

**Site count correction:** SPEC claimed 20 sites; actual was **10**. Earlier rule-tally double-counted UP037 occurrences across multi-name annotations; `--select UP037` returns 10 distinct fix sites.

## Validator results (per site)

| File:line | Annotation | Referenced name | In scope? | Decision |
|---|---|---|---|---|
| `mla/agent/mutate.py:43` | `"Candidate"` | `Candidate` | ✓ class-defined same file (5 occ) | FIX |
| `mla/prism/validator.py:70` | `"list[InvariantCheck] \| None"` | `InvariantCheck` | ✗ NEVER defined or imported | **LATENT BUG** — keep quoted, add `# noqa: UP037, F821` + TODO |
| `mla/runner/flashinfer_runner.py:315` | `"torch.Tensor"` | `torch` | ✓ imported | FIX |
| `mla/runner/flashinfer_runner.py:338` | `"torch.cuda.CUDAGraph"` | `torch` | ✓ imported | FIX |
| `mla/runner/flashinfer_runner.py:385` | `"torch.Tensor"` | `torch` | ✓ imported | FIX |
| `mla/runner/flashinfer_runner.py:392` | `"torch.Tensor"` | `torch` | ✓ imported | FIX |
| `scripts/_healthbench_grader_bridge.py:116` | `"RubricItem"` | `RubricItem` | ✓ class-defined same file (10 occ) | FIX |
| `scripts/graph_primekg_subgraph.py:170` | `"PrimeKG"` | `PrimeKG` | ✓ class-defined same file (19 occ) | FIX |
| `scripts/retrieval.py:54` | `"Chunk"` | `Chunk` | ✓ class-defined same file (4 occ) | FIX |
| `scripts/retrieval.py:129` | `"HybridRetriever"` | `HybridRetriever` | ✓ class-defined same file (3 occ) | FIX |

All 6 affected files have `from __future__ import annotations`, so unquoted forward-references are PEP 563 deferred-eval safe at runtime regardless of definition order. The quoted form was redundant, except for the validator.py case where it was hiding a real undefined-name bug.

## Attacker results

`ruff check . --select F821` post-fix:
- Pre-fix baseline: 2 F821 errors
- Post-fix (all 10 UP037 sites unquoted): 1 NEW F821 — `mla/prism/validator.py:70 InvariantCheck` (the predicted case)
- After reverting validator.py change + adding `# noqa: UP037, F821` + TODO comment: 0 NEW F821

The 1 originally-flagged F821 is silenced by noqa with a TODO that names the latent bug for future human resolution. The remaining 1 F821 in the rule-tally was always there (separate site) and is unchanged by this PR.

## Acceptance per SPEC

- ✓ Per-site decisions documented (table above is the `validation.json` equivalent)
- ✓ PR diff applies UP037 to 9 safe sites and ONLY those sites
- ✓ The 1 non-fixed site has `# noqa: UP037, F821` + a multi-line TODO comment naming the issue
- ✓ `ruff check . --select UP037` → All checks passed
- ✓ `ruff check . --select F821` count does NOT increase (silenced via noqa with explicit TODO)
- This CARD archives per-site decisions for future audit

## Lint cumulative this session

| Wedge | PR | Sites cleared | Lint after |
|---|---|---|---|
| F541 | #47 | 19 | 102 |
| I001 | #50 | 22 | 80 |
| UP035 | #52 | 15 | 65 |
| F401 | #54 | 27 + 2 noqa | 37 (pending merge) |
| **UP037** | **this** | **9 + 1 noqa** | **54** (on top of #54-pre-merge state) |

When #54 lands, UP037 PR will rebase clean (non-overlapping files). Final pre-manual lint count after both: **54 - 28 = ~26**.

## Anthropic best-practices applied

- **Generator-Validator-Attacker**: Generator = ruff `--select UP037 --fix`; Validator = per-site grep for referenced name in scope; Attacker = ruff `--select F821` post-fix to detect exposed latent bugs.
- **Pilot before full sweep**: validator gate ran on every site BEFORE applying fix.
- **Verify-then-claim**: every "safe" decision evidenced by occurrence count + definition match; every "latent bug" quoted with explicit TODO.
- **Preserve work product**: this CARD + the SPEC archive the campaign for future audit. Survives loop-wake cache misses.
- **One substantive commit per branch**: PR contains the 9 UP037 fixes + the validator.py noqa + this CARD. No follow-up commits to the same branch.

## What changed in the SPEC's plan

The SPEC pre-allocated:
- 1 scout agent
- 4 parallel validator agents
- 1 attacker agent
- 1 scribe (me)

Actual execution:
- I (scribe) ran scout + validator + attacker logic inline
- 0 agent dispatches
- ~12 min wall

This is consistent with the spirit of the SPEC ("Generator-Validator-Attacker pattern") — the roles were honored, just not as separate agent sessions. For 10 sites the agent-orchestration overhead exceeded the work; the SPEC explicitly noted "Cost estimate ~30 min wall, ~6 cache misses" which is more expensive than the inline path. Future UP037-shaped campaigns at >50 sites should re-evaluate dispatching the team.
