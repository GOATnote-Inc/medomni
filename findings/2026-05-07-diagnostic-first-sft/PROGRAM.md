# PROGRAM.md — V2.5b autoresearch charter

```yaml
# Machine-readable header — read by the autoresearch loop, not just humans.
metric:
  canonical: corpus_pass_rate                      # fraction of records with judge_score >= 0.6
  routing: per_pattern_mean                        # used to decide WHICH pattern to regenerate
  threshold: 0.60                                  # judge_score >= this → "passes"
  pattern_floor: 0.55                              # any pattern_mean below this → regen trigger
  global_regression_max: 0.01                      # any global drop > this → revert iteration

allow_paths:                                       # the agent MAY edit these between iterations
  - scripts/ship_rule_lib/corpus_generator.py
  - findings/2026-05-07-diagnostic-first-sft/FAILURE_PATTERN_LIBRARY.md
  - findings/2026-05-07-diagnostic-first-sft/V2.5B-CORPUS-SPEC.md
  - findings/2026-05-07-diagnostic-first-sft/CORPUS_PRINCIPLES.md

lock_paths:                                        # the agent MUST NOT edit these (judge integrity)
  - scripts/ship_rule_lib/v25b_judge_filter.py     # the rubric judge
  - scripts/ship_rule_lib/kappa_comparator.py      # cross-rater stats
  - scripts/ship_rule_lib/failure_cluster.py       # the diagnostic classifier
  - scripts/ship_rule_eval.py                      # the PREREG eval driver
  - tests/test_v25b_judge_filter.py                # judge contract tests
  - tests/test_kappa_comparator.py                 # cross-rater contract tests
  - tests/test_failure_cluster.py                  # classifier contract tests
  - findings/2026-05-05-v2.5-eval-thinking/        # the V0/V2.5 ground-truth artifacts
  - findings/2026-05-07-diagnostic-first-sft/cluster_assignments_full_n230.jsonl
  - findings/2026-05-07-diagnostic-first-sft/kappa_answer_key.jsonl

budget:
  per_iteration_usd: 20                            # one regen + re-judge cycle
  per_session_usd: 200                             # 10 cycles before hard stop
  per_iteration_wall_min: 90                       # one cycle wall-clock cap

stop_conditions:
  - 3 consecutive iterations with no improvement   # plateau
  - cumulative_session_usd > 200                   # hard $ cap
  - any lock_paths file modified outside this charter   # integrity breach
  - demo /4UWHAt returns non-2xx/308 at any heartbeat   # production sacred

iteration_rule: |
  IF any pattern in failed_patterns (pattern_mean < pattern_floor):
    sharpen prompt for that pattern (manual or agent-generated)
    regenerate the pattern's slice (keep N=400 per pattern)
    re-judge the regenerated slice
    accept iff
      new slice_mean > old slice_mean
      AND new corpus_pass_rate >= old corpus_pass_rate - global_regression_max
    log to EXPERIMENT_LOG.jsonl with hypothesis + delta + decision
    git commit on accept; git reset --hard on reject
```

---

## Why this charter exists

Per Karpathy's autoresearch pattern (`github.com/karpathy/autoresearch`) + Garry Tan's gstack handoff contracts, an autonomous improvement loop requires three asymmetries:

1. **Lock the judge.** The agent edits the corpus generator; it does NOT edit the judge or the eval. Without this asymmetry, every "improvement" is potentially metric-gaming. The `lock_paths` section enforces this.
2. **Single canonical metric.** `corpus_pass_rate` is one scalar, monotonically trackable across iterations. Per-pattern means are routing signals (which pattern to fix), not the headline metric.
3. **Bounded experiment budget.** Wall-clock + dollar caps prevent runaway loops. 3-strike stop on plateau prevents infinite local-optima chasing.

## What this charter is for

When the V2.5b autoresearch loop wakes (cron heartbeat or post-judge tick), it reads this file as the source of truth for:
- What it MAY change
- What it MUST NOT change
- How to decide whether to keep an iteration's result
- When to STOP

Humans iterate this file when the iteration_rule itself proves wrong (e.g., we discover that pattern_floor=0.55 is too lenient → bump to 0.60). Iterations of THIS file are version-controlled and reviewed; iterations of `corpus_generator.py` happen autonomously inside the loop.

## Current state (2026-05-09)

- corpus_pass_rate (full v25b_judged_full): **TBD** (need to compute from existing judged JSONL)
- failed_patterns (pattern_mean < 0.55): `B5_missing_context_asking_questions` (0.26), `B6_refusal_to_answer_direct_yes_no` (0.48), `B7_context_element_dropped` (0.51)
- next iteration: regenerate B5/B6/B7 with sharpened prompts (manually authored this iteration; future iterations may auto-generate)
- prior iterations: 0 (this is iteration 1 of the autoresearch loop)

## Iteration log

(append-only, one entry per iteration)

### Iteration 1 (2026-05-09 ~04:30 UTC) — pending

- **Hypothesis:** pattern-aware prompts that EXPLICITLY require the scenario to contain the target element AND the response to demonstrate the remediation will lift B5/B6/B7 means above 0.55.
- **Action:** add `_pattern_aware_prompt(section, pattern, idx)` to corpus_generator.py; regenerate ~400 examples per failing pattern on orca; re-judge; merge top-half-per-pattern with existing filtered corpus.
- **Cost cap:** $10 + 60 min orca.
- **Acceptance gate:** new B5_mean > 0.55 AND new B6_mean > 0.55 AND new B7_mean > 0.55 AND new corpus_pass_rate >= old - 0.01.
- **Decision:** TBD (post-regen).
