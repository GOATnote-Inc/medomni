# Diagnostic-first SFT V2.5b — failure-mode probe harness (step 1)

Date: 2026-05-07
Owner: kiteboard
Pod: exact-kind-orca (H100 80GB Scaleway, sole pod for ~1 month)
Repo: medomni (this repo) — laptop-side tooling only this turn

## Origin

V2.5 thinking=True ran the canonical PREREG ship-rule eval and **FAILED** (PR #122,
`findings/2026-05-05-v2.5-eval-thinking/SHIP-RULE-RESULTS.md`). On HealthBench-Hard
specifically, V2.5-thinking earns -1.31pp delta vs V0 (95% CI [-3.71, +1.13]; not
significant; Cohen d_z = -0.044). Win/loss/tie split across the 600 paired items
(200 × 3 seeds): **V0 wins 230, V2.5-thinking wins 226, ties 144**. The "negative
result" is a 230-item regression set we have not yet diagnosed.

The user's stated arc is "learn from negative results to create positive ones."
This document scopes step 1 of the Karpathy-style diagnostic-first SFT loop:
**probe the V2.5-thinking regressions, classify each into one of 5 mutually
exclusive failure modes via gpt-4.1, surface the cluster summary.** Steps 2-6
(corpus design → V2.5b training on orca → re-eval) are out of scope this turn.

## Scope

IN this turn:
- New module `scripts/ship_rule_lib/failure_cluster.py` (~80 LOC)
- New tests `tests/test_failure_cluster.py` (TDD red → green)
- New CLI subcommand `failure-cluster` in `scripts/ship_rule_eval.py`
- Small-N pilot: classify 10 of 230 regressions, inspect content, surface table
- This SPEC + a paired CARD on completion

OUT of this turn:
- Full N=230 classification run (do AFTER pilot inspection passes)
- Corpus generation (step 3)
- V2.5b training (step 4)
- Re-eval (step 6)
- Touching anything outside `scripts/ship_rule_lib/`, `tests/`, `findings/2026-05-07-diagnostic-first-sft/`

## Non-goals

- We are not building a general-purpose LLM evaluator — only a 5-category
  classifier for HealthBench-Hard regressions of V2.5-thinking vs V0.
- We are not re-grading any responses. We use the existing gpt-4.1 grader output
  (the `judge_log[].explanation` field, already in graded JSONL).
- We are not introducing scikit-learn / numpy / scipy. Stdlib + openai only,
  matching `ship_rule_lib`'s "stdlib-only" stance.

## Taxonomy (locked from Round 1B research)

5 mutually exclusive categories. Tiebreaker rules in §Tiebreakers below.

| # | Name | Definition | gpt-4.1 keys on |
|---|---|---|---|
| 1 | Knowledge Gap | V2.5 lacks factual info that V0 stated correctly | V0 asserts X; V2.5 omits X or contradicts with wrong fact |
| 2 | Reasoning Collapse | V2.5 reasons correctly then reverses or halts mid-chain | V2.5 sets up correct reasoning then contradicts itself without new info |
| 3 | Calibration Misalignment | V2.5's confidence/urgency shifts vs V0 without changed scenario data | Confidence words ("probably"/"likely"/"definitely") differ ≥2 levels between arms |
| 4 | Context Misapplication | V2.5 loses or reweights a critical scenario element (timeline, PMH, allergy, acuity) | Prompt has element E; V0 references E; V2.5 omits/contradicts E |
| 5 | Hallucinated Safeguards | V2.5 fabricates clinical info, citations, guidelines, or false reassurance | V2.5 cites X; X is invented or contradicts the actual guideline |

Sources:
- LostBench Phase 3 Class A/B/C/D framework
- Frontiers AI 2025 — medical-LLM reasoning failure analysis
- Nature Sci. Reports 2025 — LLM clinical reasoning limitations
- medRxiv 2025 — medical hallucination in foundation models

## Tiebreakers (operationalize mutual exclusivity)

When two categories appear to apply, the classifier must apply these rules in
order:

1. **Hallucinated Safeguards (#5) wins** if V2.5 asserts a fabricated guideline,
   citation, or protocol — even if a knowledge gap is also present. Fabrication
   is the more serious error class.
2. **Reasoning Collapse (#2) wins** over Knowledge Gap (#1) when V2.5 demonstrates
   awareness of the relevant fact early in its response then contradicts that
   awareness later. The hallmark is internal contradiction, not absence.
3. **Context Misapplication (#4) wins** over Knowledge Gap (#1) when the missed
   element is explicit in the prompt (timeline, PMH, allergy, acuity). If the
   element is implied but not stated, default to Knowledge Gap.
4. **Calibration Misalignment (#3) wins** over Reasoning Collapse (#2) when the
   regression is purely a confidence-language shift with otherwise unchanged
   reasoning structure. Reasoning Collapse requires actual contradiction.
5. If none of the above match cleanly: default to **Knowledge Gap (#1)** as the
   most conservative attribution.

The classifier prompt embeds these rules verbatim; the unit tests pin the rule
order via golden cases.

## Architecture

Reuse from existing `ship_rule_lib`:
- `grader.assert_openai_key`, `grader.preflight_grader` — key safety
- `grader.GPT_MODEL` (= "gpt-4.1") — same judge model as the original eval
- `grader.strip_thinking` — strip `<think>` tags before passing V2.5 response to classifier (avoid leaking thinking traces into the classifier input as a confound)
- `stats.align_paired` — pair items across arms by `item_id`

New (this turn):
- `failure_cluster.CATEGORIES: dict[int, tuple[str, str]]` — locked taxonomy
- `failure_cluster.Regression` — dataclass for one V2.5 loss (item_id, seed, prompt, both responses, both scores, missed-rubric-with-explanation list)
- `failure_cluster.ClusterAssignment` — dataclass (item_id, seed, category, category_name, justification)
- `failure_cluster.ClusterSummary` — dataclass (n_total, per_category dict)
- `failure_cluster.load_paired_graded(...)` — read graded JSONL for arm0+arm1 across seeds
- `failure_cluster.select_regressions(...)` — extract items where arm1.score < arm0.score
- `failure_cluster.classify_regression(reg, *, classify_fn=None)` — gpt-4.1 call (or injected stub)
- `failure_cluster.summarize_clusters(...)` — group + pick exemplars

CLI:
- `python scripts/ship_rule_eval.py failure-cluster --graded-dir <dir> --benchmark healthbench-hard --n-pilot 10 --out findings/2026-05-07-diagnostic-first-sft/`

Outputs:
- `cluster_assignments.jsonl` — one record per classified regression
- `CLUSTER_SUMMARY.md` — counts + 2 exemplars per category, ranked
- `REPRO.sh` — exact re-run command + env preflight

## Hard rules

1. Pre-flight `preflight_grader()` BEFORE any classification call. Per memory
   `feedback_eval_preflight_judge_key.md` — silent 401 zeros all categories.
2. Stdlib + openai only. No scikit-learn / numpy / scipy.
3. Do NOT cat the canonical .env. Source via the documented incantation.
4. Tests must pass with `pytest tests/test_failure_cluster.py -v` BEFORE the
   classifier is wired to the real openai client.
5. Pilot N=10 first, inspect JSONL by hand, surface to user. Do NOT auto-fire
   N=230 without explicit user OK.
6. The 6 graded JSONL files live in the worktree
   `/Users/kiteboard/medomni/.claude/worktrees/ship-rule-eval/findings/2026-05-05-v2.5-eval-thinking/graded/`.
   Per memory `feedback_check_worktree_status_before_cherrypick.md`, those
   files are untracked in the worktree. Do not move them this turn — read in
   place via the `--graded-dir` flag.
7. Lint: ruff (E,F,I,W,UP,B; line=100). No `--no-verify`. No `git add -A`.

## Success signal

```
pytest tests/test_failure_cluster.py -v          # all green
python scripts/ship_rule_eval.py failure-cluster \
    --graded-dir /Users/kiteboard/medomni/.claude/worktrees/ship-rule-eval/findings/2026-05-05-v2.5-eval-thinking/graded \
    --benchmark healthbench-hard \
    --n-pilot 10 \
    --out findings/2026-05-07-diagnostic-first-sft/
# produces cluster_assignments.jsonl with 10 records, each with category 1-5
# produces CLUSTER_SUMMARY.md table with counts ≤10 across categories
```

Then surface the table to the user; await explicit OK before scaling to N=230.

## Risks + mitigations

- **gpt-4.1 inter-rater unreliability**: Round 1B recommends 20-30 hand-labeled
  cases for κ ≥ 0.75 calibration BEFORE rolling out. Pilot N=10 gives an
  initial sanity check; full κ calibration is step 2 (next turn).
- **Tiebreaker ambiguity**: encode rules verbatim in classifier prompt + pin
  via golden tests. If pilot inspection shows misclassifications, refine
  prompt before scaling.
- **Confounding from `<think>` tags**: V2.5 responses include `<think>...</think>`
  segments. Pass `strip_thinking(response)` to the classifier so it scores only
  the visible answer (matches what the gpt-4.1 grader scored).
- **Missing rubric context**: a regression may have multiple failed rubrics. We
  classify the *primary* failure mode (the rubric whose v0/v25 contrast is
  largest). Tiebreaker: lowest rubric index.
