# Index — diagnostic-first SFT V2.5b loop artifacts

## Read in this order on re-engagement

1. **`FINAL.md`** — autonomous loop summary, round-by-round narrative, recommended next actions.
2. **`CARD.md`** — N=230 method + final distribution (75.7% KG / 21.7% Hallucinated / 1.3% Cal / 1.3% CM / 0% RC).
3. **`V2.5B-CORPUS-SPEC.md`** — corpus design (70/25/2/2/1 allocation, n=5000 target, generation/training/re-eval forward sections).
4. **`CORPUS_PRINCIPLES.md`** — 6 anti-patterns + 7 positive patterns + 5 hard rules.

## Open decision points

- **κ adjudication path (A/B/C)** — surfaced in conversation. Options: Claude shadow rating, literal user-CSV-fill, or narrowed-cohort physician κ. B1.1 dry-run holds until this resolves.
- **Tunnel branding** — paused per user. Three free options remain (zone migration / Tailscale Funnel / stay ad-hoc).

## State

- **`STATE.md`** — autonomous-loop state machine. Current: B1.1 HOLD on κ.
- **`SPEC.md`** — original step-1 SPEC (Round 1 entry conditions, success criteria).

## κ-calibration cohort artifacts

- **`kappa_blind_review.md`** — 30 stratified items in physician-facing review form, no gpt-4.1 categories visible.
- **`kappa_user_labels.csv`** — fill-in template (columns: item_id, seed, category, confident, notes).
- **`kappa_answer_key.jsonl`** — gpt-4.1 ground-truth labels (sealed; consumed by comparator).
- **`kappa_probe.py`** — stratified sampler that produced the above three files.

## Pilot + cohort + full classification artifacts (input → output)

| Tag | Method | N | Result file (jsonl) | Summary file (md) |
|---|---|---:|---|---|
| pilot v1 (R1-R3 result) | Round 1-3 prompt | 10 | `cluster_assignments_pilot.jsonl` | `CLUSTER_SUMMARY_PILOT.md` |
| pilot v2 (R2 surface fix) | + `rubric_deltas` | 10 | `cluster_assignments_pilot_v2.jsonl` | `CLUSTER_SUMMARY_PILOT_v2.md` |
| pilot v3 (R3 boundary sharpening) | + tighter #1 vs #5 | 10 | `cluster_assignments_pilot_v3.jsonl` | `CLUSTER_SUMMARY_PILOT_v3.md` |
| **pilot v4** (R4 polarity fix) | + "Fails to..." rewrite | 10 | `cluster_assignments_pilot_v4.jsonl` | `CLUSTER_SUMMARY_PILOT_v4.md` |
| **N=30 cohort** (R5) | v4 prompt | 30 | `cluster_assignments_cohort_n30.jsonl` | `CLUSTER_SUMMARY_cohort_n30.md` |
| **N=230 full** (R6) | v4 prompt | 230 | `cluster_assignments_full_n230.jsonl` | `CLUSTER_SUMMARY_full_n230.md` |

## Driver scripts

- **`pilot.py`** — main classification driver. Parameterized via `N_PILOT` and `RUN_TAG` env vars.
  - Pilot: `N_PILOT=10 RUN_TAG=pilot_v4 .venv/bin/python pilot.py`
  - Cohort: `N_PILOT=30 RUN_TAG=cohort_n30 .venv/bin/python pilot.py`
  - Full: `N_PILOT=230 RUN_TAG=full_n230 .venv/bin/python pilot.py`
- **`kappa_probe.py`** — emit κ review artifacts (one-shot).

## Source code (in `/Users/kiteboard/medomni/scripts/ship_rule_lib/`)

- `failure_cluster.py` — taxonomy, classifier, `rubric_deltas`, "Fails to..." rewrite. 16 tests in `tests/test_failure_cluster.py`.
- `corpus_generator.py` — V2.5b corpus scaffold + `make_orca_generation_fn` factory. 17 tests in `tests/test_corpus_generator.py`.
- `kappa_comparator.py` — Cohen's κ + disagreement matrix. 11 tests in `tests/test_kappa_comparator.py`.

**Total: 44 tests green, ruff clean across `scripts/ship_rule_lib/` + `tests/`.**

## Spend log

| Round | Calls | $ |
|---|---:|---:|
| Pilot v1-v3 (R1-R3) | 30 | ~$0.12 |
| R4 pilot v4 | 10 | ~$0.04 |
| R5 N=30 cohort | 30 | ~$0.12 |
| R6 N=230 full | 230 | ~$0.97 |
| K1+K2 (κ scaffold) | 0 | $0 |
| B1.0 (orca factory prep) | 0 | $0 |
| **Total** | **300** | **~$1.25** |

## When you re-engage

The fastest end-to-end path:
1. Pick A/B/C on κ adjudication (see FINAL.md §Recommended next actions).
2. If A or C: surface κ ≥ 0.6 → fire B1.1 dry-run (~$0.10) → fire full corpus generation B2 (~$30-100, ~12h orca).
3. Train V2.5b LoRA on orca (~24h, ~$45 orca compute).
4. Re-eval against PREREG ship rule (existing driver in `scripts/ship_rule_eval.py`, ~$1, ~12 min).
