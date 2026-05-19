# Diagnostic-first SFT V2.5b — autonomous loop state machine

**Started:** 2026-05-07 ~20:40 UTC
**Authorized window:** 5 hours (until ~01:40 UTC 2026-05-08)
**Authorized substages:** R4 → R5 → R6 → C1 → C2 → DONE
**Cadence:** q15 (cron expression `7,22,37,52 * * * *`)
**Spend cap:** ~$5 in gpt-4.1 across the window

## Current stage: **B2-RUNNING** — full 5000-example corpus generation live on orca

- **Started:** 2026-05-08 07:40 UTC
- **PID on orca:** 94917 (pgrep `python3 -u b2_runner.py`)
- **Output:** `/home/shadeform/medomni-b2/b2_corpus_v25b.jsonl` (append-only, resume-safe)
- **Log:** `/home/shadeform/medomni-b2/b2.log` (UTC timestamped)
- **Allocation:** A=1500 / B=3250 / C=250
- **Pace observed:** 8.9s/example → **ETA ~12.3h** (better than 22.6h dry-run projection)
- **Cost projection:** ~$45 orca compute (under the $84 estimate)
- **Errors so far:** 0; abort-after-50-errors safety cap in place
- **First record verified:** `v25b_secA_00000` → pattern `A1_fabricated_specific_citation`, scenario 98 chars, response 707 chars

The runner is restart-safe: any process death resumes from last-completed example by reading existing JSONL. Cron heartbeat will poll progress each iter.


User question raised: "is it designed to make future attempts easier like Karpathy autoresearch dictates?" Honest answer: B2 as scoped was one-shot. Auto mode + recommendation A → added the missing autoresearcher provenance layer this iter:
- `FAILURE_PATTERN_LIBRARY.md` — 15 named patterns (5 A + 8 B + 2 C) with smoking-gun κ-shadow evidence and per-pattern remediation
- `PATTERN_LIBRARY` dict + `pattern_for_idx()` round-robin in corpus_generator
- Each example now carries `pattern_addressed: str` field
- 5 new tests pinning round-robin balance + section coverage
- 52 tests green, ruff clean

Now V2.5b corpus is traceable: post-train you can grep by pattern to know what landed. V2.5c bootstrap is structured, not from-scratch.

**Awaiting OK to fire B2** ($84 orca compute, ~22.6h, with provenance baked in).


Decision landed: user picked "go collapsed" (3-section A/B/C corpus). Implementation:
- Collapsed taxonomy + tests landed (37 → 47 tests green; ruff clean)
- B1.1 dry-run on orca: 5/5 succeeded with max_tokens=8192 (avg 16.3s/example)
- Quality spot-check: A=anti-fabrication (defers to stewardship), B=substantive content + differentials, C=names "seductive hypothesis" mid-chain
- Full 5000-example projected: ~22.6h orca × $3.70/hr = ~$84 compute

Awaiting OK to fire B2 (full 5000-example generation).


User feedback: "no manual review for solo dev" → fired Claude shadow κ instead.
**RESULT: κ = 0.054, raw agreement 38%.** gpt-4.1 + Claude opus 4.7 disagree
heavily on the 5-category taxonomy. Implication: 75/22/2/2/0 distribution is
unreliable; corpus design (70/25/2/2/1 allocation) probably under-allocates to
calibration training.

See `KAPPA_SHADOW_REPORT.md` for the disagreement matrix + recommendation.

**Recommended pivot (solo-dev velocity):** collapse taxonomy to 2-3 sections
where models agree (fabrication 30% / omission 65% / probe 5%) and ship V2.5b
with the collapsed split. The eval re-run is the validation. Surfaced for user.

User decision (recorded 2026-05-07 ~22:50 UTC after reading FINAL.md): **κ-first**.
Reasoning per conversation: cost-of-being-wrong asymmetric (~$150 + 36h sunk if classifier biased), V2.5b corpus shape rides directly on gpt-4.1 distribution, user is a physician with comparative advantage on hand-labeling, publishability.

### Pre-authorized κ-track substages (additive to original state machine)

- [x] **K1.0** — kappa_probe.py written (~150 LOC)
- [x] **K1.1** — probe ran clean: 18 KG / 6 H / 3 CM / 3 Cal / 0 RC = 30 stratified items.
  - `kappa_blind_review.md` (89 KB physician-facing review doc)
  - `kappa_user_labels.csv` (template)
  - `kappa_answer_key.jsonl` (hidden ground truth, 13 KB)
- [x] **K2.0** — `scripts/ship_rule_lib/kappa_comparator.py` written (Cohen's κ + disagreement matrix + report renderer; stdlib only)
- [x] **K2.1** — `tests/test_kappa_comparator.py` (11 tests, all green; 37 total in repo); ruff clean
- [x] **B1.0** — `make_orca_generation_fn(base_url, model)` factory landed in `corpus_generator.py`. Reads `MEDOMNI_ORCA_VLLM_URL` / `MEDOMNI_ORCA_MODEL` env vars (defaults `http://localhost:8000/v1` / `nemotron`). Topic pools 10/10/5/5/1 across cats 1/5/3/4/2. Per-category prompt templates with hard rules (no fabricated citations, no rigid thresholds, hedge specifically). 37 tests still green, ruff clean.
- [x] **B1.0-tests** — retroactive 7 tests for `_category_prompt` (anti-fabrication rules, determinism, idx-rotation, per-category guidance) + factory (callable, env-var read) + topic-pool-size invariants. **44 tests total green, ruff clean across scripts/ + tests/.**
- [ ] **B1.1** — 5-example dry-run on orca (HOLD until κ-track decision A/B/C lands)
- [x] **B1-INDEX** — wrote `INDEX.md`: navigable map of 23 artifacts, "read in this order" guide, open decision points, spend log, end-to-end re-engagement path
- [ ] **WAIT-FOR-USER** — user adjudicates 30-item κ at their pace
- [ ] **K3.0** — comparator runs, surface κ + disagreement matrix
- [ ] **GATE** — if κ ≥ 0.6 → fire B2 (full 5000-example generation). If κ < 0.6 → refine classifier prompt + re-run N=230 deltas.



### C2 outcome (recorded 2026-05-07 ~21:25 UTC)
- `tests/test_corpus_generator.py` written (10 tests, TDD red verified)
- `scripts/ship_rule_lib/corpus_generator.py` written (skeleton with stub-injection contract; default generation_fn raises NotImplementedError so tests pass without openai)
- 26 tests green total, ruff clean, demo 200
- No real generation fired

### DONE outcome (recorded 2026-05-07 ~21:26 UTC)
- `FINAL.md` written
- All 7 hard constraints honored (demo never below 200, no public-surface edits, no commits/pushes, no DNS API calls, no orca container changes, TDD discipline maintained, openai preflight before every batch)
- Total spend $1.25 under $5 cap
- Loop complete in ~45 min wall, well under 5 hr budget

### C1 outcome (recorded 2026-05-07 ~21:18 UTC)
- V2.5B-CORPUS-SPEC.md written: 70/25/2/2/1 allocation, n=5000 target, generation/training/re-eval forward-looking sections
- CORPUS_PRINCIPLES.md written: 6 anti-patterns + 7 positive patterns + 5 hard rules + out-of-scope list
- Total spend unchanged (writing only)

### R6 outcome (recorded 2026-05-07 ~21:09 UTC)
- N=230 full classification: 171 KG (75.7%), 49 Hallucinated (21.7%), 3 Calibration (1.3%), 3 Context Misap (1.3%), 0 Reasoning Collapse, 1 low-conf, 3 errors
- Reasoning Collapse 0/230 — confirmed absent at full scale
- 21.7% fabrication rate is the safety-critical signal — V2.5b corpus must address anti-hallucination
- CARD.md written
- Spend: ~$0.97 → cumulative ~$1.25

### R5 outcome (recorded 2026-05-07 ~21:00 UTC)
- N=30 cohort distribution: 24 KG (80%), 3 Hallucinated Safeguards (10%), 2 Context Misapplication (6.7%), 1 Calibration Misalignment (3.3%), 0 Reasoning Collapse, 0 low-confidence
- Smoking-gun 06942620 still cat=1 (judge: "V0 provided guidance to advise emergency visit, V2.5 silently omitted")
- Reasoning Collapse (#2) has zero observations — either truly absent or classifier blind to it
- Spend: ~$0.12 → cumulative ~$0.28

### R4 outcome (recorded 2026-05-07 ~20:50 UTC)
- Patch landed: "Fails to <X>" rubric events pre-rewrite to positive form
- 16 tests green (added 2), ruff clean
- Pilot v4: 8 Knowledge Gap / 1 Hallucinated Safeguards (3a896889) / 1 Context Misapplication (435188d0) / 0 errors
- Δ vs v3: 1 item moved (435188d0 ERROR → #4), v3's lone ERR resolved
- Item 06942620 remained #1 — assessed as TRUE silent-omission KG on a safety-critical rubric, NOT a classifier bug. The 5-cat taxonomy doesn't have a "safety-critical" tag orthogonal to mode; flag for FINAL.md
- Spend: ~$0.04 → cumulative ~$0.16

## Hard constraints (every iter must honor)

1. **Demo first.** `curl -sIL --max-time 8 https://www.thegoatnote.com/4UWHAt/` MUST return a 308→200 chain. If not, STOP and report. Demo is sacred.
2. **No public-surface edits.** Forbidden paths:
   - `/Users/kiteboard/medomni/web/**`
   - `/Users/kiteboard/medomni/app/**`
   - `/Users/kiteboard/medomni/vercel.json` (any vercel*)
   - Anything that affects `https://www.thegoatnote.com/`, `/4UWHAt`, or `prism42-console.vercel.app/prism42-v3`
3. **No commits / pushes / merges.** Work lives in:
   - `scripts/ship_rule_lib/failure_cluster.py` (impl)
   - `tests/test_failure_cluster.py` (tests)
   - `findings/2026-05-07-diagnostic-first-sft/**` (artifacts)
4. **No DNS / CF / GoDaddy / Vercel API calls.** Tunnel-branding decision is paused.
5. **No orca container changes.** Read-only ssh probes only (`docker ps`, `nvidia-smi`).
6. **TDD discipline.** `pytest tests/test_failure_cluster.py` + `ruff check scripts/ship_rule_lib/failure_cluster.py` must be green before advancing substages.
7. **OpenAI preflight.** Source canonical .env + run `grader.preflight_grader()` before any gpt-4.1 batch. Silent 401 zeros all classifications.

## Substage state machine

### R4 — Bug-4 fix: rewrite "Fails to..." rubric events to positive form

When a rubric criterion starts with "Fails to" and `points<0`, the classifier reads the literal English as "didn't do something" → defaults to Knowledge Gap. Pre-rewrite to positive form so the classifier sees the SEMANTICS: "V0 advised X; V2.5 did NOT advise X (penalty triggered)".

- [x] **R4.0** — patch landed
- [x] **R4.1** — pytest + ruff green (16 passed)
- [x] **R4.2** — pilot v4 ran end-to-end
- [x] **R4.3** — comparison surfaced in LOOP-STATUS
- [x] **R4.4** — advanced to R5

### R5 — N=30 calibration cohort

- [x] **R5.0** — extended via N_PILOT=30 RUN_TAG=cohort_n30 env vars
- [x] **R5.1** — 30 calls clean, 0 errors
- [x] **R5.2** — distribution computed (logged in LOOP-STATUS)
- [x] **R5.3** — CLUSTER_SUMMARY_cohort_n30.md written
- [x] **R5.4** — advanced to R6

### R6 — N=230 full classification

- [x] **R6.0** — N=230 ran clean in ~6 min (faster than estimate; gpt-4.1 was responsive)
- [x] **R6.1** — distribution computed (75.7/21.7/1.3/1.3/0)
- [x] **R6.2** — CARD.md written with full reproducibility section
- [x] **R6.3** — advanced to C1

### C1 — V2.5b corpus design SPEC

- [x] **C1.0** — V2.5B-CORPUS-SPEC.md written
- [x] **C1.1** — concrete characteristics per category in SPEC
- [x] **C1.2** — CORPUS_PRINCIPLES.md written
- [x] **C1.3** — advanced to C2

### C2 — Corpus generator scaffold (TESTS + SKELETON ONLY — no generation)

- [x] **C2.0** — tests written (10 tests, red verified)
- [x] **C2.1** — scaffold impl green
- [x] **C2.2** — 26 total tests green, ruff clean
- [x] **C2.3** — advanced to DONE

### DONE

- [x] **DONE.0** — FINAL.md written
- [x] **DONE.1** — cron `7cf749c2` will fire q15; on DONE state it writes one-line "DONE" per fire
- [x] **DONE.2** — 26 tests green, ruff clean, full artifact set on disk

## Stop conditions (any → halt + surface)

- Demo `/4UWHAt/` returns non-2xx/308
- Pytest fails and can't be repaired in 1 iter
- gpt-4.1 preflight returns non-ok (401 / network)
- Cumulative gpt-4.1 spend approaches $5 (track via assignments file sizes; ~230 calls = ~$1)
- Any `git status` shows files in forbidden paths modified
- Any of the 5 conversational hard rules are about to be violated
