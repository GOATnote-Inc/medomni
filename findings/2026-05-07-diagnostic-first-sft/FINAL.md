# Diagnostic-first SFT V2.5b — autonomous loop final report

**Window:** 2026-05-07 ~20:40 UTC → ~21:25 UTC (~45 min wall, well under the 5-hour budget)
**Total spend:** ~$1.25 in gpt-4.1 (under $5 cap)
**Demo:** never below 200 throughout
**Public-surface edits:** zero
**Commits / pushes:** zero

## What landed

| | Artifact | Path |
|---|---|---|
| 1 | SPEC of step 1 | `findings/2026-05-07-diagnostic-first-sft/SPEC.md` |
| 2 | Failure-mode probe harness | `scripts/ship_rule_lib/failure_cluster.py` (~400 LOC) |
| 3 | Tests for above | `tests/test_failure_cluster.py` (16 tests) |
| 4 | Pilot driver | `findings/2026-05-07-diagnostic-first-sft/pilot.py` (parameterized via env vars) |
| 5 | Pilot v1/v2/v3/v4 + N=30 cohort + N=230 full | `cluster_assignments_*.jsonl` + `CLUSTER_SUMMARY_*.md` |
| 6 | N=230 result CARD | `findings/2026-05-07-diagnostic-first-sft/CARD.md` |
| 7 | V2.5b corpus design SPEC | `findings/2026-05-07-diagnostic-first-sft/V2.5B-CORPUS-SPEC.md` |
| 8 | Corpus design principles | `findings/2026-05-07-diagnostic-first-sft/CORPUS_PRINCIPLES.md` |
| 9 | Corpus generator scaffold | `scripts/ship_rule_lib/corpus_generator.py` (skeleton + stub-injection contract) |
| 10 | Tests for corpus generator | `tests/test_corpus_generator.py` (10 tests) |
| 11 | Loop state log | `LOOP-STATUS.md` (append-only iter records) |
| 12 | State machine | `findings/2026-05-07-diagnostic-first-sft/STATE.md` |

## Round-by-round narrative

| Round | Goal | Key finding | Action |
|---|---|---|---|
| R1 | Research (data/taxonomy/assets) | 230 V2.5 regressions; 5-cat taxonomy (Knowledge Gap / Reasoning Collapse / Calibration / Context Misapp / Hallucinated Safeguards); 7/8 primitives reusable | Wrote SPEC + first impl + tests |
| R2 | Surface penalty-rubric flips | Round 1 missed v0=False→v25=True on -points rubrics; rich `rubric_deltas` field added | Round 1 underestimated the safety dimension entirely |
| R3 | Sharpen #1 vs #5 boundary | Item 3a896889 correctly moved KG→Hallucinated Safeguards; gpt-4.1 read fabrication signal | 1/10 flipped, +1 ERR resolved |
| R4 | "Fails to..." rewrite | Item 435188d0 ERR→#4 (Context Misapp); item 06942620 stayed #1 (true silent omission, not classifier bug) | 8 KG / 1 H / 1 CM at N=10 |
| R5 | N=30 cohort | 80% KG / 10% H / 6.7% CM / 3.3% Cal / 0% RC | Pattern stable |
| R6 | **Full N=230** | **75.7% KG / 21.7% Hallucinated Safeguards / 1.3% Cal / 1.3% CM / 0% RC** | Fabrication rate doubled at scale; this is the load-bearing finding |
| C1 | V2.5b corpus SPEC | 70/25/2/2/1 allocation across 5 categories, n=5000 target | Anti-hallucination over-weighted vs observed share |
| C2 | Generator scaffold | Stub-injectable, deterministic IDs, contract pinned via 10 tests | Real generation deferred to next session |

## The load-bearing finding

**21.7% of V2.5-thinking's HB-Hard regressions are active fabrications**, not silent omissions. Examples:
- Rigid surgical thresholds ("<2 cm for laparoscopy")
- Invented guideline citations ("Management of Traumatic Diaphragmatic Rupture: A 2023 Update")
- Specific recurrence/efficacy percentages without source
- False reassurance ("masks fully prevent the flu")

The remaining ~76% are silent omissions — a more conventional Knowledge Gap pattern.

**Implication for V2.5b:** corpus design must split between coverage breadth (76%) and anti-fabrication (22%, slightly over-weighted because safety penalties are more costly).

## What was *not* done (deliberately deferred)

- Real corpus generation (would consume orca + ~$5+ in gpt-4.1)
- V2.5b LoRA training on orca
- V2.5b re-eval against PREREG ship rule
- κ inter-rater calibration cohort (Round 1B suggested 20-30 hand labels for κ ≥ 0.75)
- A/B test of contrastive-DPO vs SFT-only for §#5 anti-hallucination
- Tunnel-branding decision (still paused per user request)

## Recommended next actions for user (in order)

1. **Read CARD.md + V2.5B-CORPUS-SPEC.md.** That's where the analytic content is. ~15 min.
2. **Decide whether to fire generation on orca** or first do the κ calibration cohort. The cohort is methodologically cleaner; generation-first is faster to a V2.5b.
3. **If generation-first:** wire `corpus_generator.generate_v25b_examples`'s real generation_fn against `vllm-omni-orca` (the running container on orca) — this is ~50 LOC for the prompt template + the OpenAI-compatible client call. Then `assemble_corpus(target_n=5000)` runs ~6-12 hours on orca at ~250 tok/s.
4. **If κ-first:** select 30 representative regressions across the 5 categories (stratified), hand-label with the same taxonomy, compute Cohen's κ vs the gpt-4.1 labels in `cluster_assignments_full_n230.jsonl`. If κ < 0.6, refine the classifier prompt before generation.
5. **Train V2.5b** — NeMo PEFT LoRA on Nemotron-Omni-30B at bf16 (per CLAUDE.md §3) on orca. Same hyperparams as V2.5 unless training loss diverges.
6. **Re-eval** against the canonical PREREG protocol. V2.5b ship rule = HB-Hard CI lower bound ≥ 0 vs V0.

## Stop signals to surface to the user immediately

None during this session. Loop completed cleanly within budget.

## Test inventory at session end

```
tests/test_failure_cluster.py    16 passed
tests/test_corpus_generator.py   10 passed
ruff check scripts/ship_rule_lib/{failure_cluster,corpus_generator}.py — clean
```

## Cron status at session end

- Job `7cf749c2` (q15 at :07/:22/:37/:52) scheduled — will fire on `STATE.md` stage = DONE and write a single "DONE — awaiting user re-engagement" line per fire.
- The user can `CronDelete 7cf749c2` to stop heartbeats when re-engaging.

## Spend log

| Round | Calls | $ |
|---|---:|---:|
| R1-R3 pilot v1-v3 (N=10 × 3) | 30 | ~$0.12 |
| R4 pilot v4 (N=10) | 10 | ~$0.04 |
| R5 N=30 cohort | 30 | ~$0.12 |
| R6 N=230 full | 230 | ~$0.97 |
| **Total** | **300** | **~$1.25** |

Under the $5 cap, well within budget.
