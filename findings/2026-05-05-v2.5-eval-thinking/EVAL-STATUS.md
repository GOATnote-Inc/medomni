# V2.5 Ship-Rule Eval — Re-fire (thinking=True) — STATUS

## Mission
Re-fire canonical V2.5 ship-rule with `enable_thinking=True` per Nemotron-Omni reasoning-SFT
chat template + grader-side `</think>` stripping. Closes the methodology gap on PR #122
(prior arm thinking=False shipped 1/4 PASS).

## Phases
- A: patch + commit (DONE prior agent, commits 6c14c26 + ee618b8)
- B-cont: V0 gens 12 runs (IN PROGRESS, 4/12 in flight)
- B2: swap to v25-serve + smoke
- B3: V2.5 gens 12 runs
- D: rsync + gpt-4.1 grade
- E: stats / leakage / manifest / report
- F: PR #122 update (Ready, NOT merged)
- G: restart judge

## State log (append-only)

### 2026-05-06 — Phase B-cont kickoff
- Verified prior state: medqa v0 seed42 in flight at 81/200 (PID 3724773)
- v0-serve UP at :8003 (vllm/vllm-openai:v0.20.0, model=v0)
- Judge stopped (launch cmd at lobster:/tmp/judge-launch-cmd.json)
- GPU 123 GB used / 20 GB free
- Fired 3 additional V0 gens: medqa-123, medqa-7919, pubmedqa-42
- All 4 gens batching against v0-serve continuous-batching (vllm max-num-seqs=8)
- At T+~60s: medqa-42=101/200, medqa-123=3/200, medqa-7919=3/200, pubmedqa-42=18/200
- GPU util 100% — saturated


### 2026-05-06 21:55 PT — Phase D + E milestone (FINAL)

Ship-rule verdict: **0/4 PASS → OVERALL FAIL.**

| Benchmark | V0 | V2.5 | Δ | 95% CI | Rule | PASS? |
|---|---:|---:|---:|:---:|---|:---:|
| MedQA-USMLE | 83.50% | 84.67% | +1.17pp | [-1.00, +3.50] | delta_lower_ci > 0 | FAIL |
| PubMedQA-L | 67.33% | 64.83% | -2.50pp | [-5.00, -0.17] | delta_lower_ci ≥ -1pp | FAIL |
| MedXpertQA-Text | 33.00% | 31.67% | -1.33pp | [-4.83, +2.17] | delta_lower_ci ≥ +5pp | FAIL |
| HealthBench-Hard | 12.52% | 11.21% | -1.31pp | [-3.71, +1.13] | delta > 0 | FAIL |

Worse than Run 1 (thinking=False, 1/4 PASS). Thinking re-fire eliminated the
only PASS (HB-hard +1.48pp → -1.31pp).

A5 ablation (Δ V2.5 − V0):

| Benchmark | thinking=False (Run 1) | thinking=True (Run 2) |
|---|---:|---:|
| MedQA-USMLE | -1.17pp | +1.17pp |
| PubMedQA-L | -0.67pp | -2.50pp |
| MedXpertQA-Text | -1.33pp | -1.33pp |
| HealthBench-Hard | +1.48pp | -1.31pp |

Wall-clock + cost:
- Phase D rsync + dedupe: 5 min (HB v0 from 400→200 by item_id)
- Local-grader (medqa+pubmedqa+medxpertqa-text): ~1 min
- HB-hard gpt-4.1 grade (6 files in parallel): ~83 min wall
- gpt-4.1 calls: ~14,400 rubric grades. Cost est. $20-30 USD.

Leakage: 0 hits / 800 test prompts vs v1_train.jsonl (10,178 rows).

Outputs: SHIP-RULE-RESULTS.{md,json} + stats.json + MANIFEST.sha256 +
LEAKAGE-AUDIT.md + leakage-flags.jsonl + REPRO.sh.

Per PREREG `ship_rule.on_fail`: revert; reasoning-SFT data quality is the
likely culprit, possibly compounded by thinking-channel decode interaction.
PR #122 stays Ready (NOT merged). Decision-tree at issue #130: V2.5b vs V2.7.
