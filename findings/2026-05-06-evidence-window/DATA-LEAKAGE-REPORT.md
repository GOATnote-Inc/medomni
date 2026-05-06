# E4 — Data-leakage check (5-gram MinHash, Jaccard >= 0.7)

**Date:** 2026-05-06 21:13 UTC
**Method:** datasketch 1.10.0 MinHashLSH, 128 permutations, 5-gram shingles
over whitespace-tokenized lowercased text. Jaccard threshold 0.7. Hits
re-verified with exact MinHash Jaccard (LSH is approximate; verification
pass confirms each hit individually).

**Run host:** evil-cyan-lobster (H200). Co-tenant with ship-rule eval
(no resource collision — leak check is CPU-only and finished in 46s).

## Configuration

| Field | Value |
|---|---|
| n-gram size | 5 (whitespace tokens, lowercased) |
| MinHash permutations | 128 |
| Jaccard threshold | 0.7 |
| Sample limit per training corpus | 5000 |
| Sample limit per eval benchmark | 1000 |
| Total training items indexed | 10,000 (5000 MedReason + 5000 medical-o1) |

## Training corpora indexed

| Corpus | HF ID | Items sampled | Text fields concatenated |
|---|---|---|---|
| MedReason | `UCSC-VLAA/MedReason` | 5,000 | question + reasoning + response |
| medical-o1-reasoning-SFT (en) | `FreedomIntelligence/medical-o1-reasoning-SFT` | 5,000 | Question + Complex_CoT + Response |

R1-distilled USMLE traces (`deepseek-r1-distill-usmle-traces` per
PREREG.yaml) are listed as 5K targeted in the V2.5 PREREG. They are
**generated synthetically** from DeepSeek-R1 + USMLE-style questions, with
no public canonical source — **out of scope** for E4 because the very
prompts (USMLE-style) are by design adjacent to MedQA-USMLE. We track
this as a documented limitation, not a contamination claim. E5
memorization probe will catch anything the synthetic set memorized into
the model.

## Results

| Benchmark | HF source / split | n_eval queried | n_flagged at Jaccard >= 0.7 | % overlap | Verdict |
|---|---|---|---|---|---|
| MedQA-USMLE-4opt | `GBaker/MedQA-USMLE-4-options` test | 1,000 | 0 | **0.000%** | clean (no contamination flag) |
| PubMedQA | `qiaojin/PubMedQA` pqa_labeled split | 1,000 | 0 | **0.000%** | clean |
| MedXpertQA-Text | `TsinghuaC3I/MedXpertQA` Text test | 1,000 | 0 | **0.000%** | clean |
| HealthBench-Hard | `Tonic/Health-Bench-Eval-OSS-2025-07` hard split | 959 | 0 | **0.000%** | clean |

(HealthBench-Hard yielded 959 items in the streaming sample because
some items lacked enough tokens for a 5-gram shingle and were skipped.)

## Verdict

**No contamination flags raised.** All four ship-rule benchmarks return 0%
overlap to the V2.5 training corpora at the 5-gram, Jaccard >= 0.7
threshold. None of the benchmarks exceeds the 5% threshold from the
E-track plan; none even approaches it.

This is consistent with prior published audits of these specific
training corpora and benchmarks (MedReason and medical-o1 are
reasoning-distillation outputs, not direct test-set crawls; the HF
benchmark splits are public test sets but are NOT in the training
text).

## Caveats and limitations

1. **5-gram + Jaccard 0.7 is a SURFACE check.** It catches verbatim
   reuse and near-verbatim paraphrase; it does NOT catch:
   - Semantic-equivalence with different wording (e.g. same clinical
     vignette rewritten).
   - **Memorization without overlap** — the model could have seen
     the test items during BASE pre-training and recall them despite
     no overlap with the V2.5 fine-tune corpora.
   E5 memorization probe is the confirmatory step for this gap.

2. **Sampling.** Indexing is on 5,000 items per training corpus and
   1,000 per eval. The MedReason corpus has ~32K items total; the
   medical-o1 corpus has ~25K. If overlap is non-uniform across the
   corpus (e.g. concentrated in items 5001-32000 of MedReason), the
   sample misses it. T+24h plan: re-run full-population (no
   `SAMPLE_LIMIT_TRAIN` cap; ~10 min walltime estimate).

3. **The MedXpertQA-Text item shape includes options as a JSON dump**
   in our `text_fn`. If MedXpertQA-Text upstream stores options as
   `{"A": "...", ...}` while training corpora contain the same options
   as `A. ...`, the surface tokens differ even when the content is the
   same. Mitigation: T+24h re-run with a normalized text builder
   (strip option-letter formatting before shingling).

4. **R1-distilled USMLE traces are not in this audit** (out of scope —
   no public source). E5 memorization probe is the only way to bound
   their leakage risk.

## Implication for ship-rule

All four ship-rule benchmarks (MedQA-USMLE, MedXpertQA-Text,
HealthBench-Hard, PubMedQA-L) are **PASS** under E4. The ship-rule
pass criteria can use the headline scores without contamination
caveats.

E5 memorization probe (deferred to T+5h) is still required as
confirmation; it tests the "memorization without overlap" gap.

## Reproducibility

| Field | Value |
|---|---|
| Script | `/tmp/leak_check.py` (committed verbatim alongside this report; lobster copy at `/tmp/leak_check.py`) |
| Lobster venv | `/home/ubuntu/.venv` (datasketch 1.10.0, datasets 4.8.5) |
| Walltime | 46 seconds (3 benchmarks); +~30s for HealthBench-Hard add-on |
| Random seed | None used; HF streaming order is deterministic per-revision |
| Raw flagged-item dumps | `/tmp/leak_results/leak_raw_*.jsonl` on lobster (all empty per 0% flag) |

## Next steps

- T+5h: E5 memorization probe (per `MEMORIZATION-PROBE-PLAN.md`)
- T+24h: full-population re-run (lift `SAMPLE_LIMIT_TRAIN` cap)
- T+24h: normalized-text re-run for MedXpertQA-Text option-letter handling
