# CARD delta — v1.0 N=6 (consolidated) → v1.0 N=30 (Phase 1.7 broadened)

- **Baseline**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-consolidated-20260429-173557/heldout.json`
- **Candidate**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-n30-20260429-211018/heldout.json`
- **Mean score**: `0.385` → `0.369` (-0.016)
- **Verdict**: REGRESSION — mean dropped -0.016

## Per-fixture deltas

| Fixture | Baseline | Candidate | Delta |
|---|---|---|---|
| CLN-HELDOUT-5ARI-PROSTATE | 0.480 | 0.480 | ±0.000 |
| CLN-HELDOUT-A1C-ELDERLY | (missing) | 0.350 | n/a |
| CLN-HELDOUT-ASPIRIN-CRC | 0.480 | 0.480 | ±0.000 |
| CLN-HELDOUT-BISPHOSPHONATE-AI | 0.320 | 0.320 | ±0.000 |
| CLN-HELDOUT-BPH-MEDICAL | (missing) | 0.130 | n/a |
| CLN-HELDOUT-CKD-REFERRAL | (missing) | 0.310 | n/a |
| CLN-HELDOUT-DOAC-AFIB | (missing) | 0.480 | n/a |
| CLN-HELDOUT-DVT-PROPHYLAX-ORTHO | (missing) | 0.130 | n/a |
| CLN-HELDOUT-EZETIMIBE-ADD-ON | (missing) | 0.610 | n/a |
| CLN-HELDOUT-GERD-PPI-STEPDOWN | (missing) | 0.420 | n/a |
| CLN-HELDOUT-GLP1-DM2-INIT | (missing) | 0.100 | n/a |
| CLN-HELDOUT-H-PYLORI-ERADICATE | (missing) | 0.230 | n/a |
| CLN-HELDOUT-HPV-CATCHUP | 0.290 | 0.290 | ±0.000 |
| CLN-HELDOUT-LEVOTHYROXINE-TITRATE | (missing) | 0.430 | n/a |
| CLN-HELDOUT-NALOXONE-RX | (missing) | 0.240 | n/a |
| CLN-HELDOUT-OMEGA3-REDUCEIT | (missing) | 0.510 | n/a |
| CLN-HELDOUT-OPIOID-TAPER | (missing) | 0.210 | n/a |
| CLN-HELDOUT-OSTEOPOROSIS-INIT | (missing) | 0.460 | n/a |
| CLN-HELDOUT-PCSK9-PRIOR-AUTH | (missing) | 0.460 | n/a |
| CLN-HELDOUT-PCV13-PPSV23-ADULT | (missing) | 0.230 | n/a |
| CLN-HELDOUT-PEP-HIV-EXPOSURE | (missing) | 0.370 | n/a |
| CLN-HELDOUT-PERIOP-BRIDGING | (missing) | 0.200 | n/a |
| CLN-HELDOUT-PREP-HIV | (missing) | 0.470 | n/a |
| CLN-HELDOUT-SGLT2-HFREF | (missing) | 0.390 | n/a |
| CLN-HELDOUT-SHINGRIX-RZV | (missing) | 0.720 | n/a |
| CLN-HELDOUT-SMOKING-CESSATION-CANCER | 0.470 | 0.470 | ±0.000 |
| CLN-HELDOUT-SSRI-ADOLESCENT | (missing) | 0.300 | n/a |
| CLN-HELDOUT-STATIN-CV-CANCER | 0.270 | 0.270 | ±0.000 |
| CLN-HELDOUT-TDAP-PREGNANCY | (missing) | 0.650 | n/a |
| CLN-HELDOUT-URI-ANTIBIOTIC-STEW | (missing) | 0.360 | n/a |

## Per-axis means

| Axis | Baseline | Candidate | Delta |
|---|---|---|---|
| accuracy | +0.354 | +0.402 | +0.048 |
| communication_quality | +0.563 | +0.400 | -0.163 |
| completeness | +0.343 | +0.225 | -0.118 |
| context_awareness | +0.145 | +0.311 | +0.166 |
| instruction_following | +0.479 | +0.096 | -0.383 |

## Run config diff

| Field | Baseline | Candidate |
|---|---|---|
| clinical_system_prompt | `True` | (same) |
| embed_model | `nvidia/llama-nemotron-embed-1b-v2` | (same) |
| judge_model | `Qwen/Qwen2.5-7B-Instruct` | (same) |
| n_corpus_chunks | `107` | (same) |
| **n_per_trial** | `6` | `30` |
| rerank_model | `nvidia/llama-3.2-nv-rerankqa-1b-v2` | (same) |
| retrieval_mode | `hybrid` | (same) |
| retrieval_top_n | `8` | (same) |
| serve_model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | (same) |
| temperature | `0.0` | (same) |
| **trials** | `3` | `1` |

