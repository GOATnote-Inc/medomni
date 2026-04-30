# CARD delta — v1.0 consolidated (no PrimeKG) → v1.0+PrimeKG (Phase 2.4)

- **Baseline**: `results/ci-medomni-heldout-consolidated-20260429-173557/heldout.json`
- **Candidate**: `results/ci-medomni-heldout-primekg-pinned-20260429-204029/heldout.json`
- **Mean score**: `0.385` → `0.358` (-0.027)
- **Verdict**: REGRESSION — mean dropped -0.027

## Per-fixture deltas

| Fixture | Baseline | Candidate | Delta |
|---|---|---|---|
| CLN-HELDOUT-5ARI-PROSTATE | 0.480 | 0.400 | -0.080 |
| CLN-HELDOUT-ASPIRIN-CRC | 0.480 | 0.250 | -0.230 |
| CLN-HELDOUT-BISPHOSPHONATE-AI | 0.320 | 0.330 | +0.010 |
| CLN-HELDOUT-HPV-CATCHUP | 0.290 | 0.310 | +0.020 |
| CLN-HELDOUT-SMOKING-CESSATION-CANCER | 0.470 | 0.410 | -0.060 |
| CLN-HELDOUT-STATIN-CV-CANCER | 0.270 | 0.450 | +0.180 |

## Per-axis means

| Axis | Baseline | Candidate | Delta |
|---|---|---|---|
| accuracy | +0.354 | +0.337 | -0.017 |
| communication_quality | +0.563 | +0.507 | -0.057 |
| completeness | +0.343 | +0.283 | -0.060 |
| context_awareness | +0.145 | +0.219 | +0.074 |
| instruction_following | +0.479 | +0.256 | -0.222 |

## Run config diff

| Field | Baseline | Candidate |
|---|---|---|
| clinical_system_prompt | `True` | (same) |
| embed_model | `nvidia/llama-nemotron-embed-1b-v2` | (same) |
| judge_model | `Qwen/Qwen2.5-7B-Instruct` | (same) |
| n_corpus_chunks | `107` | (same) |
| n_per_trial | `6` | (same) |
| rerank_model | `nvidia/llama-3.2-nv-rerankqa-1b-v2` | (same) |
| **retrieval_mode** | `hybrid` | `primekg-hybrid` |
| retrieval_top_n | `8` | (same) |
| serve_model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | (same) |
| temperature | `0.0` | (same) |
| trials | `3` | (same) |

