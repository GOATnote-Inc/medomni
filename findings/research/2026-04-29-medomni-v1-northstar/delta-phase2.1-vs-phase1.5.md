# CARD delta — Phase 2.1 (numpy fallback, 78 chunks) → Phase 1.5 (107 chunks: +PHS2008 verbatim + 5 trial summaries)

- **Baseline**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-phase2.1-20260429-231731/heldout.json`
- **Candidate**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-phase1.5-20260429-171233/heldout.json`
- **Mean score**: `0.335` → `0.385` (+0.050)
- **Verdict**: MIXED — lift +0.050 but worst regression -0.340

## Per-fixture deltas

| Fixture | Baseline | Candidate | Delta |
|---|---|---|---|
| CLN-HELDOUT-5ARI-PROSTATE | 0.210 | 0.480 | +0.270 |
| CLN-HELDOUT-ASPIRIN-CRC | 0.250 | 0.480 | +0.230 |
| CLN-HELDOUT-BISPHOSPHONATE-AI | 0.470 | 0.320 | -0.150 |
| CLN-HELDOUT-HPV-CATCHUP | 0.140 | 0.290 | +0.150 |
| CLN-HELDOUT-SMOKING-CESSATION-CANCER | 0.330 | 0.470 | +0.140 |
| CLN-HELDOUT-STATIN-CV-CANCER | 0.610 | 0.270 | -0.340 |

## Per-axis means

| Axis | Baseline | Candidate | Delta |
|---|---|---|---|
| accuracy | +0.296 | +0.354 | +0.058 |
| communication_quality | +0.597 | +0.563 | -0.033 |
| completeness | +0.297 | +0.343 | +0.046 |
| context_awareness | +0.145 | +0.145 | ±0.000 |
| instruction_following | +0.296 | +0.479 | +0.182 |

## Run config diff

| Field | Baseline | Candidate |
|---|---|---|
| clinical_system_prompt | `True` | (same) |
| embed_model | `nvidia/llama-nemotron-embed-1b-v2` | (same) |
| judge_model | `Qwen/Qwen2.5-7B-Instruct` | (same) |
| **n_corpus_chunks** | `78` | `107` |
| n_per_trial | `6` | (same) |
| rerank_model | `nvidia/llama-3.2-nv-rerankqa-1b-v2` | (same) |
| retrieval_mode | `hybrid` | (same) |
| retrieval_top_n | `8` | (same) |
| serve_model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | (same) |
| temperature | `0.0` | (same) |
| trials | `1` | (same) |

