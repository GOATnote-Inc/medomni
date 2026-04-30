# CARD delta — v0 (numpy retrieval) → v0.5 Phase 2.1 (cuVS+nx-cugraph+corpus extended)

- **Baseline**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-20260429-142936/heldout.json`
- **Candidate**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-phase2.1-20260429-231731/heldout.json`
- **Mean score**: `0.273` → `0.335` (+0.062)
- **Verdict**: MIXED — lift +0.062 but worst regression -0.150

## Per-fixture deltas

| Fixture | Baseline | Candidate | Delta |
|---|---|---|---|
| CLN-HELDOUT-5ARI-PROSTATE | 0.340 | 0.210 | -0.130 |
| CLN-HELDOUT-ASPIRIN-CRC | 0.400 | 0.250 | -0.150 |
| CLN-HELDOUT-BISPHOSPHONATE-AI | 0.200 | 0.470 | +0.270 |
| CLN-HELDOUT-HPV-CATCHUP | 0.070 | 0.140 | +0.070 |
| CLN-HELDOUT-SMOKING-CESSATION-CANCER | 0.360 | 0.330 | -0.030 |
| CLN-HELDOUT-STATIN-CV-CANCER | 0.270 | 0.610 | +0.340 |

## Per-axis means

| Axis | Baseline | Candidate | Delta |
|---|---|---|---|
| accuracy | +0.221 | +0.296 | +0.075 |
| communication_quality | +0.590 | +0.597 | +0.007 |
| completeness | +0.221 | +0.297 | +0.076 |
| context_awareness | +0.025 | +0.145 | +0.120 |
| instruction_following | +0.210 | +0.296 | +0.086 |

## Run config diff

| Field | Baseline | Candidate |
|---|---|---|
| clinical_system_prompt | `True` | (same) |
| embed_model | `nvidia/llama-nemotron-embed-1b-v2` | (same) |
| judge_model | `Qwen/Qwen2.5-7B-Instruct` | (same) |
| **n_corpus_chunks** | `50` | `78` |
| n_per_trial | `6` | (same) |
| rerank_model | `nvidia/llama-3.2-nv-rerankqa-1b-v2` | (same) |
| retrieval_mode | `hybrid` | (same) |
| retrieval_top_n | `8` | (same) |
| serve_model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | (same) |
| temperature | `0.0` | (same) |
| trials | `1` | (same) |

