# CARD delta — v0 (numpy retrieval, 50 chunks, single trial unseeded) → v1.0 consolidated (107 chunks, seed=42, N=3, hybrid)

- **Baseline**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-20260429-142936/heldout.json`
- **Candidate**: `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-consolidated-20260429-173557/heldout.json`
- **Mean score**: `0.273` → `0.385` (+0.112)
- **Verdict**: PASS — significant lift, no major regression

## Per-fixture deltas

| Fixture | Baseline | Candidate | Delta |
|---|---|---|---|
| CLN-HELDOUT-5ARI-PROSTATE | 0.340 | 0.480 | +0.140 |
| CLN-HELDOUT-ASPIRIN-CRC | 0.400 | 0.480 | +0.080 |
| CLN-HELDOUT-BISPHOSPHONATE-AI | 0.200 | 0.320 | +0.120 |
| CLN-HELDOUT-HPV-CATCHUP | 0.070 | 0.290 | +0.220 |
| CLN-HELDOUT-SMOKING-CESSATION-CANCER | 0.360 | 0.470 | +0.110 |
| CLN-HELDOUT-STATIN-CV-CANCER | 0.270 | 0.270 | ±0.000 |

## Per-axis means

| Axis | Baseline | Candidate | Delta |
|---|---|---|---|
| accuracy | +0.221 | +0.354 | +0.132 |
| communication_quality | +0.590 | +0.563 | -0.027 |
| completeness | +0.221 | +0.343 | +0.122 |
| context_awareness | +0.025 | +0.145 | +0.120 |
| instruction_following | +0.210 | +0.479 | +0.269 |

## Run config diff

| Field | Baseline | Candidate |
|---|---|---|
| clinical_system_prompt | `True` | (same) |
| embed_model | `nvidia/llama-nemotron-embed-1b-v2` | (same) |
| judge_model | `Qwen/Qwen2.5-7B-Instruct` | (same) |
| **n_corpus_chunks** | `50` | `107` |
| n_per_trial | `6` | (same) |
| rerank_model | `nvidia/llama-3.2-nv-rerankqa-1b-v2` | (same) |
| retrieval_mode | `hybrid` | (same) |
| retrieval_top_n | `8` | (same) |
| serve_model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | (same) |
| temperature | `0.0` | (same) |
| **trials** | `1` | `3` |

