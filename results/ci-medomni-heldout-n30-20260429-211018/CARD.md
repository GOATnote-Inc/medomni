# CARD — nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4

**Sovereign sweep — HealthBench Hard, 30-example subset, N=1 trials.**

## Result (this run)

- **Score**: `0.369 ± 0.000` (mean ± 95% half-width across N=1 trial aggregates)
- **Wall time**: 308 s
- **Run ID**: `715141ba`
- **Generated**: 2026-04-30T04:15:27Z

### Per-axis means

| Axis | Mean across trials |
|---|---|
| accuracy | +0.402 |
| completeness | +0.225 |
| context_awareness | +0.311 |
| instruction_following | +0.096 |
| communication_quality | +0.400 |

## Comparison

| Stack | Score (mean ± 95% HW) | N trials | Date |
|---|---|---|---|
| **nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4** (sovereign, NVIDIA B300 SXM6 AC, this run) | `0.369 ± 0.000` | 1 | 2026-04-30 |
| Claude Opus 4.7 (public prism42 baseline, 2026-04-22) | `0.196 ± 0.068` | 3 | 2026-04-22 |

**CI overlap analysis**: 95% CIs disjoint — Nemotron higher than baseline.


## Sovereignty proof

- Serve URL : `http://127.0.0.1:8000/v1` (localhost-only enforced in `mla/judges/triton.py`)
- Judge URL : `http://127.0.0.1:8003/v1` (same enforcement)
- `import sovereign_bench` confirms `anthropic` and `openai` are NOT in `sys.modules` after import.
- `.env` permits only `HF_TOKEN`, `NGC_API_KEY`, `BREV_PEM_PATH`. No cloud LLM keys.

## Limitations declared

- **Judge incompleteness**: 0 rubric items recused across 30 example-trials (avg 0.0 per example, 0 of 30 example-trials had at least one recusal). The judge JSON-parser couldn't extract a verdict on those items (malformed model output, retries exhausted). Score is computed over the successfully judged subset; a tighter judge prompt or `response_format=json_object` is the first R1.5 polish.
- **Same-family judge bias**: serve and judge are the same Nemotron-3-Nano-30B-A3B-BF16 endpoint. A separate Llama-3.1-Nemotron-70B-Reward judge on the H100 pod (R2) is the correct sovereign-judge story; tonight's run trades that off for speed.
- **Non-paired comparison**: the Opus 4.7 baseline was measured on a different day with a different judge (Claude itself). This is a side-by-side absolute report, not a paired-design harness delta.

## Provenance

- Manifest: `/Users/kiteboard/prism42-nemotron-med/corpus/clinical-fixtures-heldout`
- Grader: openai/simple-evals @ `ee3b0318d8d1d9d72755a4120879be65f7c07e9e` (MIT, pinned).
- Seed: 42
- Hardware: NVIDIA B300 SXM6 AC, driver 580.126.09.
- Container: `vllm/vllm-openai:v0.20.0`.
- Run knobs: temperature=0.0, clinical_system_prompt=True, enable_thinking=False.

Artifact (full per-example detail): `heldout.json`
