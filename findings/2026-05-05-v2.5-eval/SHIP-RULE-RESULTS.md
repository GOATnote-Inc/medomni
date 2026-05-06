# V2.5 Ship-Rule Eval — FAIL

_Generated: 2026-05-06T23:05:32.926518Z_  
_Pre-registration: `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`_  
_Git HEAD: `431a57cf592571c79e6bee6c09aaf0d2a8c8fdbd`_

## Per-benchmark paired-bootstrap CI (V2.5 − V0)

| Benchmark | n | V0 mean | V2.5 mean | Δ | 95% CI | Cohen d_z | Holm reject H₀ | Rule | PASS? |
|---|---:|---:|---:|---:|:---:|---:|:---:|---|:---:|
| medqa | 600 | 0.6950 | 0.6833 | -1.17pp | [-2.83pp, +0.50pp] | -0.06 | no | delta_lower_ci > 0 | FAIL |
| pubmedqa | 600 | 0.5417 | 0.5350 | -0.67pp | [-2.67pp, +1.33pp] | -0.03 | no | delta_lower_ci >= -1pp (no regression) | FAIL |
| medxpertqa-text | 600 | 0.2917 | 0.2783 | -1.33pp | [-3.17pp, +0.50pp] | -0.06 | no | delta_lower_ci >= +5pp | FAIL |
| healthbench-hard | 600 | 0.0087 | 0.0236 | +1.48pp | [-0.66pp, +3.62pp] | 0.06 | no | delta_point_estimate > 0 | PASS |

**Overall ship-rule:** FAIL

## Ship-rule conditions (verbatim from PREREG)

- `medqa`: delta_lower_ci > 0
- `medxpertqa-text`: delta_lower_ci >= +5pp
- `healthbench-hard`: delta_point_estimate > 0
- `pubmedqa`: delta_lower_ci >= -1pp (no regression)

## Reproducibility

- `MANIFEST.sha256` — hashes for adapter, base snapshot, scripts, every output JSON
- `LEAKAGE-AUDIT.md` — 5-gram MinHash overlap + memorization probe report
- `REPRO.sh` — deterministic re-run script (seeds 42/123/7919, temp=0)
- `stats.json` — full paired-bootstrap output incl. raw resampled deltas

## What this report does NOT claim

- Does NOT replace the formal HF model card (separate doc).
- Does NOT extend to vision/audio modalities (V2.5 is text-only reasoning SFT).
- Does NOT certify safety; layer-0/1/2 guardrails are evaluated separately.

## On FAIL

Per PREREG ship_rule.on_fail: revert; debug data quality (likely insufficient
CoT diversity); re-author V2.5b PREREG.
