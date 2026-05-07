# V2.5 Ship-Rule Eval — FAIL

_Generated: 2026-05-07T04:55:39.270569Z_  
_Pre-registration: `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`_  
_Git HEAD: `490a3e59901b8e2795f0e6ad16dd7a3b8d523201`_

## Per-benchmark paired-bootstrap CI (V2.5 − V0)

| Benchmark | n | V0 mean | V2.5 mean | Δ | 95% CI | Cohen d_z | Holm reject H₀ | Rule | PASS? |
|---|---:|---:|---:|---:|:---:|---:|:---:|---|:---:|
| medqa | 600 | 0.8350 | 0.8467 | +1.17pp | [-1.00pp, +3.50pp] | 0.04 | no | delta_lower_ci > 0 | FAIL |
| pubmedqa | 600 | 0.6733 | 0.6483 | -2.50pp | [-5.00pp, -0.17pp] | -0.08 | no | delta_lower_ci >= -1pp (no regression) | FAIL |
| medxpertqa-text | 600 | 0.3300 | 0.3167 | -1.33pp | [-4.83pp, +2.17pp] | -0.03 | no | delta_lower_ci >= +5pp | FAIL |
| healthbench-hard | 600 | 0.1252 | 0.1121 | -1.31pp | [-3.71pp, +1.13pp] | -0.04 | no | delta_point_estimate > 0 | FAIL |

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
