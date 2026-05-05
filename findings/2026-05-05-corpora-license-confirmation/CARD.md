# Corpora license confirmation — V2.5 / V2.7 / V3 / V3.5

**Date:** 2026-05-05 iter-21
**Trigger:** V2.5 PREREG (`findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`) pre-flight item: confirm all training corpora are Apache-2.0 or compatibly permissive before fire. The HF release at `huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical` will publish under Apache-2.0; every training-data input must propagation-compatible.

## Method

WebFetch each corpus's HF dataset API (`https://huggingface.co/api/datasets/<id>`) and read the `cardData.license` field + the `tags` filter for `license:*`. For corpora with `license: None` on HF API, fall through to the upstream GitHub LICENSE file or paper-stated license.

## Results

| Stage | Corpus | HF id | License | Source | Status |
|---|---|---|---|---|---|
| **V2.5** | MedReason (KG-grounded CoT) | `UCSC-VLAA/MedReason` | **apache-2.0** | HF API `cardData.license` + tag | ✓ confirmed |
| **V2.5** | medical-o1-reasoning-SFT | `FreedomIntelligence/medical-o1-reasoning-SFT` | **apache-2.0** | HF API + tag | ✓ confirmed |
| **V2.5** | DeepSeek-R1-distill USMLE traces | synthetic (5K target) | **apache-2.0** | distillation output of MIT-licensed DeepSeek-R1; we author the trace generation; downstream license = Apache-2.0 by our choice | ✓ |
| **V2.7** | Hermes Function-Calling (NousResearch) | `NousResearch/hermes-function-calling-v1` | **apache-2.0** | HF API + tag | ✓ confirmed (note: PREREG referenced "v3" — the actual versioned dataset on HF is v1; will use v1) |
| **V2.7** | ToolACE | `Team-ACE/ToolACE` | **apache-2.0** | HF API + tag | ✓ confirmed (subset filtered to medical-relevant API categories per PREREG) |
| **V2.7** | Synthetic FHIR/MedAgentBench traces (5K) | factory_loop output on narwhal+catfish | **apache-2.0** | we author; OpenEM 370 source corpus is Apache-2.0 + CC-BY per its CLAUDE.md | ✓ |
| **V3** | HuatuoGPT-o1 verifiable problems | `FreedomIntelligence/medical-o1-verifiable-problem` | **apache-2.0** | HF API + tag | ✓ confirmed |
| **V3** | Clinical-R1 multi-objective subset | per arXiv 2512.00601 | per paper authors' release (typically Apache-2.0 or research-only) | needs upstream verification before fire | **VERIFY at fire-time** |
| **V3.5** | MedSafetyBench preference pairs | `AI4LIFE-GROUP/MedSafetyBench` | **license: None on HF API** | HF dataset card shows no license; GitHub repo LICENSE check needed | **VERIFY** — see below |
| **V3.5** | Synthetic HealthBench-uncertainty pairs | factory_loop output | **apache-2.0** | we author | ✓ |

## MedSafetyBench license — pending verification

HF API returns `cardData.license: None`. The dataset card likely says "research-use-only" given it's a NeurIPS 2024 paper artifact. Action items at fire-time:

1. Check `https://github.com/AI4LIFE-GROUP/MedSafetyBench/blob/main/LICENSE`
2. Read the dataset README on HF for license language
3. If MIT / Apache-2.0 / CC-BY: use as-is
4. If research-only / non-commercial: **substitute** with a synthetic refusal-pair set generated via factory_loop + dual-judge filter (the V3.5 PREREG's complementary corpus is already Apache-2.0). Fall back to ~10K synthetic pairs only.

This is a 10-min check at V3.5 fire-time. Not gating earlier stages.

## Clinical-R1 license — pending verification

Per arXiv 2512.00601 (Clinical-R1 / CRPO paper). The paper authors typically release training data under permissive license, but this hasn't been verified. At V3 fire-time, check the paper's "Data Availability" section or repo LICENSE.

If restrictive, V3 GRPO can fall back to HuatuoGPT-o1's 40K alone (already confirmed Apache-2.0). 40K verifiable problems is sufficient by itself per HuatuoGPT-o1's published recipe.

## Apache-2.0 propagation chain

medomni HF release (Apache-2.0) ← V_final merged-bf16 ← V3.5 LoRA ← V3 LoRA ← V2.7 LoRA ← V2.5 LoRA ← (Apache-2.0 corpora)

Every training corpus that contributes to weights must be Apache-2.0-compatible (or relicense-able as Apache-2.0). MedSafetyBench is the only soft spot; substitution path documented above.

## Verification commands (re-run at any time)

```bash
for ds in "UCSC-VLAA/MedReason" "FreedomIntelligence/medical-o1-reasoning-SFT" "Team-ACE/ToolACE" "NousResearch/hermes-function-calling-v1" "FreedomIntelligence/medical-o1-verifiable-problem"; do
  echo "$ds:"
  curl -s "https://huggingface.co/api/datasets/$ds" | python3 -c "import json,sys; print(' license:', json.load(sys.stdin).get('cardData',{}).get('license'))"
done
```

## Disposition for V2.5 PREREG pre-flight

V2.5 fires on three corpora — MedReason + medical-o1-reasoning-SFT + R1-distilled-USMLE traces. **All three Apache-2.0**. V2.5 pre-flight blocker `corpus license check` is **CLEARED** for V2.5 specifically.

The remaining V2.5 blockers are:
1. HF_TOKEN missing on lobster (user-action)
2. Lobster disk 94% full — needs ≥60 GB free for Omni base download (user-action)
3. Omni multimodal base download (gated on #1 + #2)

`corpus license check` was the easiest pre-flight item; it's now done. The other three remain user-action.

## What this CARD does NOT do

- Does not download the corpora to lobster. That's gated on disk + HF_TOKEN.
- Does not redistribute the corpora. Only verifies licenses for own training use.
- Does not finalize MedSafetyBench / Clinical-R1 license. Those are 10-min checks at V3.5 / V3 fire time.
