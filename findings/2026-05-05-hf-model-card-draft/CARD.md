---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - medical
  - clinical
  - radiology
  - nemotron
  - nvidia
  - peft
  - lora
  - multimodal
  - sovereign
base_model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning
datasets:
  - PubMedVision
  - MedQA
  - HealthBench
  - PubMedQA
model-index:
  - name: medomni-nemotron-3-nano-omni-medical
    results:
      - task:
          type: medical-qa
          name: HealthBench Hard (text)
        dataset:
          name: HealthBench Hard
          type: healthbench
        metrics:
          - type: pass-rate
            value: 0.054
            name: V0 baseline (gpt-4.1 graded)
      - task:
          type: medical-vqa
          name: VQA-RAD
        dataset:
          name: VQA-RAD
          type: vqa-rad
        metrics:
          - type: accuracy
            value: 0.643
            name: V0 baseline
      - task:
          type: medical-vqa
          name: SLAKE-en
        dataset:
          name: SLAKE-en
          type: slake
        metrics:
          - type: accuracy
            value: 0.744
            name: V0 baseline
---

# medomni-nemotron-3-nano-omni-medical

> **DRAFT — pending V3 PASS + safety-card review.** This card is the gating
> document for the planned Hugging Face release at
> `huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical`. Numbers
> below are V0 baselines; V1 (imaging PEFT) shipped 2026-05-03 and V2
> (multi-task SFT) is in flight. The model + adapters will publish only after
> V3 (DPO refusal calibration) PASSes the pre-registered ship rule and the
> safety datasheet (`SAFETY.md`) is co-signed by counsel + a credentialed
> physician reviewer.

A medical variant of NVIDIA's `Nemotron-3-Nano-Omni-30B-A3B-Reasoning`, fine-tuned for
clinical reasoning across text, image, and audio inputs. Released **Apache-2.0** —
key differentiator vs. MedGemma's HAI-DEF gating. Trained on a sovereign 3-GPU stack
(catfish B300 / lobster H200 / narwhal H200) with zero cloud LLM API keys in the
training-data, judge, or eval pipelines.

Reference application: [`https://www.thegoatnote.com/4UWHAt`](https://www.thegoatnote.com/4UWHAt)
(public demo, no PHI; serves the V_n adapter live via `vllm-omni-b300 --enable-lora`).

## Model details

| Attribute | Value |
|---|---|
| Base model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (NVFP4 / BF16) |
| Architecture | 23 Mamba2 + 23 MoE + 6 GQA, 256K context, multimodal in / text out |
| Active parameters | ~3 B |
| Modalities | Text + image + audio in; text out |
| License | Apache-2.0 |
| Training framework | NeMo Framework PEFT LoRA via Megatron-Bridge |
| Training hardware | NVIDIA H200 (Hopper, SM 9.0), bf16-mixed |
| Inference hardware | NVIDIA B300 (Blackwell, SM 10.x), NVFP4 quantization |
| Tool-calling | Native structured (`--tool-call-parser qwen3_coder`, `--reasoning-parser nemotron_v3`) |

## Intended use

- **Primary**: clinical reasoning research and education — explaining drug-disease
  relationships, summarizing radiology reports, computing standard clinical scores
  (CHA2DS2-VASc, HAS-BLED, MELD-Na, Wells DVT, PERC), drafting nurse-shift handoffs,
  reviewing personal health records.
- **Multimodal use cases**: visual reasoning over X-ray / MRI / panoramic dental /
  ECG / wearable display readouts; ASR over auscultation audio + medication-bottle
  photos for caregiver assistance.
- **Personas supported**: physician (full diagnostic depth + literature citations),
  nurse (clinical depth + early-warning escalation cues + teaching scaffold),
  family caregiver (plain-language analogies + when-to-call-911), patient
  (FKGL ≤ 8 + shared-decision-making tone).

## Out-of-scope use

- **Not a substitute for in-person clinical evaluation.** Diagnostic and
  treatment decisions require a credentialed clinician.
- **Not for emergency triage in production.** Demo-only routing; for life-threatening
  symptoms route to 911 / 112 / local emergency services.
- **Not for handling PHI without a HIPAA-compliant deployment.** The public demo
  is explicitly no-PHI; production deployments require a self-hosted Medplum FHIR
  server with `AccessPolicy` enforcement, S3 Object Lock 7-yr `AuditEvent` retention,
  and a signed BAA.
- **Not validated for pediatric or pregnancy-specific dosing without persona override
  + human review.**

## Training data

| Source | License | Use | Pass rate after sovereign judge filter |
|---|---|---|---|
| HealthBench-train | OpenAI permissive | text reasoning | TBD |
| MedQA-train (USMLE) | research | text reasoning | TBD |
| MedMCQA | research | text reasoning | TBD |
| PubMedQA-L | CC BY | text reasoning | TBD |
| PubMedVision | Apache-2.0 | imaging-text pairs (V1 PEFT) | shipped 2026-05-03 |
| OpenEM 370 | CC BY-SA 4.0 | structured ED conditions, FHIR R4 | TBD |
| Filtered LostBench / SG2 trajectories | internal | high-quality clinical reasoning chains | ~45% pass rate, $1.20 / 1k items |

A continuous data factory on `warm-lavender-narwhal` (H200, vllm + factory_loop.py)
generates ~21 K reasoning chains per day; an ensemble judge (gpt-4.1 + sovereign
Qwen2.5-7B-Instruct) filters at ~45% pass rate with structured rejection reasons.
The curated jsonl is then deployed to `evil-cyan-lobster` (H200) for the next PEFT
round. See [`findings/2026-05-01-medgemma-recipe/MEDOMNI-NEMOTRON-RECIPE.md`](https://github.com/GOATnote-Inc/prism42-nemotron-med/blob/main/findings/2026-05-01-medgemma-recipe/MEDOMNI-NEMOTRON-RECIPE.md)
for the full recipe.

## Training procedure

| Round | Trigger | Dataset | Wall time | Status |
|---|---|---|---|---|
| **V0** | Base `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`, no fine-tune | n/a | n/a | shipped — V0 baselines below |
| **V1** | imaging-PEFT on PubMedVision (Apache-2.0); Megatron-Bridge LoRA on lobster | PubMedVision filtered subset | 11.3 hr at 2.6 s/step (12.4× faster than HF/PEFT eager) | shipped 2026-05-03 (15594/15594 steps) |
| **V2** | multi-task SFT (HealthBench-train + MedQA-train + MedMCQA + PubMedQA-L) | curated jsonl from factory | ~28 hr est. | in flight on lobster |
| **V3** | DPO refusal calibration on factory-generated preference pairs (sovereign judge ensemble) | preference pairs from factory | TBD | gated on V2 PASS |
| **HF release** | publish under Apache-2.0 to `huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical` | n/a | n/a | gated on V3 PASS + safety-card co-sign |

Hyperparameters (V1 published; V2/V3 pending): bf16-mixed precision, LoRA rank 16,
target modules per Nemotron-Omni MoE block, learning rate 1e-4 with cosine schedule,
val-PPL kill-switch at > 10. Pin manifest in
[`fleet/monitoring-specs/v1-prod-training.yaml`](https://github.com/GOATnote-Inc/prism42-nemotron-med/blob/main/fleet/monitoring-specs/v1-prod-training.yaml).

## Evaluation

### V0 baselines (gpt-4.1 graded, paired-bootstrap CI)

| Eval | V0 score | OpenAI top | MedGemma-4B comparator |
|---|---|---|---|
| HealthBench Hard (text) | **0.054** | ≤ 0.32 | not reported |
| VQA-RAD (medical visual QA) | **0.643** | n/a | MedGemma-4B-class |
| SLAKE-en (medical visual QA) | **0.744** | n/a | MedGemma-4B-class |

V1+ numbers will publish here once the V_{n-1} → V_n paired-bootstrap eval clears
the pre-registered ship rule in `PREREG.yaml`. Numbers are
**byte-deterministic** across N=3 seeded trials with cross-family judge.

### V0 → V1 progression (preview)

V1 imaging PEFT was shipped 2026-05-03 with the goal of lifting VQA-RAD + SLAKE-en
above MedGemma-4B baselines. Full paired eval with V0 vs V1 comparator is in flight
on lobster after the V2 SFT data factory closes its current batch. Result will land
as a CARD at `results/v1-imaging-peft-2026-05/CARD.md` with paired CI vs V0.

### Reproducibility

Every eval run emits a 9-layer reproducibility manifest (sha256-verified
byte-deterministic across re-emit) per [`docs/SPEC.md §5.6`](https://github.com/GOATnote-Inc/medomni/blob/main/findings/research/2026-04-29-medomni-v1-northstar/SPEC.md).
If your manifest hash diverges from ours, please open an issue with the
"reproducibility" template — that surface is exactly what we want
debuggable in public.

## Limitations and risks

- **Hallucinated guideline versions are the dominant failure mode.** The
  agent surface uses a `guideline_currency_check` tool to detect this; the
  underlying base model has no built-in guard against citing a stale
  default (e.g. DOAC vs warfarin, H. pylori first-line, GLP-1 contraindications).
  Always pair with the tool when used in clinical workflows.
- **NVFP4 inference is Blackwell-only.** The published BF16 / FP8 variants
  run on Hopper but with 2-3× higher latency. Plan accordingly.
- **English-only training.** No reliable claims for non-English clinical
  reasoning until V4+ multilingual SFT.
- **Imaging is 2D and demo-grade.** No claim of DICOM-faithful pixel-level
  diagnosis. The reference 4UWHAt application labels its imaging panel
  "DEMO MODE · NOT DERIVED FROM PIXEL DATA" and uses CC0 / CC-BY reference
  films as visual stand-ins.
- **No pediatric / pregnancy / geriatric persona-specific safety review yet.**
  V3 DPO will start to address; full red-team cycle pending.

## Bias considerations

Training corpus drawn from English-language, US-centric clinical sources
(HealthBench-train, USMLE-style MedQA, PubMed). Known under-representation:
nursing-specific applications (only 6% of 67 medical-KG studies addressed
nursing per [JMIR AI 2025](https://ai.jmir.org/2025/1/e58670/) — the wedge
the 4-persona Records OS surface fills). Validation across non-US clinical
contexts (NHS, EU, LMIC) is research-track follow-up, not v1 deliverable.

## Safety, sovereignty, and provenance

- **Zero cloud LLM API keys** in any code path of the released training,
  judge, eval, or serve stack. The public-facing reference application
  (`/4UWHAt`) calls a local vllm-omni-b300 endpoint via Vercel reverse-proxy;
  no third-party AI APIs are called from the client or server-side.
- **Local sovereign judge stack**: Qwen2.5-7B-Instruct (Apache-2.0) on H200
  for corpus filtering. Cross-family judge for eval (gpt-4.1) is documented
  as the single optional cloud dependency, used only for paired-bootstrap CI
  against the published HealthBench Hard baseline.
- **Guardrails**: NeMo Guardrails 0.21+ with Colang 2.0 rails, backed by
  local Llama-Guard-3-8B (input rail) and Nemotron-Content-Safety-Reasoning-4B
  (output rail).

## How to use

```python
# Pending HF release. Once published:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "GOATnote-Inc/medomni-nemotron-3-nano-omni-medical",
    torch_dtype="bfloat16",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("GOATnote-Inc/medomni-nemotron-3-nano-omni-medical")

prompt = (
    "Patient: 67yo F, statin-naive, LDL-C 168 mg/dL, ASCVD 15.2%. "
    "What's the appropriate first-line therapy and dose?"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(tok.decode(out[0], skip_special_tokens=True))
```

For NVFP4 (Blackwell B300) deployment with native tool-calling:

```bash
vllm serve GOATnote-Inc/medomni-nemotron-3-nano-omni-medical \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --quantization nvfp4 \
  --max-model-len 32768
```

## Citation

```bibtex
@misc{medomni-nemotron-3-nano-omni-medical-2026,
  author       = {Dent, Brandon and {GOATnote contributors}},
  title        = {{medomni-nemotron-3-nano-omni-medical}: a sovereign Apache-2.0
                  medical fine-tune of NVIDIA Nemotron-3-Nano-Omni},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical}},
  note         = {Apache-2.0 licensed; trained on H200 with PubMedVision +
                  MedQA + HealthBench-train; sovereign judge stack
                  (Qwen2.5-7B-Instruct on H200, gpt-4.1 cross-family
                  comparator only).}
}
```

## Authors and acknowledgements

- **Lead**: Brandon Dent, MD ([@bGOATnote](https://github.com/bGOATnote))
  — physician + agent engineer. Author of `factory_loop.py`,
  `judge_reasoning_sovereign.py`, `train_peft_imaging.py`. Co-author with Claude
  Opus 4.7 on harness + scaffolding.
- **Base model**: NVIDIA Nemotron team. `Nemotron-3-Nano-Omni-30B-A3B-Reasoning`
  released 2026-04-28.
- **Training framework**: NVIDIA NeMo Framework + Megatron-Bridge.
- **Eval anchors**: OpenAI HealthBench (text), VQA-RAD + SLAKE (vision).
- **Reference application**: medomni public repo,
  [`thegoatnote.com/4UWHAt`](https://www.thegoatnote.com/4UWHAt).

## Pre-release checklist (gating for HF publish)

- [ ] V2 multi-task SFT PASS on PREREG ship rule
- [ ] V3 DPO refusal calibration PASS on PREREG ship rule
- [ ] Safety datasheet `SAFETY.md` co-signed by counsel + credentialed physician
- [ ] Red-team cycle: pediatric, pregnancy, geriatric, psychiatric, ED triage
- [ ] Reproducibility manifest sha256 published with the release
- [ ] `eval/CARD.md` on the HF model repo with V_{n-1} → V_n paired CI
- [ ] Adapter weights published separately from base model (smaller download)
- [ ] License compatibility audit (Apache-2.0 propagation through
      PubMedVision + MedQA-train + HealthBench-train + OpenEM 370)
