# HF Hub Model Card — DRAFT for V2.5 publication (T+18h after ship-rule)

**Status:** DRAFT. Do NOT publish until ship-rule eval passes AND E4 + E5
contamination verdicts are clean.

**Target Hub repo:** `GOATnote-Inc/medomni-nemotron-v2.5-lora` (private at
publication; promote to public after physician adjudication of
sample outputs).

---

```yaml
---
license: apache-2.0
base_model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
library_name: peft
tags:
  - medical
  - clinical-reasoning
  - lora
  - peft
  - usmle
  - healthcare
  - chain-of-thought
  - reasoning-sft
language:
  - en
datasets:
  - UCSC-VLAA/MedReason
  - FreedomIntelligence/medical-o1-reasoning-SFT
pipeline_tag: text-generation
---
```

# medomni-nemotron-v2.5-lora

A LoRA adapter (rank 64, alpha 128) on
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` fine-tuned with
KG-grounded medical chain-of-thought traces for clinical reasoning.
Trained on a single H200 (Brev `evil-cyan-lobster`) over 23h of
walltime. Adapter ships with sha256 verification and a full
reproducibility manifest.

## Adapter integrity

```
adapter_model.safetensors
sha256: 94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c
```

Verify locally after download:
```bash
sha256sum adapter_model.safetensors
# expected: 94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c
```

## Base model

- **Model:** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- **Architecture:** NemotronH (hybrid Mamba2 + Transformer + MoE),
  30B params (a 3B active), multimodal (text + vision + audio).
- **License:** NVIDIA Open Model License (verify on the base repo).
- Compatibility: this adapter targets `q/k/v/o_proj` (attention),
  `in_proj/out_proj` (Mamba), and `mlp1` (vision projector). MoE
  expert weights are NOT touched.

## Training data

| Dataset | HF ID | License | Size | Role |
|---|---|---|---|---|
| MedReason | `UCSC-VLAA/MedReason` | Apache-2.0 | 32,682 traces | KG-grounded medical CoT |
| medical-o1-reasoning-SFT | `FreedomIntelligence/medical-o1-reasoning-SFT` | Apache-2.0 | ~25,000 traces | o1-style scratchpad reasoning |

**Dataset SHAs and revisions** are recorded in the run manifest at
`/workspace/v2.5-prod/manifest.json` (mirrored to this repo at
`MANIFEST.json` post-upload).

## Training recipe

| Field | Value |
|---|---|
| precision | bf16-mixed |
| optimizer | AdamW |
| learning_rate | 2e-5 |
| warmup_ratio | 0.03 |
| scheduler | cosine |
| weight_decay | 0.0 |
| per_device_batch_size | 1 |
| gradient_accumulation_steps | 16 (effective batch 16) |
| max_seq_length | 1536 (post-OOM remediation; PREREG'd 8192) |
| epochs | 1 |
| total_steps | 3243 |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA target_modules | q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, mlp1 |
| trainable_params | 36,581,376 (0.116% of 31.6B) |
| seed | 42 |
| walltime | 23h 10min on a single H200 |
| final train loss | 0.992 |
| final eval loss | 1.012 |

Pre-registration: <link to PREREG.yaml in this repo>.

## Eval results — TBD post ship-rule

(This section will be replaced with the ship-rule CARD numbers
once eval completes. Placeholder format:)

| Benchmark | V0 | V2.5 | Delta | 95% CI | Cohen's d | Pass? |
|---|---|---|---|---|---|---|
| MedQA-USMLE | TBD | TBD | TBD | TBD | TBD | TBD |
| MedXpertQA-Text | TBD | TBD | TBD | TBD | TBD | TBD |
| HealthBench-Hard | TBD | TBD | TBD | TBD | TBD | TBD |
| PubMedQA-L | TBD | TBD | TBD | TBD | TBD | TBD |

Multiple-comparison correction: Holm-Bonferroni at k=4. Full protocol:
`STATS-PROTOCOL.md`.

## Contamination audit

- **N-gram overlap** (E4) computed before publication; results in
  `DATA-LEAKAGE-REPORT.md`. (Verdict TBD at draft time.)
- **Memorization probe** (E5) run post-eval; results in
  `E5-RESULTS.md`. (Run T+5h after ship-rule.)

## Intended use

- Clinical reasoning support for English-language USMLE-style and
  evidence-based-medicine queries.
- Research artifact for the medical-LLM community.
- Component in a sovereign (no cloud-LLM-keys-in-the-loop) medical
  inference stack — see the parent `medomni` repository for the full
  serve stack.

## Out-of-scope use

- Direct clinical decision-making without physician oversight.
- Triage in emergency-room settings (this model is not a triage system;
  see `safeshift`/`lostbench` for the evaluation infrastructure that
  measures emergency-medicine safety).
- Languages other than English.
- Pediatric, OBGYN, or psychiatric specialties without specialist review
  (training data weighted toward general medicine).

## Ethical considerations

- Training data is derived from publicly licensed medical reasoning
  corpora; it is NOT real patient data and contains no PHI.
- The model can produce fluent-but-incorrect clinical content. The
  model's confidence does NOT correlate with correctness — see the
  HealthBench-Hard score; high-stakes use REQUIRES a human in the loop.
- The base model's training data carries the biases documented by
  NVIDIA; the LoRA does not correct for them.

## Citation

```bibtex
@misc{medomni-v2.5-lora-2026,
  title         = {medomni-nemotron-v2.5-lora: KG-grounded medical chain-of-thought reasoning adapter for Nemotron-3-Nano-Omni},
  author        = {Dent, Brandon},
  year          = {2026},
  publisher     = {Hugging Face},
  howpublished  = {\url{https://huggingface.co/GOATnote-Inc/medomni-nemotron-v2.5-lora}},
  note          = {Adapter sha256: 94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c}
}
```

## Acknowledgments

- NVIDIA NemotronH team for the multimodal Omni base.
- UCSC-VLAA team for the MedReason corpus.
- FreedomIntelligence team for medical-o1-reasoning-SFT.
- Brev / Nebius / Hyperstack for the H200 + B300 + H100 compute.

---

## Publication command (RUN AT T+18h, NOT NOW)

```bash
# Verify HF token write scope first (see HF-TOKEN-AUDIT.md).
# Verify ship-rule passed first (see CARD.md).
# Verify E4 + E5 verdicts both green (no >5% contamination on any benchmark).

cd /workspace/v2.5-prod
huggingface-cli upload GOATnote-Inc/medomni-nemotron-v2.5-lora \
  . \
  --repo-type=model \
  --commit-message="V2.5 reasoning-SFT LoRA — initial publication; sha256 94d1f8d1eb23..."

# Verify upload integrity:
huggingface-cli download GOATnote-Inc/medomni-nemotron-v2.5-lora \
  adapter_model.safetensors \
  --local-dir /tmp/v2.5-verify
sha256sum /tmp/v2.5-verify/adapter_model.safetensors
# Must be 94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c
```

## Pre-publication blockers (must be cleared)

- [ ] Ship-rule eval CARD complete with all 4 benchmarks scored
- [ ] All 4 ship-rule pass conditions met OR documented exceptions
- [ ] E4 DATA-LEAKAGE-REPORT.md verdict per benchmark (no >10% overlap)
- [ ] E5 E5-RESULTS.md run + verdict (no >10% verbatim recall)
- [ ] HF token write scope confirmed (HF-TOKEN-AUDIT.md updated)
- [ ] Whitepaper draft (or short release blog) linked
- [ ] Physician adjudication of 20 sample outputs (post-publication
      possible, but ideally pre — coordinate with author)
