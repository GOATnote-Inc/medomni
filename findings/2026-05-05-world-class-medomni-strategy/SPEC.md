# World-Class MedOmni Trajectory — Strategy SPEC

**Date:** 2026-05-05
**Status:** DRAFT — synthesis of 4 parallel research-team reports, awaiting user OK before any further training fires.
**Predecessor:** [`findings/2026-05-01-medgemma-recipe/MEDOMNI-NEMOTRON-RECIPE.md`](../../../prism42-nemotron-med/findings/2026-05-01-medgemma-recipe/MEDOMNI-NEMOTRON-RECIPE.md) — supersedes its V2/V3 stages with May-2026-current recipe.

---

## Why this SPEC exists

User directive 2026-05-05: target **world-class clinical reasoning + tool use that exceeds all other LLMs**, on a dedicated NVIDIA B300 (catfish). Re-review all NVIDIA components + Nemotron-3-Nano-Omni capabilities; multiple agent teams agree on the path; then incorporate the existing V1-trained / V2-not-fired / catfish-base-mismatch state.

Four parallel research teams dispatched (Generator-Validator pattern). Each scoped non-overlappingly:

| Team | Scope | Top recommendation |
|---|---|---|
| #1 | Medical-LLM training SOTA May 2026 | Replace V3-DPO with GRPO on verifiable problems, preceded by reasoning-trace SFT + tool-call SFT |
| #2 | NVIDIA B300 serving stack SOTA | flashinfer_cutlass MoE + GPU util 0.90 + prefix caching + Eagle-3 spec-decode → 1.4-1.9× decode + 2-4× concurrency, zero quality loss |
| #3 | Multimodal clinical reasoning + tool use | Bolt retrieval-grounded vision bridge (BiomedCLIP/CONCH → primekg_lookup) + citation-required `guided_json` output schema |
| #4 | Eval gauntlet to claim SOTA | Beat MedAgentBench first — only benchmark stressing tool-CALL+ARG+CHAIN simultaneously with longitudinal EHR + clinical correctness |

---

## 1. Cross-team convergence (load-bearing recommendations)

### 1.1 Tool-use is the asymmetric win — beat MedAgentBench first

Three of four teams independently flagged this:

- **Team #1**: "V0→V3 plan has zero tool-call training stage; this is fatal for tool-use claims."
- **Team #3**: "Current MedAgentBench leader is Claude 3.5 Sonnet v2 at 69.67% success on 300 FHIR tasks — that is the bar."
- **Team #4**: "Beat MedAgentBench first… most defensible single claim with 100 H200-hrs."

**Decision:** the headline claim becomes "**open 30B beats Claude Opus on Stanford MedAgentBench by ≥5pp**." Tool-use SFT/RL becomes the load-bearing training addition.

### 1.2 Reasoning-trace SFT is now table-stakes (was missing)

Teams #1 and #3 converge:

- Team #1: "MedReason (KG-grounded CoT, 32K pairs) lifts DeepSeek-Distill-8B by +7.7%; medical-o1-reasoning-SFT 25K traces; existing V2 plan (HealthBench-train + MedQA + MedMCQA + PubMedQA-L) is **all answer-only**, no CoT supervision."
- Team #3: "Distill on MedR1-style traces (DeepSeek-R1 671B → Llama-8B medical CoT) — published gain on MedQA."

**Decision:** insert V2.5 Reasoning-SFT stage before any further training fires.

### 1.3 GRPO replaces DPO at V3 — the recipe must change

Teams #1, #3, #4 converge:

- Team #1: "AlphaMed-8B (GRPO) **beats DeepSeek-V3-671B and Claude-3.5-Sonnet on MedXpertQA** at 8B parameters; MediX-R1-8B at 68.8% beats MedGemma-27B at 68.4%; every May-2026 SOTA medical model wins via verifiable-reward RL, not DPO."
- Team #3: "Reasoning-RL is SOTA path."
- Team #4: GRPO-trained models lead the MedXpertQA + MedAgentBench leaderboards.

**Decision:** V3-DPO is replaced by V3-GRPO. Optional V3.5-DPO is reduced to refusal-pair calibration (no capability lift).

### 1.4 Citation-required output via `guided_json` is the dominant anti-hallucination win

Team #3 explicit, no team disagrees: vLLM `guided_json` (xgrammar backend, V1, minimal overhead) — every claim carries `{evidence_id, span}` from a retrieved PubMed/PrimeKG node; no node ⇒ field null ⇒ model says "uncertain."

**Decision:** add to inference layer alongside the existing 5-tool agent surface. No training change required.

---

## 2. Cross-team disagreements — explicit decisions

### 2.1 HealthBench-Hard SOTA number

| Team | Source | Number |
|---|---|---|
| #1 | OpenAI HealthBench-Professional PDF, 2026-04-22 | Muse Spark 42.8 / GPT-5.4 40.1 |
| #4 | HealthBench paper, May/Sep 2025 | GPT-5 0.46 |

Team #1's source is fresher. "Muse Spark" is unfamiliar — likely a recent OpenAI-anchor model. **Resolution:** WebFetch Team #1's PDF before locking the eval CARD; default to the higher number (46 on a 0-100 scale, or 0.46 on 0-1) as the to-beat floor. Either way the gap from V0 (5.4) is enormous; **MedXpertQA + MedAgentBench are the more reachable lifts**.

### 2.2 Midtraining sprinkle vs zero CPT

Team #1 recommends a 300-500M-token midtraining of MedReason CoT + PubMed abstracts blended into the SFT mix. Other teams don't address.

**Decision:** **defer** — adds ~$50-100 + setup complexity. V2.5 reasoning-SFT + V2.7 tool-call SFT carry most of the lift; revisit if V3 plateaus.

### 2.3 HuatuoGPT-Vision-34B baseline

Team #1 calls it 2024-vintage; Team #3 says rebuilt on Qwen2.5-VL backbone in April 2025, now the open-weights ceiling on PMC-VQA / OmniMedVQA. **Team #3 is correct** — 2025 rebuild. To beat HuatuoGPT-V-34B on multimodal medical VQA, retrieval-bridge (Team #3 top rec) is the path, not just LoRA.

---

## 3. The revised recipe (supersedes V2/V3 of MEDOMNI-NEMOTRON-RECIPE.md)

### Stage 0 — Base + V1 disposition (existing, already trained)

- **Base for new training**: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (multimodal Omni). **Not** the text-only BF16 variant V1 was trained against.
- **V1 imaging-PEFT** (lobster `/workspace/ckpt/v1-pathd-out/iter_0015594`): trained against `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (text-only). Cannot be deployed to catfish (Omni base, different model entirely). **Disposition:** salvage as a CARD-only exhibit; run paired V0→V1 text-only eval on lobster to know if PubMedVision PEFT moved the needle. Not production weights.
- **V2 (originally planned, answer-only multi-task SFT)**: **DO NOT FIRE.** Superseded by V2.5/V2.7 below.

### Stage 1 — V2.5 Reasoning SFT (NEW)

| Property | Value |
|---|---|
| Method | LoRA r=64, α=128, dropout=0.05 on Nemotron-Omni-30B (BF16 on lobster) |
| Target modules | Attention out_proj + Mamba in_proj/out_proj + projectors (NOT routed-MoE expert weights — keeps adapter hot-loadable on NVFP4 base per Team #2) |
| Corpus | MedReason 32.7K (KG-grounded CoT) + medical-o1-reasoning-SFT 25K + 5K DeepSeek-R1-distilled USMLE traces |
| Hyperparams | bf16-mixed, lr 2e-5, batch 32, seq 8K (medical CoT is long), 1 epoch |
| Wall | 8-10 hr H200 |
| Cost | ~$80 |
| Expected lift | +5-8% MedQA, +10-15% MedXpertQA-Text |
| PREREG | Author before fire (`findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`) |

### Stage 2 — V2.7 Tool-call SFT (NEW)

| Property | Value |
|---|---|
| Method | LoRA r=64 on V2.5 weights |
| Target modules | Same as V2.5 (hot-loadable scope) |
| Corpus | Hermes Function-Calling-v3 ~60K + ToolACE-subset 40K + 5K synthetic FHIR/MedAgentBench traces (loss-masked: only loss on tool-call structure + args, not surrounding chat) |
| Hyperparams | bf16-mixed, lr 1e-5, batch 16, seq 4K |
| Wall | 4-6 hr H200 |
| Cost | ~$40 |
| Expected lift | +15-25% BFCL, +20-30% MedAgentBench |
| PREREG | Author before fire |

### Stage 3 — V3 GRPO (NEW, replaces planned V3-DPO)

| Property | Value |
|---|---|
| Method | GRPO on V2.7 weights |
| Group size | 8 |
| KL β | 0.04 |
| Learning rate | 5e-7 |
| Sequence length | 8K |
| Reward | Composite verifiable: correctness (medical answer match) + format (CoT structure) + medical-verifier judge (sovereign Qwen2.5-7B-Instruct) |
| Corpus | HuatuoGPT-o1's 40K verifiable medical problems + Clinical-R1 multi-objective subset |
| Wall | 12-18 hr H200 |
| Cost | ~$130 |
| Expected lift | +3-6% on Hard slices over DPO baseline; reduces shortcut hallucination per MedCEG findings |
| Notes | Do **not** use SimPO — controlled study (arXiv 2603.19335) shows scale-dependent ranking inversions at 30B |

### Stage 3.5 — V3.5 DPO refusal (kept, scope reduced)

| Property | Value |
|---|---|
| Method | DPO on V3 weights, 5-10K refusal/safety pairs only |
| Wall | 4 hr H200 |
| Cost | ~$40 |
| Lift | Safety/refusal calibration only — no capability lift |

### Stage 4 — V_final → HF release

- Merge V3.5 LoRA into BF16 weights → quantize to NVFP4 via TensorRT Model Optimizer
- Smoke-test merged-NVFP4 on 30 fixtures
- Swap catfish vllm-omni-b300 to V_final
- HF push: `huggingface.co/GOATnote-Inc/medomni-nemotron-3-nano-omni-medical` (Apache-2.0)
- Wall: ~4 hr laptop + B300 quantization

### Total budget

| | |
|---|---|
| **H200-hrs** | 28-38 |
| **Cost** | ~$290-330 |
| **Wall** | ~5-6 days end-to-end on 1×H200 |
| **Decision gates** | PREREG ship rule per stage; revert if val PPL > 10 or paired-bootstrap CI fails |

---

## 4. B300 serving config upgrade (Team #2)

Apply ALL of these to the catfish vllm-omni-b300 container. Each is reversible by reverting the docker run command.

### 4.1 High-leverage configuration changes (ranked)

1. `--gpu-memory-utilization 0.90` (current 0.72) — exploits the 288 GB B300; 2-4× concurrent sessions
2. `--enable-prefix-caching` — system prompt + tool schema reuse on every agent turn
3. `--moe-backend flashinfer_cutlass` — pins TRT-LLM 1.2's CuteDSL NVFP4 grouped-GEMM kernel; 10-20% decode tok/s on MoE
4. `--speculative-config '{"method":"eagle3","num_speculative_tokens":4}'` — 1.4-1.9× decode speedup, no quality loss
5. `--video-pruning-rate 0.5` + `--media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'` — required for the 9.2× video-throughput claim
6. `--max-num-seqs 384` (current 32) — recipe default
7. Set env `VLLM_ATTENTION_BACKEND=FLASHINFER` — enables NVFP4-attention kernel for B300's 2× attention silicon
8. Keep `--kv-cache-dtype fp8` — NVFP4-KV is lossy on long-context reasoning per Red Hat Feb-2026 NVFP4 paper

### 4.2 Multi-LoRA hot-load on B300

Team #2 finding: **vLLM 0.20.x supports `--enable-lora` with NVFP4 base + BF16 LoRA adapters via `--lora-target-modules`** — but ONLY for attention/Mamba projectors, **NOT routed-MoE expert weights** (FP4 with per-block scales). For MoE-expert adaptation, must merge-then-requantize (Stage 4 of the recipe).

**Implication for V2.5/V2.7/V3:** target attention/Mamba projectors only. Adapters stay hot-loadable; intermediate versions can deploy without merge-and-requantize. Final V3.5 → V_final → merge-and-requantize for HF release.

### 4.3 Stack swaps to consider

- **Stay on vLLM 0.20.x** during the LoRA-iteration phase. tool-call-parser + reasoning-parser wired; multimodal works; supports hot-load.
- **TensorRT-LLM 1.2.1 + Triton** is the right swap once V_final ships and traffic shape stabilizes. Static engine + disaggregated prefill/decode + multi-LoRA. Trade-off: 30-60 min engine builds per LoRA; harder to hot-swap.
- **NIM container** (`build.nvidia.com → Nemotron-3-Nano-Omni-Reasoning-NIM`): use as **production endpoint after V_final** — pre-tuned, NVIDIA-supported, optimized TRT-LLM engines. Avoid during LoRA-iteration (vendor-locked, no custom adapters).

---

## 5. Inference-time additions (Team #3)

### 5.1 Citation-required output via `guided_json`

vLLM `guided_json` with xgrammar backend (V1, minimal overhead). Schema:

```json
{
  "answer": {"type": "string"},
  "claims": [{
    "text": "string — one factual claim",
    "evidence_id": "string|null — PubMed PMID or PrimeKG node_id",
    "span": "string|null — exact quote from evidence"
  }],
  "uncertainty": "string — what's missing or unknown"
}
```

No retrieved node ⇒ `evidence_id: null` ⇒ model says "uncertain in `uncertainty` field" rather than fabricate.

### 5.2 Self-consistency at inference (hard questions)

n=5-8 samples, marginalize. Apply on HealthBench Hard / MedXpertQA-class questions. Cost: 5-8× per turn but only on the hard slice. Empirical clinical-accuracy lift documented.

### 5.3 Reasoning-budget cap

`</think>` token limit at N to prevent ruminate-loop pathology that emerges past ~6K thinking tokens.

### 5.4 Anti-hallucination stack

- NeMo Guardrails 0.21+ Colang 2.0 input + output rails as **separate vLLM endpoints** (Team #2: not compiled into main graph; lets rails iterate independently of base model)
- Llama-3.1-Nemotron-Safety-Guard-8B-V3 input rail
- Nemotron-Content-Safety-Reasoning-4B output rail
- RadFact-style sentence-level entailment for image-derived claims (lightweight NLI cross-encoder over generated text vs retrieved passage; reject sentences below threshold)

---

## 6. Eval gauntlet (Team #4)

The 12-benchmark gauntlet that proves "exceeds all other LLMs in clinical reasoning + tool use":

| # | Benchmark | Axis | SOTA May-26 | Anchor to beat |
|---|---|---|---|---|
| 1 | HealthBench (full) | Clinical reasoning, text | 0.60 / 0.46 Hard | GPT-5 thinking (or Muse Spark 42.8) |
| 2 | HealthBench Responding-Under-Uncertainty | Safety/calibration | ~0.78 | GPT-5 |
| 3 | MedQA-USMLE | Closed-book knowledge | 97.0% | Med-Gemini-L 1.5 / GPT-5 saturated |
| 4 | MedMCQA | Closed-book non-US | ~83% | Med-Gemini-L |
| 5 | MMLU-Pro Medical | Hard MCQ | ~92% | GPT-5 / Claude Opus 4.5+ |
| 6 | **MedXpertQA Text + MM** | **Expert-level reasoning, multimodal** | **41.8% / 36.2%** | **o3 / GPT-5 thinking — AlphaMed-8B already beats DeepSeek-V3-671B here** |
| 7 | OmniMedVQA | Multimodal vision | ~82% | HuatuoGPT-V-34B / Med-Gemini-M |
| 8 | SLAKE-en + VQA-RAD + Path-VQA | Multimodal | 89/87/91 | LLaVA-Med-1.5 / HuatuoGPT-V-34B |
| 9 | **MedAgentBench** | **Tool-use, longitudinal EHR** | **69.7%** | **Claude 3.5 Sonnet — Opus 4.x ~75% per Stanford HAI Mar 2026** |
| 10 | BFCL v3 | Tool-CALL/ARG/CHAIN | ~83% | GPT-5 / xLAM-2-70b / Opus 4.7 |
| 11 | MedSafetyBench / med-RM | Safety/refusal | ~94% | GPT-5 thinking |
| 12 | DoctorBench (Diagens, 2026-05-01) | Real-world clinical, novelty | new | Land here for novelty claim |

### 6.1 Methodology

- N=3 seeded trials, deterministic decoding (temp=0; for thinking models, fixed thinking_budget)
- Paired bootstrap 10K resamples on per-item scores; 95% CI; same item set across compared models
- Graders: GPT-4.1 primary rubric grader; **Qwen2.5-7B-Instruct cross-family sovereignty backup** — disagreement >5pp triggers manual adjudication on 50-item slice
- Refusal/safety axis: never let the agent's own family grade itself
- Tool benchmarks: AST + executable checks (BFCL convention), not LLM grading

### 6.2 Reproducibility manifest per run

```
weights_sha256, base_model_id, lora_sha256 (if applicable),
eval_script_git_sha, eval_script_url,
prompt_template_sha256, system_prompt_sha256,
grader_model_id, grader_prompt_sha256,
decode_params (temp, top_p, max_tokens, thinking_budget),
seed_list, item_manifest_sha256, dataset_revision,
hardware (gpu, driver, cuda, vllm_version),
runtime_seconds, total_tokens_in/out, cost_usd
```

### 6.3 The headline claim

**"Open-weight 30B Nemotron-3-Nano-Omni-medical (Apache-2.0) clears Claude Opus 4.x on Stanford MedAgentBench by ≥5pp paired-bootstrap CI."** Most defensible single claim with the budget. Backed by MedXpertQA + BFCL + OmniMedVQA secondaries to demonstrate breadth.

---

## 7. How V1 / V2-not-fired / catfish-base-mismatch fits

### 7.1 V1 imaging-PEFT (already trained)

- Trained against text-only `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- **Cannot deploy to catfish** (which serves multimodal Omni-NVFP4) — base model mismatch, not just quantization
- **Disposition:** keep V1 as a research artifact + run V0→V1 paired eval on lobster against the BF16 base. CARD documents "PubMedVision PEFT lift on text-only base" as a methodology data point. Not production weights.
- iter-14 V1 export to HF format is still useful — converts the Megatron-format checkpoint into something portable. Lets V1 exist as a Hugging Face artifact even if not deployed.

### 7.2 V2 (originally planned, answer-only multi-task SFT)

- **DO NOT FIRE** as originally specified. Three of four research teams converge that answer-only SFT is dominated by reasoning-trace SFT + tool-call SFT + GRPO.
- The PREREG-V2 task (#79 completed) is now historical. Author **new PREREGs** for V2.5, V2.7, V3-GRPO, V3.5-DPO before each fires.

### 7.3 Catfish base mismatch

- Resolved by retraining the new V2.5 against the Omni multimodal base from the start, not the text-only BF16 base
- LoRA scope (attention/Mamba projectors only, per Team #2) keeps adapters hot-loadable on catfish's NVFP4 base via `--enable-lora --lora-target-modules` in vLLM 0.20.x
- Final V3.5 → V_final merge-and-requantize for the HF release

### 7.4 The autonomous loop (Karpathy autoresearcher) — what actually changes

| Loop component | Change |
|---|---|
| **Generation** (narwhal + catfish factory_loop.py) | **Continue**. Add MedAgentBench-shaped FHIR-tool-call traces to the generation tasks. |
| **Judge ensemble** (gpt-4.1 + Qwen) | **Continue**. Add medical-verifier judge for GRPO reward. |
| **Training** (lobster) | **Pivot from V2-multi-task-SFT-as-planned → V2.5 + V2.7 + V3-GRPO + V3.5-DPO chain.** This is the load-bearing change. |
| **Deploy** (catfish) | **Apply Team #2's serving config upgrade now** (independent of training pivot). LoRA hot-load enabled for V2.5/V2.7/V3 iteration; V_final merge-and-requantize for production. |
| **Eval** (lobster + laptop) | **Replace the implicit eval with the explicit 12-benchmark gauntlet.** MedAgentBench is the headline; everything else is breadth. |

---

## 8. Timeline — 5-6 days end-to-end

| Day | Action | GPU-hr | Risk |
|---|---|---|---|
| 0 (today) | Apply Team #2 serving config to catfish (low risk, reversible). Author new PREREGs (V2.5, V2.7, V3, V3.5). Author training scripts. Run V0→V1 paired eval on lobster (text-only). | 2-3 | low |
| 1 | Fire V2.5 Reasoning SFT on lobster against Omni base. Smoke-test step-1000 checkpoint (kill-switch on val PPL > 10). | 8-10 | low |
| 2 | Fire V2.7 Tool-call SFT on V2.5. Run V0→V2.5 paired eval in parallel. | 4-6 | low |
| 3-4 | Fire V3 GRPO on V2.7. Run V2.5→V2.7 paired eval in parallel. | 12-18 | medium (RL convergence) |
| 5 | V3.5 DPO refusal. V2.7→V3 paired eval. | 4 | low |
| 6 | Merge V3.5 → bf16 → NVFP4 quantize → smoke-test 30 fixtures → swap catfish to V_final. | 4 | medium (prod blip during model swap) |
| 7+ | Eval gauntlet sweep (12 benchmarks). HF model card with paired-bootstrap CIs. arXiv preprint draft. | — | low |

Total: **~32-41 H200-hrs + ~$310-360**. Budget-fit within the user's stated dedication.

---

## 9. What this SPEC does NOT do

- **Does not train the model.** All training happens on the user's pods; the babysitter loop monitors via the harmony-contract read-only fleet pulse.
- **Does not change the eval grader.** Sovereign Qwen judge remains the cross-family backup; gpt-4.1 remains the primary per HealthBench convention.
- **Does not deploy V1 to catfish.** V1 is text-only-base; catfish's Omni multimodal base is the right target for the new training.
- **Does not abandon the PubMedVision lift.** V2.5 mix can include PubMedVision + MedReason (vision-grounded CoT exists); the recipe just stops treating "pure-vision PEFT" as a deploy-shippable stage.
- **Does not exceed the user's dedicated H200 budget.** ~32-41 hr is well within the 1×H200 capacity over a week.

---

## 10. Pre-fire checklist (user-action)

Before iter-N fires V2.5 SFT:

- [ ] Confirm V2.5 PREREG ship rule
- [ ] Confirm corpus access: MedReason, medical-o1-reasoning-SFT, DeepSeek-R1-distilled-USMLE-traces (HF download, all Apache-2.0 / permissive)
- [ ] Apply Team #2 catfish serving config upgrade (no training dependency)
- [ ] Apply V0→V1 paired-eval CARD on lobster (closes the V1 question independent of trajectory)
- [ ] Confirm Omni multimodal base downloaded on lobster (currently only BF16 text-only is cached per iter-14 probe)
- [ ] Confirm the lobster `/workspace/ckpt/` has free disk for V2.5 / V2.7 / V3 checkpoints
- [ ] Confirm sovereign Qwen judge endpoint is responding for GRPO reward signal

---

## Sources (per-team)

### Team #1 (training SOTA)
- HuatuoGPT-o1 (arXiv 2412.18925), MedReason (arXiv 2504.00993), MediX-R1 (OpenReview), Med-R1 (arXiv 2503.13939v5), MedCEG (arXiv 2512.13510), Clinical-R1 (arXiv 2512.00601), MedS3 (arXiv 2501.12051), Aloe-Beta (arXiv 2505.04388), Nemotron-Research-Tool-N1 (arXiv 2505.00024), HealthBench-Professional 2026-04-22, MedQA leaderboard 2026-04-09

### Team #2 (B300 serving)
- NVIDIA Nemotron-3-Nano-Omni technical blog 2026-04-28, vLLM blog 2026-04-28, vLLM Recipes Nemotron-3-Nano docs, TensorRT-LLM 1.2.x release notes, NVIDIA Blackwell Ultra MLPerf debut, Red Hat speculative-decoding article 2026-04-16, Red Hat NVFP4 article 2026-02-04, vLLM WideEP+Blackwell blog 2026-02-03, NeMo Guardrails self-hosting docs

### Team #3 (multimodal SOTA)
- MedGemma technical report (arXiv 2507.05201), HuatuoGPT-Vision Qwen2.5-VL refresh Apr 2025, MAIRA-2 (arXiv 2406.04449), RadVLM (arXiv 2502.03333), CONCH (Nature Medicine 2024), AuscultaBase (arXiv 2411.07547v2), MedAgentBench v2 PSB 2026 paper (Chen), MedTrust-RAG (arXiv 2510.14400), MEGA-RAG (Frontiers 2026), FACTUM (arXiv 2601.05866)

### Team #4 (eval gauntlet)
- HealthBench (OpenAI), simple-evals repo, lm-evaluation-harness, MMLU-Pro (TIGER-Lab), MedXpertQA (TsinghuaC3I), OmniMedVQA (OpenGVLab), MedAgentBench (Stanford CRFM), Berkeley Function Calling Leaderboard, MedSafetyBench (AI4LIFE-GROUP), Med-Gemini paper (arXiv 2404.18416)

Full per-team transcripts archived in `/private/tmp/.../tasks/{ab9620…,a06197…,a0e214…,a11665…}.output`.
