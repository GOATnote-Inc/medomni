# NVIDIA Best Practices SPEC for medomni

Date: 2026-05-05
Audience: medomni eng (catfish/B300 prod, lobster/H200 train, narwhal/H200 factory)
Scope: May-2026 SOTA, primary-source citations, builds on existing infra.

## TL;DR — three takeaways
1. **Recover the 12.4× PEFT speedup for V2.5+.** NVIDIA shipped Day-0 Megatron-Bridge LoRA recipes for our exact base model (Nemotron-3-Nano-Omni 30B-A3B) on 2026-04-28. We're currently regressing on lobster by using HF PEFT-eager. Switch back.
2. **Use TensorRT Model Optimizer QAT, not PTQ, for V_final.** PTQ→NVFP4 silently lost 4–22% on Math-500/AIME for Llama-Nemotron-Super (NVIDIA's own data). Medical safety reasoning has the same brittle-tail profile. Budget ~1% of pre-training-equivalent steps for QAT after V3.5 DPO.
3. **Two free vLLM wins on catfish today.** `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` + `--cuda-graph-capture-size 2048` + async API server count gave gpt-oss-120B +38% throughput / +13% TTFT on B200 — same kernel path applies to Nemotron-3-Nano-Omni NVFP4 on B300. Stage one flag at a time per our iter-15 lesson.

## 1. NeMo Framework PEFT for 30B MoE-Mamba
Megatron-Bridge added Day-0 support for Nemotron-3-Nano-Omni 30B-A3B on 2026-04-28 with conversion + SFT + LoRA recipes targeting a single 8×H100 80GiB node — exactly lobster's shape. The cookbook (`usage-cookbook/Nemotron-3-Nano-Omni/Megatron-bridge`) covers HF→Megatron checkpoint conversion, LoRA train, adapter export back to HF PEFT format, and merge. Our V1 imaging-PEFT measured 12.4× over HF PEFT-eager on the same hardware; V2.5 reasoning-SFT should regain that.

## 2. TensorRT Model Optimizer NVFP4 (QAT vs PTQ)
NVFP4 = two-level scaling (FP8 E4M3 per 16-value microblock + FP32 tensor scale), 3.5× memory reduction vs FP16 / 1.8× vs FP8, generally <1% accuracy delta — but tail-task degradation is real. NVIDIA's Llama-Nemotron-Super NVFP4 PTQ lost 4–22% on Math-500 / AIME 2024; QAT recovered it at ~1% of pretraining-equivalent steps. ModelOpt is HF/PyTorch native (no PEFT integration documented yet). Plan: QAT after V3.5, before V_final HF release.

## 3. vLLM 0.20+ flag tuning for B300
Verified-on-Blackwell flags from the Feb-2026 vLLM gpt-oss post: `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` (Cutlass MoE FP8/FP4 backend), `--cuda-graph-capture-size 2048`, `--api-server-count 20` or `--stream-interval 20` (decouples HTTP from engine). +38% throughput / +13% TTFT on gpt-oss-120B B200. SM103/B300 MoE configs landed ~Mar-2026. Eagle-3 spec-decode + prefix caching not in that post — separate enable. Per iter-15: stage one flag at a time on a non-prod endpoint.

## 4. NeMo Guardrails 0.21 Colang 2.0
v0.21.0 (2025-03-12) added IORails parallel execution of content-safety + topic-safety + jailbreak rails — material latency win when stacking medical safety checks. v0.20.0 added Nemotron-Content-Safety-Reasoning-4B with configurable `/think` mode (an actual reasoning rail, not a regex). GLiNER integration replaces PrivateAI for PII (Apache 2.0, sovereignty-friendly). Colang 2.0 still beta — ship Colang 1.0 for production V_final; Colang 2.0 fine for internal eval rails. Composes orthogonally with V3.5 DPO refusal training (defense-in-depth, not redundancy).

## 5. NVIDIA Healthcare Blueprints
Ambient Healthcare Agents Blueprint bundles NeMo Microservices + Nemotron LLM NIM + Riva ASR/TTS + NeMo Retriever — overlaps our stack but the blueprint UX is the reference, not a forklift. Useful as packaging exemplar for V_final NIM. **Sovereignty flag:** AI Enterprise license is $4,500/GPU/yr — blueprints work without it but auto-update + indemnification require the subscription. We can ship Apache-2.0 NIM containers without it.

## 6. B300 Blackwell hardware optimizations
B300 = 15 PFLOPS dense NVFP4 / 20 PFLOPS sparse, 288 GB HBM3E, 8 TB/s, 1.8 TB/s NVLink-5 per GPU, 2× attention throughput vs B200 (doubled SFU throughput for softmax exponentials). To use it: NVFP4 quantization (we're doing this), kernel updates in vLLM/TRT-LLM/SGLang that ship via container updates (passive — pull latest), CUTLASS for any custom kernels. Backward CUDA-compat means existing code runs but doesn't see the gains.

## 7. NIM container packaging for V_final HF release
NIM 2026 supports HF Transformers checkpoints directly (.safetensors), with a single generic "model-free" container that picks a backend (TRT-LLM / vLLM / SGLang) per-model. For `huggingface.co/GOATnote-Inc/medomni-...`: provide the HF repo, an Apache-2.0 LICENSE, optional `trtllm_ckpt/` for TRT-LLM acceleration, and a model-card with `NIM_MODEL_NAME` example. That's the entire enterprise path.

## 8. NeMo-RL for V3 GRPO
NeMo-RL (Apache 2.0, GitHub `NVIDIA-NeMo/RL`) is the modern post-training RL toolkit — succeeds NeMo-Aligner. Native GRPO + DAPO (Clip-Higher, Dynamic Sampling, Token-Level PG Loss, Overlong Reward Shaping). Ray-based, Megatron-Core backend, HF-integrated. Reward-model training included (HelpSteer3 examples). Nemotron-3-Super was post-trained with NeMo-RL — same path open to us. TRL is fine for DPO; for V3 GRPO with verifiable medical rewards, NeMo-RL is the better fit.

## 9. TRT-LLM vs vLLM for medical-LLM
2026 H100/B200 benchmarks: TRT-LLM wins single-request and peak-batched throughput by 15–30% after a 10–30 min compile; vLLM wins TTFT at every concurrency and time-to-deploy. For medomni: vLLM stays the default on catfish (rapid V_n iteration), but a TRT-LLM build of V_final for any high-QPS enterprise customer is worth the compile cost. Don't switch the demo path.

## Top-3 actions for medomni
1. **Switch V2.5 reasoning-SFT on lobster from HF PEFT-eager → Megatron-Bridge LoRA recipe** for Nemotron-3-Nano-Omni (cookbook published 2026-04-28). Expect ~12× speedup based on V1 history.
2. **Stage `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` on a catfish non-prod endpoint** (per iter-15 protocol), validate, then promote to prod with saved `docker inspect` rollback.
3. **Plan V_final QAT in TensorRT Model Optimizer** between V3.5 DPO and HF release. Budget ~1% of pretraining-equivalent steps. Add to factory_loop schedule.

## Don't-do-this (medomni-specific)
- **Don't ship Colang 2.0 in V_final-prod.** Still beta as of v0.21. Use Colang 1.0 for the public demo; reserve 2.0 for internal eval rails where breakage is recoverable.
- **Don't PTQ the V_final NVFP4 release without QAT.** NVIDIA's own Llama-Nemotron-Super lost 4–22% on tail reasoning tasks via PTQ. Medical safety has identical brittle-tail risk; QAT is non-optional for clinical claims.
- **Don't multi-bump vLLM flags on catfish prod** (iter-15). Cutlass MoE backend, eagle-3 spec-decode, async API server count — each gets its own non-prod validation pass with 8–10 min poll budget before prod promotion.

## References (URL · date)
- Megatron-Bridge Nemotron-3-Nano-Omni cookbook — github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Nano-Omni/Megatron-bridge — Day-0 2026-04-28
- Megatron-Bridge LLM docs (LoRA / DoRA) — docs.nvidia.com/nemo/megatron-bridge/latest — accessed 2026-05-05
- TensorRT Model Optimizer QAT blog — developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery — 2025-Q4
- TRT-MO NVFP4 announcement — developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference — 2025
- vLLM gpt-oss Blackwell optimizations — vllm.ai/blog/gpt-oss-optimizations — 2026-02-01
- vLLM B300 MoE configs (SM103) — x.com/vllm_project/status/2013812840688189586 — Mar-2026
- NeMo Guardrails releases — github.com/NVIDIA-NeMo/Guardrails/releases — v0.21.0 2025-03-12, v0.20.0 2025-01-22
- Ambient Healthcare Agents Blueprint — build.nvidia.com/nvidia/ambient-healthcare-agents — accessed 2026-05-05
- Blackwell Ultra B300 architecture — developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era — 2026
- NIM for LLMs (HF checkpoint deploy) — docs.nvidia.com/nim/large-language-models/latest/getting-started.html — 2026
- NeMo-RL (GRPO/DAPO, Apache 2.0) — github.com/NVIDIA-NeMo/RL — accessed 2026-05-05
- vLLM vs TRT-LLM 2026 benchmarks — spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks — 2026
