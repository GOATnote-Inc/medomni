# MedOmni v0 — methodology status (post-red-team, 2026-04-29 evening)

## Three independent NVIDIA-researcher critiques converged on the same finding

Three parallel research agents — (1) NVIDIA NeMo Retriever / NV-Embed best practices, (2) NVIDIA GraphRAG / cuGraph integration patterns, (3) medical-RAG evaluation methodology — were dispatched after the tamoxifen acceptance test hit `0.780 ± 0.000` (CONDITIONAL PASS) on Run 3 (anchor-retrieval). All three independently flagged the same top-3 problems, and all three ranked methodology issues above architectural choices.

## Runs 1–3 are reclassified as ablations, not product results

| Run | Setup | Score | Status |
|---|---|---|---|
| 1 | freeform, T≈0.7, no system prompt | 0.630 ± 0.310 | ablation — high variance baseline |
| 2 | clinical system prompt, T=0.0, thinking=False | 0.610 ± 0.000 | ablation — variance-collapse baseline |
| 3 | + anchor retrieval (10 hand-curated facts) | **0.780 ± 0.000** | **ablation — slot-wiring smoke test only** |

These numbers validate that the retrieval slot in the bench harness works end-to-end. They are **not** defensible product numbers, for three reasons:

### 1. Rubric-corpus circularity (Soboroff arXiv:2601.13227)

`anchors.json` cites the same 10 papers (Cochrane 2020 Mirena+tamoxifen, USPSTF 2019, NCCN Risk Reduction v2.2024, ACOG 601, IBIS-II Lancet 2014, NSABP P-1, NCI BCRAT, etc.) that the rubric's 10 criteria implicitly test for. The "RAG system" is being asked to retrieve the answer key — it does not measure chemoprevention reasoning. No embedder swap or graph hop fixes a held-in test.

**Mitigation (in scope, this session)**: build held-out fixtures whose evidence base is outside the original rubric's source set. Same chemoprevention domain (so the corpus and pipeline are exercised), different drug/condition/citation set.

### 2. Same-family judge (Panickssery NeurIPS 2024; arXiv:2410.21819)

Nemotron-3-Nano-Omni grading Nemotron-3-Nano-Omni outputs runs 5–25pp self-preference inflation in pairwise; smaller in rubric-grading but real and not zero. NVIDIA's own NeMo Evaluator never self-judges Nemotron — they cross-judge with GPT-4o, Gemini-3-Flash, Qwen3-235B per the published Nemotron-3 reproducibility recipe.

**Mitigation (in scope, this session)**: deploy `nvidia/Llama-3.3-Nemotron-Super-49B-v1.5` as the headline judge on the H200 pod (`warm-lavender-narwhal`). Llama-3 base with NVIDIA post-training is a different family from Nemotron-3-Nano-Mamba2 architecture. Sovereign-stack constraint preserved (no cloud API). Nemotron-judged numbers stay as a secondary diagnostic.

### 3. N=1 fixture reported as N=5

T=0 + thinking=False produces 5 identical trial copies. The `±0.000` is judge-noise-floor, not sample variance. CIs on this are misleading.

**Mitigation (in scope, this session)**: held-out fixture pull moves us to N≥5 cases × 1 deterministic trial; report `score, gate, n_fixtures` instead of `score ± hw`.

## Architectural corrections (NVIDIA-canonical 2025-26)

| Original plan | Corrected to canonical |
|---|---|
| NV-Embed-v2 in vllm | `llama-nemotron-embed-1b-v2` in **NeMo Retriever Embedding NIM** (Triton+TRT, OpenAI `/v1/embeddings`). NV-Embed-v2 is CC-BY-NC-4.0 — non-commercial, not shippable. |
| No reranker | `llama-3.2-nv-rerankqa-1b-v2` NIM. Pipeline: embed → top-K=50 → rerank → top-N=8. |
| Dense-only retrieval | Hybrid (BM25 + dense, RRF-fused). Required for drug names, dose strings, trial acronyms (NSABP P-1, IBIS-II). |
| "chunk ~25 docs" | 1024 tokens / 15% overlap / page-aware. Tables and footnotes preserved as units. NVIDIA chunking blog tested optimum. |
| Naive co-tenant | vllm `--gpu-memory-utilization 0.70` (frees ~55 GB for the two NIMs); MIG partitioning deferred to v1. |

## Out of scope, this session (deferred but tracked)

- **Physician κ adjudication** — by user directive, lift comes after established working architecture. Optimizing a non-functional pipeline is without merit at this stage.
- **MIG B300 partitioning** — v1 polish. Current `gpu_memory_utilization=0.70` cap is the v0 isolation pattern.
- **PCST + cuGraph graph layer** — NVIDIA-canonical pattern is PyG vector-top-K → 1-hop expand → PCST prune → textualize, fused with vector + full-text. Tamoxifen+Mirena is at most a 2-hop case (graph adds <2pp per AWS/STaRK published gains). Defer until a graph-favorable polypharmacy fixture exists.

## Implementation deviations (deliberate, documented)

**Correction (2026-04-29 evening, post-red-team)**: an earlier version of this table framed v1 as "swap to NIM container when Blackwell-native NIM tag ships." That framing was **backwards** to the project's stated principle. NIM is the packaged subscription product NVIDIA assembles from open primitives; this project's principle is to build directly on the open primitives (CUDA / RAPIDS / NeMo / TensorRT-LLM / open-weights HF models). The Blackwell-NIM incompatibility we hit was a forcing function pushing us onto the right path, not a setback. v1 deepens the open-component stack; NIM is never the target.

The corrected v1 stack is in `findings/research/2026-04-29-medomni-v1-northstar/SPEC.md` §5.2 (BOM).

| Layer | NVIDIA-canonical open component (v1.0 target) | v0 actual | Status |
|---|---|---|---|
| Embedding serve | vLLM 0.20.0 → TensorRT-LLM 0.17+ (Triton OpenAI-compat frontend, no NIM packaging) | vllm 0.20.0 with same HF weights `nvidia/llama-nemotron-embed-1b-v2` | model weights canonical; engine framework one step less optimized than TRT-LLM. Phase 2.2 swap. |
| Reranker serve | TensorRT-LLM-FP8 cross-encoder engine on Hopper H100 | vllm 0.20.0 `--runner pooling --convert classify` on B300 | model weights canonical; Phase 2.2 swap to TRT-LLM-FP8 on RunPod H100 (idle, Hopper-native FP8). |
| Cross-family judge | sovereign open-weights, transformer-base (Llama or Qwen lineage), distinct from Nemotron-Mamba | Qwen2.5-7B-Instruct (vllm-served) | sovereign principle satisfied. Phase 2.2: rebuild as TRT-LLM-FP8 engine on H100. Phase 4: add ensemble-agreement signal with Llama-3.3-Nemotron-Super-49B-v1.5 (different family, NVIDIA-shipped) when capacity permits. |
| GPU isolation | MIG partitioning on Blackwell | Single CUDA context with `--gpu-memory-utilization 0.70` cap on Omni and 0.05/0.05/0.10 on co-tenants | acceptable for v1.0 demo; MIG is post-demo polish. |
| Vector index | cuVS (RAPIDS 26.04 — IVF-PQ / CAGRA / IVF-FLAT) | numpy cosine | Phase 2.1 swap (the highest-leverage component upgrade per Agent 2's BOM analysis). |
| Graph layer | nx-cugraph (RAPIDS 26.04 NetworkX backend) | (none) | Phase 2.1 add — Stage 6 of pipeline. |
| Safety rails | NeMo Guardrails 0.21.0 + NemoGuard JailbreakDetect + Nemotron-Content-Safety-Reasoning-4B (all NVIDIA-shipped open weights) | (none) | Phase 2.1 add. **Avoids Meta-gated Llama-Guard-3-8B entirely** — NemoGuard models are open and take a custom policy at inference, perfect for clinical-scope rules. |
| Corpus chunking | NeMo Curator 1.1.0 (Ray-based; exact + fuzzy MinHash dedup, 30+ heuristic filters) | hand-rolled regex chunker (50 chunks, mean 483 tokens) | Phase 2.1 swap. Also enables Phase 1's HPV + bisphosphonate corpus extension. |
| NGC subscription artifacts | none — all components are PyPI / open HF / open Docker base | `docker login nvcr.io` was performed for the initial NIM probe; **CLEANED UP 2026-04-29 evening** (`docker logout nvcr.io` on pod, NGC env tempfiles deleted) | resolved. v1.0 stack requires zero NGC runtime entitlements. |

## Sources

- [NVIDIA Nemotron-3 reproducibility recipe](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/nano-v3-reproducibility.md)
- [NeMo Retriever Embedding NIM overview](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html)
- [llama-nemotron-embed-1b-v2 model card](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2)
- [llama-3.2-nv-rerankqa-1b-v2 modelcard](https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/modelcard)
- [NVIDIA RAG Blueprint hybrid search](https://docs.nvidia.com/rag/2.4.0/hybrid_search.html)
- [NVIDIA Finding the Best Chunking Strategy](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)
- [Soboroff insider-knowledge / RAG circularity](https://www.arxiv.org/pdf/2601.13227)
- [Self-preference bias in LLM-as-Judge (NeurIPS 2024)](https://neurips.cc/virtual/2024/poster/96672)
- [HealthBench methodology](https://arxiv.org/abs/2505.08775)
