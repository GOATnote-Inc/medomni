# MedOmni v1.0 — North-Star Architecture Spec

**Status**: design brief, RFC. Drives engineering toward NVIDIA-fundable showcase demo.
**Date**: 2026-04-29.
**Authoring lenses**: Carnegie (audience emotional accounting), Munger (inversion / pre-mortem), Hormozi (dollar-denominated value stack).
**Sourced from**: four parallel research-agent reports (Omni capabilities deep-dive, NVIDIA open-component BOM April 2026, NVIDIA-funded healthcare-AI pitch-shape pattern, north-star architecture synthesis).
**Verification posture**: every component / version / URL cited inline.

---

## 1. Mission statement

MedOmni v1.0 is a **sovereign, multi-modal, persona-aware medical reasoning system** that reads the same clinical case (drug + device + image + auscultation audio + family history + prior labs) and returns four register-shaped answers — one for the **physician** (full diagnostic depth + literature citations), one for the **nurse** (clinical depth + early-warning escalation cues + teaching scaffold), one for the **family** (caregiver register + plain-language analogies + when-to-call-911), one for the **patient** (FKGL ≤ 8 + shared-decision-making tone) — each with a **cited graph path** beneath the answer that a malpractice attorney could read out at deposition. Irreducible core feature: the **persona-tagged grounded explanation**. Same case, same evidence, four registers, one auditable subgraph. Nothing else open-source does this on one stack, on-prem, with provenance, on a single GPU pod.

The literature gap that justifies the wedge: only **6%** of 67 medical-KG studies addressed nursing-specific applications (JMIR AI 2025). The four-persona stack with persona-tagged graph edges is genuinely under-served.

---

## 2. Carnegie principle — what the audience needs to feel

**The NVIDIA committee, sitting through a 12-minute demo, must feel three things**:

1. **Their own component stack made visible and beautiful.** Every layer carries an NVIDIA badge: CUDA 13.2, cuVS, nx-cugraph, NeMo Guardrails, NeMo Curator, TensorRT-LLM, NemoGuard, Llama-Nemotron-Embed/Rerank, Nemotron-3-Nano-Omni. They feel "this team understands what we built and is showing the world how the pieces lock together," not "they bolted our logo on a Llama wrapper."
2. **A win-win partnership story.** Not "fund us so we beat OpenEvidence on their corpus." OpenEvidence is already an NVIDIA-Nemotron-3-Omni partner. The pitch is "fund us as the **on-prem / VA / sovereignty / multi-modal-EM / nurse-persona** spire of the NVIDIA medical portfolio." Portfolio hedging, not displacement.
3. **Sincere appreciation made visible as telemetry.** Each tool call names the NVIDIA primitive it just used, on screen, in real time: "cuVS dense recall: 1.8 ms," "nx-cugraph 2-hop expansion: 2.4 ms," "TensorRT-LLM-FP8 judge: 92 tok/s." Carnegie's "honest sincere appreciation" — not flattery, telemetry.

**The physician end-user, opening MedOmni after the keynote, must feel**:

1. **"I could defend this in court."** Cited graph path visible under every answer — node IDs, edge types, primary-literature URLs. Malpractice-defensible by construction.
2. **"This respects my expertise."** Physician register is full clinical depth. Patient register is for patient. Nurse register acknowledges the nurse as a distinct expert, not a "physician + comms" overlay.
3. **"This works when the WiFi dies."** Airplane-mode test in the demo. Sovereignty as resilience, not abstract security.

These three feelings (each side) are concrete acceptance tests for the demo and UX, not vibes.

---

## 3. Munger inversion — designed backwards from rejection

Munger's discipline: list the failure modes first. Five sentences a hypothetical NVIDIA committee says when they pass on us. Each architectural choice below preempts a specific sentence.

| # | Rejection sentence | Architectural preemption |
|---|---|---|
| 1 | "This is a Llama-3 wrapper with our logo on it. Where's our stack?" | The v1.0 BOM (§5.2) is **all NVIDIA open components** — CUDA 13.2 Update 1, RAPIDS 26.04 (cuDF/cuML/cuVS/cuGraph/nx-cugraph), NeMo Framework 2.7, NeMo Guardrails 0.21.0, NeMo Curator 1.1.0, TensorRT-LLM 0.17+, NemoGuard JailbreakDetect + Nemotron-Content-Safety-Reasoning-4B (replaces gated Meta Llama-Guard), Llama-Nemotron-Embed-1B-v2, Llama-3.2-NV-RerankQA-1B-v2, Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4. The **only** non-NVIDIA weights in the whole pipeline are Qwen2.5-7B-Instruct (cross-family judge, sovereignty-mandated). |
| 2 | "OpenEvidence already has 757K clinicians and 18M consultations/month. You're late." | Differentiation by surface, not by corpus. We don't fight OE on the 35M-paper corpus. We win on (a) **sovereignty / on-prem / airgap** for VA + EU + strict-HIPAA (OE is SaaS-only); (b) **multi-modal medical reasoning** — ECG image, auscultation audio, ultrasound clip — Omni's strongest dimension and OE's weakest; (c) **nurse-persona depth** — under-served literature regime; (d) **reproducibility manifest** — bit-identical re-runs. |
| 3 | "Your 0.273 held-out score is below the bar." | Honest baseline, transparent methodology. Phase 1 extends corpus on weak-coverage axes (HPV, bisphosphonate-AI). Phase 2 lifts retrieval quality with cuVS + nx-cugraph + NeMo Guardrails + NeMo Curator. The pitch shows the **trajectory**, with rubric-corpus-circular Run-3 0.78 explicitly marked as ablation. The audit trail is itself a credibility asset. |
| 4 | "Demos that work on stage break in production. Where is the engineering rigor?" | The **9-layer reproducibility manifest** is shown live. Same demo runs twice. SHA256 of answer JSON byte-equal on screen. This is the rigor OE never demos because their stack isn't built for it. |
| 5 | "You're one founder with three pods. We need a defensible team." | We pitch **architecture as the deliverable**, not the team. Architecture's defensibility is the manifest, the corpus, the persona-tagged KG schema, the EM-spire positioning — assets that survive team turnover. Hiring is the next quarter, not the demo's burden. |

Cross-checking against Agent 3's predicted rejections (which were derived independently from NVIDIA-funded healthcare-AI pattern analysis):

- "Judge is same-family" — preempted in v0.5 (Qwen2.5-7B cross-family); v1.0 adds ensemble agreement scoring.
- "Not exercising Blackwell-specific features" — preempted: NVFP4 native compute on B300, FP8 on H100 (TRT-LLM engine), NVFP4 ablation table on the deck.
- "Safety hand-rolled, not Guardrails-anchored" — preempted: NeMo Guardrails 0.21.0 Colang 2.0 wrapping the FSM.
- "Pod, not cluster" — preempted: heterogeneous **four-pod** layout (Brev B300 + RunPod H100 prism + Brev H200 + Brev H100), each workload matched to silicon generation.
- "No published pre/post benchmark" — Phase 5 outputs a public eval set + held-out fixtures + manifest.

---

## 4. Hormozi value stack — dollar-denominated TAM

| Lever | Number | Source |
|---|---|---|
| Time saved per literature query | 5 min | OE's published claim |
| Queries / physician / day | 6 | mid-point from OE arithmetic |
| Working days / year | 230 | US physician avg |
| **Time saved / physician / year** | **115 hours** | 5 × 6 × 230 / 60 |
| Loaded physician cost / hour | $200 | BLS 2024 + AAMC 2025 |
| **Annual time-savings value / physician** | **$23,000 / yr** | |

**MedOmni-specific stacked levers**:

| Additional lever | Value | Source |
|---|---|---|
| Reduced 30-day readmission via teach-back to family | **$50,625 / physician / yr** | BMC Nursing 2021 (45% relative reduction × $15K avg readmission × 0.05 baseline × 1500 patients/yr) |
| Med-legal liability reduction from auditable graph-path citation | **~$5K / physician / yr** | $7-15K avg malpractice premium; auditability reduces settlement exposure |
| Nurse-persona augmentation (failure-to-rescue avoidance) | **$10K / nurse / yr** | Aiken/Silber: each additional patient/nurse ratio = 7% mortality increase |

**Conservative per-physician annual value: $78K / yr.**

**TAM ladder**:

```
EM physicians (US): 49K (ABEM-certified, ACEP 2024)        × $78K = $3.8B / yr
+ VA hospital system (~65K clinicians)                      × $78K = $5.1B / yr
+ EU / data-residency-restricted academic centers (30K)     × $78K = $2.3B / yr
─────────────────────────────────────────────────────────────────
SAM ≈ $11.2B / yr ceiling at full sovereign-spire deployment
```

**SOM (year-1 plausible)**: 5 academic ED pilots + 2 VA pilots × ~50 clinicians × $78K = **~$30M ARR by end of Y1**. This is the closing slide.

(Caveat per Munger: these are real-source-anchored back-of-napkin, not independently audited. Pitch must label them as such and show the arithmetic.)

---

## 5. The architecture (technical spec)

### 5.1 Four-pod heterogeneous compute layout

| Pod | Hardware | SM | HBM | Workload | Why this hardware |
|---|---|---|---|---|---|
| **Brev B300** `unnecessary-peach-catfish` | Blackwell Ultra | 10.3 | 288 GB HBM3E | Omni-NVFP4 + nx-cugraph KG (in-VRAM) + NemoGuard rails + dense embedder + KV cache | NVFP4 is **Blackwell-only native compute** (14 PFLOPS dense FP4). 288 GB HBM3E co-tenants the whole inference brain on one pod. |
| **RunPod H100** `prism` | Hopper | 9.0 | 80 GB / 251 GB RAM | TensorRT-LLM-FP8 cross-family judge (Qwen2.5-7B) + reranker TRT engine + orchestrator | FP8 is Hopper-native. TRT-LLM-FP8 on H100 delivers 15-30% throughput over vllm-BF16 for the judge workload. Currently 0% GPU utilization — frees B300 for safety + multimodal Omni. |
| Brev H100 `prism-mla-h100` | Hopper | 9.0 | 80 GB | Voice gateway (Parakeet STT + LiveKit + Redis) — **DO NOT TOUCH** per CLAUDE.md §1.5 | Frozen for the voice wedge; not on the v1.0 path. |
| Brev H200 `warm-lavender-narwhal` | Hopper | 9.0 | 141 GB | LoRA / PEFT training pod (NeMo Megatron-Bridge) — currently busy with voice infra; **DO NOT TOUCH for live demo path** | Becomes the R3 fine-tune pod when persona-LoRA training begins (post-demo). |

### 5.2 Open-component BOM (v1.0)

| Layer | Component | Version | Hardware | Source |
|---|---|---|---|---|
| Driver | NVIDIA driver | 580.+ | B300 + H100 | Blackwell compatibility guide |
| CUDA | CUDA Toolkit | **13.2 Update 1** (April 12, 2026) | both | required for NVFP4 native on Blackwell |
| Math | cuDNN 9.x · cuBLAS 13.2 · NCCL 2.25.x | bundled with CUDA 13.2 | both | sm_90 + sm_100 + sm_103 first-class |
| Kernels | CUTLASS 4.x + CuTeDSL | latest | both | Python-native CUDA kernel authoring |
| Attention | FlashAttention-4 | March 5, 2026 | sm_90 (FA3) + sm_100/103 (FA4) | JIT-compiled CuTeDSL Python; pure-pip |
| Compute libs | **RAPIDS 26.04** (cuDF, cuML, cuVS, cuGraph, nx-cugraph) | 26.04 | both | `pip install --extra-index-url=https://pypi.nvidia.com cudf-cu13 cuml-cu13 cuvs-cu13 cugraph-cu13 nx-cugraph-cu13` |
| Vector | cuVS (CAGRA + IVF-PQ + IVF-FLAT) | shipped with RAPIDS 26.04 | B300 | replaces numpy cosine; what NVIDIA puts inside Milvus + FAISS-GPU |
| Graph engine | nx-cugraph (NetworkX backend, GPU-accelerated) | shipped with RAPIDS 26.04 | B300 | zero-code-change Leiden/Louvain/PageRank; `NETWORKX_AUTOMATIC_BACKENDS=cugraph` |
| Graph data — factual | **PrimeKG** (Harvard mims-harvard) | v2.0 (Nature SciData 2023) | B300 (cuGraph-resident) | 129,375 nodes × 4,050,249 edges; 10 node types (disease/drug/gene-protein/phenotype/pathway/biological-process/molecular-function/cellular-component/exposure/anatomy); 30 relation types. MIT license. Source: github.com/mims-harvard/PrimeKG + Harvard Dataverse DOI 10.7910/DVN/IXA7BM |
| Graph data — register | persona-tagged hand-built graph | Phase 2.1 | B300 | NIPDS subtle-signal nodes, pedagogical-intent edges, evidence-currency, FTR patterns; complementary to PrimeKG |
| Frame | NeMo Framework | 2.7.0 (Feb 26, 2026) | H200 (training) | LLM training, PEFT, LoRA |
| Curate | **NeMo Curator** | 1.1.0 (Feb 23, 2026) | H200 (corpus build) | exact + fuzzy MinHash dedup, 30+ heuristic filters, fastText classifier, semantic dedup, Nemotron-CC pipeline. Ray-based (not Dask). |
| Ingest | NeMo Retriever extraction | open client + microservices | CPU + GPU | OCR, table/chart parse, page-aware chunking |
| Guard framework | **NeMo Guardrails 0.21.0** (March 12, 2026) — Colang 1.0 default + Colang 2.0 GA | 0.21.0 | CPU host + GPU LLM | input/output safety, fact-check (AlignScore), jailbreak, topic-control, PII (GLiNER) |
| Guard models (replaces gated Meta Llama-Guard-3) | **NemoGuard JailbreakDetect** + **llama-3.1-nemoguard-8b-content-safety** + **Nemotron-Content-Safety-Reasoning-4B** | HF revisions pinned | B300 | open-weights, takes a custom policy at inference time — perfect for clinical-scope rules |
| Engine A | vLLM | 0.20.0 | both | best for fast iteration; NVFP4 dense models serve directly |
| Engine B | **TensorRT-LLM** | 0.17+ | H100 (FP8 mature) + B300 (NVFP4 GA) | 15-30% throughput win over vllm at high concurrency; 28-min compile cost; build cache with `TRTLLM_CACHE_DIR` |
| Embed/rerank engine | TensorRT 10.x (base) | bundled with TRT-LLM | both | builds engines for `llama-bidirectional` HF arch (the embed/rerank weights' arch) |
| Embed model | nvidia/llama-nemotron-embed-1b-v2 | HF revision pinned | B300 (vllm) → H100 (TRT engine v1.5) | text embedding, 8K context, 2048-d output, Matryoshka 384/512/768/1024/2048 |
| Embed VL model (Phase 4) | nvidia/llama-nemotron-embed-vl-1b-v2 | HF revision pinned | B300 | multi-modal embedding for ECG / X-ray / dermatoscopy retrieval |
| Rerank model | nvidia/llama-3.2-nv-rerankqa-1b-v2 | HF revision pinned | H100 (TRT engine) | cross-encoder pairwise; **load-bearing per NVIDIA RAG Blueprint** |
| Rerank VL (Phase 4) | nvidia/llama-nemotron-rerank-vl-1b-v2 | HF revision pinned | H100 | multi-modal rerank |
| Brain | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | HF revision pinned | B300 | 31B/3B-active MoE, 23 Mamba2 + 23 MoE + 6 GQA, 256K context, multimodal in / text out |
| Voice out (Phase 4 optional) | nvidia/Nemotron-3-VoiceChat-12B | HF revision pinned | H200 (training pod when free) | full-duplex speech-out, fills demo's "spoken family response" gap |
| Frontend serving | **Triton Inference Server 26.03** with OpenAI-compat frontend | 26.03 | both | open-source; serves TRT-LLM + TRT + Python backends; multi-LoRA + structured outputs |
| Distributed orchestration | NVIDIA Dynamo 1.0 (March 2026 GA) | 1.0 | both | KV-cache routing across nodes when we go cross-pod for production |
| Judge (cross-family, sovereign) | Qwen/Qwen2.5-7B-Instruct | FP8 TRT-LLM engine | H100 | NVIDIA's own Nemotron-3 evaluation recipe uses Qwen-family as cross-family judge |

### 5.3 The 10-stage retrieval pipeline

```
[user query, modality bundle (text + optional image / audio / video)]
        │
   1.  NemoGuard INPUT RAIL (NeMo Guardrails 0.21 Colang 2.0)
        │   reject: jailbreak (NemoGuard JailbreakDetect), self-harm bypass, prompt-injection, PII spill
        │   pass: clinical query
        ▼
   2.  cuVS DENSE RECALL (top-50)        ◄┐
        │   embed query: llama-nemotron-embed-1b-v2 (B300)
        │   IVF-PQ index over node descriptions + corpus chunks
        │                                  │  parallel
   3.  BM25 SPARSE RECALL (top-50)         │
        │   pyserini or Tantivy CPU; required for drug names, trial acronyms (NSABP P-1, IBIS-II)
        ▼                                  │
   4.  RRF FUSION (top-50 → top-25)  ◄────┘
        │   Reciprocal Rank Fusion, k=60
        ▼
   5.  CROSS-ENCODER RERANK (top-25 → top-8)
        │   llama-3.2-nv-rerankqa-1b-v2 (TensorRT-LLM-FP8 engine on H100)
        ▼
   6.  nx-cugraph SUBGRAPH SLICE — PrimeKG (factual) + persona graph (register)
        │   FACTUAL layer: PrimeKG (Harvard mims-harvard, 129K nodes / 4M edges,
        │     MIT-licensed precision-medicine KG, 10 node types × 30 relation types).
        │     Seed entities from query NER → cuGraph k-hop expansion (default k=2,
        │     cap 100 nodes). Louvain community label per subgraph; betweenness
        │     centrality identifies bridge concepts (drug-drug interactions, multi-
        │     step reasoning chains). nx-cugraph delivers ~50-150× speedup on
        │     these algorithms vs CPU NetworkX per NVIDIA published benchmarks
        │     (Louvain ~100×, PageRank ~76×, betweenness 50-57×).
        │   REGISTER layer: persona-tagged hand-built graph (Phase 2.1, ~59 nodes).
        │     persona_mask-filtered edges; pedagogical-intent edges; NIPDS subtle-
        │     signal nodes; evidence-currency nodes.
        │   The two graphs serve different purposes; both are queried per request
        │   and composed into the subgraph rendered for the LLM.
        ▼
   7.  SUBGRAPH SERIALIZE (5–15K tokens)
        │   JSON; preserves node attrs + edge provenance + citation URLs
        ▼
   8.  OMNI INFERENCE (B300, NVFP4)
        │   vLLM 0.20.0 / later TRT-LLM-NVFP4 engine
        │   --no-async-scheduling, --kv-cache-dtype fp8, --max-model-len 131072
        │   chat_template_kwargs={"enable_thinking": <persona-conditional>}
        │   audio inputs: enable_thinking=false enforced (orchestrator silent override)
        ▼
   9.  CONSTRAINED DECODING + GROUNDING CITE RAIL
        │   guided_json against citation schema; cited node-IDs must appear in subgraph
        │   each cited passage: cosine ≥ 0.8 to retrieved node description
        │   FKGL grade gate on persona=family/patient
        ▼
  10.  NemoGuard OUTPUT RAIL (Nemotron-Content-Safety-Reasoning-4B with clinical policy)
        │   reject: hallucinated dosing, unsafe override, PII leak
        │   pass: persona-shaped grounded answer + cited subgraph
```

**Stage timing budget** (estimated, measure in Phase 5):

| Stage | p50 budget | Source |
|---|---|---|
| 1 input rail | 8 ms | Llama-style 8B FP8, single-token decision |
| 2 cuVS dense | 1.8 ms | NVIDIA cuVS blog |
| 3 BM25 sparse | 4 ms | Tantivy CPU |
| 4 RRF | 0.2 ms | trivial |
| 5 rerank | 6 ms | TRT-LLM-FP8 batch=25 |
| 6 nx-cugraph 2-hop | 2.4 ms | sub-ms at small scale |
| 7 serialize | 2 ms | host-side |
| 8 Omni TTFT | 80 ms | vLLM Nemotron recipe |
| 9 constrained decode | included in 8 | mask precomputed |
| 10 output rail | 8 ms | same as stage 1 |
| **Total p50 first-cite** | **~115 ms** | budget |

### 5.4 Where Omni's multimodal capabilities plug in

Per Agent 1's capability audit:

| Modality | Encoder | MedOmni use case | Constraint |
|---|---|---|---|
| **Text** | tokenizer | clinical note, patient question, drug name | base path |
| **Image** | C-RADIO v4-H, 1K-13K patches | ECG strip, chest X-ray, dermatoscopy frame, ultrasound still, scanned discharge summary, lab printout | OCRBenchV2-EN 65.8, MMLongBench-Doc 57.5, CharXiv 63.6, MathVista 82.8 |
| **Video** | Conv3D + EVS | ultrasound clip (POCUS), echo loop, wound-healing | Video-MME 72.2, ≤2 min, 2 FPS × 256 frames |
| **Audio** | Parakeet-TDT-0.6B-v3 | auscultation (heart, lung), patient interview, dictated history | VoiceBench 89.4, WER 5.95; **`enable_thinking=false` mandatory** |
| **Tool calls** | trained-in (Qwen3-Coder format) | graph queries, drug-interaction checker, dose calculator | BFCL v4 53.76; AIME-w/-tools jumps 89→99 |

**The audio + reasoning mutual-exclusion claim is a SOFT recommendation**, not a hard constraint. Card actually says "for ASR tasks, we recommend non-thinking mode with `temperature=0.2, top_k=1`." The orchestrator silently sets these on any audio request.

**Killer under-used capability**: **audio input + word-level timestamps native** for dictated chief-complaint, phone-triage, patient-consult-playback. Highest ROI per engineering hour.

### 5.5 Persona-tagged graph schema

Six node types, persona_mask edges. Each edge carries a `persona_mask` bitfield {physician, nurse, family, patient}; retrieval expands only edges where the bit for the active persona is set. Node attrs include `register` and `evidence_currency`.

Schema additions for the nurse persona (under-served literature regime):

1. Subtle-signal nodes (NIPDS-derived pre-deterioration indicators)
2. Context-dependent protocol variants (acuity-ratio-aware deference rules)
3. Pedagogical-intent edges (`teaches_via_[why|what|when|caution]`)
4. Failure-to-recognize patterns (anonymized FTR + staffing-context links)
5. Evidence-currency nodes (challenge-outdated-orders surface)

### 5.6 The 9-layer reproducibility manifest

Every demo run emits a `MANIFEST.yaml` with byte-identical fingerprint:

1. Container image digests (vLLM, TRT-LLM, NeMo Guardrails, NeMo Curator)
2. Weight SHAs (Omni NVFP4, NemoGuard models, embed-1b-v2, rerank-1b-v2, Qwen2.5-7B)
3. Corpus SHAs (OpenEM 370, PubMed-OA shard, fixture set)
4. Config files (Colang rails, persona prompts, retrieval params)
5. Random seeds (decoding seed, retrieval tie-break seed)
6. Hardware-foot-gun flags (`--no-async-scheduling`, `--max-model-len 131072`, `--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.70`)
7. Benchmark fixtures (chemoprevention 6 + tamoxifen 1 = 7 cases)
8. Judge model digest (Qwen2.5-7B-Instruct FP8 TRT-LLM engine)
9. Git SHA of harness + this SPEC.md SHA

---

## 6. The metrics that matter

| Category | Metric | Target | Source |
|---|---|---|---|
| **Headline accuracy** | HealthBench Hard (rubric-graded, cross-judge) | ≥ 0.55 mean (open SOTA ~0.45) | OpenAI HealthBench |
| | MedAgentBench | ≥ 0.70 (Claude 3.5 Sonnet 0.6967) | Stanford NEJM AI |
| | MedQA-USMLE | ≥ 0.85 | MedQA |
| | Held-out chemoprevention rubric (6 fixtures) | ≥ 0.55 (uplift from 0.273 baseline) | this repo |
| | Tamoxifen+Mirena rubric-v2 | ≥ 0.80 | this repo |
| **Latency** | TTFT p50 | ≤ 250 ms | budget |
| | First-cite p50 | ≤ 350 ms | budget |
| | End-to-end full answer (300 tok) p50 | ≤ 6 s | budget |
| **Sovereignty** | Cloud LLM API calls in any path | exactly **0** | grep CI gate |
| | Weights resident on local disk | 100% | manifest gate |
| | Airplane-mode demo | passes | live |
| **Safety** | NemoGuard jailbreak block rate (NR-Labs corpus + held-out) | ≥ 95% | NR-Labs |
| | Output-rail hallucination catch rate (cite-grounded) | ≥ 90% | held-out corpus |
| **Reproducibility** | Bit-identical re-run given manifest | byte-equal SHA256 of answer JSON | gate |
| | Cross-pod re-run variance | < 0.01 rubric pts | gate |

**First defensible held-out number we currently own: 0.273 mean across 6 chemoprevention fixtures** (full hybrid retrieval, cross-family judge Qwen2.5-7B, rubric-v2). Run-3 0.78 was rubric-corpus circular and self-judged. Pitch surfaces both numbers honestly with the lift trajectory.

**Consolidated v1.0 number (post Phase 2.1 + Phase 1.5 + seed=42 N=3): 0.385**, deterministic ±0.000 across trials, manifest byte-stable (sha256 `560baccbb706...`). Comparator PASS verdict on v0→v1.0 (+0.112, no major regression). 5 of 6 targeted fixtures lifted monotonically; statin flat (no targeted trial). SPEC §6 gate ≥0.45 still NOT met (-0.065); closes via Phase 1.7 scale fixtures + Phase 2.3 ensemble cross-judge. Artifact: `results/ci-medomni-heldout-consolidated-20260429-173557/`.

**Phase 2.4 PrimeKG-hybrid result (NEGATIVE; mode stays OPT-IN, not default)**: integrating PrimeKG as Stage 6's factual layer DROPPED held-out mean 0.385 → 0.358 (-0.027) on the chemoprevention-counseling fixtures. Five of six fixtures regressed; only STATIN-CV-CANCER lifted (+0.18). Per-axis diagnosis: `instruction_following` collapsed -0.222 (the 1.5K-token PrimeKG context block crowded out the 5-section structured-answer scaffold in the system prompt); `context_awareness` lifted +0.074 (graph did anchor drug-class taxonomy correctly). **Durable finding**: PrimeKG is a general biomedical KG; the held-out fixtures probe trial-vs-guideline-vs-regulator subtleties (USPSTF reversals, FDA warnings, EBCTCG postmenopausal-only signals, EAGLES neuropsych safety) that PrimeKG does not encode. PrimeKG is the right tool for rare-disease reasoning / drug-interaction chains / mechanism ladders (AWS/STaRK +30pp regime); the persona-tagged 59-node graph from Phase 2.1 remains correct for chemoprevention counseling. `--retrieval primekg-hybrid` is an OPT-IN flag for precision-medicine query strata, not the v1.0 default. Artifact: `results/ci-medomni-heldout-primekg-pinned-20260429-204029/` (manifest sha256 `90c4ec22413a...`). Path forward: cross-family judge ensemble + prompt-engineering shrink (1500→≤500 tokens) for opt-in PrimeKG calls — NOT more retrieval depth.

---

## 7. Demo script (12 minutes, on stage)

### Open (3 min) — tamoxifen + Mirena, four-persona response surface

The case from `corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/case.json`. Same case, four panels on screen. Each panel shows: the persona-shaped answer; the cited graph path (node IDs + edge types + primary-literature URLs); the FKGL grade slider (family/patient panels show ≤ 8); timing telemetry ("cuVS 1.8 ms, rerank 6 ms").

### Multimodal (3 min) — ECG + auscultation

Upload an ECG image (12-lead PNG) plus an audio clip "I have palpitations." Omni's C-RADIO v4-H reads the ECG; Parakeet transcribes the audio with word-level timestamps; response cites both. Graph path now includes image → finding → condition → drug edges. **`enable_thinking=false` enforced for the audio path** (orchestrator silently sets it; demo voice-over notes the constraint).

### Adversarial (2 min) — NemoGuard input rail

Paste the NR-Labs system-prompt-override jailbreak. NemoGuard JailbreakDetect blocks at stage 1. Show the reasoning audit log: which Colang rule fired, which category was violated.

### Reproducibility (2 min) — bit-identical re-run

Run the same demo twice. Show the SHA256 of the answer JSON on screen both times. Bytes match. The 9-layer manifest shown side-by-side. Moment OE cannot reproduce.

### Sovereignty (1 min) — airplane-mode test

Turn off WiFi mid-demo. Type a new clinical query. System keeps working. Resilience felt, not abstract security.

### Close (1 min) — Hormozi TAM

Single slide: $78K / physician / yr × ~144K reachable clinicians (EM-spire + VA + EU) = **$11B SAM ceiling**, **$30M ARR year-1 plausible** at 5+2 pilots. End on the architectural diagram with every NVIDIA logo lit.

---

## 8. Phasing to ship the demo

| Phase | Track | Workload | Pod | Acceptance |
|---|---|---|---|---|
| **1** | Track A | corpus extension on weak-coverage axes (HPV → 8 chunks, bisphosphonate → 6 chunks) via NeMo Curator 1.1.0 | local + B300 | held-out mean ≥ 0.40 (lift from 0.273) |
| **2.1** | Track B | install RAPIDS 26.04 on B300; swap numpy cosine → cuVS IVF-PQ; add nx-cugraph 2-hop; add NeMo Guardrails 0.21.0 + NemoGuard models; replace hand-rolled chunker with NeMo Curator | B300 | first-cite p50 ≤ 350 ms; held-out mean ≥ 0.45 |
| **2.2** | Track B | TensorRT-LLM-FP8 engine for Qwen2.5-7B judge on RunPod H100; offload reranker to TRT engine on H100; B300 frees up | H100 + B300 | judge throughput ≥ 80 tok/s; B300 VRAM headroom ≥ 60 GB |
| **3** | Multimodal | ECG image, auscultation audio wired through Omni; persona-tagged 4-panel response surface; FKGL slider | B300 | tamoxifen + ECG + audio fixtures all pass |
| **4** | Reproducibility + airplane-mode | 9-layer manifest emitter; bit-identical CI gate; airplane-mode demo script | B300 | manifest hash matches across 3 cold-start re-runs |

Phases 1 and 2.1 run in parallel where possible; Phase 2.2 unblocks Phase 3's B300 VRAM. Phase 4 is the demo-ready gate.

---

## 9. Decisions flagged for user

**D1. BM25 location.** Stage 3 (BM25 sparse recall) is currently CPU-resident (Tantivy or pyserini). Acceptable for first-cite p50 ≤ 350 ms. Alternative: SPLADE on B300 (sparse-on-GPU) — adds ~1 GB VRAM, ~3 ms p50, matches the "every layer is GPU-native" pitch story better. **Recommendation: ship CPU-BM25 in v1.0; SPLADE in v1.1 if pitch feedback says "why is anything on CPU?"**

**D2. Cross-family judge role.** Production options:
- (a) Stay as runtime safety/audit signal alongside Omni answer (transparency + dual-judge agreement score)
- (b) Drop once we cross-validate against physician κ ≥ 0.7 on held-out fixtures (production efficiency)
- (c) Both judges, ensemble agreement as confidence flag visible in the UI

**Recommendation: (c) for the demo — ensemble agreement tells a story. Drop to (a) post-pilot if latency binds.**

**D3. Persona-LoRA timing.** v1.0 ships with **prompt-conditional personas** (no fine-tune). Persona-LoRA on Omni router is v1.5 (post-demo). Alternative: train a minimal nurse-persona LoRA pre-demo. **Recommendation: ship prompt-conditional in v1.0. The persona story is about graph-tagged retrieval, not weight-level specialization. Save the LoRA story for the post-funding roadmap.**

---

## 10. Critical files for implementation

- `/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v0/architecture-v2.md`
- `/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v0/v2.5-b300-uplift.md`
- `/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v0/methodology-status.md`
- `/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-omni-canonical/CANONICAL.md`
- `/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-graph-rag-rethink/synthesis.md`
- `/Users/kiteboard/prism42-nemotron-med/corpus/clinical-fixtures-heldout/MANIFEST.md`
- `/Users/kiteboard/prism42-nemotron-med/results/ci-medomni-heldout-20260429-142936/CARD.md`

---

## 11. Sources (research-agent provenance)

- [Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 model card](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
- [HF blog — Nemotron-3-Nano-Omni Long-Context Multimodal Intelligence](https://huggingface.co/blog/nvidia/nemotron-3-nano-omni-multimodal-intelligence)
- [vLLM Nemotron Omni recipe](https://vllm.ai/blog/nemotron-omni)
- [arXiv 2512.20848 — Nemotron 3 Nano text LLM](https://arxiv.org/html/2512.20848v1)
- [CUDA Toolkit 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [RAPIDS](https://rapids.ai/) and [cuVS releases](https://github.com/rapidsai/cuvs/releases)
- [NeMo Framework](https://github.com/NVIDIA-NeMo/NeMo) and [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)
- [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) and [Colang 2.0 Configuration Guide](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/configuration-guide.html)
- [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [NemoGuard JailbreakDetect](https://huggingface.co/nvidia/NemoGuard-JailbreakDetect) and [Nemotron-Content-Safety-Reasoning-4B](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B)
- [llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) and [llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)
- [llama-3.2-nv-rerankqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/modelcard) and [llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2)
- [Triton Inference Server OpenAI-Compatible Frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html)
- [NVIDIA Dynamo 1.0 GA](https://nvidianews.nvidia.com/news/nvidia-enters-production-with-dynamo-the-broadly-adopted-inference-operating-system-for-ai-factories)
- [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) and [Hybrid Search](https://docs.nvidia.com/rag/2.4.0/hybrid_search.html)
- [NVIDIA chunking guide](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)
- [Soboroff RAG circularity](https://www.arxiv.org/pdf/2601.13227)
- [HealthBench](https://arxiv.org/abs/2505.08775) and [MedAgentBench NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2500144)
- [Aiken/Silber failure-to-rescue](https://www.ncbi.nlm.nih.gov/books/NBK555513/)
- [BMC Nursing teach-back meta-analysis](https://bmcnurs.biomedcentral.com/articles/10.1186/s12912-021-00622-2)

## 13. Uptime / outage / multi-pod redundancy plan

The "no stop/start" pod-rental model means total burn is **$442/day fixed** regardless of utilization — therefore every pod must do real work, AND every pod must have a documented failure-mode response. This is also the v1.0 demo's reliability narrative for the NVIDIA committee: "production-grade reliability across 3 cloud providers and 4 datacenter regions."

### 13.1 Geographic + provider distribution

| Pod | Provider | Region | Daily burn | Workload class |
|---|---|---|---|---|
| B300 `unnecessary-peach-catfish` | Verda | helsinki-finland-5 | $213/day | inference primary (NVFP4) |
| H200 `warm-lavender-narwhal` | Nebius | eu-north1 | $101/day | voice gateway + redis state store + Omni-FP8 fallback (already loaded) |
| Brev H100 `prism-mla-h100` | Hyperstack | montreal-canada-2 | $54/day | voice gateway + LiveKit |
| RunPod H100 `prism` | RunPod | unknown region | $72/day | inference secondary (FP8 TRT-LLM judge + reranker, post-Phase 2.2) |

**Three providers × four regions** — geographic correlation of simultaneous outages is extremely low. Verda Helsinki + Nebius eu-north1 + Hyperstack Montreal + RunPod region all going down at once requires correlated continental events.

### 13.2 Failure-mode runbook

| Failure | Detection | Mitigation | RTO |
|---|---|---|---|
| Single container crash on B300 (e.g., vllm-omni-b300 OOM during a long-context request) | `docker --restart unless-stopped`; health-monitor script alerts in 30s | Container auto-restarts; weight reload from local `/home/shadeform/medomni/hf_cache` (no HF Hub call) takes 8-15 min for Omni cold load | < 15 min |
| B300 entire pod outage (Verda Helsinki down) | `brev ls` reports STOPPED or unreachable; tunnel SSH dies | Failover to H200 with Nemotron Nano-30B-FP8 (already loaded; BF16 not NVFP4 so 2-3x slower but functional). Update DNS / load balancer to point read traffic at H200. RunPod prism keeps judge + reranker alive. | < 60 s if pre-warmed; ~5 min if cold-warm needed |
| RunPod outage (whole RunPod region) | TRT-LLM endpoints on prism unreachable | B300 reabsorbs judge + reranker via vllm-served fallback (proven this session — vllm-judge container on B300:8003 is the v0.5 baseline that never went away). | < 2 min |
| H200 outage (Nebius eu-north1 down) | redis state store unreachable; voice gateway down | Voice degrades to "best-effort" (LiveKit on Brev H100 keeps working with cloud STT/TTS fallback). Inference path UNAFFECTED — H200 isn't on the inference critical path. | voice impact only |
| Brev H100 outage (Hyperstack Montreal down) | parakeet/livekit unreachable | Voice degrades; inference unaffected. | voice impact only |
| Multi-region simultaneous outage (Verda + Nebius + RunPod all down) | All endpoints unreachable | Airplane-mode demo still runs on B300-laptop tunnel using cached weights. If laptop is up, demo is up. | manual / N/A in catastrophic case |
| HF Hub outage (model weights unavailable) | New container can't pull weights | All weights resident on local `HF_CACHE` volume mounts. No external dependency at runtime; only at first cold-start. | n/a (weights already local) |
| Container OOM during request | Container restarts, request fails | Bench harness retries 3x; sovereign_bench audit log captures OOM as "exhausted retries"; rubric item recused (not zero'd — preserves valid statistic). | per-request; <60s |

### 13.3 Stateless inference + persistent volumes

Every inference container is **stateless on the GPU**: no per-session state on B300 / prism. All session state lives in the H200 redis (already running for voice). KV-cache is per-request in vllm; on container restart the cache rebuilds. Model weights mount as Docker volumes from `~/medomni/hf_cache` on each pod's local disk (not container-internal layer) — survives container OOM and restart.

Every container uses `--restart unless-stopped` policy. Every pod has a `health-check.service` systemd unit (or equivalent cron) that probes `/v1/models` every 30 seconds and logs to `/var/log/medomni-health.log`.

### 13.4 Pre-demo health-check protocol

The night before any demo:
1. `make health-check-all-pods` runs full probe against B300:8000/8001/8002/8003, prism:9002/9003, H200 (read-only — voice probes), Brev H100 (read-only).
2. `make smoke-tamoxifen` runs the original tamoxifen fixture end-to-end and validates score > 0.70 on rubric-v2.
3. `make smoke-multimodal` runs an ECG image + auscultation audio fixture and validates Omni produces both image and audio reasoning.
4. `make smoke-airplane-mode` opens a SOCKS proxy, blocks all DNS except localhost, runs the tamoxifen fixture; must pass.
5. `make manifest-bit-identical` runs the same fixture twice cold-start and validates SHA256 byte-identical.

Demo green light: all five gates pass.

### 13.5 Cost-utilization economics

The "no stop/start" rule means $442/day flows whether pods are busy or idle. Therefore: **idle pod = wasted $**. The Phase 2.2 elevation (TRT-LLM on RunPod prism) is justified by this: prism at 0% GPU = $72/day burning for no value. After Phase 2.2 completes, prism serves the judge + reranker = real workload = real value-per-dollar.

Long-term: when the production deployment exists, the same compute can run autoresearch / fine-tune / dataset curation during off-peak hours. No idle-pod cost in steady state.

## 12. Blackwell foot-guns (carry forward into all phases)

1. **sm_103 ≠ sm_120 ≠ sm_121.** B300 is sm_103a; RTX 5090 is sm_120; DGX Spark is sm_121. Pin `nvcc --gpu-architecture=sm_103a` for B300; SM_100 wheels won't get NVFP4 MoE without a recent vLLM (post-issue #33416).
2. **CUDA 13 vs CUDA 12 wheels.** RAPIDS, vLLM, TRT-LLM, NeMo all ship CUDA-12 *and* CUDA-13 wheels. Mixing them silently downgrades cuBLAS at import. Pin one CUDA major and use the matching `-cu13` suffix on every package.
3. **NVFP4 needs ModelOpt PTQ first, then TRT-LLM build.** Workflow: HF checkpoint → ModelOpt PTQ → TRT-LLM build → Triton serve. Skipping ModelOpt = silent accuracy collapse.
4. **TRT-LLM compile time on Blackwell can be 28+ minutes for a 70B model.** Plan a build cache (`TRTLLM_CACHE_DIR`) and a CI artifact promotion step.
5. **FlashAttention-4 is CuTeDSL-Python.** No 30-min C++ compile but JIT path needs CUDA 13.x at runtime. Container base images shipping CUDA 13.0 will JIT-fail at first attention call on sm_103. Use CUDA 13.2 base images.
6. **NIM containers ship Hopper-only ONNX Runtime cubins.** Crash on Blackwell with `cudaErrorSymbolNotFound`. We bypass NIM entirely; this is documented as a deliberate Track-B principle, not a limitation.
7. **NeMo Curator is Ray-based** as of 1.1.0 (formerly Dask). 2024-era examples will break; check post-Ray-migration docs.
