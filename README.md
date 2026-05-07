<!--
  © 2026 GOATnote, Inc. and contributors. All rights reserved.

  Code in this repository is licensed under the Apache License, Version 2.0
  (see LICENSE). Documentation, architecture briefs, methodology notes, demo
  fixtures, and held-out benchmark artifacts are released under CC-BY-4.0
  for public benefit and to surface our research as proof-of-quality.

  You are welcome to read, fork, cite, learn from, and build on this work.
  Attribution is required (Apache 2.0 + CC-BY-4.0). Trademarks "MedOmni"
  and "GOATnote" are reserved by GOATnote, Inc.

  This is a working research repository — methodology notes include negative
  results and corrected diagnoses. Reproducibility manifests are first-class
  artifacts (sha256-verified byte-deterministic across re-emit). If you
  reproduce a result and your manifest hash diverges from ours, please open
  an issue using the "reproducibility" template — that surface is exactly
  what we want to be debuggable in public.
-->

# MedOmni

**Sovereign nurse-first medical reasoning on NVIDIA's open-component stack.**

**Headline (2026-05-07, paired-bootstrap N=600 per benchmark, gpt-4.1 grader, `thinking=True`):**

| Benchmark | V0 (Nemotron-3-Nano-Omni-30B-Reasoning, FP8) |
|---|---:|
| MedQA-USMLE | **83.50%** |
| PubMedQA-L | **67.33%** |
| MedXpertQA-Text | **33.00%** |
| HealthBench-Hard | **12.52%** |

These are the **base-model-with-thinking-enabled** numbers. The V2.5 reasoning-SFT we trained on top did NOT improve them — see PR [#122](https://github.com/GOATnote-Inc/medomni/pull/122) and [`findings/2026-05-05-v2.5-eval/DISPOSITION.md`](findings/2026-05-05-v2.5-eval/DISPOSITION.md) for the full A5 ablation (thinking=False vs thinking=True, 0/4 ship-rule criteria PASS in either polarity). The negative result + reproducibility audit is the rigor story; the strong V0 baseline is the deployment story.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900.svg)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-26.04-7400B8.svg)](https://rapids.ai/)
[![NeMo](https://img.shields.io/badge/NeMo-Framework_2.7-76B900.svg)](https://github.com/NVIDIA-NeMo/NeMo)
[![Reproducibility](https://img.shields.io/badge/manifest-9--layer-success.svg)](docs/SPEC.md#56-the-9-layer-reproducibility-manifest)

---

## Live demo — `https://www.thegoatnote.com/4UWHAt`

Public demo URL, no login, no account. Routes via Vercel edge rewrite from
the `www.thegoatnote.com` apex (owned by `v0-goat-note-landing-page-3c`)
to `medomni.vercel.app/4UWHAt`. The page is the **Records OS** dashboard:

- **Three-column patient cockpit** — left nav rail (Overview · Timeline · Labs · Meds · Conditions · Vitals · Imaging · Wearables · Visit notes · Care team · Genome · Sharing), main canvas, right rail.
- **Ask your record** command bar — voice in (Web Audio + Nemotron-Omni native ASR), voice out (Kokoro-82M TTS on H200 via Cloudflare quick-tunnel; browser `speechSynthesis` fallback), image input (camera / file → multimodal Omni reasoning), text in.
- **Imaging gallery** — three click-to-view DetailDrawer cards (X-ray · MRI · panoramic) backed by real CC0 / CC-BY reference films from Wikimedia Commons. Track D's FHIR `ImagingStudy` ingestion populates the same shape from a real EHR.
- **Five-tool agent** — `pubmed_search`, `primekg_lookup`, `guideline_currency_check`, `clinical_calculate`, `get_patient_context` (FHIR R4). Tool call provenance + verification badge under every assistant turn.
- **FHIR R4 Bundle export** — Share button produces a valid Bundle with Patient + Conditions + Observations + MedicationRequests + AllergyIntolerances + DiagnosticReports + ImagingStudies, downloadable as JSON.
- **Demo banner** — `Private by design · runs on dedicated NVIDIA hardware · no third-party AI APIs called` (for evaluation only; do not enter PHI).

Inference path: every keystroke / image / audio chunk hits **Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4** on the B300 catfish pod via vllm. No cloud LLM API keys in the path; the only external service is the public `medomni.vercel.app` Vercel deploy that proxies `/api/agent` to the catfish vllm endpoint.

Architecture decision behind the patient slice: **Pattern B (dual lookup)** — the agent hits FHIR (Medplum, self-hosted) and PrimeKG independently and merges in-prompt. Measured p95 = **11 ms** for the FHIR fetch across 12 patients, 60 samples (`findings/2026-05-04-pattern-b-spike/RESULTS.md`).

The training / eval / methodology stack below is the proof-of-quality behind that demo. Read on if you care how the model got there. Or [open the demo](https://www.thegoatnote.com/4UWHAt).

For the **world-class trajectory** (V2.5 reasoning-SFT → V2.7 tool-call SFT → V3 GRPO → V3.5 DPO refusal → V_final HF release as Apache-2.0; eval gauntlet with MedAgentBench as the headline target), see the navigable index at [`findings/INDEX.md`](findings/INDEX.md). 16+ pre-registered documents covering every stage with surgical recipe additions per the iter-38 4-agent improvement-dimensions synthesis; ~32-41 H200-hrs total.

**Live status (2026-05-07):** V2.5 reasoning-SFT eval landed (PR [#122](https://github.com/GOATnote-Inc/medomni/pull/122)) — A5 ablation closed with both arms FAILing the pre-registered ship rule. Real story is the V0 baseline strength (numbers above) once `enable_thinking=True` is set as canonical. Production demo at [`/4UWHAt`](https://www.thegoatnote.com/4UWHAt) now serves V0 FP8 from a single H100 (Hopper, 80 GB) — migrated from the prior 3-GPU B300+H200×2 fleet on 2026-05-07 with ~80% cost reduction (~$15-20/hr → $3.70/hr). The Karpathy-autoresearcher training loop (catfish + lobster + narwhal) is decommissioned; future SFT runs (V2.5b corpus-tweak, V2.7 PRM channel) wait for the next training-budget cycle (Issue [#130](https://github.com/GOATnote-Inc/medomni/issues/130)).

---

## Headline result (current — 2026-05-07)

> **V0 baseline with thinking=True is the medical reasoner.** Pre-registered paired-bootstrap eval (PR [#122](https://github.com/GOATnote-Inc/medomni/pull/122), N=600 per benchmark, 4 benchmarks × 3 seeds, gpt-4.1 grader, sha256-verified manifest):
>
> - **MedQA-USMLE 83.50%** — competitive with claimed-SOTA medical LLMs on the same harness
> - **HealthBench-Hard 12.52%** — vs published OpenAI-paper baseline on the same rubric
> - **PubMedQA-L 67.33%**, **MedXpertQA-Text 33.00%**
>
> The V2.5 reasoning-SFT we trained on top (MedReason 32K + medical-o1-reasoning-SFT 25K + R1-distill-USMLE) **did not add value** — both thinking=False (1/4 PASS, deniable) and thinking=True (0/4 PASS, canonical) FAILed the pre-registered ship-rule criteria. The A5 ablation is documented in [`findings/2026-05-05-v2.5-eval/DISPOSITION.md`](findings/2026-05-05-v2.5-eval/DISPOSITION.md).
>
> The **negative-result-with-rigor** is the publication-grade story; the strong V0 baseline is the deployment story. Both are reproducible end-to-end via [`findings/2026-05-05-v2.5-eval-thinking/REPRO.sh`](findings/2026-05-05-v2.5-eval-thinking/REPRO.sh) on a fresh GPU.

### Prior anchor (2026-05-01, superseded)

> Held-out mean 0.385 ± 0.000 across 6 chemoprevention fixtures, N=3 seeded trials, manifest byte-stable, score progression v0 0.273 → v1.0 0.385 (+41%). HPV +0.22, 5ARI +0.14, bisphosphonate +0.12, smoking +0.11, aspirin +0.08. Cross-family judge: Qwen2.5-7B-Instruct. Showcase artifact: [`results/canonical-2026-05-01-hb-hard-n1000/CARD.md`](results/canonical-2026-05-01-hb-hard-n1000/CARD.md). Superseded by the 4-benchmark paired-bootstrap eval above; preserved here for historical audit.

This is the v0.1.0 public scaffold. The held-out mean is honest, the methodology is documented
in [`docs/methodology.md`](findings/research/2026-04-29-medomni-v0/methodology-status.md), and the
roadmap to ≥ 0.55 (open-source SOTA on HealthBench Hard) is in [`docs/SPEC.md`](findings/research/2026-04-29-medomni-v1-northstar/SPEC.md) §6.

---

## What this is

A **sovereign, multi-modal, persona-aware medical reasoning system** that reads the same clinical
case (drug + device + image + auscultation audio + family history + prior labs) and returns four
register-shaped answers — one for the **physician** (full diagnostic depth + literature citations),
one for the **nurse** (clinical depth + early-warning escalation cues + teaching scaffold), one for
the **family** (caregiver register + plain-language analogies + when-to-call-911), one for the
**patient** (FKGL ≤ 8 + shared-decision-making tone) — each with a **cited graph path** beneath the
answer that a malpractice attorney could read out at deposition.

**Irreducible core feature**: the persona-tagged grounded explanation. Same case, same evidence,
four registers, one auditable subgraph, manifest-locked reproducibility. Nothing else open-source
does this on one stack, on-prem, with provenance.

The literature gap that justifies the wedge: only **6%** of 67 medical-KG studies addressed
nursing-specific applications ([JMIR AI 2025](https://ai.jmir.org/2025/1/e58670/)). The four-persona
stack with persona-tagged graph edges is genuinely under-served.

---

## Live demo + 3-GPU sovereign factory

The MedOmni stack is **operational now** — not a slide deck. Three GPUs run continuously,
each with a distinct role in the training/serving/evaluation flywheel:

| Pod | Hardware | Role | What's running |
|---|---|---|---|
| **catfish** | NVIDIA B300 (Blackwell, 288 GB HBM3E) | **Inference + agent surface** | `vllm-omni-b300` serving `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` with native structured tool-calling (`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`). Powers `medomni.vercel.app/agent` (PubMed tool, visible reasoning). NVFP4 quantization is Blackwell-only. |
| **lobster** | NVIDIA H200 (Hopper, 143 GB HBM) | **Training + sovereign judge** | Path D Megatron-Bridge LoRA training (V1 shipped 2026-05-03 in 11.3 hr at 2.6 s/step — **12.4× faster** than HF/PEFT eager). Then `judge-qwen` serving `Qwen/Qwen2.5-7B-Instruct` for sovereign corpus filtering. |
| **narwhal** | NVIDIA H200 (Hopper, 143 GB HBM) | **Data factory** | `vllm` serving `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` + `factory_loop.py` generating quality-judged medical reasoning chains for the next training round. ~21k+ items / day. |

**The autonomous loop** (Karpathy autoresearcher pattern, applied to a vertical):

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                                                                 │
        ▼                                                                 │
[narwhal + catfish: factory_loop.py generates clinical reasoning items]   │
        │                                                                 │
        │  47k+ raw items / week                                          │
        ▼                                                                 │
[laptop: scripts/judge_reasoning_sovereign.py — gpt-4.1 + Qwen ensemble]  │
        │                                                                 │
        │  ~45% pass rate, structured rejection reasons, $1.20 / 1k       │
        ▼                                                                 │
[curated jsonl → lobster: scripts/deploy_v2_corpus.sh]                    │
        │                                                                 │
        ▼                                                                 │
[lobster: Path D Megatron-Bridge LoRA on Nemotron-3-Nano-30B (≤5 epochs)] │
        │                                                                 │
        │  ~28 hr wall, $120, kill-switch on val PPL > 10                 │
        ▼                                                                 │
[lobster: HF PEFT export → catfish: vllm --enable-lora]                   │
        │                                                                 │
        ▼                                                                 │
[catfish: serve V_n adapter → /api/agent renders new behavior live]       │
        │                                                                 │
        │  (eval loop: HealthBench Hard N=1000 paired V_{n-1} vs V_n,     │
        │   gpt-4.1 graded, paired-bootstrap CI, ship rule applied        │
        │   literally per PREREG.yaml)                                    │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
```

The loop runs continuously. No GPU sits idle for >12h (idle-deletion-risk per
internal ops rule). PRs are the human-inspected checkpoints; the work between
PRs is autonomous.

**Sovereignty floor:** every box above runs on hardware we control. The only
cloud dependency is **OpenAI's gpt-4.1** as the canonical judge (and that's
optional — sovereign-only Qwen ensemble works, just with less calibration
against the OpenAI HealthBench paper baseline).

---

## Architecture

```
[user query, modality bundle (text + optional image / audio / video)]
        │
   1.  NemoGuard INPUT RAIL (NeMo Guardrails 0.21 Colang 2.0)
        ▼
   2.  cuVS DENSE RECALL (top-50)        ◄┐
        │   embed: llama-nemotron-embed-1b-v2
        │   IVF-PQ over node descriptions + corpus chunks
        │                                  │  parallel
   3.  BM25 SPARSE RECALL (top-50)         │
        ▼                                  │
   4.  RRF FUSION (top-50 → top-25)  ◄────┘
        ▼
   5.  CROSS-ENCODER RERANK (top-25 → top-8)
        │   llama-3.2-nv-rerankqa-1b-v2 (TensorRT-LLM-FP8 on H100)
        ▼
   6.  nx-cugraph SUBGRAPH SLICE
        │   FACTUAL: PrimeKG (Harvard, 129K nodes / 4M edges, MIT license)
        │   REGISTER: persona-tagged graph (NIPDS, evidence-currency, pedagogical-intent)
        ▼
   7.  SUBGRAPH SERIALIZE (5–15K tokens)
        ▼
   8.  OMNI INFERENCE (B300, NVFP4)
        │   nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
        │   23 Mamba2 + 23 MoE + 6 GQA, 256K context, multimodal in / text out
        ▼
   9.  CONSTRAINED DECODING + GROUNDING CITE RAIL
        │   guided_json against citation schema
        │   each cited passage: cosine ≥ 0.8 to retrieved node description
        ▼
  10.  NemoGuard OUTPUT RAIL (Nemotron-Content-Safety-Reasoning-4B)
        ▼
  [persona-shaped grounded answer + cited subgraph]
```

**Bill of materials** (every layer NVIDIA-shipped open):

| Layer | Component | Version |
|---|---|---|
| Driver / CUDA | NVIDIA driver 580+ / **CUDA 13.2 Update 1** (April 12, 2026) | 13.2 |
| Math | cuDNN 9.x · cuBLAS 13.2 · NCCL 2.25.x | bundled |
| Kernels | CUTLASS 4.x + CuTeDSL · FlashAttention-4 (Mar 5, 2026) | latest |
| Compute libs | **RAPIDS 26.04** (cuDF, cuML, cuVS, cuGraph, nx-cugraph) | 26.04 |
| Vector | cuVS (CAGRA + IVF-PQ + IVF-FLAT) | 26.04 |
| Graph engine | nx-cugraph | 26.04 |
| Graph data | PrimeKG (Harvard mims-harvard, MIT) + persona-tagged graph (this repo) | v2.0 |
| Frame | **NeMo Framework** | 2.7.0 |
| Curate | **NeMo Curator 1.1.0** (Feb 23, 2026) | 1.1.0 |
| Guard framework | **NeMo Guardrails 0.21.0** + Colang 2.0 | 0.21.0 |
| Guard models | NemoGuard JailbreakDetect + Nemotron-Content-Safety-Reasoning-4B | revision-pinned |
| Engine | **TensorRT-LLM** + vLLM 0.20.0 | 0.17+ |
| Embed / rerank | llama-nemotron-embed-1b-v2 + llama-3.2-nv-rerankqa-1b-v2 | revision-pinned |
| Brain | **nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4** | revision-pinned |
| Frontend serving | Triton Inference Server with OpenAI-compat | 26.03 |
| Judge (cross-family, sovereign) | Qwen/Qwen2.5-7B-Instruct | FP8 TRT-LLM engine |

The **only** non-NVIDIA weights in the whole pipeline are Qwen2.5-7B-Instruct, which acts as the
cross-family judge per NVIDIA's own published Nemotron-3 reproducibility recipe. Sovereignty by
construction: zero cloud LLM API keys in any code path.

For the full architecture spec, see [`findings/research/2026-04-29-medomni-v1-northstar/SPEC.md`](findings/research/2026-04-29-medomni-v1-northstar/SPEC.md).

---

## Quickstart

```bash
# 1. Clone + create venv
git clone https://github.com/GOATnote-Inc/medomni.git
cd medomni

# 2. Install (uv recommended; pip + venv supported)
uv venv && uv sync                 # OR: python3 -m venv .venv && pip install -e .

# 3. Configure sovereignty-only env
cp .env.example .env
# Edit .env to set HF_TOKEN, BREV_PEM_PATH (no cloud LLM keys — see CLAUDE.md §2)

# 4. Lint + test (laptop, no GPU required)
make lint
make test                          # pytest -m "not integration and not gpu"

# 5. Verify the headline manifest is byte-deterministic
make manifest-verify ARTIFACT=results/ci-medomni-heldout-consolidated-20260429-173557/heldout.json

# 6. (Optional, requires Brev pod access) full bench
make ci-medomni                    # captures snapshot + sovereign_bench + write_card + verdict
```

The `make manifest-verify` step in particular is the central reproducibility claim: it re-emits
the 9-layer manifest from the same artifact twice and asserts byte-equal output. If you can run
that and see "manifest emitter is byte-deterministic," you've reproduced the v0.1.0 result.

---

## Reproducibility manifest

Every demo run emits a `MANIFEST.yaml` with byte-identical fingerprint covering 9 layers:

1. Container image digests (vLLM, TRT-LLM, NeMo Guardrails, NeMo Curator)
2. Weight SHAs (Omni NVFP4, NemoGuard models, embed-1b-v2, rerank-1b-v2, Qwen2.5-7B)
3. Corpus SHAs (OpenEM 370, PubMed-OA shard, fixture set)
4. Config files (Colang rails, persona prompts, retrieval params)
5. Random seeds (decoding seed, retrieval tie-break seed)
6. Hardware-foot-gun flags (`--no-async-scheduling`, `--max-model-len`, `--kv-cache-dtype fp8`)
7. Benchmark fixtures (chemoprevention 6 + tamoxifen 1 = 7 cases)
8. Judge model digest (Qwen2.5-7B-Instruct FP8 TRT-LLM engine)
9. Git SHA of harness + SPEC.md SHA

The 9-layer manifest is the rigor that OE / Hippocratic / ChatGPT-for-Clinicians do not demo
because their stack isn't built for it. See [`docs/SPEC.md`](findings/research/2026-04-29-medomni-v1-northstar/SPEC.md) §5.6
and [`scripts/emit_manifest.py`](scripts/emit_manifest.py).

---

## Repo structure

```
medomni/
├── README.md                  # this file (results-first → architecture → quickstart)
├── LICENSE                    # Apache-2.0
├── CONTRIBUTING.md            # style + tests + manifest discipline
├── SECURITY.md                # disclosure email + scope + durable doctrine
├── CODE_OF_CONDUCT.md         # Contributor Covenant 2.1
├── CHANGELOG.md               # Keep-a-Changelog
├── SECURITY-INCIDENTS.md      # 2026-04-29 HF_TOKEN PTY-echo postmortem
├── CLAUDE.md                  # AI-agent operating charter (sovereignty contract, isolation rules)
├── CITATION.cff               # citation metadata
├── pyproject.toml             # PEP 621 + uv-compatible
├── Makefile                   # health, manifest, ci-medomni, demo-pre-flight
├── DEMO.md                    # 12-minute on-stage demo runbook
├── .env.example               # two secrets total (HF_TOKEN + BREV_PEM_PATH)
├── .pre-commit-config.yaml    # ruff + detect-secrets + custom hooks
├── .secrets.baseline          # detect-secrets baseline (no secrets in baseline; verified)
├── .github/                   # workflows, ISSUE_TEMPLATE, PR template, CODEOWNERS
├── corpus/                    # clinical fixtures + held-out + golden cases + medical guidelines
├── data/                      # seed_kg/ (illustrative graph), embeddings (gitignored)
├── configs/                   # guardrails rails, persona prompts, retrieval params
├── scripts/                   # bench, retrieval, manifest, serve_*, _runpod_ssh.sh
├── tests/                     # unit (laptop), integration / gpu (Brev pod required)
├── mla/                       # medical-LLM agent harness (legacy from prism42 lift)
├── findings/                  # research provenance — north-star SPEC + POSITIONING + DEMO + PITCH
│   └── research/
│       ├── 2026-04-29-medomni-v0/        # methodology audit
│       └── 2026-04-29-medomni-v1-northstar/  # the SPEC + the moat
├── results/                   # per-run CARDs + MANIFEST.yaml (showcase entries committed)
└── reproducibility/           # manifest emitter design + captured templates
```

---

## Sovereignty contract

Per [CLAUDE.md §2](CLAUDE.md), MedOmni runs with **exactly two secrets**:

- `HF_TOKEN` — Hugging Face read-only, gated-model access.
- `BREV_PEM_PATH` — path to the Brev SSH key on disk.

**No cloud LLM API keys exist in any code path.** The judge runs locally on H100. The serve runs
locally on H200. Guardrails run locally. RAG runs locally. External keys defeat the entire premise.

The 2026-04-29 HF_TOKEN PTY-echo incident and its durable mitigation (the `_runpod_ssh.sh` secret-grep
guard) are documented in [SECURITY-INCIDENTS.md](SECURITY-INCIDENTS.md). Operational maturity is
shipping the postmortem alongside the fix.

---

## Status & roadmap

- **v0.1.0 (this release)** — initial public scaffold; held-out 0.385; reproducibility manifest
  shipped; 9-layer byte-deterministic emitter.
- **v0.2 (Phase 2.4 — in flight)** — PrimeKG (Harvard mims-harvard, 129K nodes / 4M edges) wired into
  Stage 6 nx-cugraph subgraph slice; projected lift +0.05–0.10 closes the SPEC §6 ≥ 0.45 gate.
- **v0.3 (Phase 1.6 / 1.7)** — broader verbatim PMC corpus + N=30 fixture scale.
- **v0.4 (Phase 2.3)** — ensemble cross-judge agreement scoring.
- **v1.0 — pre-launch SOTA gate**: HealthBench Hard ≥ 0.55, MedAgentBench ≥ 0.70, MedQA-USMLE ≥ 0.85.
  If we miss the gate, we don't ship — accuracy degradation = revenue collapse.

**Image-aware MedOmni (training imminent — next 24h)** — V_image-aware extends the Nemotron-Omni
reasoning surface with native radiology comprehension via continued pretraining on the
[GOATnote-Inc/medimage-corpus](https://github.com/GOATnote-Inc/medimage-corpus) registry: 134
open-source medical imaging datasets across CT, X-ray, MRI, ultrasound, and image-text paired
(VLM) collections, with download dispatchers and format converters wired for H200 ingest.
Manifests cover ~7.4 PB cataloged and ~700 TB pullable in the open + registration tiers; the
demo's `Imaging` rail (X-ray · MRI · panoramic) is the first surface that benefits.

Detailed phasing and acceptance criteria: [`SPEC.md`](findings/research/2026-04-29-medomni-v1-northstar/SPEC.md) §8.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The manifest discipline is the load-bearing rule:
**every PR that touches the inference path must attach a fresh manifest hash from a re-bench.**

Bug reports + feature requests + reproducibility issues all have templates in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).

---

## Citation

```
@software{medomni_v0_1_2026,
  author       = {Dent, Brandon},
  title        = {{MedOmni: Sovereign nurse-first medical-LLM stack on NVIDIA's open-component stack}},
  version      = {0.1.0},
  date         = {2026-04-29},
  url          = {https://github.com/GOATnote-Inc/medomni},
  organization = {GOATnote Inc.}
}
```

Full machine-readable citation metadata: [`CITATION.cff`](CITATION.cff).

---

## Acknowledgements

Built on NVIDIA's open-component stack — every layer carries an NVIDIA badge: CUDA 13.2, cuVS,
nx-cugraph, NeMo Guardrails, NeMo Curator, TensorRT-LLM, NemoGuard,
Llama-Nemotron-Embed/Rerank, Nemotron-3-Nano-Omni. Architecture derives from the
[NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) reference,
[NVIDIA Nemotron-3 reproducibility recipe](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/nano-v3-reproducibility.md),
and [Stanford CRFM HELM](https://github.com/stanford-crfm/helm) reproducibility-first patterns.

Held-out clinical fixtures sourced from [USPSTF](https://uspreventiveservicestaskforce.org/),
[NCCN](https://nccn.org/), [ACOG](https://acog.org/), [IBIS-II](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736%2814%2960140-3),
[NSABP P-1](https://academic.oup.com/jnci/article/97/22/1652/2521376), and the
[Nurse.org](https://nurse.org/) RN-perspective corpus.

Provenance: derived from `github.com/GOATnote-Inc/prism42` (private fork lineage); squash-imported
with no shared commit history. The medical-LLM eval harness was lifted from public prism42 with
zero prod-surface entanglement.
