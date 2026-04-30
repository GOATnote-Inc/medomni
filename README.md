# MedOmni

**Sovereign nurse-first medical-LLM stack on NVIDIA's open-component stack.**
Held-out **0.385** mean across 6 chemoprevention fixtures, **+41%** over v0 baseline,
deterministic across N=3 seeded trials, manifest sha256 `560baccbb706` byte-stable.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900.svg)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-26.04-7400B8.svg)](https://rapids.ai/)
[![NeMo](https://img.shields.io/badge/NeMo-Framework_2.7-76B900.svg)](https://github.com/NVIDIA-NeMo/NeMo)
[![Reproducibility](https://img.shields.io/badge/manifest-9--layer-success.svg)](docs/SPEC.md#56-the-9-layer-reproducibility-manifest)

---

## Headline result

> **Held-out mean 0.385 ± 0.000** across 6 chemoprevention fixtures, **N=3 seeded trials**,
> manifest byte-stable, score progression **v0 0.273 → v1.0 0.385 (+41%)**.
>
> 5 of 6 targeted fixtures lifted monotonically vs v0:
> HPV +0.22, 5ARI +0.14, bisphosphonate +0.12, smoking +0.11, aspirin +0.08.
>
> Cross-family judge: Qwen2.5-7B-Instruct (sovereign, no cloud LLM keys in any path).
> Comparator verdict: **PASS** on v0 → v1.0 (+0.112, no major regression).
>
> SPEC §6 ≥ 0.45 gate **NOT YET MET** (-0.065). Closes via Phase 1.6 broader corpus +
> Phase 1.7 fixture scale + Phase 2.3 ensemble cross-judge.
>
> Showcase artifact: [`results/ci-medomni-heldout-consolidated-20260429-173557/CARD.md`](results/ci-medomni-heldout-consolidated-20260429-173557/CARD.md)

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
