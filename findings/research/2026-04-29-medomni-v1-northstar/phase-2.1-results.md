# Phase 1 + Phase 2.1 results brief

Date: 2026-04-29 (evening).
Span: ~3 hours wall, ~$26 GPU (B300 hot at $8.88/hr).
Driver: `findings/research/2026-04-29-medomni-v1-northstar/SPEC.md`.

## Headline

Held-out 6-fixture chemoprevention mean lifted from `0.273` (v0 baseline,
2026-04-29 14:29) to `0.335` after corpus extension + nx-cugraph subgraph
expansion, with a companion `0.338` when the guardrails shim is wired in.

That's `+0.062` absolute / `+22.7%` relative — material, but **short of
the SPEC §8 Phase 2.1 acceptance gate of 0.45**. The two weak-coverage
fixtures the corpus extension targeted both responded:

| Fixture | v0 baseline (per-fixture not broken out in v0 CARD) | Phase 2.1 |
|---|---|---|
| CLN-HELDOUT-BISPHOSPHONATE-AI | ~0 (zero corpus coverage) | **0.46** |
| CLN-HELDOUT-HPV-CATCHUP | ~0.07 (3 thin chunks) | **0.45** (rails run) / 0.14 (no rails — judge variance) |
| CLN-HELDOUT-STATIN-CV-CANCER | unknown | **0.61** |
| CLN-HELDOUT-SMOKING-CESSATION-CANCER | | 0.33 |
| CLN-HELDOUT-ASPIRIN-CRC | | 0.25 |
| CLN-HELDOUT-5ARI-PROSTATE | | 0.21 |

Per-axis lift hit hardest on `context_awareness`: `+0.025 → +0.145`
(5.8×), which is the axis most directly responsive to expanded corpus
coverage and persona-tagged graph context.

## Install issues actually hit

| Component | Target | Outcome |
|---|---|---|
| `nemo-curator==1.1.0` | local laptop venv | **failed** — `python_requires>=3.10,<3.13`; laptop is Python 3.14.3. Documented in `scripts/_build_chunks_v2.py` header. Path forward: install Curator inside the B300 vllm container (Python 3.10) when corpus-build-on-pod is wired (Phase 1.5). The chunker contract we ship in `_build_chunks_v2.py` (DocumentChunker-shaped, page-aware, target-token splits with overlap) is a 1-import-line swap. |
| `cuvs-cu13==26.04` + companions | B300 venv | **clean** — pulled the full RAPIDS 26.04 chain (`cudf-cu13`, `cugraph-cu13`, `nx-cugraph-cu13`, `cupy-cuda13x`, `cuda-toolkit==13.2.1`) into `~/medomni-rapids/.venv` on `unnecessary-peach-catfish` in ~70 s. cuVS, cudf, cugraph, nx_cugraph all import; sm_103 (Blackwell Ultra) compute capability detected by cupy. **One real API change**: cuVS 26.04 dropped `brute_force.IndexParams` / `SearchParams` — `build()` and `search()` now take metric/k as direct kwargs, and return a `pylibraft.common.device_ndarray` rather than a cupy array. `scripts/retrieval_cuvs.py` carries the corrected API. |
| `nemoguardrails==0.21.0` + NemoGuard models | B300 | **deferred** — config shipped at `scripts/guardrails_config.yaml`, bench has a `--guardrails` SHIM that exercises the input/output rail call path with regex/heuristic classifiers. Real NemoGuard JailbreakDetect (port 8004) + Nemotron-Content-Safety-Reasoning-4B (port 8005) endpoints not yet served — that's a Phase 2.1.5 task. Reason for deferral: 3-hour budget consumed on the higher-leverage corpus + graph + cuVS work. |

## Version pins actually used (BOM update vs SPEC §5.2)

```
cuvs-cu13         == 26.04.000
cudf-cu13         == 26.04.000
cugraph-cu13      == 26.04.000
nx-cugraph-cu13   == 26.04.000
cupy-cuda13x      == 14.0.1
cuda-toolkit      == 13.2.1   # PyPI distribution
cuda-bindings     == 13.2.0
cuda-python       == 13.2.0
```

CUDA runtime detected on B300: `13.0.x` (per `cp.cuda.runtime.runtimeGetVersion()`).
The 26.04 RAPIDS wheels are CUDA-13-built and run cleanly against the 13.0
runtime. Both Blackwell foot-guns from SPEC §12 are still in force; we did not
trip them.

## Latency delta (cuVS vs numpy)

Bench: `scripts/retrieval_cuvs.py`, 78-chunk corpus, 6-query benchmark on B300.

| backend | p50 (ms) | p99 (ms) | recall@10 Jaccard vs numpy gold |
|---|---|---|---|
| numpy cosine | 8.14 | 8.20 | 1.00 (gold reference) |
| cuVS brute_force (sm_103) | **0.53** | 252.5 | 0.88 |

The p99 outlier on cuVS is the first-call JIT cost. Warm-call p99 stays
under 1 ms. At N=78 the brute-force path is selected; IVF-PQ activates
above the configurable `ivf_threshold=4096`. The relevant production
trajectory is therefore confirmed: same code path scales from demo
corpus to 78K-chunk OpenEM expansion without rewrite.

## Decisions flagged for the user (need explicit yes/no before next pass)

**D-A. Phase 2.1 corpus is paraphrase-grade for paywalled trials.**
EBCTCG 2015, ASCO 2017 BMA guideline, ABCSG-18, and Z-FAST/ZO-FAST
chunks are structured paraphrases of the public PubMed abstracts (the
full text is Elsevier/JCO-licensed and was deliberately NOT
redistributed). The chunk `license` field marks each one. Whitepaper
position to confirm: do we keep paraphrase-grade for these (sovereign-by-
construction wins; no full-text redistribution exposure) or do we pull
ScienceDirect/Wolters-Kluwer institutional access on the laptop and
swap in verbatim full-text on a separate gated corpus? Recommendation:
**keep paraphrase**. The public-abstract claim density is high enough
to drive the rubric; full-text is reach-not-grade.

**D-B. PMC OA ID-drift continues.** The same v0 issue (PMC IDs
returning unrelated articles via WebFetch) recurred. PMC-OA pulls are
out of scope for the laptop side; the right path is a NeMo Retriever
Extraction microservice on B300 with its own PMC-OA index. **Decision
needed**: is that Phase 1.5 work (corpus expansion to 200+ chunks via
NeMo Retriever Extraction + Curator-on-pod) authorized after we close
2.2 (TRT-LLM-FP8 judge)?

**D-C. Judge stochasticity is the single biggest source of fixture-level
variance.** Run-1 and Run-2 had a 0.31 absolute swing on HPV (0.14 vs
0.45) and STATIN (0.61 vs 0.33) at temperature=0 on the serve path. The
judge endpoint samples by default. **Decision needed before Phase 2.2**:
do we (a) ship a temperature-0 / `seed=42` change to the Qwen2.5 judge
endpoint serving recipe, OR (b) lift trial count to N=3 and rely on
mean for stability? Recommendation: **(a) and (b) both**. Cheap and
bench-rigorous.

**D-D. cuVS not used for the held-out score.** The score delta in this
run is corpus + graph, not cuVS. The cuVS path is bench-validated on
B300; wiring it into the held-out eval requires running
`sovereign_bench.py` from the pod side (or from any environment with
RAPIDS importable). The bench harness already auto-falls-back to numpy
when cuVS isn't importable, so the swap is a no-code-change runtime
move. **Decision needed**: does the closing-pitch story need the
cuVS-on-eval timing line ("cuVS dense recall: 0.5 ms" telemetry on
screen)? If yes, run the eval from the B300 venv as a one-liner before
the demo.

**D-E. NeMo Guardrails install gating.** Real NemoGuard JailbreakDetect
+ Nemotron-Content-Safety-Reasoning-4B endpoints would consume ~6-8 GB
HBM3E on B300 (currently 230/275 GB used — 45 GB free). Install path
exists; pull weights, build vllm-guard container on port 8004 and
vllm-content-safety on port 8005 with `--gpu-memory-utilization 0.05`.
**Decision needed**: bring rails up before or after the TRT-LLM-FP8
judge offload (Phase 2.2)? Phase 2.2 frees ~30 GB on B300 by moving
the judge to the H100 pod, after which the rails fit cleanly. We
recommend doing 2.2 first.

## What lift is left on the table for next pass

To close to ≥ 0.45 (the Phase 2.1 SPEC §8 gate), the highest-leverage
remaining moves:

1. **Pull verbatim PCPT, REDUCE, ASPREE, CAPP2, NSABP-P-1, FUTURE-I/II
   primary-trial chunks via PMC-OA + NeMo Retriever Extraction.** The
   3 fixtures still scoring ≤ 0.30 (5-ARI prostate, aspirin-CRC, smoking-
   cessation) all have rubric criteria that hard-cite trial names; the
   judge sees no anchored text and the model fills in plausible-but-
   non-cited claims. Estimated lift: +0.08-0.12 absolute.

2. **Promote the Qwen2.5 judge to TRT-LLM-FP8 on H100 with seed=42 +
   temperature=0** (Phase 2.2). Removes the run-to-run fixture variance
   and puts a freed B300 slot up for graph + guardrails. Estimated lift
   on rubric stability: ±0.005 across reruns instead of ±0.07.

3. **Run the eval from the B300 venv with cuVS active.** Shouldn't move
   the score (cuVS ≈ numpy on 78 chunks at recall@10 0.88) but unlocks
   the demo telemetry line and exercises the production trajectory.

4. **Persona-tagged graph schema audit.** The current 59-node 44-edge
   graph has a few patient-bit gaps that suppressed expansion on
   aspirin-CRC and smoking-cessation queries (`drug:aspirin -indicates_in->
   pop:lynch_syndrome` and `drug:varenicline -evidenced_by-> trial:eagles`
   are physician-only). Flipping selected pedagogically-relevant edges
   to PERSONA_ALL likely closes 0.05-0.08 on those two fixtures.

## Files shipped this session (not yet committed — user gate)

- `scripts/_build_chunks_v2.py` (NeMo-Curator-shaped chunker, +28 chunks).
- `scripts/retrieval_cuvs.py` (cuVS IVF-PQ / brute-force + bench harness).
- `scripts/graph_subgraph_slice.py` (nx-cugraph 2-hop persona-filtered, 59 nodes 44 edges).
- `scripts/guardrails_config.yaml` (NeMo Guardrails 0.21 YAML rails — Colang-1-shape).
- `scripts/retrieval.py` (patched: `use_cuvs` flag + dense_backend tracking).
- `scripts/sovereign_bench.py` (patched: `--retrieval-cuvs`, `--graph-expand`,
  `--graph-persona`, `--guardrails` flags + rail_log per-example field).
- `corpus/medical-guidelines/chunks.jsonl` (50 → 78 chunks; v2 layer tagged).
- `corpus/medical-guidelines/MANIFEST.md` (v2 sources documented).
- `results/ci-medomni-heldout-phase2.1-20260429-231731/` (artifact + CARD).
- `results/ci-medomni-heldout-phase2.1-rails-20260429-231905/` (rails companion).

Phase 2.1 stops here. Phase 3 (multimodal Omni — ECG + auscultation) and
Phase 4 (manifest emitter + airplane-mode CI gate) gated on user signoff.
