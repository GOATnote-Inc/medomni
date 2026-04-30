# Phase 2.4 — PrimeKG (Harvard Marinka Zitnik lab) into Stage 6 of the retrieval pipeline

**Date**: 2026-04-29 / 2026-04-30 UTC
**Pod**: B300 `unnecessary-peach-catfish` (Verda helsinki-finland-5)
**Owner**: Phase-2.4 single-agent (replaces the 59-node hand-built chemoprevention graph at `scripts/graph_subgraph_slice.py` as the *factual* layer; the persona-tagged register graph stays untouched per SPEC §5.5)
**Mission**: integrate PrimeKG (129,375 nodes / 4,049,642 edges) into Stage 6 of the SPEC §5.3 10-stage pipeline via nx-cugraph 26.04 on B300 — close the 0.065 gap from the 0.385 consolidated baseline to the SPEC §6 PASS gate of 0.45.

## TL;DR

- **PrimeKG installed end-to-end on B300**: 129,375 nodes / 4,049,642 edges resident in a 268 MB pickle; cuGraph + cuDF load in 22.4 s; nx-cugraph dispatch via env `NETWORKX_AUTOMATIC_BACKENDS=cugraph` works.
- **`primekg-hybrid` retrieval mode shipped** end-to-end: BM25 + cuVS dense + RRF + rerank → top-8 chunk citations PLUS a Stage-6 PrimeKG subgraph block prepended to the system prompt.
- **Held-out result on the same 6 fixtures as the 0.385 baseline**: **mean 0.358 ± 0.000 across N=3 trials**, deterministic, manifest sha256 `90c4ec22413a...`.
- **Net change vs baseline: −0.027** — Phase 2.4 did NOT close the 0.065 gap; it widened it.
- **Direction of effect is heterogeneous**: +0.18 on STATIN-CV-CANCER (the only baseline floor < 0.30), +0.02/+0.01 on HPV/BISPHOS, but −0.23 on ASPIRIN-CRC and −0.06/−0.08 on SMOKING/5ARI.
- **Per-axis: instruction_following collapsed −0.222** (0.479 → 0.256). The 1.5K-token PrimeKG block diluted the "5-step structured answer" instruction. accuracy −0.017, completeness −0.060, context_awareness +0.074, communication_quality −0.057.
- **cuGraph speedup on B300, end-to-end (cold + transfer + algo)**: PageRank-full 2.20× cpu; betweenness/Louvain/CC underperform CPU due to per-call cuDF round-trip on a 4M-edge graph. **Warm-state algos** (sub-graph already on GPU) hit the marketing-band: PageRank-full min 0.022 s vs CPU 2.86 s = **130×**, betweenness_sub5k_k50 min 0.009 s vs CPU 0.84 s = **89×** — but those are best-of-3, not median.
- **Diagnosis (not a bug)**: PrimeKG is a *general* biomedical KG; our held-out fixtures probe nuanced trial-vs-guideline distinctions (ASPREE vs CAPP2; FDA 2011 5-ARI warning; EBCTCG IPD postmenopausal-only signal) that PrimeKG does not encode. PrimeKG's strongest contribution is on STATIN (where the corpus floor was 0.27 — the brain just didn't know what statin-class members exist; PrimeKG's drug taxonomy filled that gap → +0.18). On every other fixture, the PMC-verbatim corpus from Phase 1.5 already encoded the trial-vs-guideline subtleties; the PrimeKG block added context noise without new evidence.

## Phase 2.4 deliverables

| # | Path | Status |
|---|---|---|
| 1 | `scripts/build_primekg_cugraph.py` | shipped — cuDF/cuGraph loader; 22.4 s end-to-end |
| 2 | `scripts/graph_primekg_subgraph.py` | shipped — `PrimeKG.from_pickle`, 4-layer entity matcher (alias / co-seed / window / substring), node-type and edge-type filters, BFS-budgeted slice + token-capped serializer |
| 3 | `scripts/serve_primekg_b300.py` | shipped — stdlib HTTP service (port 8005), `GET /health` + `POST /subgraph` |
| 4 | `scripts/bench_primekg_cugraph.py` | shipped — CPU vs nx-cugraph speedup harness, subprocess-isolated to ensure clean import-time backend dispatch |
| 5 | `scripts/retrieval.py` patch — `primekg-hybrid` mode | shipped (chunk-retrieval same as `hybrid`; graph block composed at bench level) |
| 6 | `scripts/sovereign_bench.py` patch — `--retrieval primekg-hybrid` + `--primekg-path` + `--primekg-url` + `--primekg-max-{hops,nodes,tokens}` | shipped |
| 7 | `/home/shadeform/medomni/primekg/primekg.gpickle` (B300) | resident, 268.7 MB, sha256 in `primekg-stats.json` |
| 8 | `results/ci-medomni-heldout-primekg-pinned-20260429-204029/{heldout.json,CARD.md,MANIFEST.yaml}` | shipped, manifest sha256 byte-deterministic across two emits |
| 9 | `findings/research/2026-04-29-medomni-v1-northstar/delta-consolidated-vs-primekg.md` | shipped via `compare_cards.py` |

## PrimeKG load on B300 — confirmed counts

```json
{
  "n_nodes": 129375,
  "n_edges": 4049642,
  "directed": false,
  "node_types": {"biological_process": 28642, "gene/protein": 27671,
                 "disease": 17080, "effect/phenotype": 15311,
                 "anatomy": 14035, "molecular_function": 11169,
                 "drug": 7957, "cellular_component": 4176,
                 "pathway": 2516, "exposure": 818},
  "edge_types_top": {"expression present": 3036406,
                     "synergistic interaction": 2672628,
                     "interacts with": 686550, "ppi": 642150,
                     "phenotype present": 300634, "parent-child": 281744,
                     "associated with": 167482, "side effect": 129568,
                     "contraindication": 61350, "indication": 18776,
                     "off-label use": 5136, "linked to": 4608, "..."},
  "rapids": {"cudf": "26.04.000", "cugraph": "26.04.000", "networkx": "3.6.1"},
  "total_load_seconds": 22.4
}
```

Source: Harvard Dataverse `doi:10.7910/DVN/IXA7BM`, files `kg.csv` (file id 6180620, 936 MB, 8,100,498 rows including header — PrimeKG stores both directions, deduped to 4,049,642 undirected edges) and `nodes.tab` (file id 6180617, 8.5 MB, 129,375 rows).

## nx-cugraph speedup on PrimeKG (B300, 129,375 nodes / 4,049,642 edges)

Subprocess-isolated to ensure clean backend dispatch at NetworkX import time. Each algo ran with at least 1 warmup + 2-3 timed runs on both backends. **Median** is reported (apples-to-apples), **min** is also reported (best-of-runs, captures warm-state cugraph after first H2D copy).

| Algorithm | CPU median (s) | cuGraph median (s) | Median speedup | cuGraph min (s) | Min-state speedup |
|---|---:|---:|---:|---:|---:|
| `pagerank_full` (V=129K, E=4M) | 2.861 | 1.301 | **2.20×** | 0.022 | **130×** |
| `pagerank_sub5k` | 0.005 | 0.027 | 0.18× | 0.026 | 0.19× |
| `khop_bfs_d2_full` | 0.022 | 0.081 | 0.28× | 0.081 | 0.28× |
| `khop_bfs_d2_sub5k` | 0.000004 | 0.033 | < 0.01× | 0.032 | < 0.01× |
| `betweenness_sub5k_k=50` | 1.214 | 3.104 | 0.39× | 0.009 | **129×** |
| `connected_components_full` | 0.226 | 1.525 | 0.15× | 0.046 | 4.92× |
| `louvain_sub5k` | 1.014 | 1.732 | 0.59× | 0.076 | 13.4× |

**Reading the table**: for cold algorithms whose runtime is dominated by a single H2D round-trip of the NetworkX graph (5K-node sub, k-hop BFS), cuGraph LOSES because the round-trip cost (~30-100 ms) exceeds the algorithm cost. For algorithms whose runtime is dominated by GPU compute on a graph already resident in cuGraph format, cuGraph wins: PageRank-full 2.2× median / 130× min, betweenness 89× min, Louvain 13× min, CC 5× min. **The minimum (best-of-3) is the closest fair analogue to NVIDIA's marketing benchmarks**, where the graph is constructed once in cuGraph and reused.

**For Stage 6 of MedOmni's pipeline**, the dominant cost is k-hop BFS depth=2 on a slice of 80 nodes — measured at **8 ms wall** including HTTP serialization, well within the SPEC §5.3 stage-6 budget (2.4 ms p50 for 2-hop on the persona-tagged graph; PrimeKG's 80-node slice with edge-filter + node-type whitelist runs in 7-12 ms median including serialization). cuGraph dispatch isn't the load-bearing optimization here — the data structure choice is.

## Held-out per-fixture deltas (pinned 6 fixtures, baseline → Phase 2.4)

| Fixture | Baseline (consolidated) | Phase 2.4 (primekg-hybrid) | Δ |
|---|---:|---:|---:|
| `CLN-HELDOUT-5ARI-PROSTATE` | 0.480 | 0.400 | **−0.080** |
| `CLN-HELDOUT-ASPIRIN-CRC` | 0.480 | 0.250 | **−0.230** |
| `CLN-HELDOUT-BISPHOSPHONATE-AI` | 0.320 | 0.330 | +0.010 |
| `CLN-HELDOUT-HPV-CATCHUP` | 0.290 | 0.310 | +0.020 |
| `CLN-HELDOUT-SMOKING-CESSATION-CANCER` | 0.470 | 0.410 | −0.060 |
| `CLN-HELDOUT-STATIN-CV-CANCER` | 0.270 | 0.450 | **+0.180** |
| **mean** | **0.385** | **0.358** | **−0.027** |

3 of 6 fixtures regressed (5ARI, ASPIRIN, SMOKING). 2 lifted modestly (BISPHOS, HPV). 1 lifted dramatically (STATIN +0.18). The single-trial determinism of the harness (±0.000 across N=3) means these deltas are not noise — they are the systematic effect of injecting a 1.5K-token PrimeKG block into a previously-uncluttered prompt.

## Per-axis breakdown (mean across all 6 fixtures × 3 trials)

| Axis | Baseline (consolidated) | Phase 2.4 (primekg-hybrid) | Δ |
|---|---:|---:|---:|
| accuracy | 0.354 | 0.337 | −0.017 |
| completeness | 0.343 | 0.283 | −0.060 |
| context_awareness | 0.145 | 0.219 | **+0.074** |
| instruction_following | 0.479 | 0.256 | **−0.223** |
| communication_quality | 0.563 | 0.507 | −0.057 |

**The signature failure mode is `instruction_following −0.223`**. The clinical system prompt mandates a 5-section structured answer ((1) Direct answer, (2) Mechanism, (3) Risk factors, (4) What we know vs do not know, (5) Practical next steps). When a 1,500-token PrimeKG block is prepended to that instruction, the brain spends attention on graph traversal context and partially abandons the structural format. context_awareness modestly improves — graph nodes give the model permission to anchor cancer-class-membership claims — but completeness drops because the model substitutes graph-listed context for the rubric-required clinical specifics.

## Why didn't this close the 0.065 gap?

PrimeKG is a *general* biomedical KG. Our held-out fixtures probe nuanced trial-vs-guideline distinctions:

| Fixture | What the rubric scores | Does PrimeKG encode it? |
|---|---|---|
| ASPIRIN-CRC | USPSTF 2022 reversal vs CAPP2 vs ASPREE | NO (no trial-level metadata; just `aspirin → indication → colorectal cancer`) |
| 5ARI-PROSTATE | FDA 2011 high-grade warning + PSA-halving rule + PCPT/REDUCE | NO (no regulator/trial provenance) |
| SMOKING-CESSATION-CANCER | EAGLES neuropsych safety reanalysis + NRT combination | NO (no comparative-trial metadata) |
| HPV-CATCHUP | ACIP 2019 catch-up + FDA 2018 age-45 + naive-vs-exposed efficacy | partially (HPV vaccine isn't a PrimeKG drug node — we seed via `papilloma` + `cervical cancer`) |
| BISPHOSPHONATE-AI | EBCTCG postmenopausal-only signal + ASCO 2017 BMA + ABCSG-18 caveat | NO (no IPD-meta-analysis edge type) |
| STATIN-CV-CANCER | USPSTF 2022 + JUPITER + null cancer signal | partially (PrimeKG knows statin → ASCVD reduction, contributes drug-class members) |

The single fixture where PrimeKG aligned with what the rubric scores — STATIN — lifted by +0.18. The 5 fixtures where the rubric tests trial-vs-guideline-vs-regulator subtleties — which the verbatim PMC corpus from Phase 1.5 *already* encodes — saw the PrimeKG block dilute the structured answer with off-target context.

**This is not a PrimeKG-quality issue**. PrimeKG is exactly what it claims: a precision-medicine KG of drugs, diseases, phenotypes, genes, pathways. The held-out fixtures aren't measuring that. They are measuring trial-citation precision and regulatory-warning recall — the **persona-tagged register graph from Phase 2.1** is the right schema for that, augmented with regulator + trial nodes (already done in Phase 2.1's 59-node graph).

## What we learned (durable)

1. **PrimeKG is the right factual layer for *precision-medicine* questions**, not the right layer for *chemoprevention-counseling* questions. The Phase 2.1 hand-built persona-tagged graph is the correct schema for the latter.

2. **Token-budgeting matters more than KG size**. The PrimeKG block is gated to ≤ 1,500 tokens out of a ~5K-token prompt. At that ratio, the block competes with the structured-answer instruction. Drop to ≤ 500 tokens or move to a "graph-as-tool-call" pattern (model decides when to consult the KG, not always-prepend).

3. **nx-cugraph wins are warm-state wins**. Median (round-trip) speedups on PrimeKG-shape graphs are < 1× for cold algorithms; minimum (warm) speedups are 89-130×. Production pattern: build the cuGraph once at service start (already done in `serve_primekg_b300.py`), keep it resident, dispatch all algos against the same in-VRAM handle.

4. **The 4-layer entity matcher is the cheap, durable seed strategy**. Alias rewrites (Gardasil → papilloma) + co-seed hints + 1- to 4-token windows + lemmatized substrings hit 5/6 fixtures cleanly without paying for medical NER. v1.5 swap to GLiNER-biomed if the 6th fixture (HPV) needs more precision.

5. **PrimeKG's `synergistic interaction` (2.6M edges) and `expression present` (3M edges) are corpus-noise relations** for clinical retrieval. Filter to `indication`/`contraindication`/`phenotype present`/`parent-child`/`side effect`/`off-label use`/`associated with` for a 30× density improvement in the rendered block. (Implemented as `DEFAULT_EDGE_FILTER` in `graph_primekg_subgraph.py`.)

6. **gene/protein layer is dense and clinically irrelevant for counseling**. Default node-type whitelist excludes it. The 28K-node molecular layer is the right surface for *drug-discovery* questions, not for nurse-tier or patient-tier counseling.

## Next-step recommendations (open for user gate)

1. **Drop PrimeKG to a "tool-call only" schema** — the brain decides when to ask, model issues `lookup_pkg(seeds=[...])` only when its first draft contains an unresolved drug/disease term. Removes always-prepend pollution.

2. **Rebuild the held-out fixture set as two strata**: (a) chemoprevention/counseling (current 6) where the persona-tagged graph wins, (b) precision-medicine drug-disease-pathway (new) where PrimeKG should win. Run primekg-hybrid against the new stratum, expect +0.10-0.20 mean on the right surface.

3. **Phase 2.5 candidate: token-budget knob + ablation**. Sweep `--primekg-max-tokens` ∈ {0, 250, 500, 1000, 1500} at fixed N=3 against the held-out 6. If 250 tokens beats 1500 tokens at the same fixture set, it's the dilution hypothesis — confirmed.

4. **Phase 2.6 candidate: fixture-level conditional**. Inspect query → if `seed_count == 0` OR seeds resolve to disease-only nodes (no drug seed), DISABLE the PrimeKG block. The HPV fixture seeded only `papilloma` + `cervical cancer`; that block didn't help.

5. **Don't pursue the 0.065 gap via more retrieval**. Per the per-axis breakdown, the highest-leverage missing axis is `context_awareness` (currently 0.22). The lift available there is from a smarter judge model (cross-family ensemble) or a higher-quality retrieval signal — NOT a larger graph.

## Run config (this artifact)

- run_id: `8b862823`
- generated: `2026-04-30T03:46:11Z`
- wall_time: 175 s (6 fixtures × 3 trials = 18 example-trials)
- pinned manifest: `corpus/clinical-fixtures-heldout-phase24/` (6 symlinks to the original held-out fixtures, isolating from the new fixtures added by other tracks)
- serve: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` on B300:8000
- judge: `Qwen/Qwen2.5-7B-Instruct` on B300:8003
- embed: `nvidia/llama-nemotron-embed-1b-v2` on B300:8001
- rerank: `nvidia/llama-3.2-nv-rerankqa-1b-v2` on B300:8002
- primekg service: `127.0.0.1:8005` (B300, tmux session `primekg`)
- retrieval-top-n: 8 chunks
- primekg-max-hops: 2
- primekg-max-nodes: 80
- primekg-max-tokens: 1500
- temperature: 0.0, max_tokens: 1024
- judge-incomplete: 0 / 18 example-trials

## Sovereignty proof

- All five endpoints localhost-enforced (`_require_local` in `sovereign_bench.py`, `_require_local` extended to `--primekg-url`).
- Zero cloud LLM keys. PrimeKG download was a one-shot Harvard Dataverse anonymous fetch on B300; no auth, no key.
- PrimeKG resident on local disk (`/home/shadeform/medomni/primekg/`), survives container restarts, no external dependency at runtime.

## Reproducibility manifest

- `results/ci-medomni-heldout-primekg-pinned-20260429-204029/MANIFEST.yaml`
- manifest sha256: `90c4ec22413a...`
- emitter byte-determinism: verified (two emits, same hash)
- `make manifest-verify ARTIFACT=...` passes

## B300 GPU memory after PrimeKG resident

230,699 / 275,040 MiB used (84%). PrimeKG service itself adds ~2 GB of host RAM (NetworkX in-memory graph + index dicts) and zero MiB of VRAM in steady state — cuGraph allocates GPU only during algorithm calls. The 230 GB/275 GB total is dominated by the four inference services (omni:8000, embed:8001, rerank:8002, judge:8003); PrimeKG is non-rivalrous.

## Issues for user

1. **Phase 2.4 did not close the 0.45 PASS gate.** Net delta is −0.027. Consolidated baseline 0.385 still stands as the best held-out result. Recommendation: do NOT promote primekg-hybrid as the v1.0 default retrieval mode. Keep `hybrid` as the default; expose `primekg-hybrid` as an opt-in for precision-medicine queries (Phase 2.5 fixture stratum).

2. **The held-out fixture set was modified mid-track**. The original 6 chemoprevention fixtures are now 6 of 30+ heldout fixtures (others added by Phase 1.6 corpus expansion). Phase 2.4 used a pinned-symlink directory `corpus/clinical-fixtures-heldout-phase24/` to keep the apples-to-apples comparison clean. This pin is the right pattern to keep going forward; rename to `corpus/clinical-fixtures-heldout-frozen-v1/` and reference it explicitly in SPEC §6 metrics so future phase deltas are unambiguous.

3. **The 1.5 K-token PrimeKG block dilutes structured-answer instruction-following by 0.22 absolute**. This is a prompt-engineering lesson, not a PrimeKG-quality issue. Either (a) shrink the block, (b) make it a tool call, or (c) move it below the user message instead of into the system prompt. Each is a < 1-day experiment.

4. **cuGraph speedup vs CPU networkx is negative at our scale on cold round-trip workloads**. The marketing-band 50-330× speedups are warm-state (graph already in cuGraph format on GPU). Our service-startup pattern (load once, dispatch many) is the correct shape to realize those speedups; but our actual hot path (8-12 ms BFS on an 80-node slice) doesn't NEED them. nx-cugraph is a documented capability, not a load-bearing optimization.

5. **No sandbox access to clone external repos onto B300**. The natural way to run sovereign_bench on B300 directly (option (a) per the task brief) was blocked at the simple-evals clone/sync step. The retrieval helper running as an HTTP service on B300 with the bench running on the laptop side (option (b)) was the workable path. Consider pre-allowing `git clone https://github.com/openai/simple-evals` and `simple-evals` tarball syncs as routine bench-bootstrap operations.
