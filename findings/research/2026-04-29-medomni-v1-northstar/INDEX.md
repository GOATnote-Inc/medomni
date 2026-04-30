# MedOmni v1.0 — north-star findings index

This directory holds the architectural, strategic, and ops artifacts for the v1.0 milestone (the NVIDIA-fundable nurse-first sovereign medical RAG demo). The artifacts are written by parallel agents and consolidated by the orchestrator.

## Cardinal documents (read these first, in order)

| # | Path | Owner | Purpose |
|---|---|---|---|
| 1 | `SPEC.md` | orchestrator | The architectural north star. Mission, Carnegie/Munger/Hormozi lenses, four-pod heterogeneous compute, open-component BOM, 10-stage retrieval pipeline, multimodal plug-in, persona-tagged graph schema, 9-layer reproducibility cake, metrics, demo script outline, 5-phase shipping plan, three flagged decisions, uptime/HA plan, Blackwell foot-guns. **The durable charter.** |
| 2 | `POSITIONING.md` | T3 (in flight) | Strategic positioning brief. Nurse-first thesis. Trust ladder. Five-element moat. Competitive landscape. |
| 3 | `REVENUE-MODEL.md` | T3 (in flight) | Three-tier model: Free (ad-supported), Pro ($9.99-$14.99/mo), Hospital (long-tail). DTC ad model + brand list + CPM benchmarks + ARPU arithmetic. |
| 4 | `DEMO-RUNBOOK.md` | T3 (in flight) | Minute-by-minute on-stage demo flow (12 minutes, 7 scenes, nurse-first). |
| 5 | `PITCH-DECK-OUTLINE.md` | T3 (in flight) | Slide-by-slide NVIDIA-funding deck structure (12-16 slides, with voice-over scripts). |

## Phase-execution artifacts (one per agent run)

| # | Path | Owner | Phase |
|---|---|---|---|
| P2.1 | `phase-2.1-results.md` | Team Alpha (landed) | RAPIDS 26.04 + cuVS + nx-cugraph + Guardrails shim on B300; held-out 0.273 → 0.335 |
| P1.5 | `phase-1.5-results.md` | T1 (in flight) | NeMo Curator + verbatim PMC primary-trial corpus extension; targets the 5ARI/aspirin/smoking floor |
| P2.2 | `phase-2.2-results.md` | Team Bravo (in flight, post-rotation) | TensorRT-LLM-FP8 judge engine on RunPod H100 prism; throughput delta vs vllm-BF16 |
| P3-fix | `phase-3-fixtures-results.md` | T4 (in flight) | Multimodal demo fixtures (ECG image + auscultation audio + pill image + CXR) with smoke-tested Omni outputs |
| P4 | `phase-4-results.md` | T2 (in flight) | 9-layer reproducibility manifest emitter + airplane-mode demo script |

## Comparator artifacts (orchestrator-generated, evergreen)

| Path | Source |
|---|---|
| `delta-v0-vs-phase2.1.md` | `scripts/compare_cards.py` baseline=v0 candidate=Phase-2.1 |
| `delta-phase2.1-vs-phase1.5.md` | (will be generated when P1.5 lands) |
| `delta-phase2.1-vs-prism-judge.md` | (will be generated when P2.2 lands) |
| `delta-CONSOLIDATED.md` | (final consolidated comparison: v0 baseline → v1.0 PASS-gate run) |

## Methodology + safety doctrine

| Path | Purpose |
|---|---|
| `../2026-04-29-medomni-v0/methodology-status.md` | Red-team finding from earlier in 2026-04-29: rubric-corpus circularity, same-family judge bias, N=1 reporting hygiene; documents the v1 deviation table (open-component stack, NOT NIM) |
| `../2026-04-29-omni-canonical/CANONICAL.md` | Omni capabilities reference (multimodal matrix, NVFP4 specifics, RULER long-context, MoE architecture) |

## Cross-references

- Architecture sprint plan + decision log: `SPEC.md` §8 (phasing) + §9 (decisions D1/D2/D3)
- Memory note for cross-session retention: `~/.claude/.../memory/feedback_runpod_proxy_pty_echo.md` (the 2026-04-29 PTY-echo lesson; durable rule)
- The 9-layer reproducibility manifest emitter spec: `SPEC.md` §5.6
- Hardware foot-guns: `SPEC.md` §12

## Live status (orchestrator-maintained, may be stale)

The orchestrator updates this section after each agent landing. Treat as point-in-time, not authoritative.

```
2026-04-29 evening — agent state:
  T2     (Phase 4 manifest)       LANDED  — emitter byte-deterministic, 9 layers populate, Makefile gates added
  T3     (POSITIONING + 3 docs)   LANDED  — POSITIONING + REVENUE-MODEL + DEMO-RUNBOOK + PITCH-DECK-OUTLINE shipped (~9.3K words), critical correction surfaced
  T4     (multimodal fixtures)    LANDED  — ECG STEMI + warfarin pill + CXR pneumonia all PASS Omni smoke; lung-crackles audio fixture-ready but vllm-omni-b300 missing audio extras
  T1     (Phase 1.5 PMC corpus)   LANDED  — 0.273→0.335→0.385 monotonic lift; 3-of-3 targeted fixtures hit (+0.27/+0.23/+0.14); 29 chunks added (78→107); NeMo Curator 1.1.0 installed clean on B300
  Bravo  (Phase 2.2 prism)        LANDED PARTIAL — vllm-BF16 Qwen judge on prism healthy + benchmarked (170/1286 tok/s); TRT-LLM-FP8 deferred to 2.3; tunnel blocked on user pubkey install
  Phase 2.2 consolidated (orch.)  LANDED  — seed=42 + N=3 trials = ±0.000 deterministic; mean 0.385; comparator PASS verdict on v0→v1.0 (+0.112); manifest sha256 560baccbb706 byte-stable
  Phase 2.4 (PrimeKG via nx-cgr)  LANDED — NEGATIVE RESULT: mean 0.385 → 0.358 (-0.027). PrimeKG resident on B300 (129K/4M, 22s load, 0 VRAM steady), cuGraph 130× warm-state confirmed. BUT 5 of 6 fixtures regressed; instruction_following -0.222 collapsed (graph block crowded out structured-answer scaffold). Durable finding: PrimeKG general, chemoprevention rubrics specific — wrong fixture class. primekg-hybrid stays OPT-IN. Agent redirect: gap closes via cross-family judge ensemble (Phase 2.3, blocked on prism tunnel), NOT more retrieval.
  R1     (repo establishment)     LANDED — Apache 2.0 + README.md.medomni-public + CONTRIBUTING + SECURITY + 22KB REPO-ESTABLISHMENT.md research synthesis with 14 frontier-lab decisions; final summary tripped content-filter, gaps filled by orchestrator (.github/ tree, CODE_OF_CONDUCT, CHANGELOG, .gitattributes). Staged not pushed; user gates gh repo create
  Agent F (Phase 1.7 N=6→30)      LANDED — 24 new fixtures across 9 subdomains, all weights=1.0
  Phase 1.7 N=30 bench (orch.)    LANDED — mean 0.369 (vs N=6 0.385, only -0.016; F predicted -3-7pp). Per-axis: accuracy +0.048 + context_awareness +0.166 LIFTED (architecture generalizes); instruction_following -0.383 COLLAPSED on fixtures probing 2024-2025 guideline updates with must_NOT_state on outdated defaults. New binding failure mode: model has stale guideline knowledge. Phase 2.6 (Task #29): guideline-currency output rail. Manifest sha256 7e6a31925480 byte-stable.

T4 demo readiness:
  Scene 2 (pill camera)            DEMO-READY  — fixture-pill-001-warfarin-tablets.png; Omni names + bleeding-risk + interactions in 1.4s
  Scene 3 (auscultation)           BLOCKED     — fixture valid; serve-side vllm[audio] missing (soundfile, av); fix is one-line + bounce, queued
  Scene 5 ECG/CXR stretch beats    DEMO-READY  — STEMI ST-elevation correctly localized; CXR pneumonia laterality defensible

Consolidated v1.0 result: **mean 0.385, deterministic ±0.000 across N=3 seeded trials**, manifest byte-stable (sha256 `560baccbb706...`). Score progression: v0 0.273 → Phase 2.1 0.335 → Phase 1.5 0.385 → consolidated 0.385 (variance collapsed; mean unchanged — seed=42 confirmed precision but didn't lift accuracy). 5 of 6 targeted fixtures lifted monotonically vs v0 (HPV +0.22, 5ARI +0.14, bisphosphonate +0.12, smoking +0.11, aspirin +0.08); statin flat (Phase 1.5 didn't target a statin trial). instruction_following +0.269 absolute — the canonical-citation anchor effect. PASS verdict from comparator (lift ≥0.05, no major regression); SPEC §6 gate ≥0.45 still NOT met (-0.065). Closing the residual gap requires Phase 1.6 (broader verbatim corpus including statin trials), Phase 1.7 (scale to N=30 fixtures), or Phase 2.3 (ensemble cross-judge once prism tunnel unblocks).

CRITICAL POSITIONING CORRECTION (T3): OpenEvidence ALREADY has free NPI-verified RN/NP/PA/pharmacist tier — verified via library-cuanschutz FAQ + Nurse.org coverage + Mount Sinai/Epic March 2026 enterprise rollout. The original premise "OpenEvidence doesn't allow most RN users now but will soon" was inverted. Moat rebuilt around persona-depth + iPhone-bedside form factor + sovereignty + cited-subgraph mechanism + RN-credentialed-advisor brand-trust loop. The wedge survives but pitch language MUST NOT claim "RN tier closed."

NEW COMPETITOR FLAGGED: Hippocratic AI launched Nurse Co-Pilot in April 2026 — but it's EHR-embedded enterprise-only, NOT consumer iOS. Different surface, MedOmni's iPhone-at-bedside wedge still intact. Track quarterly for consumer-app expansion.

OE PRODUCTIZED AD-MANAGER: ads.openevidence.com is a Google-AdWords-shaped buying surface for pharma + device. Validates the ad-tech-on-clinical-AI thesis as live infrastructure, not theory.

Demo-readiness gates live:
  make health             — all-pod multi-endpoint probe
  make manifest           — emit 9-layer reproducibility manifest from a bench artifact
  make manifest-verify    — proves emitter byte-determinism
  make airplane-test      — WiFi-off bench reproduction (BUILT, awaiting quiet window to run live)
  make demo-pre-flight    — runs health + manifest-verify together
```
