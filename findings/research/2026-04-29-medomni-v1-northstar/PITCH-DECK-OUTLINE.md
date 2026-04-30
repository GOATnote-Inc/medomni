# MedOmni v1.0 — NVIDIA Pitch Deck Outline

**Status**: slide-by-slide structure for the NVIDIA-funding deck.
**Date**: 2026-04-29.
**Length target**: 14 main slides + 4 appendix.
**Pair with**: POSITIONING.md (positioning), REVENUE-MODEL.md (numbers), DEMO-RUNBOOK.md (the live demo embedded as scene-set).
**Verification posture**: every quantitative claim cited or marked `[uncertain — verify before pitch]`. The accuracy moat is binding; the SPEC §6 metric gates are non-negotiable.

Each slide block: **(a) headline · (b) on-screen visual · (c) 30–90s voice-over · (d) NVIDIA logo overlay** if applicable.

---

## Slide 1 — Title

**(a) Headline**: MedOmni for Nurses. Sovereign. Cited. SOTA. iOS.

**(b) Visual**: full-bleed background — a hand holding an iPhone in scrubs, the MedOmni app surface in mid-render. Subtle telemetry strip ghosted at the bottom: *cuVS 1.8 ms · nx-cugraph 2.4 ms · TRT-LLM-FP8 · sovereign · airplane-mode-capable*.

**(c) Voice-over (15s)**: *"MedOmni v1.0 — sovereign nurse-first medical reasoning, on the iPhone, free at point of use, supported by RN-relevant brand advertising. Built on the NVIDIA open stack. SOTA accuracy is the moat."*

**(d) NVIDIA logo**: small "Built on NVIDIA" mark, bottom right.

---

## Slide 2 — Carnegie hook (the bedside iPhone moment)

**(a) Headline**: The 30-second moment between rooms.

**(b) Visual**: a real RN (advisor) `[uncertain — RN advisor confirmed before pitch]` quoted: *"I have a tamoxifen patient with a Mirena IUD. What do I need to watch for?"* — with a cropped iPhone screenshot of the four-panel MedOmni response: nurse panel highlighted, family-teach-back panel ghosted underneath. Cited subgraph icons (NCCN, USPSTF, ACOG) visible.

**(c) Voice-over (60s)**: *"This is the moment. A nurse, in scrubs, has 30 seconds between two rooms. She has a question. Today she asks one of three things — Epocrates for a drug, Google for everything else, Claude or ChatGPT for hard problems. None of these answer her in nurse register. None cite their sources to a primary literature URL. None work when the WiFi is gone. Today, MedOmni does. We answer her question in nurse register, with a graph-traceable citation she could read out at deposition, in under six seconds, on her iPhone, with no PHI ever leaving her device. That moment is the wedge."*

**(d) NVIDIA logo**: none on this slide. Make the audience feel the use case before naming the stack.

---

## Slide 3 — The gap

**(a) Headline**: Physician AI is solved. Nurse AI is not.

**(b) Visual**: a 2x2 table.

|  | Physician | Nurse |
|---|---|---|
| **Workforce size (US)** | ~1M | **~3.4M** (BLS) |
| **AI surface today** | OpenEvidence, Doximity, Heidi, Abridge | scattered, none nurse-register-native |

Plus an under-the-table line: *Only 6% of 67 medical-KG studies addressed nursing-specific applications — JMIR AI 2025.*

**(c) Voice-over (60s)**: *"Three-and-a-half million US registered nurses. The largest single profession in American healthcare. Three times the size of the physician workforce. And the dedicated AI surface — built for their register, their protocols, their teach-back — does not exist. OpenEvidence rewrites a physician answer for an RN. Hippocratic AI is enterprise-procured voice automation. Claude and GPT are generic. The literature itself flags the gap — six percent of medical knowledge graph studies address nursing. We're building for the other 94 percent."*

**(d) NVIDIA logo**: none.

---

## Slide 4 — The moat (5 architectural bullets)

**(a) Headline**: Five things Anthropic, OpenAI, and OpenEvidence cannot replicate in a quarter.

**(b) Visual**: a five-bullet stack, each with a small icon:

1. **Nurse-persona graph + register** (graph icon) — NIPDS, NEWS2, evidence-currency, pedagogical-intent edges.
2. **Sovereignty + offline mode** (lock icon) — zero cloud LLM keys; airplane-mode test passes.
3. **Cited responses with graph-traceable provenance** (citation icon) — constrained-decoding cite rail; manifest-locked.
4. **Mobile-first iOS UX with voice + camera + audio** (iPhone icon) — pill camera, lung-sound audio, ECG image; one stack.
5. **Brand-trust loop with credentialed advisors** (NCCN/USPSTF icons) — primary-source association at the cite layer + RN advisory board.

**(c) Voice-over (75s)**: *"The moat is architectural plus credentialing, not weights-and-data. A larger Claude does not displace it. A stronger GPT does not displace it. The persona-tagged graph is six months of nurse-register engineering. The offline-mode resilience is a sovereign on-prem stack — Claude and OpenEvidence are SaaS-only. The cited subgraph is a constrained-decoding rail, not a UI overlay. The mobile-multimodal is one Omni stack with C-RADIO-v4-H, Parakeet-TDT, and word-level timestamps. The brand-trust association is RN advisory board signoff plus primary-source icons under every cite. Each rung is enforceable at the pipeline level, not promised in a deck."*

**(d) NVIDIA logo**: small Nemotron-Omni mark on bullet 4.

---

## Slide 5 — The architecture

**(a) Headline**: Every layer is NVIDIA open. One stack, four pods, two GPU generations.

**(b) Visual**: SPEC §5.2 BOM diagram. 12 logo'd boxes, color-coded by hardware (Blackwell B300 = green; Hopper H100/H200 = blue). Telemetry overlay shows live numbers: cuVS 1.8 ms, nx-cugraph 2.4 ms, TRT-LLM-FP8 92 tok/s, first-cite p50 87 ms.

Stack from top: *NemoGuard JailbreakDetect · Nemotron-Content-Safety-Reasoning-4B · NeMo Guardrails 0.21 · Nemotron-3-Nano-Omni-NVFP4 · Llama-Nemotron-Embed-1B-v2 · Llama-3.2-NV-RerankQA-1B-v2 · cuVS · nx-cugraph · NeMo Curator 1.1.0 · TensorRT-LLM · Triton · CUDA 13.2 · RAPIDS 26.04*.

**(c) Voice-over (75s)**: *"Every layer is an NVIDIA open component. CUDA 13.2 — the April 12 release, NVFP4-native on Blackwell. RAPIDS 26.04 — cuVS for vector recall, nx-cugraph for graph traversal, all GPU-resident. NeMo Curator for the corpus. NeMo Guardrails 0.21 for the safety rails. NemoGuard JailbreakDetect and Nemotron-Content-Safety-Reasoning-4B replace the gated Meta Llama-Guard. Nemotron-3-Nano-Omni-NVFP4 is the brain — 31-billion params, three-billion active, 256K context, multimodal. TensorRT-LLM-FP8 for the cross-family Qwen judge on Hopper. Triton for the OpenAI-compatible serving frontend. The only non-NVIDIA weight in the entire pipeline is the 7-billion Qwen judge, kept for cross-family methodological honesty per NVIDIA's own Nemotron-3 evaluation recipe."*

**(d) NVIDIA logos**: every box. The slide IS the logo lineup.

---

## Slide 6 — Sovereignty proof

**(a) Headline**: Airplane mode at the bedside.

**(b) Visual**: a screen-capture sequence, four frames left-to-right: WiFi on → user toggles airplane mode → fresh clinical query submitted → answer renders, telemetry shows *Network: OFFLINE · Pod: localhost · Sovereign mode active*.

**(c) Voice-over (45s)**: *"Hospital floors don't have great WiFi. Code-blue corridors don't either. The thing the nurse needs most when WiFi is bad is the answer engine she trusts. Airplane mode on the iPhone. Fresh query. Answer in six seconds. No PHI leaves the device. Cloud LLM API calls in any path: exactly zero. This is sovereignty as resilience, not abstract security. OpenEvidence cannot do this — they're SaaS-only."*

**(d) NVIDIA logo**: subtle — the on-screen telemetry shows TRT-LLM-FP8 running on local pod; no other NVIDIA mark.

---

## Slide 7 — Accuracy benchmarks

**(a) Headline**: SOTA is the moat. Here's where we are and where we land.

**(b) Visual**: a four-row table.

| Metric | v0 baseline | Phase 2.1 actual | Phase 2.2 target | v1 launch gate | Open SOTA |
|---|---|---|---|---|---|
| Held-out chemoprevention 6-fixture mean | 0.273 | 0.335 | ≥ 0.45 | ≥ 0.55 | n/a (held-out) |
| HealthBench Hard | not yet | not yet | (Phase 4) | ≥ 0.55 | ~0.45 |
| MedAgentBench | not yet | not yet | (Phase 4) | ≥ 0.70 | 0.6967 (Claude 3.5 Sonnet) |
| MedQA-USMLE | not yet | not yet | (Phase 4) | ≥ 0.85 | varies |

Cross-family judge methodology callout: *Qwen2.5-7B-Instruct (FP8 TRT-LLM engine on H100) judges Nemotron output. Sovereign-mandated; aligned with NVIDIA's own Nemotron-3 reproducibility recipe.*

**(c) Voice-over (75s)**: *"Honest baseline. Honest trajectory. We started at 0.273 on held-out chemoprevention. Phase 1 corpus extension and Phase 2.1 retrieval lifts brought us to 0.335. Phase 2.2 — TensorRT-LLM-FP8 on the judge plus full nx-cugraph plus the rest of the corpus extension — targets 0.45. The launch gate is 0.55, above the open SOTA on HealthBench Hard. We don't ship below the gate. The cross-family judge is Qwen2.5-7B, deliberately not Nemotron, because Nemotron-judges-Nemotron has a documented 5–25 percentage-point self-preference inflation. We chose methodological honesty before optics."*

**(d) NVIDIA logo**: TensorRT-LLM mark on the cross-family judge callout.

---

## Slide 8 — Munger preempt

**(a) Headline**: Five rejections, each preempted architecturally.

**(b) Visual**: two columns. Left: *"It hallucinates / Where's the citation? / I'll just ask Claude / Another subscription / If accuracy isn't SOTA I leave"*. Right: each rejection mapped to the architectural feature that preempts it (constrained-decoding cite rail / cited subgraph / sovereignty + offline mode + nurse register / free tier is default / SOTA gate is non-negotiable launch criterion).

**(c) Voice-over (60s)**: *"Munger's discipline — list the failure modes first. Five rejections. The hallucination concern is preempted by the constrained-decoding cite rail; no claim survives output without resolving to a retrieved node. The 'where's the citation' concern is preempted by the cited subgraph below every answer. The 'I'll just ask Claude' concern is preempted by the sovereignty plus nurse register plus offline mode that Claude does not have. The subscription-fatigue concern is preempted by free-tier-as-default. The accuracy concern is the binding one — the launch gate is non-negotiable. If we miss it, we slip the launch. This is the discipline."*

**(d) NVIDIA logo**: none.

---

## Slide 9 — Hormozi value stack + TAM

**(a) Headline**: $11B SAM ladder.

**(b) Visual**: a stacked-bar TAM diagram.

```
Year-1 SOM:
  250K MAU × $30 ARPU + 5% Pro ≈ $9M ARR (consumer)

Year-2 plausible:
  1M MAU × $50 ARPU + 8% Pro ≈ $50M ARR (consumer)

Year-3 plausible:
  + hospital long-tail ~$27M ≈ $70-100M ARR

Long-term SAM:
  ~3.4M US RNs × ~$30-60 ARPU ≈ $100-200M consumer ceiling
  + ~6,090 US hospitals × tier-mix ≈ $50-200M hospital ceiling
  + Pro conversion ≈ +20%
  = ~$200M-$500M+ blended SAM
```

Anchor labels visible: *OpenEvidence at $150M ARR run-rate, ~1M users, $124 ARPU (April 2026 Sacra)*.

**(c) Voice-over (75s)**: *"Nurse-audience CPMs are not pharma CPMs — Hoka, FIGS, Aya, NSO Insurance, Walden Nursing pay less per impression than the pharma launch inventory OpenEvidence runs. We model a $30 to $60 ARPU band; OpenEvidence runs $124 today. Year one we target a quarter-million MAU and roughly nine million ARR consumer-only. Year two: one million MAU, fifty million ARR consumer-only. Year three: layer the hospital tier — six thousand US hospitals at twenty to two-fifty thousand each — and the ladder runs to a hundred million plus. SAM ceiling at full RN coverage plus hospital tier, two to five hundred million ARR. Real-source-anchored back-of-napkin per Munger discipline; not audited."*

**(d) NVIDIA logo**: none.

---

## Slide 10 — Distribution

**(a) Headline**: iOS App Store, < 90 days post-funding.

**(b) Visual**: a vertical timeline. Day 0 (funding) → Day 30 (TestFlight beta with 100 RN advisors) → Day 60 (App Store submission) → Day 90 (public iOS launch). Below the timeline: *OpenEvidence's enterprise EHR procurement cycle: 12–24 months. We're winning the iPhone-bedside surface while they win the keyboard-at-workstation surface.*

**(c) Voice-over (60s)**: *"App Store distribution. No procurement. No BAA. No IT-department onboarding. No EHR-integration consulting engagement. Day zero post-funding to public iOS launch in under ninety days. OpenEvidence just announced their first enterprise Mount Sinai Epic embed in March — that's a twelve-to-eighteen-month sales cycle to win each large system. We're racing them on a different surface, on a different clock."*

**(d) NVIDIA logo**: none.

---

## Slide 11 — Ad model

**(a) Headline**: $30–$80 CPM. Hoka, FIGS, Aya, NSO. No pharma.

**(b) Visual**: brand-logo mosaic — Hoka, Dansko, Brooks, FIGS, Cherokee, Jaanuu, 3M Littmann, MDF, NSO Insurance, Aya Healthcare, Trusted Health, UWorld, Walden Nursing, WGU Nursing, Laurel Road, SoFi, Carhartt. Footer: *Ad and answer always temporally separated. Pharma and medical-device excluded by policy. Self-declared profile targeting; no PHI in ad delivery.*

**(c) Voice-over (60s)**: *"OpenEvidence runs $70 to $150-plus CPM on pharma. We exclude pharma in version one — that's the trust differentiator. Our brand inventory is the brands nurses already buy: Hoka, Dansko, Brooks for the twelve-hour-shift footwear. FIGS, Cherokee, Jaanuu for scrubs. Littmann for stethoscopes. NSO for malpractice insurance. Aya, Trusted, Vivian for travel nursing. UWorld and Walden for NCLEX and BSN-MSN. We model a thirty-to-eighty CPM blended floor — to be revised against ninety days of live pilot inventory. Different audience than OpenEvidence, different brand list, similar shape."*

**(d) NVIDIA logo**: none.

---

## Slide 12 — Moat-breaker scenario analysis

**(a) Headline**: What if Anthropic ships nurse-Claude? Or GPT-5.5 with citations? Or OpenEvidence ramps RN?

**(b) Visual**: three columns, each a hypothetical move-and-response.

| Move | Their advantage | Our defense |
|---|---|---|
| Anthropic ships nurse-Claude wrapper | brand gravity + foundation-model lead | nurse-persona graph (not a prompt); offline mode; cited subgraph; RN advisory credentialing — none replicable in a quarter |
| GPT-5.5 ships with citations | citation parity | persona graph; sovereignty; ad-supported free tier vs paid Plus; protocol-aware (NEWS2/SBAR/IPASS) |
| OpenEvidence ramps RN tier (already free with NPI) | physician-network → RN spillover | nurse-register depth; mobile-first form factor; multimodal at bedside; sovereignty / offline; BAA-by-construction |

**(c) Voice-over (60s)**: *"OpenEvidence's RN tier is already open — we adjusted the positioning to reflect that. Their wedge is the EHR keyboard surface; ours is the iPhone-bedside surface. If Anthropic ships nurse-Claude, our defense is the persona graph, the offline mode, and the credentialing — none is foundation-model-replicable. If GPT-5.5 ships citations, our defense is the persona depth and the sovereignty. The architectural moat is twelve to eighteen months ahead and getting deeper each Phase release."*

**(d) NVIDIA logo**: none.

---

## Slide 13 — Team

**(a) Headline**: Founder + AI agent team + advisor bench. Hiring Q1 post-funding.

**(b) Visual**:

- **Founder**: Brandon Dent, MD — physician, GOATnote founder.
- **Engineering velocity**: Claude Code (Opus 4.7) + agent teams (kernel correctness research, clinical reasoning auditing) — same harness used for the prism42 911-PSAP voice work.
- **On-call physician advisor**: GOATnote network `[uncertain — name confirmed before pitch]`.
- **RN advisory board**: 5–7 RNs across ED / ICU / med-surg / NP / nursing-informatics. **Placeholder — first hire post-funding.**
- **Year-1 hires**: iOS lead + ML/inference lead + RN clinical lead.

**(c) Voice-over (45s)**: *"One founder. Agent-team-amplified engineering velocity. The same harness pattern that built the prism42 911 PSAP voice stack and the HealthCraft RL training environment. Physician advisor — GOATnote network. RN advisory board is the first hire post-funding. Year one: iOS lead, inference lead, RN clinical lead. We're pitching architecture as the deliverable; team is the next quarter, not the demo's burden."*

**(d) NVIDIA logo**: none.

---

## Slide 14 — Ask

**(a) Headline**: $X seed + DGX Cloud credits + GTC keynote slot + RN advisory access.

**(b) Visual**: a four-line ask.

- **$X seed** `[uncertain — final amount calibrated to 18-month runway before pitch]`.
- **NVIDIA DGX Cloud credits**: B300 + H100 capacity for v1.5 training (router LoRA, persona-LoRA, nurse-register fine-tune).
- **GTC 2027 keynote slot or sovereign-medical-AI session**: distribution + credibility.
- **NVIDIA healthcare-partner network**: warm intros to Mount Sinai / Mayo / Cleveland Clinic / OhioHealth nursing leadership for RN advisory recruiting.

**(c) Voice-over (45s)**: *"The ask. Seed capital sized to eighteen months of runway. DGX Cloud credits for v1.5 router-LoRA and nurse-persona fine-tune work. A GTC stage to demo this — sovereign medical AI is a session NVIDIA wants on its program. And warm intros through the healthcare-partner network for RN advisory recruiting."*

**(d) NVIDIA logo**: small DGX Cloud + GTC marks.

---

## Slide 15 — Close

**(a) Headline**: The bedside iPhone moment. Again.

**(b) Visual**: same image as Slide 1. The iPhone in scrubs, the four-panel response, the cited subgraph. Closing line beneath: *"This shipped on consumer hardware. The world's largest healthcare workforce just got their answer engine."*

**(c) Voice-over (30s)**: *"Three-and-a-half million US nurses. The largest healthcare workforce. Smallest AI surface. We're winning the nurse mobile shelf — not in three years, in twelve months. The architecture is sovereign. The accuracy is best-in-class. The trust is graph-traceable. The window is open and it does not stay open. Thank you."*

**(d) NVIDIA logo**: full-bleed BOM lineup at the very end — every NVIDIA mark from Slide 5, lit one more time before the screen goes dark.

---

## Appendix slides (4)

### A1 — Reproducibility manifest deep-dive

The 9-layer manifest (SPEC §5.6) shown line by line with example SHA hashes. Container digests, weight SHAs, corpus SHAs, configs, seeds, foot-gun flags, fixtures, judge engine digest, harness git SHA. *Demo-show: bit-identical re-run.*

### A2 — The 9-layer cake (SPEC §5.3 retrieval pipeline)

10 stages from input rail through cuVS recall, BM25 sparse, RRF fusion, cross-encoder rerank, nx-cugraph 2-hop, subgraph serialize, Omni inference, constrained-decoding cite rail, output rail. p50 stage budget table.

### A3 — Held-out methodology

Three independent NVIDIA-researcher critiques (methodology-status.md). Rubric-corpus circularity. Same-family-judge bias. N=1 reported as N=5. The mitigations: held-out fixtures, cross-family judge (Qwen2.5-7B), N≥5 cases × 1 deterministic trial. *This is the rigor that beats vibes.*

### A4 — 4-pod uptime + 1-year roadmap

4-pod heterogeneous compute (B300 + H200 + 2x H100), 3 cloud providers × 4 datacenter regions, $442/day fixed burn, RTO under 60 seconds for most failure modes (SPEC §13.2). 1-year roadmap: v1.0 launch (90d) → v1.5 router-LoRA + persona-LoRA (180d) → v2.0 hospital BAA tier (365d).

---

## Sources

- [SPEC.md](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v1-northstar/SPEC.md)
- [POSITIONING.md](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v1-northstar/POSITIONING.md)
- [REVENUE-MODEL.md](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v1-northstar/REVENUE-MODEL.md)
- [DEMO-RUNBOOK.md](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v1-northstar/DEMO-RUNBOOK.md)
- [Sacra equity research, OpenEvidence April 2026](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf)
- [BLS Registered Nurses 2024 OOH](https://www.bls.gov/ooh/healthcare/registered-nurses.htm)
- [JMIR AI 2025 medical KG nursing-application gap](https://ai.jmir.org/2025/1/e58670/)
- [Mount Sinai × OpenEvidence Epic embed March 2026](https://www.mountsinai.org/about/newsroom/2026/mount-sinai-health-system-collaborates-with-openevidence-to-provide-evidence-based-knowledge-within-electronic-medical-record)
