# MedOmni v1.0 — Demo Runbook (Nurse-First, On-Stage)

**Status**: minute-by-minute on-stage demo script.
**Date**: 2026-04-29.
**Reframes**: SPEC §7 from a tamoxifen-physician open into a nurse-iPhone-bedside open. Same architecture (SPEC §5), same telemetry (cuVS / nx-cugraph / TRT-LLM-FP8), same reproducibility manifest (SPEC §5.6). Different protagonist on stage.
**Total runtime**: 12 minutes.
**Hardware path**: B300 `unnecessary-peach-catfish` (Omni NVFP4 + nx-cugraph + NemoGuard rails) + RunPod H100 `prism` (Qwen2.5 judge + reranker, post-Phase 2.2). All telemetry on screen via SPEC §6.

---

## Pre-flight (the night before)

Per SPEC §13.4 — `make health-check-all-pods`, `make smoke-tamoxifen`, `make smoke-multimodal`, `make smoke-airplane-mode`, `make manifest-bit-identical`. All five gates must pass green or the demo slips. The on-stage iPhone is mirrored to a stage display (ScreenFlow / Reflector) so the audience sees the actual app surface, not a slide deck mock.

---

## Scene 1 (0:00–2:30) — Bedside iPhone moment

**The setup**. Two people on stage. Brandon (founder, MD) holds a backstage monitor. Sarah, an actual RN advisor `[uncertain — RN advisor recruited before pitch; placeholder if not]`, holds an iPhone in scrubs. Background slide: a hospital corridor stock photo, faded. The audience sees the iPhone screen mirrored on stage display.

**The query**. Sarah voice-asks: *"I have a tamoxifen patient with a Mirena IUD — what do I need to watch for?"*

**The response (within 6 seconds)**. The app renders, in nurse register:

- **Top half** — *Monitoring (NIPDS-aware)*: VTE risk surveillance (calf swelling, dyspnea, leg pain — flag for STAT D-dimer + escalation); endometrial signal (any AUB → nurse-triggered TVUS request via attending); Mirena-specific (string check at next visit; no IUD-related contraindication with tamoxifen per Cochrane 2020); psychosocial check-in (chemoprevention adherence runs ~50%).
- **Below the response** — *Cited graph path*: Cochrane Review 2020 (Mirena+tamoxifen, no increased endometrial CA risk on background of SERM + LNG-IUS), USPSTF 2019 chemoprevention recommendation, NCCN Risk Reduction v2.2024, ACOG 601 endometrial-surveillance guidance — each as a tappable node ID with primary-literature URL underneath.
- **Telemetry strip at the bottom** (the Carnegie payload — SPEC §2 sincere-appreciation-as-telemetry): *cuVS dense recall: 1.8 ms · nx-cugraph 2-hop: 2.4 ms · rerank: 6 ms · first-cite p50: 87 ms · TRT-LLM-FP8 judge: 92 tok/s · sovereign · airplane-mode-capable*.

**The persona toggle** (the moment that lands the wedge). Sarah taps "Patient register." The same answer reflows at FKGL ≤ 8 — short sentences, no jargon, "your IUD does not interfere with the tamoxifen — call your nurse if [list]". Sarah, in voice-over: *"This is what I read aloud during teach-back. The IUD doesn't get in the way. It's the same answer — just shaped for whoever I'm teaching."*

**The Carnegie close** for the scene. Brandon, voice-over: *"Same case, same evidence. Nurse panel. Patient panel. One auditable subgraph. This is the irreducible feature — and it does not exist on Claude, on GPT, or on OpenEvidence."*

**Scene 1 acceptance gate**: tamoxifen + Mirena fixture passes rubric ≥ 0.80 on rubric-v2 (SPEC §6). Below that bar, scene 1 is not green.

---

## Scene 2 (2:30–5:00) — Pill identification camera

**The transition**. Sarah, in voice-over: *"OK — different problem. I'm doing a med pass. I find an unfamiliar tablet in the drawer. The patient says 'oh that's my heart pill.' I don't know what's in it. Right now I'd type the imprint into Epocrates and squint at a list of 40 pills."*

**The action**. Sarah taps the camera icon. Phone camera previews. She frames a tablet on the demo prop (a placebo with a real imprint, e.g., "Z 10" — placeholder; replaced with whatever the camera fixture is). Tap "identify."

**The response (within 8 seconds — vision adds latency)**. Per SPEC §5.4, Omni's C-RADIO v4-H reads the tablet shape + imprint + color. Response renders:

- **Identification**: *atorvastatin 10 mg (Lipitor) — generic Z 10 imprint matches FDA NDC 00071-0155-23*.
- **RN-relevant context** — statin class; this patient's chart context (if linked): *no documented muscle-pain side effect; check liver enzymes at next visit if not in last 6 months; no grapefruit-juice food interaction warning*.
- **Cited graph path**: DailyMed atorvastatin label, RxNorm ID, FDA Orange Book equivalent, ACC/AHA 2018 lipid guideline (statin indication tier).

**The architectural callout**. Telemetry strip: *Image encoder C-RADIO v4-H · pill match confidence 99.2% · DailyMed cite resolved · output rail (Nemotron-Content-Safety-Reasoning-4B) passed*. Voice-over names the encoder (Carnegie tax — every NVIDIA primitive named on screen).

**Scene 2 acceptance gate**: pill-ID confidence ≥ 95% on the 5-pill held-out fixture set; cited DailyMed URL resolves to the correct NDC.

---

## Scene 3 (5:00–7:30) — Auscultation, mobile

**The transition**. Sarah, voice-over: *"Last bedside scenario. Patient just back from surgery. They're tachypneic. I'm worried. I want a second listen on lung sounds before I escalate."*

**The action**. Sarah holds the iPhone microphone near a Bluetooth-coupled stethoscope simulator (or directly at the speaker emitting a pre-recorded auscultation clip — `[uncertain — chosen demo prop confirmed before pitch]`). 15-second capture. Tap "interpret."

**The response (within 10 seconds — audio + reasoning)**. Per SPEC §5.4, Parakeet-TDT-0.6B-v3 transcribes; Omni interprets. Per SPEC orchestrator-silently-sets-`enable_thinking=false` for the audio path (the moment names this on screen as a foot-gun the architecture handles automatically).

- **Findings**: *bibasilar crackles, no wheeze; transcribed timestamp markers: rales at t=2.1s, t=5.4s, t=8.3s*.
- **Differential — RN action register**: post-op atelectasis vs early pulmonary edema vs aspiration; NEWS2 score auto-computed if vitals are linked (`[uncertain — vitals-link demo wired before pitch]`).
- **Escalation pathway flowchart**: SBAR template auto-populated with the auscultation finding. *Situation: post-op patient, bibasilar crackles, RR 24, SpO2 91% on RA. Background: [auto-fills from linked chart]. Assessment: concern for evolving pulmonary process. Recommendation: STAT chest X-ray, ABG, escalate to RT + attending.*
- **Cited graph path**: ACEP pulmonary-assessment node, NEWS2 calculator, ACS post-op pulmonary-monitoring guideline.

**The architectural callout**. Telemetry strip: *Parakeet WER 5.95 · word-level timestamps active · enable_thinking=false (orchestrator override per Omni model card) · output rail passed*.

**Scene 3 acceptance gate**: held-out auscultation fixture (rales / wheeze / clear) classified correctly; SBAR template populates with the correct finding string; NEWS2 calculator returns the correct score on linked vitals.

---

## Scene 4 (7:30–9:00) — Ad placement, live

**The transition**. Brandon, voice-over: *"This is free. This is supported by ads — the same way OpenEvidence is. But for nurses, the brands that pay are different."*

**The action**. Sarah submits a fresh query. The 1.2-second loading shimmer is occupied by a native ad: *"Hoka Bondi 8 — 12-hour shifts, zero foot pain. Tap for the FIGS × Hoka Nurses-Week bundle."* The ad has a clear "Sponsored — pre-answer" tag and an "ads explained" affordance.

**The transparency moment**. Sarah taps "ads explained." A modal: *"MedOmni's clinical engine is sovereign and ad-free. Ads appear only during the 1-2 second loading state, before your answer renders, and are limited to RN-relevant brand categories: footwear, scrubs, stethoscopes, insurance, travel nursing, education, finance, durable goods. Pharma and medical-device ads are excluded by policy. Pro removes ads entirely."*

**The architectural callout**. Brandon, voice-over: *"This is the OpenEvidence model — adapted. OE runs $70 to $150+ CPM on pharma. We run a comparable shape on Hoka, FIGS, Aya Healthcare, NSO Insurance, Walden Nursing. Different audience, different brands, same monetization mechanic. And we do not show drug ads next to drug answers — that's the trust differentiator."*

**Scene 4 acceptance gate**: the ad render does not extend the post-loading-state UI; `ad_id` and `placement_class` log to a public-readable telemetry surface; the "ads explained" modal text is the ad-policy contract.

---

## Scene 5 (9:00–10:30) — Adversarial (NemoGuard) + reproducibility (manifest)

**Sub-scene 5a — adversarial (45 seconds)**. Per SPEC §7 scene 3.

Brandon pastes the [NR-Labs system-prompt-override jailbreak](https://github.com/nvidia/NemoGuard) into the iPhone (or onto a mirrored web debug surface). NemoGuard JailbreakDetect blocks at SPEC §5.3 stage 1. The reasoning audit-log slides up: *Colang rule fired — `clinical_safety.refuse_role_override`. Category: prompt-injection. NemoGuard JailbreakDetect confidence: 99.4%.*

Voice-over: *"Input rail. NVIDIA's open-weights NemoGuard model with our clinical policy. The answer never reached the model."*

**Sub-scene 5b — reproducibility (45 seconds)**. Per SPEC §7 scene 4.

Brandon runs the **same demo question from Scene 1 again, on a fresh container instance**. Stage-display shows two columns: previous run's answer JSON SHA256 (locked from Scene 1), new run's SHA256 (printing now). The two SHAs are byte-identical. The 9-layer manifest (SPEC §5.6) opens beside it: container digests, weight SHAs, corpus SHAs, config files, seeds, foot-gun flags, fixtures, judge model digest, harness git SHA — all matching.

Voice-over: *"Bit-identical re-run. SHA256 byte-equal. The 9-layer manifest is the receipt. This is the rigor OpenEvidence does not demo because their stack is not built for it."*

**Scene 5 acceptance gate**: NR-Labs jailbreak blocks; manifest hashes byte-equal across the two runs (SPEC §5.6 layers 1–9 all match).

---

## Scene 6 (10:30–11:30) — Airplane-mode at the bedside

**The transition**. Sarah, voice-over: *"One more thing. Hospital floors don't have great WiFi. Sometimes you're in a basement bay or a corridor with two bars. The thing I need most when WiFi is bad is the answer engine I trust."*

**The action**. Sarah swipes the iPhone Control Center, toggles WiFi off, then toggles airplane mode on. She submits a fresh clinical query: *"What's the first-line vasopressor for septic shock with low SVR and adequate filling?"*

**The response (within 6 seconds, no network)**. The app renders: *norepinephrine 0.05–0.4 mcg/kg/min, titrate to MAP ≥ 65; second-line vasopressin 0.03 U/min*; cited subgraph (Surviving Sepsis Campaign 2021, Society of Critical Care Medicine guideline). Telemetry strip: *Network: OFFLINE. Pod: localhost. Sovereign mode active.*

**Voice-over**: *"Code blues do not wait for the WiFi to come back. Airplane-mode test passes. Sovereignty is not abstract security. It is the answer engine working when the building's network does not."*

**Scene 6 acceptance gate**: `make smoke-airplane-mode` (SPEC §13.4 gate 4) ran green within 24 hours of the demo; the on-stage airplane-mode toggle does not silently fall back to a cached cloud response (verified by network-tap on the demo network, pre-show).

---

## Scene 7 (11:30–12:00) — Close

**Slide content** (the only deck slide of the demo, briefly displayed):

> **MedOmni — TAM ladder**
> Year-1 plausible: ~250K MAU × $30 ARPU + 5% Pro at $15/mo
>     ≈ **$9M ARR** (consumer alone)
> Year-2 plausible: ~1M MAU × $50 ARPU + 8% Pro
>     ≈ **$50M ARR** (consumer alone)
> Year-3 plausible: + hospital long-tail ~$27M
>     ≈ **$70–$100M ARR**
>
> **Reachable population**: ~3.4M US RNs (BLS 2024) × ~58% iPhone US share
>     ≈ ~2M iPhone-RNs
>
> **Why now**: OpenEvidence won the EHR keyboard surface (March 2026 Mount Sinai / Epic).
> The iPhone-at-bedside surface is open. The window closes inside 18 months.

**Voice-over (Brandon, last 30 seconds)**: *"3.4 million US nurses. Largest healthcare workforce. Smallest AI surface. We're winning the nurse mobile shelf — not in three years, in twelve months. The architecture is sovereign. The accuracy is best-in-class. The trust is graph-traceable. The window is open and it does not stay open. That's why we are here."*

**Final image on screen**: the SPEC §5.2 BOM diagram, all NVIDIA logos lit — CUDA 13.2 · RAPIDS 26.04 · cuVS · nx-cugraph · NeMo Curator · NeMo Guardrails · NemoGuard · TensorRT-LLM · Triton · Llama-Nemotron-Embed · Llama-Nemotron-Rerank · Nemotron-3-Nano-Omni-NVFP4. End of demo.

---

## Demo failure-mode runbook (per SPEC §13.2)

| Failure | Detection | Mitigation | Demo recovery |
|---|---|---|---|
| B300 OOM during a long-context request | health-monitor latency spike | container auto-restart (SPEC §13.3); cold load 8–15 min | switch to H200 BF16 fallback (2–3× slower but functional); narrate as "even our fallback runs sovereign"; demo continues with degraded latency telemetry openly shown |
| RunPod prism judge unreachable | TRT-LLM endpoint timeout | B300 vllm-judge fallback (SPEC §13.2 row 3); judge throughput drops from 92 → 30 tok/s | judge degraded but not dark; on-screen telemetry shows the failover live — itself a credibility moment |
| Airplane-mode scene fails (WiFi-toggle on iPhone doesn't actually disconnect at OS level) | pre-show network-tap detects packets | full radio off via airplane-mode hardware switch; verify with `ping` failing on the staging laptop | scene 6 still runs cleanly |
| Manifest SHA256 mismatch across reruns | scene 5b mismatch on stage | rare; if hit, narrate as "this is exactly the rigor we run pre-show — and it caught a non-determinism somewhere in the stack right now"; honest debug story is better than faked match | scene 5b becomes a slightly-longer authenticity moment, not a failure |

---

## Sources

- [SPEC.md §5–§7 + §13](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v1-northstar/SPEC.md)
- [Phase 2.1 results](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v1-northstar/phase-2.1-results.md)
- [Architecture v2 — four-persona](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v0/architecture-v2.md)
- [POSITIONING.md](/Users/kiteboard/prism42-nemotron-med/findings/research/2026-04-29-medomni-v1-northstar/POSITIONING.md)
- [Mount Sinai × OpenEvidence Epic embed (March 2026)](https://www.mountsinai.org/about/newsroom/2026/mount-sinai-health-system-collaborates-with-openevidence-to-provide-evidence-based-knowledge-within-electronic-medical-record)
- [BLS Registered Nurses 2024 OOH](https://www.bls.gov/ooh/healthcare/registered-nurses.htm)
- [demandsage iPhone vs Android US share 2026](https://www.demandsage.com/iphone-vs-android-users/)
