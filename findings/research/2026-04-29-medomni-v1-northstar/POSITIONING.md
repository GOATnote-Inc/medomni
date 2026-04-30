# MedOmni v1.0 — Strategic Positioning Brief

**Status**: positioning doctrine. Drives DEMO, REVENUE-MODEL, PITCH-DECK siblings.
**Date**: 2026-04-29.
**Reframing source**: user (Brandon Dent, MD) directive: "nurse-first DTC iPhone app, ad-supported, mobile-first, with a SOTA-accuracy moat against Anthropic Claude / OpenAI GPT / OpenEvidence."
**Verification posture**: every external claim cited inline; speculative claims marked `[uncertain — verify before pitch]`.

---

## 1. Mission, restated for the nurse-first wedge

MedOmni v1.0 is a **sovereign, nurse-first, multi-modal medical reasoning iPhone app**, free at point of use, supported by RN-relevant brand advertising, with a Pro tier that strips ads and adds CEU/NCLEX/SBAR scaffolding. Same architecture as the SPEC §1 four-persona engine — physician / nurse / family / patient registers over one auditable cited subgraph — but the **default app surface is the nurse register**, the **default form factor is iPhone-at-bedside**, and the **default revenue meter runs on impressions, not seat licenses**.

The clinical depth, sovereignty, reproducibility, and persona-tagged graph from SPEC §1 are unchanged. Only the go-to-market wedge changes: instead of academic ED + VA pilots first, the wedge is the App Store. The hospital tier (SPEC's original GTM) becomes the year-3 long-tail surface, not the year-1 reference customer.

The under-served literature regime that justifies the wedge is unchanged: only **6%** of 67 medical-KG studies addressed nursing-specific applications ([JMIR AI 2025](https://ai.jmir.org/2025/1/e58670/)). The persona-tagged-edge architecture — NIPDS subtle-signal nodes, evidence-currency nodes, pedagogical-intent edges, failure-to-recognize patterns, context-dependent protocol variants per SPEC §5.5 — is what makes the nurse register two architectural levels deeper than a re-prompted Claude or GPT.

---

## 2. Target user

**Primary**: US-based RN at the bedside, smartphone in scrub pocket, taking 30-second peeks between rooms. Median RN: ~3.4M employed, $86K mean annual wage ([BLS OOH 2024](https://www.bls.gov/ooh/healthcare/registered-nurses.htm)). iPhone share in the US runs ~58% ([demandsage 2026](https://www.demandsage.com/iphone-vs-android-users/)) and skews higher among healthcare professionals — `[uncertain — RN-specific iPhone share verify before pitch]`. Reachable iPhone-RN population ≈ ~2M, year-1 SOM target ~250K monthly active.

**Secondary**: LPN (~700K), CNA (~1.4M), NP (~350K), CRNA (~60K), MSN/DNP students. NPs are also the OpenEvidence overlap user — see §4.

**The 30-second iPhone-at-bedside use case** is the design north star. The user types or voice-asks one question between rooms. They get one cited answer in nurse register, with a plain-language family-teach-back below it, in under 6 seconds end-to-end (SPEC §6 latency budget). They do not log in to a web portal, do not wait through procurement, do not fight a corporate VPN. The app is their cognitive lever, not their employer's tool.

---

## 3. The trust ladder — what makes a nurse trust this over Claude/GPT

Five rungs, in order of how much weight each carries with a clinical user. Each rung maps to architecture, not promises.

**Rung 1 — cited responses with graph-traceable provenance.** Every answer ships with the cited subgraph: NCCN / USPSTF / ACOG / ACEP / NICE node IDs and the primary-literature URL beneath each claim. Per SPEC §5.6 (manifest layer 1) and §5.3 stage 9 (constrained decoding + grounding cite rail), no claim survives output without a cite that resolves into the retrieved subgraph. This is the Abridge–NEJM/JAMA pattern — *Linked Evidence map AI-generated documentation to source data, helping clinicians quickly trust and verify the output* ([Abridge–JAMA partnership, April 2026](https://hitconsultant.net/2026/04/15/abridge-nejm-jama-partnership-clinical-decision-support-ehr/)) — operationalized at inference time. Recent literature pegs medical-domain hallucination rates at 28%+ without grounding ([medRxiv 2025.02.28](https://www.medrxiv.org/content/10.1101/2025.02.28.25323115v2.full.pdf)); MedOmni's grounding cite rail is the architectural answer.

**Rung 2 — nurse register, not physician-overlay.** Per SPEC §5.5, the persona graph encodes NIPDS subtle-signal nodes, context-dependent protocol variants (acuity-ratio-aware deference rules), pedagogical-intent edges (`teaches_via_[why|what|when|caution]`), failure-to-recognize patterns, and evidence-currency nodes. Nurses are not a "physician + comms" overlay; they are a distinct expert register. The architecture honors this; Claude / GPT / OpenEvidence's physician-register rewrites do not.

**Rung 3 — protocol awareness (NEWS2, NIPDS, SBAR, IPASS, RACE).** Nurse responses include the operational scaffold the nurse uses to escalate or hand off. SBAR template auto-populates; NEWS2 score auto-computes; teach-back script auto-generates at FKGL ≤ 8. Claude and GPT do not own these protocols; OpenEvidence is physician-register-first and does not surface them by default.

**Rung 4 — sovereignty + offline mode.** Per CLAUDE.md §2 and SPEC §6 (cloud LLM API calls = exactly **0**), no PHI ever leaves the device or pod. The airplane-mode demo (SPEC §7 scene 5) is the visible proof. For nurses on hospital floors with 1-bar WiFi, code-blue corridors, or rural ER bays, "the WiFi went down so the AI went down" is a real failure mode. MedOmni does not have it. ChatGPT, Claude, and OpenEvidence (SaaS-only, per SPEC §3 row 2) do.

**Rung 5 — brand association.** NCCN / USPSTF / ACOG / ACEP / AAP / NICE icons appear directly under the cited claim. The visible association of a primary-source guideline body — not "according to medical literature" — is the trust signal that takes Claude/GPT a year of brand work to replicate, and even then only at the foundation-model level, not the app surface. This is the same trust mechanism that earned OpenEvidence 757K clinicians and $150M ARR ([Sacra equity research, April 2026](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf)) — operationalized at the nurse register.

The trust ladder is empirical, not aspirational: each rung is enforceable at the SPEC §5.3 pipeline level, with manifest-layer evidence (SPEC §5.6) that the rung was not skipped at inference time.

---

## 4. Competitive landscape

| Surface | Persona | Form factor | Citation mechanism | Sovereignty | RN access today |
|---|---|---|---|---|---|
| **MedOmni** | nurse-first, four-persona | iOS-first, ad-supported free + Pro | graph-traceable subgraph, manifest-locked | sovereign on-prem | targeted year-1 default |
| **OpenEvidence** | physician-first, NP/PA/RN expanding | web + iOS ([Aug 2024 launch](https://www.openevidence.com/announcements/openevidence-iphone-and-android-apps-now-available)) | peer-reviewed citations, opacity flagged in clinical reviews | SaaS-only | free with NPI; enterprise EMR rollout via Mount Sinai/Epic ([March 2026](https://www.mountsinai.org/about/newsroom/2026/mount-sinai-health-system-collaborates-with-openevidence-to-provide-evidence-based-knowledge-within-electronic-medical-record)) **already includes RN, NP, PA, pharmacist** |
| **Anthropic Claude** | generic | web + iOS | none (foundation model only) | cloud-only | no NPI gate, but no clinical wrapper either |
| **OpenAI GPT** | generic | web + iOS | none default; ChatGPT for Clinicians launched 2026 | cloud-only | no NPI gate; no nurse register |
| **Epocrates** | drug reference | iOS / Android | drug-monograph-only | SaaS | broad RN/MD use, ~1M HCPs |
| **Medscape** | encyclopedic + news + free CE | iOS / Android | ad-supported, free | SaaS | broad RN/MD use |
| **UpToDate** | physician-first | web + iOS | textbook-style synthesis | SaaS, paid (~$559/yr individual) | physician-skewed; ~2M clinicians 190 countries |
| **Hippocratic AI Nurse Co-Pilot** | nurse-task-automaton | EHR-embedded voice; no consumer app | task-script-grounded, not literature-grounded | enterprise-only | enterprise procurement only ([April 2026 launch](https://hitconsultant.net/2026/04/17/hippocratic-ai-front-door-nurse-co-pilot-voice-automation/)); not consumer-distributed |

**Critical correction surfaced by April 2026 research**: OpenEvidence already grants free NPI-verified access to nurses, NPs, PAs, and pharmacists ([library-cuanschutz FAQ](https://library-cuanschutz.libanswers.com/faq/426983), [Nurse.org coverage](https://nurse.org/news/openevidence-ai-nurses-clinical-support/)). The user's reframing — "OpenEvidence which doesn't allow most RN users now but will soon" — is partially inverted. The window is not "before they open the gate"; the gate is open. The wedge instead is **persona depth + form factor + sovereignty** — same conclusions as SPEC §3 row 2, sharper now that the RN-tier-closure premise is removed. The pitch language must reflect this.

What MedOmni still beats OpenEvidence on, with the gate open:
1. **Nurse register depth** — OE rewrites physician answers; MedOmni encodes nurse-distinct nodes (NIPDS, NEWS2, SBAR scaffolds, evidence-currency).
2. **Multimodal at the bedside** — pill camera, lung-sound audio, ECG image. OE is text-first; this is Omni's strongest dimension and OE's weakest (SPEC §3 row 2).
3. **Sovereignty + offline** — OE is SaaS-only; MedOmni's airplane-mode test runs.
4. **Reproducibility manifest** — bit-identical re-runs (SPEC §5.6); OE has not published this.
5. **Nurse-relevant ad inventory** — Dansko / Hoka / FIGS / Littmann / Aya are not the same advertiser set that OE has built around pharma + medical-device. Different ad market, different CPM curve (see REVENUE-MODEL.md).

---

## 5. The moat — exactly five things that keep Anthropic/GPT/OpenEvidence out

1. **Nurse-persona graph + register.** The persona-tagged edges (SPEC §5.5) are not a prompt; they are graph data. Re-prompting Claude or GPT with "answer as a nurse" produces tone, not the NIPDS graph hop, the evidence-currency check, or the deference-rule edge. This is six-month-build-out architecture, not weekend-replicable.
2. **Sovereignty + offline mode (HIPAA-by-construction).** Per CLAUDE.md §2, zero cloud LLM keys, weights resident on device or on-prem pod. Claude / GPT / OpenEvidence cannot replicate this without rebuilding their stack from foundation up. For the BAA-required hospital tier (year-3 surface), MedOmni is BAA-ready by construction.
3. **Cited responses with graph-traceable provenance.** Same provenance mechanism as Abridge–NEJM/JAMA but operationalized at the inference layer (SPEC §5.3 stage 9), not as a UI overlay. Constrained decoding enforces that no cited node ID exists outside the retrieved subgraph; manifest layer 7 (SPEC §5.6) makes the evidence audit-trail bit-identical reproducible.
4. **Mobile-first iOS UX with voice + camera + audio capture.** Omni's C-RADIO v4-H image encoder (pill ID, ECG strip), Parakeet-TDT audio (lung sound, dictated complaint), word-level timestamps, all native to one stack (SPEC §5.4). Claude has no audio-in; GPT-5 has voice-in but no clinical wrapper or citations; OpenEvidence is text-first. This is the single highest-ROI under-used capability per SPEC §5.4.
5. **Brand-trust loop with credentialed advisors.** NCCN / USPSTF / ACOG / ACEP / AAP / NICE icons under cited claims; RN advisory board (placeholder per pitch deck slide 13) signs off on register and protocol scaffolds. Claude and GPT cannot produce this surface without a clinical specialty wrapper they have not built; OpenEvidence has it for physician-facing literature but not for nurse-facing protocols.

The moat is **architectural + clinical-credentialing**, not weights-and-data. A larger Claude or a stronger GPT does not displace it; a year of nurse-register graph engineering, partner credentialing, and on-prem deployment does. That gap is the year-1 window.

---

## 6. Why now

Three time-bound forces compress the window to roughly 12–18 months.

**Force 1 — OpenEvidence's enterprise EMR push.** The Mount Sinai / Epic deployment (March 2026, [Mount Sinai newsroom](https://www.mountsinai.org/about/newsroom/2026/mount-sinai-health-system-collaborates-with-openevidence-to-provide-evidence-based-knowledge-within-electronic-medical-record)) puts OpenEvidence in front of RNs at-keyboard, inside the EHR. **It does not, however, win the iPhone-at-bedside surface.** EHR-embedded sessions are at the workstation; the at-bedside surface is the smartphone in scrub pocket. That surface is open. The race is for the App Store shelf, not the EHR widget.

**Force 2 — ChatGPT for Clinicians launched 2026.** [Iatrox 2026 landscape review](https://www.iatrox.com/blog/clinical-ai-landscape-2026-chatgpt-openevidence-iatrox-medwise) documents OpenAI's free clinician tier. Without nurse register, without offline mode, without cited subgraph — but with brand gravity. Every quarter MedOmni does not occupy nurse mindshare, ChatGPT habituates the user. Habit is harder to displace than absence.

**Force 3 — Hippocratic AI Nurse Co-Pilot (April 2026).** Voice-AI nurse co-pilot launched with Cleveland Clinic / Cincinnati Children's / OhioHealth ([HIT Consultant April 2026](https://hitconsultant.net/2026/04/17/hippocratic-ai-front-door-nurse-co-pilot-voice-automation/)). Enterprise-procured, EHR-embedded, no consumer app — therefore not a direct App Store competitor, but a signal that the nurse-AI thesis is now venture-validated. The capital follows. Either MedOmni occupies the consumer wedge before someone else's enterprise pivot DTC, or it doesn't.

The window is **own-the-shelf-before-they-do**, on a 12–18-month clock. Past 18 months, mindshare is locked.

---

## 7. Munger inversion — the nurse-app-specific rejection sentences

Five rejections a skeptical nurse, NVIDIA committee member, or App Store reviewer might surface, each preempted architecturally.

| # | Rejection | Architectural preemption |
|---|---|---|
| 1 | "It hallucinates. I asked it about a drug I know and it made up the dose." | Constrained-decoding cite rail (SPEC §5.3 stage 9) — every dose claim must resolve to a retrieved DailyMed / RxNorm / FDA-label node ID. Output rail (stage 10, Nemotron-Content-Safety-Reasoning-4B with clinical policy) blocks unsourced dose strings. Demo scene includes the adversarial probe (SPEC §7 scene 3). |
| 2 | "Where's the citation? OpenEvidence shows me the source." | Cited subgraph below every answer, NCCN/USPSTF/ACOG/ACEP icons, primary-literature URL. Manifest layer 7 (SPEC §5.6) makes the evidence stable across reruns. |
| 3 | "I'll just ask Claude. It's free and I already use it." | Sovereignty + nurse register + protocol awareness + offline mode. Free tier matches Claude on price; nurse-register depth + cited subgraph + offline-mode resilience exceed it. The `[uncertain — A/B retention vs Claude on a clinical-fixture set verify before pitch]` claim is the year-1 traction proof. |
| 4 | "Another subscription. I already pay for UpToDate / Epocrates." | **Free tier is the default**. Pro tier ($9.99–$14.99/mo, ad-free + CEU/NCLEX/SBAR templates) is the upgrade, not the gate. Hospital tier (year-3) is enterprise-procured, not nurse-out-of-pocket. |
| 5 | "If the medical knowledge is anything but best-in-class SOTA, I go back to GPT or Claude." | Direct from user reframing. SOTA accuracy is the moat. Phase 1 + 2.1 lifted held-out chemoprevention from 0.273 → 0.335 (Phase 2.1 results). Phase 2.2 + corpus extension target ≥ 0.55 mean before launch (SPEC §6 metrics). Pre-launch SOTA gate: HealthBench Hard ≥ 0.55, MedAgentBench ≥ 0.70, MedQA-USMLE ≥ 0.85. If we miss the gate, we don't ship — accuracy degradation = revenue collapse (REVENUE-MODEL §7). |

The brand work and content credentialing (RN advisory board, ACEP / ENA partnerships, the NCCN icon visibility) are tractable on a 6-month timeline post-funding; the SOTA-accuracy gate is the binding constraint and the daily engineering focus.

---

## 8. Sources

- [OpenEvidence iOS launch (Aug 2024)](https://www.openevidence.com/announcements/openevidence-iphone-and-android-apps-now-available)
- [Mount Sinai × OpenEvidence Epic embed, March 2026](https://www.mountsinai.org/about/newsroom/2026/mount-sinai-health-system-collaborates-with-openevidence-to-provide-evidence-based-knowledge-within-electronic-medical-record)
- [Sacra equity research, OpenEvidence April 2026](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf)
- [OpenEvidence NPI verification FAQ (CU Anschutz)](https://library-cuanschutz.libanswers.com/faq/426983)
- [Nurse.org — OpenEvidence for nurses](https://nurse.org/news/openevidence-ai-nurses-clinical-support/)
- [Iatrox 2026 clinical-AI landscape review](https://www.iatrox.com/blog/clinical-ai-landscape-2026-chatgpt-openevidence-iatrox-medwise)
- [Hippocratic AI Nurse Co-Pilot launch April 2026](https://hitconsultant.net/2026/04/17/hippocratic-ai-front-door-nurse-co-pilot-voice-automation/)
- [Abridge × NEJM / JAMA April 2026](https://hitconsultant.net/2026/04/15/abridge-nejm-jama-partnership-clinical-decision-support-ehr/)
- [Medical hallucination in foundation models (medRxiv 2025)](https://www.medrxiv.org/content/10.1101/2025.02.28.25323115v2.full.pdf)
- [BLS Registered Nurses 2024 OOH](https://www.bls.gov/ooh/healthcare/registered-nurses.htm)
- [demandsage iPhone vs Android US share 2026](https://www.demandsage.com/iphone-vs-android-users/)
- [JMIR AI 2025, medical KG nursing-application gap](https://ai.jmir.org/2025/1/e58670/)
