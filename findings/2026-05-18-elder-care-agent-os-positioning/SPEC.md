# Elder-care agent OS — positioning SPEC

**Status:** SPEC, awaiting founder ratification before any outbound investor send.
**Date:** 2026-05-18
**Author:** Brandon Dent, MD (b@thegoatnote.com)
**Supersedes (at the GTM layer only):** the four prior investor-pitch arcs from the 2026-05-16 strategy thread (mental-health pivot, ED operating system, EM CMG disruption, pure clinical assurance OS). The architectural skeleton from `findings/2026-05-06-goatnote-v2-thesis-em-correction/SPEC.md` is preserved verbatim. This document re-frames the surface and the buyer, not the engine.

---

## 0. TL;DR

The 2026-05-16 thread converged on a four-persona consumer giveaway (medomni × rituals) as the Trojan horse for global ambient healthcare triage. The 2026-05-17 deep-research pass broke that frame: consumer eldercare is a graveyard with named bodies (Best Buy Health $800M+ write-down, Amazon Alexa Together discontinued June 2024, Babylon $4.2B → $0, Embodied/Moxie bankrupt, 23andMe bankrupt March 2025, Pillo, Pear, Care.coach, 98point6 B2C). Adult-child willingness to pay caps below the CAC line. AgeTech raised ~$700M in 2025 — <2% of VC flows; a16z has no published eldercare thesis. Pitching "free iPad app for grandma" walks into a non-thesis at the wrong firm.

What survives, sharper than any prior frame in the thread: **sovereign clinical agent OS — the Android for the elder-care surface — with a Medicare-Advantage-paid RPM/CCM distribution wedge in which the elderly user pays nothing and Brandon-as-supervising-physician is the regulatory load-bearing asset.** Same engine. Same vision. Different buyer, different surface count, different investor sequence.

---

## 1. The reframe (what was wrong, what replaces it)

| Axis | 2026-05-16 frame (broken) | 2026-05-18 frame (this SPEC) |
|---|---|---|
| Product positioning | Free consumer iPad app for elderly | Sovereign clinical agent OS, hardware-agnostic |
| Demo surface | iPad rituals app | iPad rituals app — *as demo, not as company* |
| Primary buyer | Adult children @ $20–$50/mo subscription | Medicare Advantage plans @ $80–$200 PMPM via Stars-rating economics + RPM/CCM CMS billing |
| Secondary buyer | Family premium tier | Health systems (white-label discharge-to-home), pharma adherence data, robotics OEMs (2027+) |
| User pays | $20–$50/mo | $0 |
| CAC owner | Brandon (consumer marketing) | MA plan (member acquisition is already paid for) |
| Lead VC | a16z Bio+Health (Jay) | General Catalyst Health Assurance (Hemant Taneja / Holly Maloney) |
| a16z role | Lead | Syndicate participant (Bio+Health + Apps + American Dynamism) |
| Pitch narrative | Trojan horse for ambient healthcare | Clinical brain for every elder-care surface — iPad today, Echo Show next quarter, ElliQ + 1X Neo + Figure + Apptronik tomorrow |
| Moat | Consumer brand + data flywheel | Sovereign architecture + 9-layer reproducibility manifest + supervising-physician network + 4-persona FDA-non-device line |

The frame failure was treating consumer surface as the company. Consumer surface is the demo; agent infrastructure is the company.

---

## 2. The five load-bearing claims of the new positioning

### 2.1 The buyer is the MA plan, not the user

2026 CMS PFS rates (verified via ThoroughCare / CircleLink / Rimidi 2026 update):

| Code | Service | 2026 rate |
|---|---|---:|
| 99454 | RPM device + data, 16–30 days | $52.11 |
| 99457 | First 20 min RPM mgmt | $51.77 |
| 99458 | Add'l 20 min RPM | $41.00 |
| 99490 | First 20 min CCM (clinical staff) | $66.30 (+9.6% YoY) |
| 99439 | Add'l 20 min CCM | $50.56 (+10.1% YoY) |
| 99491 | 30 min physician CCM | ~$89 |
| 99437 | Add'l 30 min physician CCM | ~$63 |

PMPM stack with one physician contact: ~$170 Medicare-billable. Add ~$80 PMPM MA-plan engagement contract (Stars-bonus economics: each star ≈ $500/member/yr bonus to the plan). Blended ~$250 PMPM = $3,000/year/patient ARR, **payor-paid CAC**.

- 5,000 enrolled patients = $15M ARR (Series A milestone)
- 50,000 enrolled patients = $150M ARR (Series B milestone)

### 2.2 The 4-persona architecture is the FDA-non-device line

FDA's 21st Century Cures Act §3060(a) carves out CDS software from "device" status if it meets 4 criteria; FDA's January 2026 CDS guidance explicitly flagged "time-critical triage or risk-stratification tools" as NOT exempt — these are Class II SaMD requiring 510(k) (6–18 mo, $50–$200K filing cost).

Medomni's 4-persona output is the architectural feature that lets the company stay non-device while still being clinically deep:

- **Patient persona** (FKGL ≤8, shared-decision-making tone): never claims diagnostic certainty, always recommends contact-your-physician. *Non-device.*
- **Family persona** (caregiver register, when-to-call-911 cues): structured escalation prompts, never ranks acuity, always defers to clinician. *Non-device.*
- **Nurse persona** (clinical depth + early-warning escalation cues): visible only to supervising clinical staff under Brandon's MD-of-record license. *Non-device because the patient never sees it.*
- **Physician persona** (full diagnostic depth + literature citations + cited graph path): visible only to the supervising physician. *Non-device for the same reason.*

This is genuinely clever and worth a deck slide. Competitors collapse the persona, cross the SaMD line by accident, and either pay $50–$200K to file 510(k) or get FDA letters.

### 2.3 The 9-layer reproducibility manifest is the trust moat

Per medomni README + `docs/SPEC.md` §5.6, every demo run emits a `MANIFEST.yaml` with byte-identical fingerprint covering: container image digests, weight SHAs, corpus SHAs, config files, random seeds, hardware-foot-gun flags, benchmark fixtures, judge model digest, git SHA.

In a post-23andMe-collapse world, "your data and your model's reasoning never leave a sovereign stack and every inference is byte-deterministically auditable" is the only trust moat that actually compounds. No other AI eldercare company has this. Parachute (YC S25) and Qualified Health ($155M raised) ship dashboards without the science. ElliQ has hardware without the engine. Hippocratic has the engine but is non-diagnostic by design and B2B-only.

This is the same contrarian moat the 2026-05-17 assurance-OS research surfaced — the legally-admissible audit trail that gets subpoenaed. It transfers directly to the consumer-grade surface as the trust mark.

### 2.4 The agent is hardware-agnostic — robotics is a free option

Surfaces, in order of shipping:

1. **iPad (today)** — rituals UI canvas, voice-first, family-looped, multimodal in/out.
2. **Echo Show / smart speaker (Q3 2026)** — same agent, voice-only client. Echo Show wins the daily voice-ritual layer that iPad loses.
3. **ElliQ (2027 license deal)** — Intuition Robotics has hardware + state-aging-office distribution but no clinical engine. License the medomni agent as their clinical brain.
4. **1X Neo (2026 shipping at $20K or $499/mo subscription per 2026-04-30 GlobeNewswire), Figure 02/03 (BMW pilot live), Apptronik Apollo (Mercedes pilot), Tesla Optimus, Sanctuary, Unitree** — every humanoid OEM needs a clinical brain. NVIDIA Isaac GR00T is foundation-model-only; it doesn't ship a clinical persona layer with audit manifest. Medomni's 4-persona agent + reproducibility manifest sits on top of GR00T, not against it.

Same agent, four surface families, one sovereign engine. Brandon manufactures nothing.

### 2.5 The supervising-physician + PC/MSO structure unlocks multi-state Medicare billing

OIG issued a favorable advisory June 2025 on the PC/MSO structure for physician-supervised tech platforms. Brandon owns the PC; the company is the MSO providing tech. ~$50–$150K legal to stand up across 5 launch states (CA, TX, NY, FL, IL). This is the gatekeeper that converts physician credentials into recurring Medicare RPM/CCM revenue across state lines.

---

## 3. Hard tactical changes the deep research forces

### 3.1 Rituals must wire to a connected pill device — patient self-report does NOT qualify for 99454

CMS requires the RPM device transmit physiologic data automatically. A free iPad app where grandma logs pills manually does NOT bill 99454. Options:

- **Hero** (Hero Health, Inc.) — connected pill dispenser, BLE-paired, ~$30/mo consumer pricing today; B2B2C licensing achievable
- **MedMinder** — locked pill dispenser, cellular-connected, focused on senior living
- **AdhereTech** — smart pill bottle with cellular telemetry, more pharma-aligned
- **TimerCap** — lower-cost, BLE only, less mature data pipeline

Recommendation: pilot with Hero (consumer maturity + BLE simplicity), evaluate AdhereTech as the pharma-RWE-aligned secondary integration once data partnerships open.

This is the single most actionable constraint from the research. Without it, the entire RPM revenue line is $0.

### 3.2 Stand up PC/MSO across launch states before any Medicare billing

See §2.5. Required before code 99490 / 99491 can be billed under Brandon's NPI across multiple states.

### 3.3 Stay on the FDA non-device side until Series A closes

Until the company has $5M+ in the bank, the patient and family personas never rank acuity. They always recommend contact-your-physician. The nurse and physician personas can do acuity work because they are not patient-facing. If the company eventually wants to ship a Class II SaMD triage product, that's a deliberate Series B decision, not an accidental one.

### 3.4 Stop pitching "free iPad app." Start pitching "agent OS with payor-paid distribution"

The narrative shift is the single highest-leverage change. The same product, the same engine, the same demo, repositioned for the buyer (MA plan / health system / pharma / robotics OEM) instead of the user (elderly + family).

---

## 4. The 90-second demo video (this weekend)

Single take, Brandon's iPad, rituals UI loaded against medomni production at thegoatnote.com.

| Time | Frame |
|---|---|
| 0:00–0:15 | Open rituals on iPad. Voice greeting from "Lily." Stand-in user. Morning ritual screen: connected pill bottle status (Hero stub for v0), weight, BP, glass of water. |
| 0:15–0:35 | User confirms pills by voice. Glucometer reading shows high. Lily switches to nurse-register: "Margaret, your blood sugar is a little high this morning. Did you take your insulin?" Two-turn dialog. |
| 0:35–0:55 | User says "I'm having a little chest discomfort." Lily switches register again, runs structured red-flag check (3 questions), generates a structured assessment with the cited graph path visible in a corner. Patient persona stays non-acuity-ranking; structured output is for the supervising clinician. |
| 0:55–1:15 | Lily: "Margaret, this might be nothing, but I'd like to call Sarah and Dr. Lee. Is that okay?" iPad calls Sarah. Sarah's screen shows the structured assessment + recommended action. |
| 1:15–1:30 | Bottom-third caption: "Sovereign clinical agent. $3.70/hr H100. 4-persona output (FDA non-device). MA-plan paid. Hardware-agnostic. Ships to robot." |

The caption is what reframes who watches the video. "Free for grandparents" sends it to consumer teams. The reframed caption sends it to Bio+Health partners.

---

## 5. Investor sequence (revised, ordered)

1. **General Catalyst Health Assurance** — Hemant Taneja / Holly Maloney. GC has the published Health Assurance thesis, allocated $1B+ of the $8B fund to healthcare, has Sprinter Health (last-mile in-home) and Commure (provider infra) in portfolio. This is their pattern exactly. First call.
2. **a16z Bio+Health** — Jay Rughani is already in the thread. Frame as agent OS, not consumer app. Suggest syndication with Connie Chan / Olivia Moore (Apps) for the consumer surface and Katherine Boyle (American Dynamism) for the sovereign-robotics narrative. Three-team syndication carries the partner meeting.
3. **AgeTech-dedicated funds** — Equitage Ventures ($47.3M fund), AgeTech Capital ($50M first fund). Smaller checks but thesis-aligned signal value.
4. **8VC, F-Prime, GV** — follow-on healthcare specialists.

Jay no longer leads. He stays in the round because (a) he opened the door and (b) the a16z brand carries the MA-plan enterprise sales call. But he is not the term-setter for this thesis.

---

## 6. What survives from the existing asset base

Every artifact already shipped in the GOATnote-Inc public org becomes load-bearing or feature-level under the new framing. Nothing is wasted.

| Asset | Role under new positioning |
|---|---|
| medomni (this repo) | Sovereign clinical agent OS — the engine |
| rituals HTML (in `~/Downloads/`, unwired) | iPad demo surface — to be productized into `web/rituals/` once the agent OS pitch lands |
| OpenEM (370 conditions, FHIR R4) | Clinical ontology grounding the engine |
| HealthCraft (195 tasks, 2,255 criteria) | Pre-ship safety eval gate — every release passes pass^k before shipping to any patient |
| LostBench (CEIS 3-layer framework) | Adversarial probe set for the regression suite |
| Receipts (Merkle-attested audit ledger) | The trust mark — every patient interaction lands with κ-graded judges |
| RadSlice (330 DICOM tasks) | Imaging surface for V_image-aware Q3 2026 |
| ED Decision Rules MCP (live at mcp.thegoatnote.com) | Escalation surface — when patient persona hands off, the supervising clinician gets the decision-rule MCP output |
| Brandon's MD + EM attending + former professor | Supervising physician of record — the load-bearing regulatory asset |
| medimage-corpus (134 datasets, 7.4 PB cataloged) | Training corpus for V_image-aware |

The "Clinical AI Assurance OS" pitch from 2026-05-17 is not dead. It becomes the trust-mark feature of this product, not a separate company.

---

## 7. Open questions (must answer before Series A pitch)

1. **Pilot site selection.** First 10–25 elderly users — Brandon's clinical network, hometown, EMDR-network parents. Need names + addresses + Hero-bottle provisioning by 2026-06-01.
2. **PC/MSO legal counsel.** Need recommendation for friendly-PC/MSO counsel in CA + TX + FL. ~$50K initial budget.
3. **MA plan first contact.** Target 1 regional MA plan (Bright Health, Devoted, Alignment) for design-partner conversation by 2026-07-01.
4. **Connected pill device partnership term.** Hero vs MedMinder vs AdhereTech — first-call this week.
5. **FDA pre-sub.** Do we file a pre-sub now to confirm non-device classification, or rely on the 21st Century Cures Act CDS exemption on its face? Recommendation: pre-sub by 2026-09-01 once Series A capital is in.
6. **Whether to re-domain Stealth-TIC** (trauma-informed MH pipeline, currently sidequest) as a vertical agent on top of the same OS — defer to Series A+1.

---

## 8. Next 30-day action list

- [ ] Record 90-second demo video per §4 (founder action, this weekend)
- [ ] Wire rituals HTML into `web/rituals/` route as deployable demo surface (separate PR)
- [ ] Email Hemant Taneja or Holly Maloney at General Catalyst Health Assurance with the video + this SPEC as a one-pager
- [ ] Reply to Jay Rughani with the reframed positioning (agent OS, not consumer app) and the demo video — propose 30-min Zoom in 2 weeks
- [ ] First call with Hero / MedMinder / AdhereTech for connected-device partnership terms
- [ ] PC/MSO counsel intake call (3 firms)
- [ ] Recruit 10 pilot elderly users via Brandon's clinical + family network
- [ ] Stand up Stripe + first MA-plan billing pipeline behind the demo (not for charging users — for billing Medicare RPM/CCM once the first physician-supervised encounter clears)

---

## 9. Provenance

This SPEC is the synthesized output of the 2026-05-16 strategy thread + two parallel deep-research passes on 2026-05-17 (highest-ARR VC pitches given Brandon's asset stack; elderly-consumer-as-Trojan-horse thesis under graveyard scrutiny). The graveyard evidence (Best Buy Health, Alexa Together, Babylon, Embodied, 23andMe) is what forced the reframe. The unit-economics ($170 PMPM Medicare + $80 PMPM MA) is what makes the reframe a real business. The 4-persona FDA-non-device line is what makes Brandon's existing architecture the load-bearing differentiator.

Read alongside `findings/2026-05-06-goatnote-v2-thesis-em-correction/SPEC.md` for the architectural skeleton this re-frames at the GTM layer.
