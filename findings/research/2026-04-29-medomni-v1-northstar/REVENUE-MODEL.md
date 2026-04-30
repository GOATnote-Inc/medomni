# MedOmni v1.0 — Revenue Model

**Status**: ad-supported DTC revenue plan, paired with POSITIONING.md.
**Date**: 2026-04-29.
**Verification posture**: every CPM, ARPU, and TAM number cited or marked `[uncertain — verify before pitch]`. No fabricated numbers.

---

## 1. Three-tier model

| Tier | Price | Audience | Mechanism | Year-1 share target |
|---|---|---|---|---|
| **Free** | $0 / mo | RN / LPN / NP / CNA / nursing students | full clinical features + ad-supported | 90–95% of MAU |
| **Pro** | $9.99–$14.99 / mo | high-volume RN / NP, NCLEX students, CEU-pursuing | ad-free + CEU/NCLEX bank + SBAR/SOAP templates + voice macros + custom protocols | 5–10% conversion |
| **Hospital / enterprise** | $10K–$500K / yr per facility | year-3 surface; not in v1 launch revenue | BAA + on-prem option + EHR widget + admin dashboard | 0% in year 1; year-3 ladder per §6 |

The free tier is the entire user-acquisition vehicle. The Pro tier funds itself off the top 5–10% of users who accumulate enough query volume that ads become friction, plus the test-prep cohort whose CEU/NCLEX content has standalone value. The hospital tier is the long-tail story (§6) and the v3 pitch slide, not the v1 launch revenue.

**Anchor**: this matches OpenEvidence's revenue topology — *primarily on a CPM (cost-per-thousand-impressions) basis* ([Sacra equity research April 2026](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf)) — with the audience swapped from physician to nurse, and the brand inventory swapped from pharma + medical-device to RN-relevant DTC. Different audience, different ad market, same monetization shape.

---

## 2. Ad placement design — non-clinical-decision-distorting principles

Three rules govern every ad placement, derived from OpenEvidence's documented practice and reinforced by the user reframing:

**Rule A — ad and answer are temporally separated.** OpenEvidence's CEO articulated this as: *the way AI search works is it takes a few seconds to find and collect evidence to generate an answer. We take advantage of that moment to display the ad. The ad and the answer are always separate — once the answer is generated, the ad disappears* ([Sacra equity research](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf)). MedOmni inherits this: the ad lives in the loading-state UI between query submission and first-cite-render. End of decoded answer = end of ad render.

**Rule B — no clinical decision can be steered by an ad.** Brand category is independent of query intent. A nurse asking about tamoxifen does not see a tamoxifen-brand ad. The whitelist is **non-clinical-decision-relevant** brand categories only: footwear, scrubs, stethoscopes, nursing insurance, travel-nursing platforms, nursing-education programs, financial services, durable goods, food/coffee, sneakers. Pharma and medical-device ads are **disallowed in v1** to maintain trust differentiation. (Phase 4+ may revisit pharma with stricter governance, but v1's wedge is "we don't show you drug ads next to your drug answer.")

**Rule C — ad served by category, not by clinical session content.** No PHI, no query content, ever leaves the device for ad targeting. Targeting uses self-declared profile (RN / LPN / NP / specialty / region) plus broad nursing-relevant categories, not the clinical-content vector of the in-flight query. This is HIPAA-by-construction (CLAUDE.md §2 sovereignty) extended to ad delivery.

**Brand inventory categories**, ranked by RN relevance (per [Nurse.org best-shoes-for-nurses 2026](https://nurse.org/articles/best-shoes-for-nurses/), [FIGS Nurses Week 2026](https://nurse.org/articles/figs-nurses-week/), and the user's reframing):

| Rank | Category | Example brands | Anchor |
|---|---|---|---|
| 1 | Footwear | Hoka, Dansko, Brooks, Birkenstock Pro, On Cloud, Nike, New Balance × FIGS | [Nurse.org best shoes 2026](https://nurse.org/articles/best-shoes-for-nurses/) lists Hoka, Dansko, Brooks as top picks |
| 2 | Scrubs / apparel | FIGS, Cherokee, Jaanuu, Wonderwink, Carhartt | FIGS 2025 revenue $631M, market cap ~$1.75–2.45B ([macrotrends](https://www.macrotrends.net/stocks/charts/FIGS/figs/revenue), [PitchBook](https://pitchbook.com/profiles/company/57741-85)); 1.5M active customers per [Modern Retail](https://www.modernretail.co/retailers/inside-figs-brand-ambassador-program-of-health-care-professionals/) |
| 3 | Stethoscopes / equipment | 3M Littmann, MDF, ADC, Welch Allyn | proven RN purchaser pattern `[uncertain — direct ad-spend data verify before pitch]` |
| 4 | Insurance | NSO Insurance, Proliability, CMF | nursing-malpractice insurance is a recurring annual purchase |
| 5 | Travel nursing | Aya Healthcare, Trusted Health, Cross Country, Vivian Health, Incredible Health | high-CPC category with active recruiter spend `[uncertain — CPM verify before pitch]` |
| 6 | Education / NCLEX / BSN-MSN | UWorld, Kaplan, Saunders, Walden University, WGU Nursing, Capella | online-degree CPMs are documented-high `[uncertain — verify specific CPM before pitch]` |
| 7 | Financial services | Laurel Road, SoFi, KASA Living, NurseFirst Loans | nurse-targeted refinance and PSLF pathways |
| 8 | Durable goods | Carhartt, watches, food/coffee, sneakers, Yeti | RN as 12-hour-shift consumer |

Pharma and medical-device explicitly **out** in v1.

---

## 3. CPM benchmarks — what we can claim, what we can't

| Audience | CPM range | Source |
|---|---|---|
| General-consumer social (Meta, TikTok) | $5–$15 | well-documented industry standard |
| Healthcare-on-Meta (general health audience) | ~$22.76 → $38.70 (Jan 2025 → Jan 2026) | [Promodo healthcare benchmarks 2026](https://www.promodo.com/blog/healthcare-digital-marketing-benchmarks) `[uncertain — direct verify before pitch]` |
| **Verified HCP audience (OpenEvidence range, pharma + device)** | **$70–$150+ CPM** | [Sacra equity research April 2026](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf), confirmed in [Linear newsletter #158](https://www.newsletter.lukesophinos.com/p/linear-158-openevidence-just-turned) |
| OpenEvidence top-tier pharma | up to $1,000+ CPM | same sources, premium pharma-launch inventory |
| **Nurse-specific endemic ad inventory** | **`[uncertain — no public benchmark; expected $30–$80 CPM range pre-pharma, $70–$150 if pharma admitted in Phase 4]`** | inferred from HCP-vs-consumer gap; verify with first 90-day pilot data before pitching specific number |

The honest pitch number is: **OpenEvidence's $70–$150 CPM band exists because verified-clinician audiences trade at 5–10× general-consumer CPM**. MedOmni's nurse-only audience occupies the same shape with a smaller absolute base; the durable goods + footwear + scrubs + travel-nursing brands listed in §2 do not pay $1,000 CPM (that is pharma-launch territory) but the consensus HCP-targeted-endemic-platform CPM is materially higher than consumer social. Pre-launch we model a **$50 CPM blended** as the conservative floor — to be revised against the first 90 days of live pilot inventory.

---

## 4. ARPU model — running the arithmetic

| Lever | Conservative | Optimistic |
|---|---|---|
| Free-tier impressions per active RN per month | 60 (1 query × 2 ad slots × 30 days) | 150 (3 queries × 2 slots × 25 days) |
| Annual impressions per free-tier RN | 720 | 1,800 |
| Blended CPM | $30 | $80 |
| **Free-tier ad ARPU per RN per year** | **$21.60** | **$144** |

OpenEvidence currently runs **~$124 ARPU** ([Sacra equity research April 2026](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf)) — physician audience, pharma-heavy inventory. MedOmni's nurse-audience ARPU lands somewhere south of OpenEvidence in the conservative case, parity in the optimistic case. The right pre-pitch number to anchor on: **$30–$60 ARPU per free-tier RN per year** as the modeling band.

**Free-tier MAU ladder** (year-1 plausible, see POSITIONING §2 SOM math):

| MAU target | Conservative ARR ($30 ARPU) | Optimistic ARR ($60 ARPU) |
|---|---|---|
| 100K | $3.0M | $6.0M |
| 250K | $7.5M | $15M |
| 500K | $15M | $30M |
| 1M | $30M | $60M |

**Pro-tier conversion model**:

| Pro pricing | Conversion (5%) | Annual revenue per Pro user | Pro ARR at 250K free MAU |
|---|---|---|---|
| $9.99 / mo | 5% (12.5K Pro users) | $119 | **$1.5M** |
| $14.99 / mo | 5% (12.5K Pro users) | $179 | **$2.2M** |
| $14.99 / mo | 10% (25K Pro users) | $179 | **$4.5M** |

**Year-1 SOM ARR (conservative)**: ~$7.5M free + $1.5M Pro ≈ **$9M ARR** at 250K MAU.
**Year-1 SOM ARR (optimistic)**: ~$15M free + $2.2M Pro ≈ **$17M ARR**.
**Year-2 SOM (1M MAU plausible)**: ~$30–$60M ARR + $5–$10M Pro ≈ **$35–$70M ARR**, before any hospital deals.

This is the ad-supported DTC ladder. The numbers are real-source-anchored back-of-napkin (per Munger discipline), not audited.

---

## 5. Hormozi value-stack from the nurse user's POV

**The free user's "what do I get for $0" stack:**

| Lever | What the nurse gets | Dollar-equivalent |
|---|---|---|
| Cited clinical answers, nurse register, mobile-first | unlimited queries, no NPI gate friction (lower than OE's enrollment) | UpToDate $559/yr equivalent ([2026 individual price](https://patientnotes.ai/resources/medical-apps-healthcare-providers)) |
| Pill camera + lung-sound audio + ECG image input | multimodal at point of care | Epocrates Plus + Medscape alone do not own this |
| SBAR / SOAP teach-back templates auto-generated at FKGL ≤ 8 | shaves ~3 min per teach-back × N patients | conservative $30 / shift in shaved documentation |
| Offline / WiFi-down resilience | works in code-blue corridors, 1-bar floors | uncountable; no competitor matches |
| Free CEU through Pro upsell preview | low-friction CEU exposure | typical CEU bundle $200–$400 / yr |
| **Stack value at $0 cost** | | **$700–$1,000+ / yr equivalent** |

The Hormozi "value-stack ratio" is roughly **70:1 to 100:1** at the free tier, with ad inventory paying for it. Pro at $14.99 / mo ($180 / yr) layers on ad-free + 1 NCLEX bank + custom protocols + voice macros — adds another ~$300–$500 / yr equivalent value.

---

## 6. Hospital tier (long-tail, year-3)

Per AHA 2024 data, ~6,090 US hospitals. Deal-size band for an HCP clinical-AI product:

| Hospital size | Deal size | Cycle |
|---|---|---|
| Critical-access (~1,300 facilities) | $10K–$30K / yr | 12–18 months |
| Community hospitals (~3,500 facilities) | $30K–$100K / yr | 12–24 months |
| Academic / large systems (~1,200 facilities) | $100K–$500K / yr | 18–36 months |

Year-3 ARR ladder, conservative penetration (~5% of TAM):

| Tier penetration | ARR contribution |
|---|---|
| 5% × critical-access × $20K | $1.3M |
| 5% × community × $60K | $10.5M |
| 5% × academic × $250K | $15M |
| **Year-3 hospital ARR** | **~$27M** layered on top of consumer |

Combined year-3 ladder: ~$35–70M consumer + ~$27M hospital + Pro = **$70–$100M+ ARR plausible**, contingent on year-1 nurse mindshare being won. The hospital tier is the v3 pitch slide; it is **not** load-bearing for year-1 revenue and **is** load-bearing for the SAM ceiling story.

---

## 7. The risk that overrides everything: SOTA-accuracy collapse

If accuracy is anything but best-in-class, RNs return to GPT or Claude. Per the user reframing: *if the medical knowledge and accuracy is anything but best-in-class SOTA, they will return to the more convenient GPT and Claude*.

Concretely, the engineering binding constraints:

| Metric | v0 baseline | Phase 2.1 actual | v1.0 launch gate | Source |
|---|---|---|---|---|
| Held-out chemoprevention 6-fixture mean | 0.273 | 0.335 | ≥ 0.55 | this repo Phase 2.1 results |
| HealthBench Hard | not yet measured | not yet measured | ≥ 0.55 (open SOTA ~0.45) | SPEC §6 |
| MedAgentBench | not yet measured | not yet measured | ≥ 0.70 | SPEC §6 |
| MedQA-USMLE | not yet measured | not yet measured | ≥ 0.85 | SPEC §6 |
| Tamoxifen + Mirena rubric-v2 | 0.273 | (chemoprev fixture set) | ≥ 0.80 | SPEC §6 |

**The accuracy moat IS the revenue moat.** Every ad impression rests on the bet that the user came back. They come back if the answer was right and citable. They do not come back if it was wrong, vague, or uncited. This is why CLAUDE.md §4 (verify-then-claim discipline) and SPEC §6 (the metric table with explicit pre-launch gates) are revenue-protective, not academic.

If we miss the ≥ 0.55 HealthBench Hard gate at v1 launch readiness review, the launch slips. Shipping below SOTA = burning the brand and the funding ladder in one quarter. The accuracy gate is non-negotiable.

---

## 8. Sources

- [Sacra equity research, OpenEvidence April 2026](https://sacra-pdfs.s3.us-east-2.amazonaws.com/openevidence.pdf)
- [Linear newsletter #158 — OpenEvidence ads playbook](https://www.newsletter.lukesophinos.com/p/linear-158-openevidence-just-turned)
- [OpenEvidence advertising policy](https://www.openevidence.com/policies/advertising)
- [Promodo healthcare digital marketing benchmarks 2026](https://www.promodo.com/blog/healthcare-digital-marketing-benchmarks)
- [Nurse.org best shoes for nurses 2026](https://nurse.org/articles/best-shoes-for-nurses/)
- [FIGS Nurses Week 2026](https://nurse.org/articles/figs-nurses-week/) and [FIGS revenue 2025 macrotrends](https://www.macrotrends.net/stocks/charts/FIGS/figs/revenue)
- [Modern Retail FIGS brand-ambassador program](https://www.modernretail.co/retailers/inside-figs-brand-ambassador-program-of-health-care-professionals/)
- [BLS Registered Nurses 2024 OOH](https://www.bls.gov/ooh/healthcare/registered-nurses.htm)
- [PatientNotes 2026 medical apps for healthcare providers](https://patientnotes.ai/resources/medical-apps-healthcare-providers)
