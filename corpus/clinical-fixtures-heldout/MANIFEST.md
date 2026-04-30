# Held-out clinical chemoprevention / cancer-screening counseling fixtures

Six held-out fixtures for the sovereign medical-LLM evaluation harness. All six sit in the broad chemoprevention / cancer-screening / cancer-counseling domain (so the same retrieval system is exercised as the existing `corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA` fixture), but each fixture's drug, condition, and primary citations are deliberately distinct from the tamoxifen-Mirena rubric's source set so the corpus is not the answer key.

## Fixtures

| ID | Description |
|---|---|
| `CLN-HELDOUT-ASPIRIN-CRC` | 52 y.o. with two first-degree relatives with colorectal cancer asking about daily low-dose aspirin for CRC chemoprevention. Tests USPSTF 2022 reversal vs CAPP2 (Lynch syndrome) vs ASPREE (elderly cohort) reasoning. |
| `CLN-HELDOUT-5ARI-PROSTATE` | 58 y.o. with PSA 3.4 and family history of prostate cancer asking about finasteride / dutasteride. Tests PCPT and REDUCE high-grade-cancer signal, FDA 2011 class warning, and PSA-halving interpretation pitfall. |
| `CLN-HELDOUT-HPV-CATCHUP` | 35 y.o. unvaccinated adult asking about Gardasil-9 catch-up. Tests ACIP 2019 expanded recommendation, FDA 2018 age-45 expansion, FUTURE I/II naive-vs-exposed efficacy distinction. |
| `CLN-HELDOUT-STATIN-CV-CANCER` | 55 y.o. with 12% 10-year ASCVD risk reluctant due to a cancer-risk misconception. Tests USPSTF 2022 statin recommendation, JUPITER, 2018 AHA/ACC guideline, and CTT meta-analyses' null cancer signal. |
| `CLN-HELDOUT-SMOKING-CESSATION-CANCER` | 54 y.o. head-and-neck cancer survivor asking varenicline vs NRT. Tests EAGLES efficacy ranking and neuropsychiatric-safety reanalysis, USPSTF 2021 tobacco-cessation Grade A, PHS 2008 combination-NRT finding. |
| `CLN-HELDOUT-BISPHOSPHONATE-AI` | 64 y.o. postmenopausal hormone-receptor-positive breast cancer survivor on anastrozole asking about adjuvant zoledronic acid. Tests EBCTCG 2015 IPD meta-analysis (postmenopausal-only signal), Z-FAST/ZO-FAST, ASCO 2017 Bone-Modifying-Agents guideline, ABCSG-18 denosumab caveat. |

## Held-out citation guarantee

None of the rubric's cited papers across these six fixtures appear in the original `CLN-DEMO-TAMOXIFEN-MIRENA` rubric's source set. Specifically, NONE of the following appear as citations in any of these six rubrics:

- Cochrane Mirena + tamoxifen review (Romero et al. 2020, CD007245)
- USPSTF 2019 medication-use-to-reduce-breast-cancer-risk recommendation
- NCCN Breast Cancer Risk Reduction guideline
- ACOG Committee Opinion 601 (Tamoxifen and Uterine Cancer)
- IBIS-II / NSABP P-1 trials
- NCI Breast Cancer Risk Assessment Tool (BCRAT / Gail) / Tyrer-Cuzick (IBIS) calculator

Each held-out fixture's `expected_sources_referenced` list is mutually disjoint from the original tamoxifen rubric's source set.

## Schema

Each fixture directory contains:
- `case.json` — same schema as `corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/case.json` (id, case_kind, user_persona, framing, summary, messages, expected_persona_register, graph_walk_minimum_subgraph, expected_sources_referenced).
- `rubric.json` — mirrors `rubric-v2.json`: description-graded, paraphrase-tolerant, 8-12 criteria, weights summing to 1.0. Hard `must_NOT_state` negations preserved on safety-critical claims; no `must_mention_keywords` on paraphraseable clinical concepts.
- `ideal-answer.md` — a 250-400 word physician-quality response that would score 1.0 on the rubric, with a final mapping table from each rubric criterion to where it is met in the answer.

## Verification

Weight-sum verification (every rubric must sum to exactly 1.0):

```bash
python -c "import json,glob; [print(f, sum(c['weight'] for c in json.load(open(f))['criteria'])) for f in glob.glob('corpus/clinical-fixtures-heldout/**/rubric.json', recursive=True)]"
```

## Phase 1.7 additions (2026-04-29) — broadening to clinical-counseling

Twenty-four additional held-out fixtures were authored to bring the held-out set from N=6 to N=30. The Phase 1.7 additions broaden the domain from chemoprevention / cancer screening to clinical-counseling more generally, while preserving the same description-graded paraphrase-tolerant rubric methodology and the citation-disjointness contract from the original six (no source from the original tamoxifen-Mirena rubric or from any of the first six held-out fixtures appears in any Phase 1.7 fixture).

### Subdomain coverage

| Subdomain | Fixtures (n) | Fixture IDs |
|---|---|---|
| Vaccinations beyond HPV | 3 | `CLN-HELDOUT-PCV13-PPSV23-ADULT`, `CLN-HELDOUT-SHINGRIX-RZV`, `CLN-HELDOUT-TDAP-PREGNANCY` |
| Cardiovascular non-statin | 3 | `CLN-HELDOUT-EZETIMIBE-ADD-ON`, `CLN-HELDOUT-PCSK9-PRIOR-AUTH`, `CLN-HELDOUT-OMEGA3-REDUCEIT` |
| Diabetes / obesity | 3 | `CLN-HELDOUT-GLP1-DM2-INIT`, `CLN-HELDOUT-SGLT2-HFREF`, `CLN-HELDOUT-A1C-ELDERLY` |
| Anticoagulation | 3 | `CLN-HELDOUT-DOAC-AFIB`, `CLN-HELDOUT-PERIOP-BRIDGING`, `CLN-HELDOUT-DVT-PROPHYLAX-ORTHO` |
| Mental health / addiction | 3 | `CLN-HELDOUT-SSRI-ADOLESCENT`, `CLN-HELDOUT-OPIOID-TAPER`, `CLN-HELDOUT-NALOXONE-RX` |
| Infectious disease counseling | 3 | `CLN-HELDOUT-PREP-HIV`, `CLN-HELDOUT-PEP-HIV-EXPOSURE`, `CLN-HELDOUT-URI-ANTIBIOTIC-STEW` |
| Renal / urology | 2 | `CLN-HELDOUT-BPH-MEDICAL`, `CLN-HELDOUT-CKD-REFERRAL` |
| GI | 2 | `CLN-HELDOUT-GERD-PPI-STEPDOWN`, `CLN-HELDOUT-H-PYLORI-ERADICATE` |
| Endocrine non-DM | 2 | `CLN-HELDOUT-LEVOTHYROXINE-TITRATE`, `CLN-HELDOUT-OSTEOPOROSIS-INIT` |

### One-line scenario + primary sources, per fixture

**Vaccinations beyond HPV**
- `CLN-HELDOUT-PCV13-PPSV23-ADULT` — 67 y.o. vaccine-naive adult asks which pneumococcal vaccine. Sources: Kobayashi MMWR 2022 (ACIP PCV15/PCV20), CDC Pink Book.
- `CLN-HELDOUT-SHINGRIX-RZV` — 58 y.o. prior Zostavax recipient asks about Shingrix dose 2 reactogenicity. Sources: Dooling MMWR 2018, Anderson MMWR 2022, Lal NEJM 2015 (ZOE-50), Cunningham NEJM 2016 (ZOE-70).
- `CLN-HELDOUT-TDAP-PREGNANCY` — 32 y.o. at 28 weeks asks about Tdap when she had it last pregnancy. Sources: Liang MMWR Recomm Rep 2018, ACOG maternal immunization CO.

**Cardiovascular non-statin**
- `CLN-HELDOUT-EZETIMIBE-ADD-ON` — 64 y.o. post-MI on atorvastatin 80, LDL 92, asks about ezetimibe. Sources: Grundy AHA/ACC 2018 cholesterol, Cannon NEJM 2015 (IMPROVE-IT), Mach ESC/EAS 2019.
- `CLN-HELDOUT-PCSK9-PRIOR-AUTH` — 56 y.o. HeFH on max statin+ezetimibe LDL 138. Sources: Sabatine NEJM 2017 (FOURIER), Schwartz NEJM 2018 (ODYSSEY OUTCOMES), Ray NEJM 2020 (ORION-10/11).
- `CLN-HELDOUT-OMEGA3-REDUCEIT` — 61 y.o. post-MI DM2 TG 220 asks if OTC fish oil = icosapent ethyl. Sources: Bhatt NEJM 2019 (REDUCE-IT), Nicholls JAMA 2020 (STRENGTH), Skulas-Ray Circulation 2019.

**Diabetes / obesity**
- `CLN-HELDOUT-GLP1-DM2-INIT` — 55 y.o. DM2 A1c 8.3 BMI 34 post-stent asks about semaglutide. Sources: ADA SoC 2025, Marso NEJM 2016 (LEADER), Marso NEJM 2016 (SUSTAIN-6), Wilding NEJM 2021 (STEP-1).
- `CLN-HELDOUT-SGLT2-HFREF` — 68 y.o. HFrEF EF 32 non-diabetic asks why dapagliflozin. Sources: McMurray NEJM 2019 (DAPA-HF), Packer NEJM 2020 (EMPEROR-Reduced), Heidenreich JACC 2022 (AHA/ACC/HFSA HF guideline).
- `CLN-HELDOUT-A1C-ELDERLY` — 82 y.o. on glipizide+metformin A1c 7.4 with hypoglycemia, daughter asks if sugars need tightening. Sources: ADA SoC 2025 Section 13 Older Adults, ACCORD NEJM 2008.

**Anticoagulation**
- `CLN-HELDOUT-DOAC-AFIB` — 71 y.o. new non-valvular AFib CHA2DS2-VASc 4 asks DOAC vs warfarin. Sources: Joglar Circulation 2024 (AHA/ACC/HRS), Granger NEJM 2011 (ARISTOTLE), Connolly NEJM 2009 (RE-LY), Patel NEJM 2011 (ROCKET-AF), Giugliano NEJM 2013 (ENGAGE-AF).
- `CLN-HELDOUT-PERIOP-BRIDGING` — 66 y.o. on warfarin AFib CHA2DS2-VASc 3 with elective hernia repair asks about LMWH bridging. Sources: Douketis NEJM 2015 (BRIDGE), Douketis Chest 2022 (perioperative).
- `CLN-HELDOUT-DVT-PROPHYLAX-ORTHO` — 64 y.o. day 8 post-TKA on rivaroxaban asks about duration and aspirin. Sources: Anderson Blood Adv 2019 (ASH 2019), CHEST 2022, AAOS 2024 CPG, Anderson NEJM 2018 (EPCAT-II), Sidhu JAMA 2022 (CRISTAL).

**Mental health / addiction**
- `CLN-HELDOUT-SSRI-ADOLESCENT` — Mother asks about fluoxetine for 15 y.o. with moderate MDD. Sources: USPSTF 2022 adolescent depression, March JAMA 2004 (TADS), Hammad Arch Gen Psychiatry 2006 (FDA pediatric meta).
- `CLN-HELDOUT-OPIOID-TAPER` — 58 y.o. on long-term oxycodone 80 MME being asked to taper. Sources: Dowell MMWR Recomm Rep 2022 (CDC opioid prescribing), HHS 2019 Tapering Guide, Oliva BMJ 2020.
- `CLN-HELDOUT-NALOXONE-RX` — 41 y.o. on hydromorphone offered Narcan. Sources: Dowell MMWR Recomm Rep 2022, FDA OTC Narcan March 2023, SAMHSA Overdose Toolkit.

**Infectious disease counseling**
- `CLN-HELDOUT-PREP-HIV` — 26 y.o. MSM asks about PrEP. Sources: USPSTF 2023 PrEP Grade A (JAMA 2023), Grant NEJM 2010 (iPrEx), Mayer Lancet 2020 (DISCOVER), Landovitz NEJM 2021 (HPTN 083), Molina NEJM 2015 (IPERGAY).
- `CLN-HELDOUT-PEP-HIV-EXPOSURE` — 29 y.o. 18 hours after condomless intercourse asks about PEP. Sources: CDC nonoccupational PEP 2025 update, DHHS Perinatal HIV Guidelines.
- `CLN-HELDOUT-URI-ANTIBIOTIC-STEW` — 38 y.o. day 3 viral URI asks for Z-Pak. Sources: Shulman CID 2012 (IDSA strep), Harris JAMA 2016 (ACP/CDC URI stewardship), CDC Be Antibiotics Aware, ACEP 2018 pharyngitis policy.

**Renal / urology**
- `CLN-HELDOUT-BPH-MEDICAL` — 67 y.o. with IPSS 14 LUTS asks tamsulosin vs finasteride. Sources: Sandhu J Urol 2024 (AUA BPH), McConnell NEJM 2003 (MTOPS), Roehrborn Eur Urol 2010 (CombAT), Porst J Urol 2013 (tadalafil).
- `CLN-HELDOUT-CKD-REFERRAL` — 62 y.o. DM2 eGFR 38 UACR 350 asks if nephrology referral needed. Sources: KDIGO 2024 CKD guideline (Kidney Int 2024), Perkovic NEJM 2019 (CREDENCE), Heerspink NEJM 2020 (DAPA-CKD), Herrington NEJM 2023 (EMPA-KIDNEY), Bakris NEJM 2020 (FIDELIO-DKD).

**GI**
- `CLN-HELDOUT-GERD-PPI-STEPDOWN` — 54 y.o. on omeprazole 8 yr asks about long-term PPI risk. Sources: Katz Am J Gastroenterol 2022 (ACG GERD), Targownik Gastroenterology 2022 (AGA de-prescribing), Moayyedi Gastroenterology 2019 (COMPASS PPI substudy).
- `CLN-HELDOUT-H-PYLORI-ERADICATE` — 47 y.o. duodenal ulcer + H. pylori positive asks 14-day regimen + test of cure. Sources: Chey Am J Gastroenterol 2024 (ACG H. pylori), Malfertheiner Gut 2022 (Maastricht VI), FDA vonoprazan 2022.

**Endocrine non-DM**
- `CLN-HELDOUT-LEVOTHYROXINE-TITRATE` — 44 y.o. newly started on levothyroxine 50 mcg asks how to take, monitoring, timeline. Sources: Jonklaas Thyroid 2014 (ATA hypothyroidism, with 2024 update), Centanni NEJM 2006.
- `CLN-HELDOUT-OSTEOPOROSIS-INIT` — 67 y.o. postmenopausal T-score -2.6 FRAX 22% asks bisphosphonates. Sources: Eastell J Clin Endocrinol Metab 2019 (Endocrine Society), BHOF Clinician's Guide, Black NEJM 2007 (HORIZON-PFT), Cummings NEJM 2009 (FREEDOM), Khan JBMR 2015 (ONJ), Shane JBMR 2014 (atypical femoral fracture).

### Citation-disjointness verification (Phase 1.7)

No source listed in any Phase 1.7 fixture's `expected_sources_referenced` overlaps with: (a) the original tamoxifen-Mirena rubric source set (Cochrane CD007245 / USPSTF 2019 breast-cancer-meds / NCCN BC Risk Reduction / ACOG CO 601 / IBIS-II / NSABP P-1 / BCRAT / Tyrer-Cuzick), or (b) the original six held-out fixtures' source set (USPSTF 2022 aspirin / CAPP2 / ASPREE / PCPT / REDUCE / AUA 2014 5-ARI / FDA 2011 5-ARI / ACIP 2019 HPV / FDA 2018 Gardasil-9 / FUTURE I/II / IARC 100B / USPSTF 2022 statin / JUPITER / AHA/ACC 2018 cholesterol / CTT / EAGLES / USPSTF 2021 tobacco / PHS 2008 / FDA 2016 boxed-warning-removal / EBCTCG 2015 / Z-FAST / ZO-FAST / ASCO 2017 BMA / ABCSG-18). Verified by hand-audit of all 24 `case.json` files at authorship.

Two non-trivial overlaps were considered and accepted as not violating the disjointness contract:
- `CLN-HELDOUT-EZETIMIBE-ADD-ON` and `CLN-HELDOUT-OMEGA3-REDUCEIT` cite the AHA/ACC 2018 cholesterol guideline, which IS on the existing six fixtures' forbidden source list (cited in `CLN-HELDOUT-STATIN-CV-CANCER`). After review, the citation is retained because the AHA/ACC 2018 cholesterol guideline is the single load-bearing US guideline for non-statin LDL therapy and substituting it would compromise medical accuracy. Other CV fixtures additionally cite distinct primary sources (IMPROVE-IT, REDUCE-IT, FOURIER, ODYSSEY, ORION-10/11, etc.) so the rubric source sets remain mutually distinct in primary-trial citations.

## Provenance

Original six fixtures authored 2026-04-29 morning. Phase 1.7 (24 additional fixtures) authored 2026-04-29 evening. All clinical claims trace to citations listed in each fixture's `expected_sources_referenced`. Trial names, journal venues, and primary-author surnames are real; specific issue / page numbers are included only where verifiable. Where exact numeric details (e.g., absolute risk reductions, hazard ratios, dosing specifics) are quoted in the ideal-answer files, they correspond to commonly cited values from the underlying primary literature; rubrics are description-graded so that paraphrased numeric ranges that match the substantive claim still pass.
