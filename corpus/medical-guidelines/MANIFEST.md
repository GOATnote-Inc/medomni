# Medical-guideline corpus — MANIFEST

Sovereign chemoprevention RAG corpus for the Nemotron-70B-Med stack.

- Output: `chunks.jsonl` — 78 chunks across 29 source documents (50 v0 chunks
  + 28 v2 chunks; both layers coexist for A/B inspection via the
  `corpus_layer` field on every record).
- Builders:
  - `_build_chunks.py` (v0, hand-rolled section-aware splitter) —
    deterministic; rerun to rebuild the v0 50-chunk slice. Idempotent.
  - `scripts/_build_chunks_v2.py` (v2, NeMo-Curator-shaped chunker) —
    appends 28 chunks for the HPV catch-up and bisphosphonate-AI
    breast-cancer-bone-modifying-agent axes (Phase 1 corpus extension).
    See file header for the NeMo-Curator-API mirror contract and the
    Python 3.14 install-issue note.
- Chunk parameters (NVIDIA-canonical): target ~1024 tokens, 15% overlap (~150
  tokens), page-aware section boundaries, lists/tables preserved as units.
  Heuristic: 4 chars per token. Most chunks land in the 400-700 token range
  because each natural section (page-aware unit) is preserved as its own chunk
  rather than packed to the maximum; consecutive sections within one source
  doc share ~150-600 chars of paragraph-aligned overlap.
- Verify: `python -c "import json; lines=open('corpus/medical-guidelines/chunks.jsonl').readlines(); print('chunks:', len(lines)); print('source docs:', len({json.loads(l)['source_doc_id'] for l in lines}))"`

## Source documents

All sources are public domain (US federal — USPSTF, NCI, CDC) and were
extracted via WebFetch from the canonical public URLs listed below. No
fabricated text — every chunk body traces to one of these URLs.

| # | source_doc_id | title | authors | year | URL | n_chunks | license |
|---|---|---|---|---|---|---|---|
| 1 | `USPSTF-2019-breast-cancer-meds-risk-reduction` | Medication Use to Reduce Risk of Breast Cancer | US Preventive Services Task Force | 2019 | https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/breast-cancer-medications-for-risk-reduction | 3 | USPSTF public domain |
| 2 | `USPSTF-2022-aspirin-CVD-CRC` | Aspirin Use to Prevent Cardiovascular Disease | US Preventive Services Task Force | 2022 | https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/aspirin-to-prevent-cardiovascular-disease-preventive-medication | 2 | USPSTF public domain |
| 3 | `USPSTF-2019-BRCA-risk-assessment` | BRCA-Related Cancer: Risk Assessment, Genetic Counseling, and Genetic Testing | US Preventive Services Task Force | 2019 | https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/brca-related-cancer-risk-assessment-genetic-counseling-and-genetic-testing | 3 | USPSTF public domain |
| 4 | `USPSTF-2022-statin-primary-prevention` | Statin Use for the Primary Prevention of CVD in Adults | US Preventive Services Task Force | 2022 | https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/statin-use-in-adults-preventive-medication | 3 | USPSTF public domain |
| 5 | `USPSTF-2021-tobacco-cessation` | Tobacco Smoking Cessation in Adults, Including Pregnant Persons | US Preventive Services Task Force | 2021 | https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/tobacco-use-in-adults-and-pregnant-women-counseling-and-interventions | 1 | USPSTF public domain |
| 6 | `CDC-ACIP-HPV-immunization-schedule` | Child and Adolescent Immunization Schedule by Age (HPV section) | CDC; ACIP | 2025 | https://www.cdc.gov/vaccines/hcp/imz-schedules/child-adolescent-age.html | 1 | CDC public domain |
| 7 | `CDC-smoking-cessation-quit` | How to Quit Smoking — CDC Tips From Former Smokers | CDC | 2024 | https://www.cdc.gov/tobacco/campaign/tips/quit-smoking/index.html | 3 | CDC public domain |
| 8 | `NCI-PDQ-breast-cancer-prevention-HP` | Breast Cancer Prevention (PDQ) — Health Professional Version | NCI PDQ Editorial Board | 2025 | https://www.cancer.gov/types/breast/hp/breast-prevention-pdq | 4 | NCI public domain |
| 9 | `NCI-PDQ-breast-cancer-prevention-patient` | Breast Cancer Prevention (PDQ) — Patient Version | NCI PDQ Editorial Board | 2025 | https://www.cancer.gov/types/breast/patient/breast-prevention-pdq | 1 | NCI public domain |
| 10 | `NCI-PDQ-colorectal-cancer-prevention-HP` | Colorectal Cancer Prevention (PDQ) — Health Professional Version | NCI PDQ Editorial Board | 2025 | https://www.cancer.gov/types/colorectal/hp/colorectal-prevention-pdq | 3 | NCI public domain |
| 11 | `NCI-PDQ-colorectal-cancer-prevention-patient` | Colorectal Cancer Prevention (PDQ) — Patient Version | NCI PDQ Editorial Board | 2025 | https://www.cancer.gov/types/colorectal/patient/colorectal-prevention-pdq | 2 | NCI public domain |
| 12 | `NCI-PDQ-prostate-cancer-prevention-HP` | Prostate Cancer Prevention (PDQ) — Health Professional Version | NCI PDQ Editorial Board | 2025 | https://www.cancer.gov/types/prostate/hp/prostate-prevention-pdq | 3 | NCI public domain |
| 13 | `NCI-PDQ-cervical-cancer-prevention-HP` | Cervical Cancer Prevention (PDQ) — Health Professional Version | NCI PDQ Editorial Board | 2025 | https://www.cancer.gov/types/cervical/hp/cervical-prevention-pdq | 2 | NCI public domain |
| 14 | `NCI-oral-contraceptives-cancer-risk` | Oral Contraceptives and Cancer Risk — NCI Fact Sheet | NCI | 2024 | https://www.cancer.gov/about-cancer/causes-prevention/risk/hormones/oral-contraceptives-fact-sheet | 3 | NCI public domain |
| 15 | `NCI-breast-cancer-risk-women` | Breast Cancer Risk in American Women — NCI Fact Sheet | NCI | 2024 | https://www.cancer.gov/types/breast/risk-fact-sheet | 1 | NCI public domain |
| 16 | `NCI-menopausal-hormone-therapy` | Menopausal Hormone Therapy and Cancer — NCI Fact Sheet | NCI | 2024 | https://www.cancer.gov/about-cancer/causes-prevention/risk/hormones/mht-fact-sheet | 4 | NCI public domain |
| 17 | `NCI-alcohol-cancer-risk` | Alcohol and Cancer Risk — NCI Fact Sheet | NCI | 2024 | https://www.cancer.gov/about-cancer/causes-prevention/risk/alcohol/alcohol-fact-sheet | 3 | NCI public domain |
| 18 | `NCI-obesity-cancer-fact-sheet` | Obesity and Cancer — NCI Fact Sheet | NCI | 2024 | https://www.cancer.gov/about-cancer/causes-prevention/risk/obesity/obesity-fact-sheet | 5 | NCI public domain |
| 19 | `NCI-diet-cancer-overview` | Diet and Cancer Risk — NCI Overview | NCI | 2024 | https://www.cancer.gov/about-cancer/causes-prevention/risk/diet | 2 | NCI public domain |
| 20 | `NCI-antiperspirants-breast-cancer` | Antiperspirants/Deodorants and Breast Cancer — NCI Fact Sheet | NCI | 2024 | https://www.cancer.gov/about-cancer/causes-prevention/risk/myths/antiperspirants-fact-sheet | 1 | NCI public domain |

**v0 total: 20 source documents, 50 chunks.**

## v2 source documents (Phase 1 corpus extension, 2026-04-29)

Phase 1 of the MedOmni v1.0 north-star plan added these nine sources to lift
HPV (3 -> 13 chunks) and bisphosphonate-AI breast-cancer-bone-modifying-
agents (0 -> 15 chunks) coverage. All chunks carry `corpus_layer: v2`.

| # | source_doc_id | title | year | URL | n_chunks | license |
|---|---|---|---|---|---|---|
| 21 | `ACIP-MMWR-2019-HPV-adults-Meites` | ACIP HPV Vaccination for Adults — Updated Recommendations (Meites et al.) | 2019 | https://www.cdc.gov/mmwr/volumes/68/wr/mm6832a3.htm | 5 | MMWR / CDC public domain |
| 22 | `NCI-HPV-vaccine-fact-sheet` | HPV Vaccines — NCI Fact Sheet | 2024 | https://www.cancer.gov/about-cancer/causes-prevention/risk/infectious-agents/hpv-vaccine-fact-sheet | 5 | NCI public domain |
| 23 | `CDC-adult-imz-schedule-HPV-notes` | Adult Immunization Schedule — HPV Notes | 2025 | https://www.cdc.gov/vaccines/hcp/imz-schedules/adult-notes.html | 1 | CDC public domain |
| 24 | `CDC-STI-treatment-guidelines-HPV` | STI Treatment Guidelines — HPV | 2021 | https://www.cdc.gov/std/treatment-guidelines/hpv.htm | 2 | CDC public domain |
| 25 | `EBCTCG-2015-Lancet-bisphosphonate-meta` | Adjuvant bisphosphonate treatment in early breast cancer (EBCTCG IPD meta-analysis) | 2015 | https://pubmed.ncbi.nlm.nih.gov/26211824/ | 3 | structured paraphrase of public abstract; full text Elsevier-licensed |
| 26 | `ASCO-2017-BMA-early-breast-cancer-Dhesy-Thind` | ASCO/CCO Bone-Modifying-Agents Guideline (Dhesy-Thind et al.) | 2017 | https://ascopubs.org/doi/10.1200/JCO.2016.70.7257 | 6 | structured paraphrase of public abstract; full text JCO-licensed |
| 27 | `ABCSG-18-2015-Gnant-Lancet` | Adjuvant denosumab in breast cancer (ABCSG-18) | 2015 | https://pubmed.ncbi.nlm.nih.gov/26244780/ | 2 | structured paraphrase of public abstract |
| 28 | `Z-FAST-ZO-FAST-summary` | Z-FAST/ZO-FAST upfront-vs-delayed zoledronic acid | 2009 | https://pubmed.ncbi.nlm.nih.gov/19470937/ | 2 | structured paraphrase of public abstract |
| 29 | `FDA-bisphosphonate-ONJ-AFF-safety` | FDA bisphosphonate ONJ / atypical-femoral-fracture safety information | 2014 | https://www.fda.gov/drugs/postmarket-drug-safety-information-patients-and-providers | 2 | FDA public domain |

**v2 total: 9 source documents, 28 chunks.**

**Combined v0 + v2: 29 source documents, 78 chunks.**

## v3 source documents (Phase 1.5 corpus extension, 2026-04-29)

Phase 1.5 of the MedOmni v1.0 north-star plan added these eight sources to
target the trial-citation gaps that drove the held-out floor scores in the
5ARI-PROSTATE, ASPIRIN-CRC, and SMOKING-CESSATION fixtures. All chunks
carry `corpus_layer: v3` and the new Phase 1.5 metadata fields:
`source_tier` ∈ {`primary-verbatim`, `secondary-summary`, `regulatory-extract`},
`verbatim` (bool), and the existing `license` field.

Provenance-tier policy: where the primary trial publication is paywalled
(NEJM, Lancet), this corpus extension does NOT redistribute the paywalled
text. Instead it ingests verbatim text from authoritative public-access
secondary sources (NCI PDQ, USPSTF, PMC commentary) that describe each
trial — the chunk body text is verbatim from the secondary source, but
the trial *citation* (PCPT, REDUCE, CAPP2, ASPREE, EAGLES) is what the
chunk anchors. The PHS 2008 Treating Tobacco Use guideline IS public
domain (US PHS / AHRQ) and ships as `primary-verbatim`.

Builder: `scripts/_build_chunks_pmc_verbatim.py` — NeMo-Curator-API-shaped
chunker (target ~1024 tokens, 15% overlap, page-aware section boundaries,
tables / numbered lists preserved as units). Curator 1.1.0 was installed
on B300 in `~/medomni-rapids/.venv` (Python 3.12.3) for on-pod chunking;
this builder script mirrors the Curator pipeline composition
(`UnicodeReformatter` + `NewlineNormalizer` + `get_paragraphs` + target-
token packer) so the local-laptop and on-pod chunkers produce equivalent
output.

| # | source_doc_id | title | year | URL | tier | n_chunks | license |
|---|---|---|---|---|---|---|---|
| 30 | `NCI-PDQ-PCPT-finasteride-trial-detail` | NCI PDQ Prostate Cancer Prevention HP — PCPT trial detail (Thompson 2003 finasteride) | 2025 | https://www.cancer.gov/types/prostate/hp/prostate-prevention-pdq | secondary-summary | 5 | NCI public domain |
| 31 | `NCI-PDQ-REDUCE-dutasteride-trial-detail` | NCI PDQ Prostate Cancer Prevention HP — REDUCE trial detail (Andriole 2010 dutasteride) | 2025 | https://www.cancer.gov/types/prostate/hp/prostate-prevention-pdq | secondary-summary | 4 | NCI public domain |
| 32 | `NCI-PDQ-CAPP2-aspirin-lynch-detail` | NCI PDQ Colorectal Cancer Prevention HP — CAPP2 aspirin-Lynch detail (Burn 2011 + 2020) | 2025 | https://www.cancer.gov/types/colorectal/hp/colorectal-prevention-pdq | secondary-summary | 4 | NCI public domain |
| 33 | `PMC6678038-ASPREE-elderly-aspirin` | PMC review of ASPREE (McNeil 2018 NEJM x3 — elderly aspirin primary prevention) | 2019 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6678038/ | secondary-summary | 3 | PMC OA |
| 34 | `USPSTF-2022-aspirin-ASPREE-detail` | USPSTF 2022 Aspirin Use to Prevent CVD — ASPREE evidence + 2022 reversal rationale | 2022 | https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/aspirin-to-prevent-cardiovascular-disease-preventive-medication | secondary-summary | 2 | USPSTF public domain |
| 35 | `USPSTF-2021-tobacco-EAGLES-pharmacotherapy-detail` | USPSTF 2021 Tobacco Cessation — EAGLES findings + pharmacotherapy comparative effectiveness | 2021 | https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/tobacco-use-in-adults-and-pregnant-women-counseling-and-interventions | secondary-summary | 2 | USPSTF public domain |
| 36 | `EAGLES-trial-summary-NERDCAT` | NERDCAT critical-appraisal summary of EAGLES (Anthenelli 2016 Lancet) | 2016 | https://nerdcat.org/studysummaries/eagles | secondary-summary | 2 | Open educational |
| 37 | `PHS-2008-Treating-Tobacco-Use-Fiore` | Treating Tobacco Use and Dependence: 2008 Update (US PHS Clinical Practice Guideline, Fiore et al.) | 2008 | https://www.ncbi.nlm.nih.gov/books/NBK63952/ | primary-verbatim | 7 | US PHS / AHRQ public domain |

**v3 total: 8 source documents, 29 chunks.**

**Combined v0 + v2 + v3: 37 source documents, 107 chunks.**

### Verification command (Phase 1.5)

```bash
.venv/bin/python -c "import json; \
    L=[json.loads(l) for l in open('corpus/medical-guidelines/chunks.jsonl')]; \
    from collections import Counter; \
    print('total:', len(L)); \
    print('by layer:', Counter(c.get('corpus_layer','v0') for c in L)); \
    print('by tier:', Counter(c.get('source_tier','none') for c in L))"
```

Expected output: `total: 107`, `by layer: {v0: 50, v3: 29, v2: 28}`,
`by tier: {none: 78, secondary-summary: 22, primary-verbatim: 7}` (the
`none` tier is v0+v2 chunks predating the source-tier metadata; their
provenance is documented in the v0/v2 license fields above).

### Source-tier audit and Phase 1.5 deferred items

**Trials NOT in primary-verbatim form (paywalled at primary venue):**
- PCPT (Thompson 2003 NEJM 348:215) — secondary via NCI PDQ
- REDUCE (Andriole 2010 NEJM 362:1192) — secondary via NCI PDQ
- CAPP2 (Burn 2011 Lancet 378:2081 + Burn 2020 Lancet 395:1855) — secondary via NCI PDQ + PMC4093362
- ASPREE (McNeil 2018 NEJM 379:1499/1509/1519, three companion papers) — secondary via PMC6678038 + USPSTF 2022
- EAGLES (Anthenelli 2016 Lancet 387:2507) — secondary via USPSTF 2021 + NERDCAT

The whitepaper position from Phase 2.1 §D-A holds: keep secondary-source
verbatim text; do not redistribute paywalled NEJM/Lancet full-text. The
chunk body is auditable verbatim text from the public secondary source;
the cited primary trial is anchored by name, year, and journal venue so
the model can cite the trial directly even though the corpus does not
contain its full primary text.

**Trial in primary-verbatim form:**
- PHS 2008 (Fiore et al. — Treating Tobacco Use and Dependence) — primary-verbatim via NCBI Bookshelf NBK63952; US-government public-domain clinical practice guideline.

**Skipped this pass:**
- FDA Drug Safety Communication on Chantix/Zyban boxed-warning removal
  (December 2016): the original FDA URL returned HTTP 404 / 403 across
  multiple variants attempted. Regulatory-extract tier therefore ships
  as 0 chunks in v3; the EAGLES regulatory context is captured indirectly
  through the USPSTF 2021 + NERDCAT chunks, which describe the boxed-
  warning-removal context.

License note: PMC abstract chunks (EBCTCG, ABCSG-18, Z-FAST/ZO-FAST, ASCO
2017) are structured paraphrases of the public PubMed/journal abstract,
labeled in each chunk's `license` field. Full-text Elsevier/Wolters Kluwer
content was NOT redistributed; chunk bodies are paraphrase-grade with short
verbatim fragments under fair use for the sovereign RAG corpus. Federal-
government chunks (CDC, FDA, ACIP/MMWR, NCI) are verbatim public-domain.

## Chemoprevention domain coverage

- Breast: USPSTF-2019 risk-reducing meds; NCI PDQ HP + Patient; NCI breast risk fact sheet; NCI antiperspirant fact sheet (held-out negative-control); NCI oral contraceptives + MHT (cross-references breast).
- Colorectal: USPSTF-2022 aspirin (CRC section); NCI PDQ HP + Patient; CAPP2 trial in Lynch; NCI alcohol + obesity (cross-references CRC).
- Prostate: NCI PDQ HP (PCPT/REDUCE/SELECT).
- Cervical / HPV: NCI PDQ HP (FUTURE I/II, 9-valent); CDC ACIP immunization schedule.
- Cardiovascular / preventive cardiology: USPSTF-2022 aspirin + statin.
- Tobacco: USPSTF-2021 cessation; CDC quit-smoking; NCI alcohol fact sheet (synergy).
- BRCA / genetic risk: USPSTF-2019 BRCA recommendation.
- Modifiable risk: NCI obesity (13 cancers); NCI alcohol (Group 1 carcinogen); NCI diet overview; NCI MHT.

## Skipped sources

The following sources were attempted but skipped — content was either gated,
returned 404, or returned content unrelated to the requested topic. No
fabricated text was used to fill in these gaps.

| Source | URL attempted | Reason skipped |
|---|---|---|
| ACOG Committee Opinion 601 (Tamoxifen and Uterine Cancer) | https://www.acog.org/clinical/clinical-guidance/committee-opinion/articles/2014/06/tamoxifen-and-uterine-cancer | HTTP 402 — paywall / member-gated |
| Cochrane Romero 2020 — Mirena + tamoxifen for endometrial protection | https://pmc.ncbi.nlm.nih.gov/articles/PMC7390505/ | PMC ID returned an unrelated Cochrane review (low back pain ultrasound), not the chemoprevention article. PMC ID drift / wrong-id; would need a different PMC ID to recover the Romero paper. |
| NSABP P-1 / chemoprevention trial PMC OA | https://pmc.ncbi.nlm.nih.gov/articles/PMC2528959/ | PMC ID returned an unrelated neuroscience article about songbird brain gene expression. PMC ID drift; would need the correct PMC ID for Fisher et al. NSABP P-1 to recover. |
| IBIS-II / anastrozole chemoprevention PMC OA | https://pmc.ncbi.nlm.nih.gov/articles/PMC4537443/ | PMC ID returned an unrelated neuroscience article (cortical microcircuits). PMC ID drift; would need a verified PMC ID for Cuzick et al. IBIS-II to recover. |
| NCI BCRAT/Gail tool description | https://www.cancer.gov/bcrisktool/about-tool.aspx | 404 after 301 redirect to bcrisktool.cancer.gov subdomain; the redirected URL also 404'd. |
| CDC HPV vaccination consumer / public page | https://www.cdc.gov/vaccines/vpd/hpv/public/index.html | HTTP 404 — page moved. Replaced with CDC ACIP child-adolescent schedule page (source #6). |
| CDC ACIP HPV recommendations HCP page | https://www.cdc.gov/vaccines/vpd/hpv/hcp/recommendations.html | Returned empty content; replaced with CDC schedule page (source #6). |
| ACOG #601 (tamoxifen-uterine, 2014) | https://www.acog.org/clinical/... | Member paywall. |
| NCI fruit/vegetable fact sheet | https://www.cancer.gov/about-cancer/causes-prevention/risk/diet/fruit-vegetables-fact-sheet | HTTP 404. |
| NCI menopausal hormones (legacy URL) | https://www.cancer.gov/about-cancer/causes-prevention/research/menopausal-hormones-fact-sheet | HTTP 404 — recovered via current URL `/risk/hormones/mht-fact-sheet` (source #16). |
| CDC breast cancer risk factors | https://www.cdc.gov/cancer/breast/basic_info/risk_factors.htm | HTTP 404 — page moved. |

## License summary

All 20 included sources are US federal government works (USPSTF, NCI, CDC).
US federal works are not subject to copyright (17 USC §105); they are in the
public domain. Verbatim reuse for the sovereign RAG corpus is permitted
without attribution requirements beyond standard scholarly courtesy. Each
chunk record retains `source_url`, `source_title`, and `source_authors` so
downstream RAG retrievals can cite back to the canonical page.

The PMC OA full-text articles intended for inclusion (NSABP P-1, IBIS-II,
Cochrane Romero) were not recovered (see skipped sources). Any future
expansion of this corpus that adds PMC OA articles must check the article's
specific license — typically CC-BY or CC-BY-NC — before redistribution.
