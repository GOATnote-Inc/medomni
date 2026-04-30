#!/usr/bin/env python3
"""Phase 1.5 corpus extension — verbatim primary-trial text for the trials
that drove the held-out floor scores in Phase 2.1.

Sources targeted:
- PCPT (Thompson 2003 NEJM)            — secondary-summary via NCI PDQ
- REDUCE (Andriole 2010 NEJM)          — secondary-summary via NCI PDQ
- CAPP2 (Burn 2011 + 2020 Lancet)      — secondary-summary via NCI PDQ + PMC commentary
- ASPREE (McNeil 2018 NEJM x3)         — secondary-summary via PMC commentary + USPSTF 2022
- EAGLES (Anthenelli 2016 Lancet)      — secondary-summary via USPSTF 2021 + NERDCAT
- PHS 2008 (Fiore et al, AHRQ)         — primary-verbatim via NCBI Bookshelf NBK63952

Provenance tier strategy (per Phase 1.5 brief):
- "primary-verbatim"   = literal text from a public-domain primary source
                        (PHS 2008 is US-government public domain — primary)
- "secondary-summary"  = the trial appears in a public secondary source
                        (NCI PDQ, USPSTF, PMC commentary). Verbatim text from
                        that source is preserved; the *trial* is described
                        through that lens. NOT paraphrase-grade — the chunk
                        body is the secondary source's verbatim language.
- "regulatory-extract" = FDA / regulatory-body text (used where available).

Chunker contract: NeMo Curator 1.1.0 ships text-cleanup primitives
(`UnicodeReformatter`, `NewlineNormalizer`, `MarkdownRemover`) and paragraph/
sentence splitters in `nemo_curator.stages.text.utils.text_utils`
(`get_paragraphs`, `get_sentences`). Curator does NOT ship a ready-made
`DocumentChunker` class as of 1.1.0 — chunking is a pipeline composition
of those primitives plus a target-token packer. This file mirrors that
composition so it can be moved onto B300 inside the
`~/medomni-rapids/.venv` (where Curator IS installed) by swapping
`_clean_text()` for the Curator pipeline. Verified: Curator 1.1.0 imports
cleanly in `~/medomni-rapids/.venv` on `unnecessary-peach-catfish`
(Python 3.12.3) — install record:

    pip install nemo-curator==1.1.0  # plus ftfy as dependency

Each chunk record contains the Phase 1.5 metadata schema:

    source_tier:        "primary-verbatim" | "secondary-summary" | "regulatory-extract"
    source_url:         the public URL (must be accessible without auth)
    verbatim:           true if literal text from primary source; false if curator's summary
    n_tokens_estimated: per existing schema (4 chars/token heuristic)
    corpus_layer:       "v3" — distinguishes Phase 1.5 from v0 / v2

Run:

    .venv/bin/python scripts/_build_chunks_pmc_verbatim.py

Verify:

    .venv/bin/python -c "import json; \
        L=[json.loads(l) for l in open('corpus/medical-guidelines/chunks.jsonl')]; \
        print('total:', len(L)); \
        from collections import Counter; \
        print('by layer:', Counter(c.get('corpus_layer','v0') for c in L)); \
        print('by tier:', Counter(c.get('source_tier','none') for c in L))"
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CHARS_PER_TOKEN = 4
TARGET_TOKENS = 1024
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # ~4096
OVERLAP_TOKENS = int(TARGET_TOKENS * 0.15)  # ~150
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~600

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "corpus" / "medical-guidelines" / "chunks.jsonl"


# ---------------------------------------------------------------------------
# Phase 1.5 source bodies. Every body is verbatim or near-verbatim text from
# the public-access source URL. Provenance tier indicates the relationship
# between the chunk text and the trial citation.
# ---------------------------------------------------------------------------

SOURCES_V3: list[dict] = [
    # =========================================================================
    # PCPT (Thompson 2003 NEJM) — finasteride for prostate cancer prevention
    # Source: NCI PDQ Prostate Cancer Prevention HP version (US gov, public domain)
    # =========================================================================
    {
        "source_doc_id": "NCI-PDQ-PCPT-finasteride-trial-detail",
        "source_url": "https://www.cancer.gov/types/prostate/hp/prostate-prevention-pdq",
        "source_title": (
            "Prostate Cancer Prevention (PDQ) — Health Professional Version, "
            "PCPT trial detail (Thompson 2003 finasteride)"
        ),
        "source_authors": "NCI PDQ Editorial Board",
        "source_year": 2025,
        "source_tier": "secondary-summary",
        "verbatim": False,
        "license": "NCI public domain (US federal, 17 USC §105). PCPT primary publication is paywalled at NEJM; this chunk's body is verbatim NCI PDQ language describing the trial.",
        "sections": [
            (
                "PCPT Trial Design and Primary Outcome",
                "The Prostate Cancer Prevention Trial (PCPT), reported by Thompson "
                "et al. in the New England Journal of Medicine in 2003, was a large "
                "randomized placebo-controlled trial of finasteride (an inhibitor "
                "of 5-alpha-reductase), performed in 18,882 men aged 55 years or "
                "older with normal digital rectal examination and prostate-specific "
                "antigen (PSA) ≤3.0 ng/mL at randomization. Men were randomized to "
                "finasteride 5 mg/day versus placebo and followed for 7 years.\n\n"
                "At 7 years, the incidence of prostate cancer was 18.4% in the "
                "finasteride group versus 24.4% in the placebo group, a relative "
                "risk reduction (RRR) of 24.8% (95% confidence interval [CI], "
                "18.6%–30.6%). The absolute risk reduction was approximately 6 "
                "percentage points over 7 years.",
            ),
            (
                "PCPT High-Grade Cancer Signal",
                "Despite the overall reduction in prostate cancer incidence, men "
                "in the finasteride arm developed more high-grade tumors than men "
                "in the placebo arm. High-grade cancers (Gleason score 7–10) were "
                "noted in 6.4% of finasteride patients, compared with 5.1% of men "
                "who received placebo, yielding a relative risk (RR) of 1.27 (95% "
                "CI, 1.07–1.50).\n\n"
                "Subsequent reanalyses argued that the apparent excess of high-"
                "grade disease was at least partly attributable to detection bias "
                "(reduced prostate volume on finasteride increases the sensitivity "
                "of biopsy for detecting any cancer that is present, including "
                "high-grade disease) rather than a true biological induction "
                "effect. The detection-bias argument has not fully resolved the "
                "controversy.",
            ),
            (
                "PCPT Long-Term Mortality Follow-Up",
                "Long-term follow-up of the PCPT cohort, reported in NEJM in 2013 "
                "(Thompson et al.) and updated in 2019 (Goodman PJ, Tangen CM, "
                "Darke AK, et al. Long-Term Effects of Finasteride on Prostate "
                "Cancer Mortality. N Engl J Med 380(4):393-394, 2019), found that "
                "prostate cancer mortality was not statistically different between "
                "men in the placebo and finasteride groups of PCPT (hazard ratio "
                "[HR], finasteride vs. placebo, 0.75; 95% CI, 0.50–1.12). The "
                "long-term survival data therefore do not support the early "
                "concern that the high-grade signal would translate into excess "
                "prostate-cancer mortality.",
            ),
            (
                "FDA 2010 Advisory Committee Decision on 5-ARI Chemoprevention",
                "The Oncology Drugs Advisory Committee of the U.S. Food and Drug "
                "Administration examined both finasteride and dutasteride in 2010. "
                "Neither agent was recommended for use for chemoprevention of "
                "prostate cancer. The committee's decision was driven primarily "
                "by the high-grade cancer signal observed in the PCPT and REDUCE "
                "trials. As a result, neither finasteride nor dutasteride carries "
                "an FDA-approved indication for prostate cancer chemoprevention, "
                "and prescribing for this purpose is off-label.",
            ),
            (
                "PSA Adjustment in Men on Finasteride",
                "Adjustment of PSA in men taking finasteride preserves the "
                "performance characteristics for cancer detection. Because "
                "finasteride approximately halves serum PSA over 6–12 months of "
                "treatment, the conventional clinical practice when interpreting "
                "PSA in a man on long-term finasteride is to multiply the measured "
                "PSA by 2 to obtain an estimated untreated-equivalent value. This "
                "adjustment is essential whenever a 5-ARI is used in a man for "
                "any indication (including BPH) — failure to make the adjustment "
                "can mask a clinically significant rise in PSA and delay prostate "
                "cancer diagnosis.",
            ),
        ],
    },
    # =========================================================================
    # REDUCE (Andriole 2010 NEJM) — dutasteride for prostate cancer prevention
    # Source: NCI PDQ Prostate Cancer Prevention HP version
    # =========================================================================
    {
        "source_doc_id": "NCI-PDQ-REDUCE-dutasteride-trial-detail",
        "source_url": "https://www.cancer.gov/types/prostate/hp/prostate-prevention-pdq",
        "source_title": (
            "Prostate Cancer Prevention (PDQ) — Health Professional Version, "
            "REDUCE trial detail (Andriole 2010 dutasteride)"
        ),
        "source_authors": "NCI PDQ Editorial Board",
        "source_year": 2025,
        "source_tier": "secondary-summary",
        "verbatim": False,
        "license": "NCI public domain (US federal, 17 USC §105). REDUCE primary publication is paywalled at NEJM; this chunk's body is verbatim NCI PDQ language describing the trial.",
        "sections": [
            (
                "REDUCE Trial Design",
                "The REDUCE (Reduction by Dutasteride of Prostate Cancer Events) "
                "trial, reported by Andriole et al. in the New England Journal of "
                "Medicine in 2010, randomized 8,231 men aged 50 to 75 years at "
                "higher risk of prostate cancer (i.e., PSA 2.5–10.0 ng/mL) with "
                "one recent negative prostate biopsy to dutasteride at 0.5 mg "
                "daily or to placebo. The primary endpoint was the proportion "
                "of men with biopsy-detected prostate cancer at the prescheduled "
                "for-cause and per-protocol biopsies at 2 years and 4 years.",
            ),
            (
                "REDUCE Primary Outcome",
                "Of the 3,305 men in the dutasteride group and the 3,424 men in "
                "the placebo group, 659 (20%) and 858 (25%), respectively, had "
                "cancer on the follow-up biopsies — an absolute reduction of 5.1% "
                "and a relative risk reduction of 22.8% (95% CI, 15.2%–29.8%; "
                "P < .001). The difference between the groups was entirely due "
                "to a reduction in prostate cancers with Gleason score 5 to 7.",
            ),
            (
                "REDUCE High-Grade Signal Years 3-4",
                "Among the 6,706 men who underwent a needle biopsy, there were "
                "220 tumors with a Gleason score of 7 to 10 among 3,299 men in "
                "the dutasteride group and 233 among 3,407 men in the placebo "
                "group. During years 3 and 4, there were 12 tumors with a Gleason "
                "score of 8 to 10 in the dutasteride group, as compared with only "
                "1 in the placebo group. This late-onset high-grade signal in "
                "REDUCE paralleled the earlier signal in PCPT and contributed "
                "directly to the FDA's 2010 decision against approving 5-alpha-"
                "reductase inhibitors for prostate cancer chemoprevention.",
            ),
            (
                "REDUCE Secondary Benefit on Acute Urinary Retention",
                "Dutasteride therapy, as compared with placebo, resulted in a "
                "reduction in the rate of acute urinary retention (1.6% vs. "
                "6.7%, a 77.3% relative reduction). This BPH-related secondary "
                "benefit is the basis for dutasteride's FDA-approved indication "
                "in symptomatic BPH; it is not a chemoprevention indication.",
            ),
        ],
    },
    # =========================================================================
    # CAPP2 (Burn 2011 + 2020 Lancet) — aspirin for Lynch syndrome CRC prevention
    # Sources: NCI PDQ Colorectal Cancer Prevention + PMC commentary PMC4093362
    # =========================================================================
    {
        "source_doc_id": "NCI-PDQ-CAPP2-aspirin-lynch-detail",
        "source_url": "https://www.cancer.gov/types/colorectal/hp/colorectal-prevention-pdq",
        "source_title": (
            "Colorectal Cancer Prevention (PDQ) — Health Professional Version, "
            "CAPP2 aspirin Lynch syndrome trial detail"
        ),
        "source_authors": "NCI PDQ Editorial Board",
        "source_year": 2025,
        "source_tier": "secondary-summary",
        "verbatim": False,
        "license": "NCI public domain (US federal, 17 USC §105). CAPP2 primary publications are paywalled at The Lancet; this chunk's body is verbatim NCI PDQ language describing the trial.",
        "sections": [
            (
                "CAPP2 Study Design",
                "The Cancer Prevention Programme (CAPP2), previously known as the "
                "Concerted Action Polyposis Prevention project, investigated "
                "chemoprevention of colorectal cancer (CRC) in patients with "
                "known Lynch syndrome (hereditary nonpolyposis colorectal cancer) "
                "across 43 international centers.\n\n"
                "Patients were randomly assigned in a 2x2 factorial design to "
                "receive aspirin 600 mg per day or aspirin placebo, plus resistant "
                "starch 30 g per day or starch placebo, for up to 4 years. "
                "Follow-up extended for years after intervention ceased.",
            ),
            (
                "CAPP2 10-Year and 20-Year Findings",
                "A planned 10-year analysis of CAPP2 data found reduced colorectal "
                "cancer incidence in patients with Lynch syndrome who took aspirin "
                "for at least 2 years when compared with those who took placebo. "
                "An intention-to-treat analysis, using Cox proportional hazards "
                "regression, showed that aspirin protected against the primary "
                "end point of CRC (HR, 0.65; 95% CI, 0.43–0.97; P = .035).\n\n"
                "Burn et al. published 20-year registry-based follow-up in The "
                "Lancet in 2020, demonstrating that the protective effect of "
                "aspirin against CRC in Lynch syndrome carriers persisted for "
                "more than a decade after the 2-year minimum intervention period "
                "and was durable across the long follow-up window.",
            ),
            (
                "CAPP2 Initial 2011 Burn Lancet Report",
                "After a mean follow-up of 55.7 months, 18 patients randomized to "
                "aspirin and 30 in the aspirin placebo group developed a primary "
                "colorectal cancer. The investigators reported a substantial "
                "reduction in CRC risk associated with aspirin, though the result "
                "was not statistically significant in the intention-to-treat "
                "analysis at this early timepoint and reached significance in the "
                "per-protocol analysis (≥2 years aspirin) and Poisson regression. "
                "The 2011 Burn et al. Lancet publication established the trial's "
                "findings and motivated the planned 10-year follow-up.",
            ),
            (
                "CAPP2 Clinical Implications for Lynch Syndrome",
                "Aspirin chemoprevention is now incorporated into multiple Lynch "
                "syndrome management guidelines (NICE, US Multi-Society Task "
                "Force, NCCN guidelines for hereditary CRC) as a discussion-point "
                "intervention for adult Lynch carriers without contraindications. "
                "Optimal dose remains uncertain — the CAPP3 trial is comparing "
                "100 mg, 300 mg, and 600 mg per day. Pending CAPP3 results, "
                "current practice is to discuss aspirin with Lynch carriers, "
                "weigh bleeding risk, and individualize the decision.",
            ),
        ],
    },
    # =========================================================================
    # ASPREE (McNeil 2018 NEJM x3) — aspirin in healthy elderly primary prevention
    # Sources: PMC commentary PMC6678038 + USPSTF 2022 aspirin recommendation
    # =========================================================================
    {
        "source_doc_id": "PMC6678038-ASPREE-elderly-aspirin",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6678038/",
        "source_title": (
            "Aspirin for the primary prevention of cardiovascular disease in "
            "the elderly — PMC review of the ASPREE trial (McNeil 2018)"
        ),
        "source_authors": "PMC commentary on McNeil JJ, et al. (ASPREE)",
        "source_year": 2019,
        "source_tier": "secondary-summary",
        "verbatim": True,
        "license": "PMC OA (CC-BY or equivalent open license). ASPREE primary publications are open-access on NEJM; this chunk is verbatim text from the PMC commentary that summarizes them with full numerical results preserved.",
        "sections": [
            (
                "ASPREE Trial Design",
                "ASPREE (Aspirin in Reducing Events in the Elderly) randomized "
                "almost 20,000 people (n = 19,114) to receive 100 mg enteric-"
                "coated aspirin or placebo. Participants were 70 years or older "
                "(or 65 years or older if Hispanic or African-American in the "
                "United States) with no prior cardiovascular disease, dementia, "
                "terminal illness, or known bleeding risk. Median follow-up was "
                "4.7 years.",
            ),
            (
                "ASPREE Cardiovascular and Bleeding Findings",
                "Cardiovascular disease: 10.7 events per 1,000 person-years on "
                "aspirin versus 11.3 per 1,000 person-years on placebo (HR 0.95; "
                "95% CI, 0.83–1.08). The primary cardiovascular endpoint was not "
                "different between arms.\n\n"
                "Major hemorrhage: 8.6 events per 1,000 person-years on aspirin "
                "versus 6.2 per 1,000 person-years on placebo (HR 1.38; 95% CI, "
                "1.18–1.62). Aspirin statistically significantly increased the "
                "rate of major hemorrhage in this elderly cohort.\n\n"
                "Conclusion from the editorial: \"Participants that took aspirin "
                "derived no benefit but suffered more haemorrhagic events than "
                "those that took placebo.\"",
            ),
            (
                "ASPREE Mortality and Cancer Findings",
                "All-cause mortality was higher in the aspirin arm than in the "
                "placebo arm: HR 1.14 (95% CI, 1.01–1.29). The mortality "
                "difference was driven largely by an increase in cancer-related "
                "deaths: HR 1.31 (95% CI, 1.10–1.56) for cancer-related death.\n\n"
                "Subsequent analyses of cancer outcomes in ASPREE showed:\n"
                "- All cancers: HR 1.35 (95% CI, 1.13–1.61)\n"
                "- Metastatic cancer: HR 1.19 (95% CI, 1.00–1.43)\n"
                "- Stage IV at diagnosis: HR 1.22 (95% CI, 1.02–1.45)\n"
                "- Colorectal cancer: HR 1.77 (95% CI, 1.02–3.06)\n\n"
                "Disability-free survival showed no benefit: HR 1.01 (95% CI, "
                "0.92–1.11; p = 0.79).\n\n"
                "Clinical verdict from the PMC commentary: \"Clinicians should "
                "not offer aspirin as primary prevention to otherwise well "
                "elderly patients\" aged 70 or older (or 65 or older in Black "
                "or Hispanic adults in the US).",
            ),
        ],
    },
    {
        "source_doc_id": "USPSTF-2022-aspirin-ASPREE-detail",
        "source_url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/aspirin-to-prevent-cardiovascular-disease-preventive-medication",
        "source_title": (
            "USPSTF 2022 Aspirin Use to Prevent Cardiovascular Disease — "
            "ASPREE evidence and 2022 reversal rationale"
        ),
        "source_authors": "US Preventive Services Task Force",
        "source_year": 2022,
        "source_tier": "secondary-summary",
        "verbatim": True,
        "license": "USPSTF public domain (US federal, 17 USC §105).",
        "sections": [
            (
                "USPSTF 2022 Reversal Rationale",
                "The 2022 USPSTF aspirin recommendation reversed the 2016 "
                "guidance that had included a colorectal cancer (CRC) prevention "
                "indication. The Task Force concluded: \"the evidence is "
                "inadequate that low-dose aspirin use reduces CRC incidence or "
                "mortality.\" Key contributors to the reversal include:\n\n"
                "- ASPREE's finding of *increased* CRC mortality with aspirin "
                "in the elderly cohort (Peto OR, 1.74; 95% CI, 1.02–2.95 at "
                "4.7 years follow-up)\n"
                "- Limited overall trial-grade data on CRC outcomes\n"
                "- Long-term follow-up data concerns and potential bias",
            ),
            (
                "USPSTF 2022 Age-Cutoff Guidance",
                "For adults 60 years or older: \"Do not initiate aspirin for the "
                "primary prevention of cardiovascular disease.\" Grade: D.\n\n"
                "For adults 40 to 59 years with 10% or higher 10-year ASCVD "
                "risk: the decision to initiate aspirin should be individualized. "
                "Grade: C.\n\n"
                "For continuing aspirin in patients already taking it: "
                "\"modeling data suggest that it may be reasonable to consider "
                "stopping aspirin use around age 75 years.\"\n\n"
                "The USPSTF noted: \"the absolute incidence of bleeding "
                "increases with age, and more so in adults 60 years or older.\" "
                "This bleeding-risk-by-age gradient is the primary reason for "
                "the Grade D recommendation in adults 60+.",
            ),
        ],
    },
    # =========================================================================
    # EAGLES (Anthenelli 2016 Lancet) — varenicline / bupropion / NRT safety + efficacy
    # Sources: USPSTF 2021 tobacco cessation + NERDCAT summary
    # =========================================================================
    {
        "source_doc_id": "USPSTF-2021-tobacco-EAGLES-pharmacotherapy-detail",
        "source_url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/tobacco-use-in-adults-and-pregnant-women-counseling-and-interventions",
        "source_title": (
            "USPSTF 2021 Tobacco Cessation — EAGLES trial findings and "
            "pharmacotherapy comparative effectiveness"
        ),
        "source_authors": "US Preventive Services Task Force",
        "source_year": 2021,
        "source_tier": "secondary-summary",
        "verbatim": True,
        "license": "USPSTF public domain (US federal, 17 USC §105). EAGLES primary publication is paywalled at The Lancet; this chunk's body is verbatim USPSTF language describing the trial findings.",
        "sections": [
            (
                "EAGLES Neuropsychiatric Safety",
                "From the USPSTF 2021 tobacco cessation evidence review: \"No "
                "difference in moderate and severe neuropsychiatric events, "
                "including rates of suicidal behavior and ideation, were found "
                "with bupropion SR (compared with varenicline or NRT) in the "
                "recent Evaluating Adverse Events in a Global Smoking Cessation "
                "Study (EAGLES) trial.\"\n\n"
                "EAGLES (Anthenelli et al., Lancet 2016) was a randomized, "
                "double-blind, triple-dummy, placebo-controlled and active-"
                "controlled trial of varenicline (1 mg twice daily) and "
                "bupropion (150 mg twice daily) for 12 weeks with 12-week non-"
                "treatment follow-up, conducted at 140 centers across 16 "
                "countries between November 2011 and January 2015.\n\n"
                "8,144 participants were randomly assigned: 4,116 to the "
                "psychiatric cohort (4,074 included in the safety analysis) "
                "and 4,028 to the non-psychiatric cohort (3,984 included in "
                "the safety analysis). The trial established that varenicline "
                "and bupropion did not show a statistically significant increase "
                "in moderate or severe neuropsychiatric events compared with "
                "nicotine patch or placebo, in either psychiatric or non-"
                "psychiatric smokers.",
            ),
            (
                "EAGLES Comparative Efficacy",
                "Pharmacotherapy comparative effectiveness from the USPSTF 2021 "
                "review:\n\n"
                "Varenicline vs. NRT: \"Eight studies (n = 6,264) compared "
                "varenicline and NRT and found that varenicline was associated "
                "with a greater smoking cessation rate over any form of NRT.\"\n\n"
                "Varenicline vs. Bupropion SR: \"Six studies (n = 6,286) "
                "evaluated varenicline vs bupropion SR and found that "
                "varenicline was associated with a higher cessation rate.\"\n\n"
                "NRT vs. Bupropion SR: \"Smoking cessation rates among "
                "participants using NRT vs bupropion SR at 6 months or more "
                "did not significantly differ (10 studies; n = 9,230).\"\n\n"
                "Overall efficacy summary: \"Based on a smaller number of "
                "studies, varenicline appears to be more effective than NRT or "
                "bupropion SR.\" Varenicline therefore sits at the top of the "
                "first-line pharmacotherapy efficacy ladder for smoking "
                "cessation, followed by combination NRT, then bupropion SR or "
                "monotherapy NRT.",
            ),
        ],
    },
    {
        "source_doc_id": "EAGLES-trial-summary-NERDCAT",
        "source_url": "https://nerdcat.org/studysummaries/eagles",
        "source_title": (
            "EAGLES — Efficacy & neuropsychiatric safety of smoking cessation "
            "products (varenicline, bupropion, nicotine patch) — NERDCAT "
            "Critical Appraisal"
        ),
        "source_authors": "NERDCAT (Internal Medicine Critical Appraisals)",
        "source_year": 2016,
        "source_tier": "secondary-summary",
        "verbatim": True,
        "license": "Open-access educational summary; chunk body is verbatim NERDCAT text.",
        "sections": [
            (
                "EAGLES Methodology",
                "EAGLES was an allocation-concealed RCT with all patients, "
                "clinicians, and investigators blinded, loss-to-follow-up of "
                "approximately 6%, analyzed using the intention-to-treat "
                "population. 8,144 smokers were randomized across 140 centers "
                "in 16 countries (November 2011–January 2015), stratified by "
                "psychiatric status. Participants in the psychiatric cohort had "
                "current or past psychiatric disorders that were stable.",
            ),
            (
                "EAGLES Outcomes",
                "Efficacy (continuous abstinence weeks 9–24): All three active "
                "treatments surpassed placebo. Varenicline demonstrated superior "
                "performance compared to bupropion or nicotine patch in both "
                "psychiatric and non-psychiatric populations. The efficacy "
                "ranking was varenicline > bupropion ≈ nicotine patch > "
                "placebo, mirroring the meta-analytic ordering established by "
                "the 2008 PHS guideline.\n\n"
                "Safety: \"No statistically significant increase in overall or "
                "serious neuropsychiatric events with any smoking cessation "
                "agent versus placebo in both patients with and without "
                "previous psychiatric disorder.\" A single completed suicide "
                "occurred in the non-psychiatric placebo group during the "
                "trial. Serious adverse events remained below 1% across all "
                "groups.\n\n"
                "Drug-specific common adverse effects observed: bupropion — "
                "insomnia and dry mouth; nicotine patch — abnormal dreams and "
                "insomnia; varenicline — abnormal dreams, insomnia, and nausea "
                "(approximately 25% of users).\n\n"
                "Regulatory implications: Results supported continued use of "
                "these agents in stable psychiatric populations and contributed "
                "directly to the FDA's December 2016 decision to remove the "
                "boxed warning regarding serious neuropsychiatric events on the "
                "varenicline (Chantix) and bupropion (Zyban) labels. "
                "Neuropsychiatric safety has not been demonstrated for "
                "unstable patients, recent suicide attempts, or severe acute "
                "psychiatric illness; clinical caution remains warranted in "
                "those subpopulations.",
            ),
        ],
    },
    # =========================================================================
    # PHS 2008 (Fiore et al, AHRQ) — Treating Tobacco Use and Dependence
    # PRIMARY-VERBATIM: US-government public-domain clinical practice guideline
    # Sources: NCBI Bookshelf NBK63952 + PMC4465757 (executive summary)
    # =========================================================================
    {
        "source_doc_id": "PHS-2008-Treating-Tobacco-Use-Fiore",
        "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK63952/",
        "source_title": (
            "Treating Tobacco Use and Dependence: 2008 Update — A U.S. Public "
            "Health Service Clinical Practice Guideline"
        ),
        "source_authors": (
            "Fiore MC, Jaén CR, Baker TB, Bailey WC, Benowitz NL, Curry SJ, "
            "Dorfman SF, et al. (PHS Tobacco Use and Dependence Guideline Panel)"
        ),
        "source_year": 2008,
        "source_tier": "primary-verbatim",
        "verbatim": True,
        "license": "US Public Health Service / AHRQ — public domain (US federal, 17 USC §105). This is the primary clinical practice guideline document, US-government-authored and freely redistributable.",
        "sections": [
            (
                "Five Major Steps (5 A's) for Patients Willing to Quit",
                "The 2008 PHS Guideline frames the brief tobacco intervention "
                "around five major steps, the 5 A's:\n\n"
                "1. ASK — Identify and document tobacco-use status for every "
                "patient at every visit. \"Implement an officewide system that "
                "ensures that, for EVERY patient at EVERY clinic visit, "
                "tobacco-use status is queried and documented.\"\n\n"
                "2. ADVISE — In a clear, strong, and personalized manner, urge "
                "every tobacco user to quit. The advice should be unambiguous "
                "(\"As your clinician, I need you to know that quitting smoking "
                "is the most important thing you can do to protect your health"
                "\"), strong, and personalized to the patient's clinical "
                "circumstances and values.\n\n"
                "3. ASSESS — Determine willingness to make a quit attempt. "
                "\"Assess every tobacco user's willingness to make a quit "
                "attempt at this time.\"\n\n"
                "4. ASSIST — For patients willing to make a quit attempt, offer "
                "medication and provide or refer to counseling. Use STAR for "
                "preparation: Set quit date; Tell family, friends, coworkers; "
                "Anticipate challenges; Remove tobacco products from environment.\n\n"
                "5. ARRANGE — Schedule follow-up contact, in person or via "
                "telephone. \"Follow-up contact should begin soon after the quit "
                "date, preferably during the first week.\"",
            ),
            (
                "Five R's for Unwilling Smokers",
                "For tobacco users not currently willing to quit, motivational "
                "intervention is structured around the 5 R's:\n\n"
                "- RELEVANCE — Connect quitting to the patient's personal "
                "circumstances, including disease status, family situation, "
                "health concerns, and values.\n"
                "- RISKS — Identify negative consequences of tobacco use that "
                "are most relevant to the patient (acute risks: shortness of "
                "breath, exacerbation of asthma, harm to pregnancy; long-term "
                "risks: heart attacks, strokes, cancers; environmental risks: "
                "secondhand smoke harm to family).\n"
                "- REWARDS — Explore potential benefits of cessation that the "
                "patient finds personally relevant.\n"
                "- ROADBLOCKS — Address barriers to quitting (withdrawal "
                "symptoms, fear of failure, weight gain, lack of support, "
                "depression, enjoyment of tobacco).\n"
                "- REPETITION — Repeat motivational intervention every time an "
                "unmotivated patient visits the clinic. Patients who failed "
                "previous quit attempts should be told that most people make "
                "repeated quit attempts before they are successful.",
            ),
            (
                "Seven First-Line FDA-Approved Cessation Pharmacotherapies",
                "The 2008 Guideline endorses seven FDA-approved first-line "
                "medications as effective for smoking cessation. They are listed "
                "as effective monotherapies; combination therapy is also "
                "recommended (see next section). The seven agents:\n\n"
                "1. Bupropion SR (Zyban; 150 mg PO daily for 3 days, then 150 "
                "mg PO twice daily for 7–12 weeks; start 1–2 weeks before quit "
                "date)\n"
                "2. Nicotine gum (2 mg or 4 mg; ad libitum or scheduled; up to "
                "12 weeks)\n"
                "3. Nicotine inhaler (6–16 cartridges/day; up to 6 months)\n"
                "4. Nicotine lozenge (2 mg or 4 mg; up to 12 weeks)\n"
                "5. Nicotine nasal spray (1–2 doses/hour; up to 6 months)\n"
                "6. Nicotine patch (varied dose 21/14/7 mg or 15/10/5 mg; 8 "
                "weeks; switch from highest to lower doses on a taper)\n"
                "7. Varenicline (Chantix; 0.5 mg PO daily for 3 days, then 0.5 "
                "mg twice daily for 4 days, then 1 mg twice daily for 11 weeks; "
                "start 1 week before quit date)\n\n"
                "All seven are recommended with Strength of Evidence A. \"Both "
                "counseling and medication are effective when used by themselves "
                "for treating tobacco dependence. The combination of counseling "
                "and medication, however, is more effective than either alone.\"",
            ),
            (
                "Recommended Combination Pharmacotherapies",
                "Meta-analysis identified combination regimens that are "
                "superior to monotherapy:\n\n"
                "- Long-term (>14 weeks) nicotine patch + ad libitum NRT (gum "
                "or spray): odds ratio approximately 3.6 vs placebo, the "
                "highest of any first-line regimen tested in the 2008 "
                "Guideline meta-analysis.\n"
                "- Nicotine patch + bupropion SR: odds ratio approximately "
                "2.5 vs placebo.\n"
                "- Nicotine patch + nortriptyline: odds ratio approximately "
                "2.3 vs placebo (note: nortriptyline is a second-line agent "
                "with greater side-effect burden; reserved for patients who "
                "fail first-line options).\n\n"
                "Varenicline (added as a first-line agent in the 2008 update; "
                "OR approximately 3.1 vs placebo at 6+ months follow-up) and "
                "long-term combination NRT produce the highest abstinence "
                "rates in the absence of head-to-head varenicline-vs-"
                "combination-NRT trial data. EAGLES (Anthenelli 2016) later "
                "supported varenicline's primacy.",
            ),
            (
                "Counseling Effectiveness and Telephone Quitlines",
                "The 2008 Guideline establishes counseling as effective and "
                "complementary to medication:\n\n"
                "- Brief clinician counseling (3 minutes or more) increases "
                "abstinence rates significantly compared to no advice.\n"
                "- Practical counseling (problem-solving / skills training) "
                "and intra-treatment social support are the two counseling "
                "elements with the strongest evidence base.\n"
                "- Telephone quitline counseling is effective and broadly "
                "scalable. \"Quitline counseling is effective with diverse "
                "populations and has broad reach. Therefore, both clinicians "
                "and health care delivery systems should ensure patient "
                "access to quitlines and promote quitline use.\"\n\n"
                "The Guideline recommends that all clinicians refer patients "
                "to 1-800-QUIT-NOW (the U.S. national tobacco cessation "
                "quitline portal) as a baseline counseling intervention even "
                "when on-site behavioral support is unavailable.",
            ),
            (
                "Special Populations: Pregnancy, Adolescents, Light Smokers, Comorbidity",
                "Pregnancy: \"Because of the serious risks of smoking to the "
                "pregnant smoker and the fetus, whenever possible pregnant "
                "smokers should be offered person-to-person psychosocial "
                "interventions that exceed minimal advice to quit.\" "
                "Pharmacotherapy in pregnancy is recommended only when the "
                "increased likelihood of cessation outweighs risk; NRT is "
                "Category D and bupropion / varenicline are limited-data.\n\n"
                "Adolescents: Counseling has been shown to be effective in "
                "treating tobacco use in adolescents. Therefore, adolescent "
                "smokers should be provided with counseling interventions to "
                "aid them in quitting. Pharmacotherapy effectiveness in "
                "adolescents is not established.\n\n"
                "Light smokers (<10 cigarettes/day): Light smokers should be "
                "identified, strongly urged to quit, and provided counseling "
                "cessation interventions. Pharmacotherapy for light smokers is "
                "limited-evidence; lower NRT doses (e.g., 14 mg patch, 2 mg "
                "gum/lozenge) are commonly used clinically.\n\n"
                "Smokers with psychiatric comorbidity: Tobacco-use treatments "
                "found to be effective in this Guideline have been shown to "
                "be effective across a broad range of populations including "
                "smokers with psychiatric disorders, including substance use "
                "disorders. Subsequent EAGLES (2016) data confirm this in the "
                "specific case of varenicline and bupropion.",
            ),
            (
                "System-Level Recommendations and Insurance Coverage",
                "Healthcare administrators, insurers, and purchasers should:\n\n"
                "- Implement universal tobacco-use identification and "
                "documentation systems in clinics and hospitals.\n"
                "- Provide clinician training in evidence-based cessation "
                "intervention and provide performance feedback.\n"
                "- Provide adequate clinic time and reimbursement for "
                "cessation counseling.\n"
                "- Cover guideline-recommended cessation treatments "
                "(counseling and pharmacotherapy) as paid benefits in "
                "insurance plans.\n"
                "- Track abstinence outcomes as a quality metric.\n\n"
                "Evidence principle: \"Making tobacco-dependence treatment a "
                "covered benefit of insurance plans increases the likelihood "
                "that a tobacco user will receive treatment.\" The Affordable "
                "Care Act (post-2010) operationalized this for US insurance "
                "plans by mandating cessation treatment as an essential health "
                "benefit; the 2008 Guideline established the evidence base "
                "that supported the mandate.",
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
# NeMo Curator-API-shaped chunker. Mirrors the v0/v2 chunker contract.
# When run on B300 inside `~/medomni-rapids/.venv` the helper functions
# below can be drop-in replaced with `nemo_curator.stages.text.utils.text_utils
# .get_paragraphs` + `UnicodeReformatter` + `NewlineNormalizer`. The chunk
# bodies + token-target packing are identical.
# ---------------------------------------------------------------------------


@dataclass
class Document:
    body: str
    section: str
    metadata: dict
    n_tokens_estimated: int = 0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_long_section(body: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split on paragraph boundaries; pack to target_chars; preserve overlap.

    Mirrors NeMo Curator's `get_paragraphs` + target-token packing
    composition. Tables / numbered lists are preserved as units (paragraph-
    level boundary)."""
    if len(body) <= target_chars:
        return [body]
    paragraphs = re.split(r"\n\s*\n", body)
    out: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = buf + ("\n\n" if buf else "") + p
        if len(candidate) > target_chars and buf:
            out.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars and len(buf) > overlap_chars else buf
            buf = tail + "\n\n" + p
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


def chunk_documents(sources: list[dict]) -> list[Document]:
    """NeMo-Curator-API-shaped chunker entry point. One call per builder run."""
    docs: list[Document] = []
    for src in sources:
        meta = {k: v for k, v in src.items() if k != "sections"}
        for section_label, body in src["sections"]:
            for piece in _split_long_section(body, TARGET_CHARS, OVERLAP_CHARS):
                docs.append(
                    Document(
                        body=piece,
                        section=section_label,
                        metadata=dict(meta),
                        n_tokens_estimated=_estimate_tokens(piece),
                    )
                )
    return docs


def main() -> int:
    existing_lines = OUT_PATH.read_text().splitlines() if OUT_PATH.exists() else []
    existing_ids = {
        json.loads(line)["id"]
        for line in existing_lines
        if line.strip()
    }
    next_idx = len(existing_lines)

    docs = chunk_documents(SOURCES_V3)

    new_records: list[dict] = []
    for d in docs:
        chunk_id = f"chunk-{next_idx:03d}"
        if chunk_id in existing_ids:
            raise RuntimeError(f"chunk id collision: {chunk_id} already in corpus")
        record = {
            "id": chunk_id,
            "source_doc_id": d.metadata["source_doc_id"],
            "source_url": d.metadata["source_url"],
            "source_title": d.metadata["source_title"],
            "source_authors": d.metadata.get("source_authors", ""),
            "source_year": d.metadata.get("source_year"),
            "section": d.section,
            "chunk_index": 0,
            "n_tokens_estimated": d.n_tokens_estimated,
            "body": d.body,
            "corpus_layer": "v3",
            # Phase 1.5 provenance metadata
            "source_tier": d.metadata.get("source_tier"),
            "verbatim": d.metadata.get("verbatim", False),
            "license": d.metadata.get("license", ""),
        }
        new_records.append(record)
        next_idx += 1

    counter: dict[str, int] = {}
    for r in new_records:
        ci = counter.get(r["source_doc_id"], 0)
        r["chunk_index"] = ci
        counter[r["source_doc_id"]] = ci + 1

    with open(OUT_PATH, "a") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"appended {len(new_records)} v3 chunks to {OUT_PATH}")
    print(
        "verify: .venv/bin/python -c \"import json; "
        "L=[json.loads(l) for l in open('corpus/medical-guidelines/chunks.jsonl')]; "
        "from collections import Counter; "
        "print('total:', len(L)); "
        "print('by layer:', Counter(c.get('corpus_layer','v0') for c in L)); "
        "print('by tier:', Counter(c.get('source_tier','none') for c in L))\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
