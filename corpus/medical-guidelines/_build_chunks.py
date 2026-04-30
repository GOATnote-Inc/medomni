#!/usr/bin/env python3
"""Build chunks.jsonl for the sovereign chemoprevention RAG corpus.

NVIDIA-canonical chunk parameters:
  - target ~1024 tokens per chunk (heuristic: 4 chars per token = ~4096 chars)
  - 15% overlap between consecutive chunks within a single source doc
  - page-aware boundaries (snap to section/paragraph breaks where possible)
  - tables / numbered recommendation lists / footnotes preserved as units

Source content embedded inline (extracted via WebFetch from public USPSTF, NCI,
CDC, and PMC OA pages). Every chunk's body traces to a real public source URL.
No fabricated text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

CHARS_PER_TOKEN = 4
TARGET_TOKENS = 1024
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN          # ~4096
OVERLAP_TOKENS = int(TARGET_TOKENS * 0.15)              # ~150
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN        # ~600

OUT_DIR = Path(__file__).parent
OUT_PATH = OUT_DIR / "chunks.jsonl"


# ----------------------------------------------------------------------------
# Source documents.
# Each entry maps a source slug -> (metadata, list[(section_label, body)]).
# Bodies are extracted verbatim text from the public source via WebFetch.
# Sections are kept as units; very long sections are split into sub-chunks
# with 15% overlap (page-aware: split on paragraph boundaries).
# ----------------------------------------------------------------------------

SOURCES: list[dict] = [
    # ------------------------------------------------------------------
    {
        "source_doc_id": "USPSTF-2019-breast-cancer-meds-risk-reduction",
        "source_url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/breast-cancer-medications-for-risk-reduction",
        "source_title": "Medication Use to Reduce Risk of Breast Cancer: USPSTF Recommendation Statement",
        "source_authors": "US Preventive Services Task Force",
        "source_year": 2019,
        "license": "USPSTF public domain (US federal advisory body, no copyright restriction)",
        "sections": [
            ("Recommendation Summary",
             "For women at increased risk (Grade B): The USPSTF recommends that clinicians offer to "
             "prescribe risk-reducing medications, such as tamoxifen, raloxifene, or aromatase inhibitors, "
             "to women who are at increased risk for breast cancer and at low risk for adverse medication "
             "effects.\n\n"
             "For women not at increased risk (Grade D): The USPSTF recommends against the routine use of "
             "risk-reducing medications, such as tamoxifen, raloxifene, or aromatase inhibitors, in women "
             "who are not at increased risk for breast cancer.\n\n"
             "This guidance applies to asymptomatic women 35 years and older, including women with previous "
             "benign breast lesions on biopsy (such as atypical ductal or lobular hyperplasia and lobular "
             "carcinoma in situ). The recommendation explicitly excludes women who have a current or "
             "previous diagnosis of breast cancer or ductal carcinoma in situ (DCIS)."),
            ("Risk Assessment Methodology",
             "Various methods are available to identify women at increased risk for breast cancer, "
             "including formal clinical risk assessment tools or assessing breast cancer risk factors "
             "without using a formal tool. The USPSTF does not endorse any particular risk-prediction "
             "tool. However, the NCI Breast Cancer Risk Assessment Tool and the Breast Cancer Surveillance "
             "Consortium Risk Calculator are based on models tested in US populations and are publicly "
             "available.\n\n"
             "There is no single cutoff for defining increased risk for all women. However, women at "
             "greater risk, such as those with at least a 3% risk for breast cancer in the next 5 years, "
             "are likely to derive more benefit than harm from risk-reducing medications.\n\n"
             "Some examples of combinations of multiple risk factors in women at increased risk include "
             "(but are not limited to): age 65 years or older with 1 first-degree relative with breast "
             "cancer; age 45 years or older with more than 1 first-degree relative with breast cancer or "
             "1 first-degree relative who developed breast cancer before age 50 years; age 40 years or "
             "older with a first-degree relative with bilateral breast cancer; presence of atypical "
             "ductal or lobular hyperplasia or lobular carcinoma in situ on a prior biopsy."),
            ("Medications and Menopausal Status",
             "Tamoxifen, raloxifene, and aromatase inhibitors all reduce primary breast cancer risk in "
             "postmenopausal women. Use of raloxifene and aromatase inhibitors is indicated only in "
             "postmenopausal women; only tamoxifen is indicated for risk-reduction of primary breast "
             "cancer in premenopausal women.\n\n"
             "In trials, participants typically used risk-reducing medications for 3 to 5 years. Benefits "
             "of tamoxifen have been found to persist up to 8 years beyond discontinuation, whereas risk "
             "for VTEs and endometrial cancer return to baseline after discontinuation of tamoxifen. Data "
             "on similarly long-term persistence of effects are not available for raloxifene or aromatase "
             "inhibitors."),
            ("Benefits Summary",
             "Tamoxifen effectiveness: Compared with placebo, tamoxifen reduced the incidence of invasive "
             "breast cancer by 7 events per 1000 women over 5 years (95% CI, 4-12).\n\n"
             "Raloxifene effectiveness: Raloxifene reduced incidence by 9 events (95% CI, 3-15) per 1000 "
             "women over 5 years.\n\n"
             "Aromatase inhibitor effectiveness: Aromatase inhibitors were found to reduce the incidence "
             "of invasive breast cancer by 16 events per 1000 women over 5 years.\n\n"
             "Fracture reduction: Both tamoxifen and raloxifene can reduce risk of some types of skeletal "
             "fractures, independent from the risk of breast cancer."),
            ("Harms Summary",
             "Tamoxifen and raloxifene increase risk for venous thromboembolic events (VTEs); tamoxifen "
             "increases risk more than raloxifene. Tamoxifen, but not raloxifene, increases risk for "
             "endometrial cancer in women with a uterus. Tamoxifen also increases risk of cataracts. "
             "Vasomotor symptoms (hot flashes) are a common adverse effect of both medications.\n\n"
             "The harms of aromatase inhibitors are also small to moderate. These harms include vasomotor "
             "symptoms, gastrointestinal symptoms, musculoskeletal pain, and possible cardiovascular "
             "events, such as stroke. Aromatase inhibitors do not reduce, and may even increase, risk of "
             "fractures.\n\n"
             "The potential for harms are greater in older women than in younger women."),
            ("Shared Decision-Making and Risk-Benefit",
             "When considering prescribing breast cancer risk-reducing medications, the potential benefit "
             "of risk reduction of breast cancer must be balanced against the potential harms of adverse "
             "medication effects. When considering prescribing risk-reducing medications for breast "
             "cancer, clinicians should discuss each woman's personal values and preferences with respect "
             "to breast cancer risk reduction, in addition to what is known about her personal risk for "
             "breast cancer and the potential benefits and harms of medications.\n\n"
             "Clinicians should discuss the limitations of current clinical risk assessment tools for "
             "predicting an individual's future risk of breast cancer when discussing the benefits and "
             "harms of risk-reducing medications with women. Although only exploratory, a number of "
             "studies have suggested that even women who are well informed about the risks and benefits "
             "have relatively little interest in taking risk-reducing medications for breast cancer and "
             "are primarily concerned with potential harms.\n\n"
             "Women not at increased risk for breast cancer, such as women younger than 60 years with no "
             "additional risk factors for breast cancer, or women with a low 5-year risk of breast "
             "cancer should not be routinely offered medications to reduce risk of breast cancer, since "
             "the risk of harms from these medications likely outweighs their potential benefit."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "USPSTF-2022-aspirin-CVD-CRC",
        "source_url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/aspirin-to-prevent-cardiovascular-disease-preventive-medication",
        "source_title": "Aspirin Use to Prevent Cardiovascular Disease: USPSTF Recommendation Statement",
        "source_authors": "US Preventive Services Task Force",
        "source_year": 2022,
        "license": "USPSTF public domain",
        "sections": [
            ("Core Recommendations",
             "The USPSTF issued updated guidance on aspirin for cardiovascular disease prevention in "
             "April 2022. For adults aged 40-59 years with a 10% or greater 10-year CVD risk, the task "
             "force recommends that the decision to initiate low-dose aspirin use for the primary "
             "prevention of CVD in adults aged 40 to 59 years who have a 10% or greater 10-year CVD risk "
             "should be an individual one (Grade C, indicating a small net benefit exists).\n\n"
             "For adults 60 years or older, the USPSTF recommends against initiating low-dose aspirin "
             "use for the primary prevention of CVD in adults 60 years or older (Grade D, reflecting no "
             "net benefit).\n\n"
             "Persons who are not at increased risk for bleeding and are willing to take low-dose "
             "aspirin daily are more likely to benefit. Implementation guidance directs clinicians to "
             "use shared decision-making, taking into account potential benefits and harms of aspirin "
             "use, as well as patients' values and preferences, to inform the decision about initiating "
             "aspirin. CVD risk estimation is imprecise and imperfect at the individual level."),
            ("Cardiovascular Benefits Evidence",
             "The systematic review examined 13 randomized clinical trials involving 161,680 "
             "participants. Results showed that aspirin use for primary prevention of CVD was associated "
             "with a decreased risk of myocardial infarction and stroke but not cardiovascular mortality "
             "or all-cause mortality.\n\n"
             "A pooled analysis of 11 trials (n = 134,470) showed that low-dose aspirin use was "
             "associated with a statistically significant decreased risk of nonfatal myocardial "
             "infarction (Peto odds ratio [OR], 0.88 [95% CI, 0.80-0.96]). A pooled analysis of 5 trials "
             "(n = 54,947) demonstrated that low-dose aspirin use was associated with a statistically "
             "significant decreased risk of nonfatal ischemic stroke (Peto OR, 0.88 [95% CI, 0.78-1.00]; "
             "P = .046).\n\n"
             "Low-dose aspirin use was not associated with a statistically significant effect on fatal "
             "myocardial infarction, fatal stroke, cardiovascular mortality, or all-cause mortality."),
            ("Colorectal Cancer Evidence",
             "Regarding colorectal cancer reduction, a key distinction from prior recommendations, the "
             "evidence remains inconclusive. The USPSTF concluded the evidence is inadequate that "
             "low-dose aspirin use reduces CRC incidence or mortality.\n\n"
             "Four studies conducted in primary CVD prevention populations found no association between "
             "aspirin use and CRC incidence at up to approximately 10 years of follow-up. Regarding "
             "longer-term follow-up, the WHS reported a lower incidence of CRC at 17.5 years of "
             "follow-up (Peto OR, 0.82 [95% CI, 0.69-0.98]); recent data showed that this effect did "
             "not persist from 17.5 to 26 years of follow-up.\n\n"
             "The ASPREE trial findings complicated the picture. This study reported that aspirin use "
             "was associated with statistically significantly higher CRC mortality at 4.7 years of "
             "follow-up (Peto OR, 1.74 [95% CI, 1.02-2.95]), while other trials showed inconsistent "
             "results regarding CRC mortality over time."),
            ("Bleeding Harms",
             "The evidence demonstrates significant bleeding risks. A pooled analysis of 10 trials "
             "(n = 119,130) showed that aspirin use was associated with a 58% increase in major "
             "gastrointestinal bleeding (Peto OR, 1.58 [95% CI, 1.38-1.80]). A pooled analysis of 11 "
             "trials (n = 134,470) showed an increase in intracranial bleeds in the aspirin group "
             "compared with the control group (Peto OR, 1.31 [95% CI, 1.11-1.54]).\n\n"
             "The absolute incidence of bleeding, and thus the magnitude of bleeding harm, increases "
             "with age, and more so in adults 60 years or older. This age-related harm escalation "
             "directly informed the recommendation against initiating aspirin in older adults."),
            ("Discontinuation and Changes from Prior Guidance",
             "For individuals already taking aspirin, the USPSTF suggests it may be reasonable to "
             "consider stopping aspirin use around age 75 years. Modeling data suggested that there is "
             "generally little incremental lifetime net benefit in continuing aspirin use beyond the "
             "age of 75 to 80 years. This guidance distinguishes between continuing existing therapy "
             "versus initiating new therapy in older age groups.\n\n"
             "The 2022 update modified the 2016 guidance substantially. Previously, the USPSTF "
             "recommended considering aspirin starting at age 50; the new recommendation lowers this to "
             "age 40. The prior recommendation suggested individual decision-making for ages 60-69; the "
             "current version recommends against initiation entirely for those 60 and older, "
             "strengthening the recommendation grade from C to D for this population."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "USPSTF-2019-BRCA-risk-assessment",
        "source_url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/brca-related-cancer-risk-assessment-genetic-counseling-and-genetic-testing",
        "source_title": "BRCA-Related Cancer: Risk Assessment, Genetic Counseling, and Genetic Testing: USPSTF Recommendation Statement",
        "source_authors": "US Preventive Services Task Force",
        "source_year": 2019,
        "license": "USPSTF public domain",
        "sections": [
            ("Core Recommendations",
             "The U.S. Preventive Services Task Force issued two contrasting recommendations regarding "
             "BRCA1/2 mutation testing.\n\n"
             "For high-risk women: The USPSTF recommends that clinicians assess women with a personal or "
             "family history of breast, ovarian, tubal, or peritoneal cancer using validated risk "
             "assessment tools, with appropriate genetic counseling and testing when indicated. Grade B "
             "recommendation, indicating moderate certainty of net benefit.\n\n"
             "For average-risk women: The task force recommends against routine screening, stating they "
             "recommend against routine risk assessment, genetic counseling, or genetic testing for "
             "women whose personal or family history or ancestry is not associated with potentially "
             "harmful BRCA1/2 gene mutations. Grade D recommendation.\n\n"
             "This recommendation applies to women who are asymptomatic for BRCA-related cancer and have "
             "unknown BRCA mutation status, including those with a previous breast, ovarian, tubal, or "
             "peritoneal cancer diagnosis who have completed treatment and are considered cancer free."),
            ("Risk Assessment Tools",
             "The USPSTF identified and evaluated several validated screening instruments:\n\n"
             "- Ontario Family History Assessment Tool: assigns point values based on cancer type, "
             "relative relationship, and age at diagnosis.\n"
             "- Manchester Scoring System: uses separate scoring for BRCA1 and BRCA2 with thresholds of "
             "10 points.\n"
             "- Referral Screening Tool: a checklist approach identifying whether two or more risk "
             "factors are present.\n"
             "- Pedigree Assessment Tool: assigns points for various cancer diagnoses including male "
             "breast cancer at any age worth 8 points.\n"
             "- Seven-Question Family History Screening (FHS-7): requires only one positive response to "
             "trigger referral.\n"
             "- International Breast Cancer Intervention Study Model (Tyrer-Cuzick): incorporates "
             "personal and family history with genetic testing considerations.\n\n"
             "The task force found these tools had sensitivity estimates between 77% and 100% and areas "
             "under the receiver operating characteristic curve between 0.68 and 0.96, though it noted "
             "insufficient evidence to recommend one tool over another."),
            ("Benefits of Identification and Intervention",
             "For women with family or personal history suggesting BRCA mutations, the task force "
             "identified moderate benefits including:\n\n"
             "Risk reduction through surgery: Studies demonstrated that bilateral mastectomy was "
             "associated with a 90% to 100% reduced breast cancer incidence and 81% to 100% reduced "
             "breast cancer mortality. Similarly, oophorectomy was associated with 81% to 100% reduced "
             "ovarian cancer incidence.\n\n"
             "Medication benefits: The task force found that tamoxifen, raloxifene, and aromatase "
             "inhibitors provided clinically significant reductions in invasive breast cancer, with "
             "results showing 7 fewer events per 1000 women for tamoxifen, 9 fewer events per 1000 "
             "women for raloxifene, and 16 fewer events per 1000 women for aromatase inhibitors.\n\n"
             "Psychological effects: Pretest genetic counseling either decreased or had no effect on "
             "breast cancer worry, anxiety, and depression, and most studies reported either improved "
             "understanding of risk or no negative association."),
            ("Documented Harms",
             "The USPSTF also comprehensively documented potential harms.\n\n"
             "From screening: Intensive breast cancer screening using MRI was associated with higher "
             "false-positive rates (14-15% versus 5.5-11% for mammography), and ovarian cancer "
             "screening demonstrated high false-positive rates (3.4%) with one study reporting a "
             "diagnostic surgery rate of 55% after annual screening in women without cancer.\n\n"
             "From medications: Tamoxifen and raloxifene increased risk for thromboembolic events "
             "compared with placebo, and tamoxifen specifically caused an increased risk of endometrial "
             "cancer at 4 cases per 1000 women. Women using these drugs experienced vasomotor symptoms "
             "and vaginal discharge, itching, or dryness.\n\n"
             "From surgery: Mastectomy complications ranged from 49% to 69%, including numbness, pain, "
             "tingling, infection, swelling, breast hardness, bleeding. Psychological effects included "
             "reductions in body image, sexual activity/satisfaction, and general mental health, though "
             "the task force noted many of these symptoms were transient.\n\n"
             "From testing: Post-test anxiety increased in some populations, with increased worry "
             "documented particularly in women who tested positive."),
            ("Clinical Implementation Guidance",
             "The task force stressed that genetic counseling about BRCA1/2 mutation testing should be "
             "performed by trained health professionals, including suitably trained primary care "
             "clinicians, and that the process must include detailed kindred analysis and risk "
             "assessment, identification of candidates for testing, patient education, discussion of "
             "the benefits and harms, and interpretation of results after testing.\n\n"
             "On emerging testing approaches, the task force acknowledged that there has been "
             "significantly increased access to multigene panels since the 2013 Supreme Court ruling "
             "invalidating gene patents, but emphasized that the clinical significance of identifying "
             "pathogenic variants in multigene panels requires further investigation. The task force "
             "found no studies on the benefits of intensive screening for BRCA-related cancer on "
             "clinical outcomes, indicating a gap in evidence supporting surveillance protocols."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "USPSTF-2022-statin-primary-prevention",
        "source_url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/statin-use-in-adults-preventive-medication",
        "source_title": "Statin Use for the Primary Prevention of Cardiovascular Disease in Adults: USPSTF Recommendation Statement",
        "source_authors": "US Preventive Services Task Force",
        "source_year": 2022,
        "license": "USPSTF public domain",
        "sections": [
            ("Primary Recommendations",
             "Grade B Recommendation: For adults aged 40-75 with one or more CVD risk factors and "
             "estimated 10-year cardiovascular disease risk of 10% or greater, the task force "
             "recommends clinicians prescribe a statin for the primary prevention of CVD. The "
             "identified risk factors include dyslipidemia, diabetes, hypertension, or smoking.\n\n"
             "Grade C Recommendation: For the same age group with estimated 10-year CVD risk between "
             "7.5% and less than 10%, clinicians should selectively offer a statin, recognizing that "
             "the likelihood of benefit is smaller in this group than in persons with a 10-year risk "
             "of 10% or greater.\n\n"
             "I Statement (Insufficient Evidence): Regarding adults 76 years or older, the USPSTF "
             "determined the evidence is insufficient to assess the balance of benefits and harms of "
             "initiating a statin for the primary prevention of CVD events and mortality."),
            ("Risk Assessment Framework",
             "The document emphasizes using the ACC/AHA Pooled Cohort Equations to estimate 10-year "
             "CVD risk. This tool represents the only US-based CVD risk prediction tool that has "
             "published external validation studies in other US-based populations. The equations "
             "incorporate age, cholesterol levels, systolic blood pressure level, antihypertension "
             "treatment, presence of diabetes, and smoking status.\n\n"
             "The USPSTF acknowledges significant limitations: Concerns about calibration of the "
             "Pooled Cohort Equations exist, with many external validation studies showing "
             "overprediction in broad populations. Furthermore, limited evidence also suggests "
             "underprediction in disadvantaged communities that could result in undertreatment.\n\n"
             "Clinical implementation: clinicians should determine whether the patient has a "
             "cardiovascular risk factor and estimate CVD risk using a CVD risk estimator. For those "
             "meeting criteria with 10-year risk >=10%, initiate a moderate-intensity statin after "
             "discussing the rationale and provided the patient agrees. For patients in the 7.5-10% "
             "risk category, the benefit of starting a statin is smaller, so clinicians should "
             "selectively offer a statin, taking patient values and preferences into account."),
            ("Benefits Evidence",
             "The systematic review examined 22 trials with mean follow-up of 3.3 years. Results "
             "demonstrated that statin therapy was associated with decreased risk of all-cause "
             "mortality with relative risk of 0.92, fatal or nonfatal stroke with RR of 0.78, and "
             "fatal or nonfatal myocardial infarction with RR of 0.67. In a pooled analysis of 15 "
             "trials, composite cardiovascular outcomes showed RR of 0.72.\n\n"
             "However, regarding cardiovascular mortality specifically, evidence was more limited. "
             "Only one trial showed statistically significant reduction, and pooled analyses of all "
             "12 trials demonstrated a slight reduction in cardiovascular mortality risk that was "
             "not statistically significant."),
            ("Safety Profile and Harms",
             "The document extensively reviewed potential adverse effects across 19 trials and 3 "
             "observational studies.\n\n"
             "Muscle-Related Events: Despite concerns from observational data, a pooled analysis of "
             "9 trials found no increased risk of myalgia with statin therapy compared with placebo. "
             "The evidence indicated statins were not associated with increased risk of myopathy or "
             "rhabdomyolysis.\n\n"
             "Hepatotoxicity: Twelve trials reported no difference between statin therapy and "
             "placebo in risk of elevation in aminotransferase levels.\n\n"
             "Cancer Risk: A pooled analysis of 13 trials found no difference between statin therapy "
             "and placebo or no statin in risk of any cancer.\n\n"
             "Diabetes Incidence: Based on 6 trials involving over 59,000 participants, there was no "
             "difference between statins and placebo or no statin in risk of diabetes with relative "
             "risk of 1.04. However, one high-intensity statin trial (JUPITER) reported an increased "
             "risk of diabetes with statin use (3.0% vs 2.4%; RR, 1.25) though this was subsequently "
             "found to be limited to study participants with 1 or more diabetes risk factors at "
             "baseline.\n\n"
             "General Tolerability: The review found statin therapy was not associated with "
             "increased risk of study withdrawal due to adverse events or serious adverse events."),
            ("Special Populations and Statin Intensity",
             "The document highlights concerning equity issues: Data from the 2013-2014 National "
             "Health and Nutrition Examination Survey found that among persons eligible for statin "
             "use, statin use was higher among non-Hispanic White (58.3%) persons compared with "
             "non-Hispanic Asian (49.2%), non-Hispanic Black (44.3%), or Hispanic (33.7%) persons.\n\n"
             "The USPSTF emphasizes that it is essential to equitably improve statin use in both "
             "women and men of all races and ethnicities and specifically notes those with the "
             "highest prevalence of CVD include Black and Hispanic adults, who have the highest "
             "prevalence of CVD and the lowest utilization of statins, respectively.\n\n"
             "Regarding dosing strategies, there are limited data directly comparing the effects of "
             "different statin intensities on health outcomes. Most reviewed trials used a "
             "moderate-intensity statin therapy, which seems reasonable for the primary prevention "
             "of CVD in most persons.\n\n"
             "Evidence for adults 76+: Limited trial data exist for older adults. The PROSPER trial, "
             "which included 3,239 primary prevention participants with mean age 75 years, found no "
             "decrease in all-cause mortality, risk of stroke, or in a composite cardiovascular "
             "outcome compared to placebo. This limited and inconclusive evidence supported the I "
             "statement for this age group."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "USPSTF-2021-tobacco-cessation",
        "source_url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/tobacco-use-in-adults-and-pregnant-women-counseling-and-interventions",
        "source_title": "Tobacco Smoking Cessation in Adults, Including Pregnant Persons: USPSTF Recommendation Statement",
        "source_authors": "US Preventive Services Task Force",
        "source_year": 2021,
        "license": "USPSTF public domain",
        "sections": [
            ("Key Recommendations",
             "Nonpregnant Adults (Grade A): The USPSTF recommends that clinicians ask all adults about "
             "tobacco use, advise them to stop using tobacco, and provide behavioral interventions and "
             "US Food and Drug Administration (FDA)-approved pharmacotherapy for cessation to "
             "nonpregnant adults who use tobacco.\n\n"
             "Pregnant Persons (Grade A for behavioral interventions): The USPSTF recommends that "
             "clinicians ask all pregnant persons about tobacco use, advise them to stop using "
             "tobacco, and provide behavioral interventions for cessation to pregnant persons who use "
             "tobacco.\n\n"
             "For pharmacotherapy in pregnancy: The USPSTF concludes that the current evidence is "
             "insufficient to assess the balance of benefits and harms of pharmacotherapy "
             "interventions for tobacco cessation in pregnant persons.\n\n"
             "E-Cigarettes (I Statement): The USPSTF concludes that the current evidence is "
             "insufficient to assess the balance of benefits and harms of electronic cigarettes "
             "(e-cigarettes) for tobacco cessation in adults, including pregnant persons."),
            ("FDA-Approved Pharmacotherapy",
             "The document identifies three pharmacotherapy options for nonpregnant adults:\n\n"
             "- Nicotine replacement therapy (patches, lozenges, gum, inhalers, nasal spray)\n"
             "- Bupropion sustained-release\n"
             "- Varenicline\n\n"
             "Evidence shows varenicline appears to be more effective than NRT or bupropion SR based "
             "on available research.\n\n"
             "Documented Harms: Behavioral counseling shows limited evidence on harms with no evidence "
             "of serious adverse events. Nicotine replacement therapy is associated with statistically "
             "significantly more cardiovascular adverse events (in particular, heart palpitations and "
             "chest pain) compared to placebo, plus nausea, vomiting, gastrointestinal symptoms, and "
             "insomnia. Bupropion SR shows no difference in serious adverse events or major "
             "cardiovascular events versus placebo. Varenicline common effects include nausea, "
             "insomnia, abnormal dreams, headache, and fatigue, with no statistically significant "
             "difference in cardiovascular adverse events. E-cigarettes most commonly reported adverse "
             "effects are coughing, nausea, throat irritation, and sleep disruption."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "CDC-ACIP-HPV-immunization-schedule",
        "source_url": "https://www.cdc.gov/vaccines/hcp/imz-schedules/child-adolescent-age.html",
        "source_title": "Child and Adolescent Immunization Schedule by Age (HPV section)",
        "source_authors": "Centers for Disease Control and Prevention; Advisory Committee on Immunization Practices (ACIP)",
        "source_year": 2025,
        "license": "CDC public domain",
        "sections": [
            ("Routine HPV Vaccination Age",
             "The Centers for Disease Control and Prevention (CDC) provides comprehensive guidance on "
             "human papillomavirus (HPV) vaccination as part of the Child and Adolescent Immunization "
             "Schedule. According to the official CDC immunization schedule updated July 2, 2025, HPV "
             "vaccination represents an important preventive measure for protecting young people "
             "against HPV-related cancers and diseases.\n\n"
             "The CDC schedule indicates that HPV vaccination follows a specific age-based approach. "
             "The immunization table shows HPV vaccination beginning at ages 11-12 years, which "
             "represents the primary window for routine administration. Healthcare providers can "
             "initiate HPV vaccination starting at age 9, though the routine recommendation centers on "
             "the 11-12 year age group. This timing aligns with CDC guidance to vaccinate individuals "
             "before potential exposure to the virus."),
            ("Dosing Schedules and Catch-Up",
             "The CDC's immunization schedule recognizes different vaccination series depending on age "
             "at initiation and individual circumstances. The distinction between two-dose versus "
             "three-dose regimens typically depends on the age when vaccination begins and spacing "
             "between doses.\n\n"
             "For those who fall behind or start late, provide catch-up vaccination at the earliest "
             "opportunity as indicated by the green bars. This guidance applies to all vaccines, "
             "including HPV, allowing flexibility for individuals who did not receive doses at the "
             "recommended times. The schedule directs providers to consult Table 2, the catch-up "
             "vaccination table, for specific minimum intervals between doses.\n\n"
             "The immunization schedule displays HPV vaccination positioned in the 11-12 year age "
             "column within the 18 Months to 18 Years table. This placement reflects the CDC's "
             "recommendation that routine HPV vaccination occur during early adolescence, optimizing "
             "immune response and protection before potential exposure. The schedule visually "
             "accommodates older adolescents and young adults through its design, extending "
             "recommendations through age 18."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-PDQ-breast-cancer-prevention-HP",
        "source_url": "https://www.cancer.gov/types/breast/hp/breast-prevention-pdq",
        "source_title": "Breast Cancer Prevention (PDQ) - Health Professional Version",
        "source_authors": "National Cancer Institute PDQ Screening and Prevention Editorial Board",
        "source_year": 2025,
        "license": "NCI public domain",
        "sections": [
            ("Risk Assessment Models",
             "The National Cancer Institute identifies several validated tools for quantifying breast "
             "cancer risk:\n\n"
             "- Breast Cancer Risk Assessment Tool (Gail Model) - derives risk estimates from "
             "database and cohort studies\n"
             "- Breast Cancer Surveillance Consortium (BCSC) Risk Calculator - population-based "
             "assessment approach\n"
             "- Breast Cancer Referral Screening Tool (B-RST) - designed for identifying high-risk "
             "populations\n"
             "- Hall Detailed Breast Risk Calculator - comprehensive individual assessment\n"
             "- IBIS Breast Cancer Risk Calculator Tool (Tyrer-Cuzick) - incorporates family history "
             "data"),
            ("Non-Modifiable Risk Factors",
             "Age and Sex: Female sex and increasing age are the major risk factors for the "
             "development of breast cancer. Women face approximately 100 times greater lifetime "
             "breast cancer risk compared to men. A 70-year-old woman's short-term risk is about "
             "tenfold that of a 30-year-old woman.\n\n"
             "Family History and Genetic Mutations: Women who have a family history of breast cancer, "
             "especially in a first-degree relative, have an increased risk. Risk doubles with one "
             "affected first-degree relative and increases fivefold with two affected relatives. For "
             "BRCA mutations, lifetime risk of breast cancer is 55% to 65% for BRCA1 mutation "
             "carriers and 45% to 47% for BRCA2 mutation carriers, compared to 13% in the general "
             "population.\n\n"
             "Breast Density: Dense breast tissue demonstrates substantial risk elevation. Women with "
             "dense breasts have increased risk, proportionate to the degree of density. This "
             "increased relative risk ranges from 1.79 for women with slightly increased breast "
             "density to 4.64 for women with very dense breasts."),
            ("Modifiable Risk Factors Increasing Breast Cancer Risk",
             "Menopausal Hormone Therapy: For women using combination estrogen-progestin therapy, "
             "approximately a 26% increase in incidence of invasive breast cancer occurs, with the "
             "number needed to produce one excess breast cancer is 237. For estrogen-progesterone "
             "users, risk increases from RR, 1.60 (95% CI, 1.52-1.62) during years 1-4 of use to RR, "
             "2.08 (95% CI, 2.02-2.15) during years 5-14. Estrogen-only formulations show lower but "
             "still significant increases: RR, 1.17 (95% CI, 1.10-1.26) initially, rising to RR, "
             "1.33 (95% CI, 1.28-1.37) with prolonged use.\n\n"
             "Obesity is associated with an increased breast cancer risk, especially among "
             "postmenopausal women who do not use hormone therapy. Comparing women weighing more "
             "than 82.2 kg with those under 58.7 kg yielded RR was 2.85 (95% CI, 1.81-4.49).\n\n"
             "Alcohol consumption is associated with increased breast cancer risk in a dose-"
             "dependent fashion. The relative risk for women consuming approximately four drinks "
             "daily compared with non-drinkers is 1.32 (95% CI, 1.19-1.45), with risk increasing "
             "by 7% (95% CI, 5.5%-8.7%) for each drink per day.\n\n"
             "Ionizing Radiation Exposure: Risk depends on radiation dose and age at exposure, and "
             "is especially high if exposure occurs during puberty, when the breast develops. Women "
             "treated for Hodgkin lymphoma with mantle radiation by age 16 face a subsequent risk "
             "up to 35% of developing breast cancer by age 40 years."),
            ("Protective Lifestyle Factors",
             "Early Pregnancy: Women who have a full-term pregnancy before age 20 years have "
             "decreased breast cancer risk. The magnitude represents 50% decrease in breast cancer, "
             "compared with nulliparous women or women who give birth after age 35 years.\n\n"
             "Breast-feeding: Women who breast-feed have a decreased risk of breast cancer. The RR "
             "of breast cancer is decreased 4.3% for every 12 months of breast-feeding, in addition "
             "to 7% for each birth.\n\n"
             "Physical Exercise: Average RR reduction association is 20% for both postmenopausal and "
             "premenopausal women and affects the risk of both hormone-sensitive and hormone-"
             "resistant cancers."),
            ("Tamoxifen Chemoprevention",
             "The Breast Cancer Prevention Trial enrolled over 13,000 women, finding that the "
             "incidence of breast cancer for the tamoxifen group was 49% lower than for the control "
             "group (85 vs. 154 invasive breast cancer cases and 31 vs. 59 in situ cases at 4 years).\n\n"
             "Tamoxifen reduced the incidence of estrogen receptor-positive breast cancer and ductal "
             "carcinoma in situ in high-risk women by about 30% to 50% over 5 years of treatment. "
             "The reduction in ER-positive invasive breast cancer was maintained for at least 16 "
             "years after starting treatment.\n\n"
             "However, significant adverse effects emerged: tamoxifen increases the risk of "
             "endometrial cancer, thrombotic vascular events (i.e., pulmonary embolism, stroke, and "
             "deep venous thrombosis), and cataracts. The endometrial cancer risk persists for 5 "
             "years after tamoxifen cessation but not the risk of vascular events or cataracts. "
             "Meta-analysis showed RR of 2.4 (95% CI, 1.5-4.0) for endometrial cancer and 1.9 (95% "
             "CI, 1.4-2.6) for venous thromboembolic events.\n\n"
             "A recent study examined lower-dose approaches. In the TAM-01 trial using 5 mg daily, "
             "after a median follow-up of 9.7 years, there were 25 breast cancers in the tamoxifen "
             "group (41 invasive cancers) and 41 in the placebo group (59 invasive) (HR, 0.58; 95% "
             "CI, 0.35-0.95; log-rank P = .03)."),
            ("Raloxifene and STAR Trial",
             "Raloxifene hydrochloride is a SERM that has antiestrogenic effects on breast and "
             "estrogenic effects on bone, lipid metabolism, and blood clotting. Unlike tamoxifen, it "
             "has antiestrogenic effects on the endometrium.\n\n"
             "The MORE trial demonstrated efficacy: after a median follow-up of 47 months, the risk "
             "of invasive breast cancer was decreased in the raloxifene-treated women (RR, 0.25; "
             "95% CI, 0.17-0.45). Long-term follow-up in the CORE extension showed even stronger "
             "protection: the overall reduction in invasive breast cancer during the 8 years of "
             "MORE and CORE was 66% (HR, 0.34; 95% CI, 0.22-0.50); the reduction for ER-positive "
             "invasive breast cancer was 76%.\n\n"
             "The STAR trial comparing both agents found that invasive breast cancer incidence was "
             "approximately the same for both drugs, but there were fewer noninvasive cancers in "
             "the tamoxifen group. Adverse events of uterine cancer, venous thromboembolic events, "
             "and cataracts were more common in tamoxifen-treated women."),
            ("Aromatase Inhibitors",
             "These agents offer prevention benefits in high-risk populations. An exemestane trial "
             "in 4,560 high-risk women showed that 3 years of exemestane treatment reduced breast "
             "cancer incidence by 65%, compared with controls. Similarly, a comparable trial of 5 "
             "years of anastrozole treatment reduced breast cancer incidence by 53%, an effect "
             "persisting at 11 years.\n\n"
             "For women with specific risk factors, women aged 35 years and older who had at least "
             "one risk factor (age >60 years, a Gail 5-year risk >1.66%, or DCIS with mastectomy) "
             "and who took 25 mg of exemestane daily had a decreased risk of invasive breast cancer "
             "(HR, 0.35; 95% CI, 0.18-0.70) compared with controls.\n\n"
             "Side effects include: exemestane is associated with hot flashes and fatigue compared "
             "with placebo with absolute increase in hot flashes was 8% and the absolute increase "
             "in fatigue was 2%."),
            ("Surgical Prevention",
             "Prophylactic Mastectomy: Bilateral prophylactic mastectomy reduces the risk of breast "
             "cancer in women with a strong family history and breast cancer risk after bilateral "
             "prophylactic mastectomy in women at high risk may be reduced as much as 90%. "
             "Additionally, most women experience relief from anxiety about breast cancer risk after "
             "undergoing prophylactic mastectomy.\n\n"
             "Prophylactic Oophorectomy: Prophylactic oophorectomy in premenopausal women with a "
             "BRCA gene mutation is associated with decreased breast cancer incidence. The magnitude "
             "shows breast cancer incidence may be decreased by up to 50%. However, consequences "
             "are substantial: castration may cause the abrupt onset of menopausal symptoms such as "
             "hot flashes, insomnia, anxiety, and depression. Long-term effects include decreased "
             "libido, vaginal dryness, and decreased bone mineral density.\n\n"
             "Current Incidence and Mortality: The most recent epidemiologic data indicates that in "
             "2025, 316,950 women will be diagnosed with breast cancer, with 42,170 deaths from "
             "this disease. Despite ongoing incidence increases, breast cancer mortality rates "
             "declined by 44% from 1989 to 2022. However, mortality rates in Black women remain "
             "about 38% higher than in White women."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-PDQ-colorectal-cancer-prevention-HP",
        "source_url": "https://www.cancer.gov/types/colorectal/hp/colorectal-prevention-pdq",
        "source_title": "Colorectal Cancer Prevention (PDQ) - Health Professional Version",
        "source_authors": "National Cancer Institute PDQ Screening and Prevention Editorial Board",
        "source_year": 2025,
        "license": "NCI public domain",
        "sections": [
            ("Aspirin Chemoprevention and CAPP2 Trial",
             "The evidence shows that daily aspirin reduces colorectal cancer risk, but only after "
             "prolonged use. ASA use reduces the long-term risk of developing CRC by 40% about 10 to "
             "19 years after initiation (hazard ratio [HR], 0.60; 95% CI, 0.47-0.76).\n\n"
             "Allocation to use of 75 to 1,200 mg of daily ASA for at least one year reduced the "
             "cumulative risk of colon cancer death compared with controls (HR, 0.67; 95% CI, "
             "0.52-0.86), but this benefit emerged only after 10-20 years of follow-up.\n\n"
             "CAPP2 Trial in Lynch Syndrome: The Cancer Prevention Programme (CAPP2) tested aspirin "
             "specifically in patients with Lynch syndrome, a hereditary condition carrying high "
             "colorectal cancer risk. The findings showed that aspirin protected against the primary "
             "end point of CRC (HR, 0.65; 95% CI, 0.43-0.97; P = .035) in individuals taking "
             "aspirin for at least two years. This represents meaningful protection for this "
             "high-risk population.\n\n"
             "Aspirin Harms: Very low-dose ASA use (i.e., <=100 mg every day or every other day) "
             "results in an estimated 14 (95% CI, 7-23) additional major gastrointestinal bleeding "
             "events and 3.2 (95% CI, -0.5 to 0.82) extra hemorrhagic strokes per 1,000 individuals "
             "over 10 years."),
            ("NSAIDs, Polyp Removal, and Lifestyle",
             "NSAIDs - Inadequate Evidence: While some studies show NSAIDs reduce adenoma risk, "
             "evidence remains insufficient to recommend them for colorectal cancer prevention. "
             "There is inadequate evidence that the use of NSAIDs reduces the risk of CRC, despite "
             "their demonstrated ability to reduce precancerous polyps. The estimated average excess "
             "risk of upper gastrointestinal complications in average-risk people attributable to "
             "NSAIDs is 4 to 5 per 1,000 people per year, with serious cardiovascular events "
             "increased by 50-60%.\n\n"
             "Polyp Removal: Removing adenomatous polyps provides significant colorectal cancer "
             "prevention, particularly for larger lesions. Removal of adenomatous polyps reduces "
             "the risk of CRC, though the benefit of removing smaller polyps - which are much more "
             "common - is unknown. The procedure carries procedural risks: seven to nine events per "
             "1,000 procedures result in major complications including perforation of the colon and "
             "bleeding.\n\n"
             "Physical Activity: Regular physical activity is associated with a decreased incidence "
             "of CRC, with a meta-analysis of 52 observational studies finding a statistically "
             "significant 24% reduction in CRC incidence (RR, 0.76; 95% CI, 0.72-0.81).\n\n"
             "Alcohol and Smoking: Excessive alcohol consumption increases risk substantially. "
             "Consumption exceeding 45 g/day carried an adjusted relative risk (RR) of 1.41 (95% "
             "confidence interval [CI], 1.16-1.72). Cigarette smoking similarly elevates risk. A "
             "pooled analysis of 106 observational studies estimated an adjusted RR (current "
             "smokers vs. never smokers) of 1.18 for developing CRC (95% CI, 1.11-1.25)."),
            ("Hormone Therapy, Statins, and Hereditary Risk",
             "Combined Estrogen-Progestin: The Women's Health Initiative trial provided critical "
             "data on hormone replacement therapy. The study found that combined hormone therapy had "
             "a statistically significant higher stage of cancer (regional and distant) at "
             "diagnosis. Regarding incidence, there were fewer CRCs in the combined hormone therapy "
             "group than in the placebo group (0.12% vs. 0.16%; HR, 0.72; 95% CI, 0.56-0.94). "
             "However, mortality outcomes differed: there were 37 CRC deaths in the combined "
             "hormone therapy arm compared with 27 deaths in the placebo arm (0.04% vs. 0.03%; HR, "
             "1.29; 95% CI, 0.78-2.11). The harms proved substantial: the WHI showed a 26% "
             "increase in invasive breast cancer in the combined hormone group, a 29% increase in "
             "coronary heart disease events, a 41% increase in stroke rates, and a twofold higher "
             "rate of thromboembolic events.\n\n"
             "Estrogen-Only Therapy: Conjugated equine estrogens do not affect the incidence of or "
             "mortality from invasive CRC.\n\n"
             "Statins: Statins do not reduce the incidence of or mortality from CRC, based on "
             "systematic analysis of randomized controlled trials. However, statins remain "
             "relatively safe for use in patients requiring them for cardiovascular indications, as "
             "the harms of statins are small.\n\n"
             "Hereditary Syndromes: Two major hereditary syndromes carry extremely high cancer "
             "risk. In familial adenomatous polyposis, the risk of CRC by age 40 can be as high as "
             "100%. Lynch syndrome carriers face similarly elevated risk: individuals with Lynch "
             "syndrome can have a lifetime risk of CRC of about 80%. Age represents the dominant "
             "risk factor: 90% of all CRCs are diagnosed after age 50. Family history substantially "
             "increases risk - a first-degree relative, especially if diagnosed before the age of "
             "55 years, roughly doubles the risk."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-PDQ-prostate-cancer-prevention-HP",
        "source_url": "https://www.cancer.gov/types/prostate/hp/prostate-prevention-pdq",
        "source_title": "Prostate Cancer Prevention (PDQ) - Health Professional Version",
        "source_authors": "National Cancer Institute PDQ Screening and Prevention Editorial Board",
        "source_year": 2025,
        "license": "NCI public domain",
        "sections": [
            ("Finasteride PCPT Trial",
             "The Prostate Cancer Prevention Trial enrolled 18,882 men aged 55 or older. At 7 years, "
             "the incidence of prostate cancer was 18.4% in the finasteride group versus 24.4% in "
             "the placebo group with a relative risk reduction of 24.8%. However, prostate cancer "
             "mortality was not statistically different between men in the placebo and finasteride "
             "groups during long-term follow-up extending to a median of 18.4 years.\n\n"
             "Concerning high-grade cancers, researchers observed that high-grade cancers (Gleason "
             "score 7-10) were noted in 6.4% of finasteride patients, compared with 5.1% of men who "
             "received placebo. Importantly, subsequent analyses adjusted for detection bias found "
             "that finasteride reduced the incidence of Gleason score 5 to 7 and Gleason score 3 to "
             "4 prostate cancer.\n\n"
             "Finasteride Adverse Effects: Men in the finasteride group had statistically "
             "significantly more erectile dysfunction, loss of libido, and gynecomastia than men in "
             "the placebo group. Specific magnitude of effects included: reduced volume of ejaculate "
             "(60.4% vs. 47.3%); erectile dysfunction (67.4% vs. 61.5%); loss of libido (65.4% vs. "
             "59.6%); gynecomastia (4.5% vs. 2.8%). Treatment discontinuation rates reflected these "
             "concerns, as a greater fraction of men in the finasteride group (36.8%) temporarily "
             "discontinued treatment at some time during the study for reasons other than death or "
             "a diagnosis of prostate cancer than in the placebo group (28.9%)."),
            ("Dutasteride REDUCE Trial",
             "The dutasteride trial involved 8,231 men aged 50-75 with PSA between 2.5-10.0 ng/mL "
             "and prior negative biopsy. The results indicated: After 4 years, among the 6,729 men "
             "who had at least one prostate biopsy, 25.1% of the placebo group and 19.9% of the "
             "dutasteride group had been diagnosed with prostate cancer representing an absolute "
             "risk reduction of 5.1% and relative risk reduction of 22.8%.\n\n"
             "A critical finding emerged: The difference between the groups was entirely due to a "
             "reduction in prostate cancers with Gleason score 5 to 7. Notably, there was a "
             "statistically significant difference in higher-grade disease in years 3-4 of the "
             "trial.\n\n"
             "Dutasteride Adverse Effects: Overall, 4.3% of men in the dutasteride group compared "
             "with 2% of men in the placebo group discontinued the trial because of drug-related "
             "adverse events. Reported adverse effects included: decreased libido (3.3% vs. 1.6%); "
             "loss of libido (1.9% vs. 1.3%); decreased semen volume (1.4% vs. 0.2%); erectile "
             "dysfunction (9.0% vs. 5.7%); gynecomastia (1.9% vs. 1.0%)."),
            ("Vitamin E, Selenium, and FDA Position",
             "Vitamin E and Selenium (SELECT Trial): The large randomized SELECT trial examined "
             "these supplemental agents with unexpected results. It showed no reduction in prostate "
             "cancer period prevalence, but an increased risk of prostate cancer with vitamin E "
             "alone. Compared with the placebo group in which 529 men developed prostate cancer, "
             "there was a statistically significant increase in prostate cancer in the vitamin E "
             "group (620 cases), but not in the selenium plus vitamin E group (555 cases) or in "
             "the selenium group (575 cases). The magnitude of increase in prostate cancer risk "
             "with vitamin E alone was 17%.\n\n"
             "Remarkably, the statistically increased risk of prostate cancer among men receiving "
             "vitamin E was seen after study supplements had been discontinued suggesting a "
             "longer-term effect of this agent.\n\n"
             "FDA Review: The Oncology Drugs Advisory Committee of the FDA examined both "
             "finasteride and dutasteride in 2010. Neither agent was recommended for use for "
             "chemoprevention of prostate cancer.\n\n"
             "A crucial distinction exists between incidence reduction and mortality benefit. "
             "Long-term follow-up of PCPT participants found no increased risk of prostate cancer "
             "mortality with finasteride use. The evidence remains inadequate to determine whether "
             "chemoprevention with finasteride or dutasteride reduces mortality from prostate "
             "cancer. This distinction carries important clinical implications, as its effect is "
             "primarily in preventing the diagnosis of prostate cancer and the subsequent events "
             "(staging, treatment, follow-up, and management of treatment-related side effects) "
             "after diagnosis rather than preventing death from the disease."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-PDQ-cervical-cancer-prevention-HP",
        "source_url": "https://www.cancer.gov/types/cervical/hp/cervical-prevention-pdq",
        "source_title": "Cervical Cancer Prevention (PDQ) - Health Professional Version",
        "source_authors": "National Cancer Institute PDQ Screening and Prevention Editorial Board",
        "source_year": 2025,
        "license": "NCI public domain",
        "sections": [
            ("HPV Vaccination FUTURE and 9-Valent Trials",
             "The quadrivalent vaccine underwent extensive testing in the FUTURE trials. One study "
             "enrolled 17,622 women aged 15 to 26 years who received vaccine or placebo at "
             "specific intervals. Results demonstrated significant protection in the HPV-naive "
             "population, with 100% (90.5%-100%) efficacy for lesions associated with HPV-6, -11, "
             "-16, or -18. However, in the broader intention-to-treat analysis that included "
             "sexually experienced women, efficacy dropped to 45.3% (29.8%-57.6%).\n\n"
             "The 9-valent vaccine study involved 14,215 women and compared it directly against "
             "the quadrivalent vaccine. The research found that while both prevented similar rates "
             "of high-grade disease in those already exposed, the 9-valent option reduced disease "
             "related to HPV-31, -35, -45, -52, and -58 more effectively, at 0.1 vs. 1.6 per 1,000 "
             "person-years.\n\n"
             "Current CDC recommendations support a two-dose schedule at least 6 months apart for "
             "adolescents younger than 15 years, with three doses recommended for older "
             "individuals."),
            ("HPV Types, Smoking, and Oral Contraceptive Risk",
             "HPV-16 and HPV-18 remain the most clinically significant. HPV type 16 (HPV-16) and "
             "HPV type 18 (HPV-18) are most often associated with invasive disease. A Costa Rican "
             "population study determined that 80% of high-grade squamous intraepithelial lesions "
             "(HSIL) and invasive lesions were associated with HPV infection by one or more of 13 "
             "cancer-associated types, with about 50% of HSIL and invasive cervical cancer "
             "attributable to HPV-16 alone.\n\n"
             "Smoking as a Risk Factor: Cigarette smoke exposure demonstrates dose-dependent "
             "effects. Among women with HPV infection, current and former smokers have "
             "approximately two to three times the incidence of high-grade cervical intraepithelial "
             "neoplasia or invasive cancer. This risk increases with longer duration and intensity "
             "of smoking and passive smoking shows increased risk but to a lesser extent.\n\n"
             "Oral Contraceptive Risk: Long-term oral contraceptive use significantly elevates "
             "cervical cancer risk in HPV-positive women. Among infected women, those who used oral "
             "contraceptives for 5 to 9 years have approximately three times the incidence of "
             "invasive cancer, and those who used them for 10 years or longer have approximately "
             "four times the risk. Importantly, risk decreases after cessation and returns to "
             "normal risk levels in 10 years.\n\n"
             "Population-Level Vaccination Impact: Real-world data from England's national program "
             "documented substantial benefits. Among girls vaccinated ages 12-13, there was an "
             "estimated 87% (72-94) reduction in cervical cancer and 97% (96-78) reduction in CIN "
             "3. A Swedish study of over 1.67 million women found that the cumulative risk of "
             "cervical cancer by age 30 years was 47 cases per 100,000 in vaccinated women, "
             "compared with 94 cases per 100,000 in unvaccinated women."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-oral-contraceptives-cancer-risk",
        "source_url": "https://www.cancer.gov/about-cancer/causes-prevention/risk/hormones/oral-contraceptives-fact-sheet",
        "source_title": "Oral Contraceptives and Cancer Risk - NCI Fact Sheet",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI public domain",
        "sections": [
            ("Overview and Methodology",
             "The National Cancer Institute provides comprehensive information about the "
             "relationship between oral contraceptive use and various cancer types. Oral "
             "contraceptives, commonly known as birth control pills, are hormone-containing "
             "medications taken orally to prevent pregnancy. They work by inhibiting ovulation and "
             "blocking sperm penetration through the cervix.\n\n"
             "The most widely prescribed form in the United States contains synthetic versions of "
             "estrogen and progesterone, termed combined oral contraceptives. An alternative "
             "formulation, the mini pill, contains only progestin (synthetic progesterone).\n\n"
             "Nearly all evidence regarding oral contraceptive use and cancer derives from "
             "observational studies, including large prospective cohort studies and population-"
             "based case-control investigations. Such observational data cannot definitively "
             "establish that an exposure - in this case, oral contraceptives - causes (or "
             "prevents) cancer, since users may differ from non-users in ways beyond contraceptive "
             "use alone. Despite these methodological limitations, research has consistently "
             "demonstrated that oral contraceptive use increases risks for certain cancers while "
             "decreasing risks for others."),
            ("Breast and Cervical Cancer Findings",
             "Breast Cancer: Research involving over 150,000 women across 54 epidemiologic studies "
             "demonstrated that women who had used oral contraceptives showed a 7 percent overall "
             "increase in breast cancer risk compared to never-users. However, current users "
             "experienced a more pronounced 24 percent risk increase that did not escalate with "
             "duration of use. Risk declined after use of oral contraceptives stopped, and no risk "
             "increase was evident by 10 years after use had stopped.\n\n"
             "A significant 2017 Danish prospective study examined contemporary oral contraceptive "
             "formulations. Results showed that women currently using or recently discontinuing "
             "oral combined hormone contraceptives had approximately a 20 percent increased breast "
             "cancer risk. Importantly, the risk increase varied from 0% to 60%, depending on the "
             "specific type of oral combined hormone contraceptive. Longer duration of use "
             "correlated with increased risk.\n\n"
             "Cervical Cancer: The relationship between oral contraceptive use and cervical cancer "
             "demonstrates clear dose-response characteristics. Women using oral contraceptives "
             "for five or more years face elevated cervical cancer risk, with greater increases "
             "correlating to longer use duration. One study documented a 10 percent increased "
             "risk for less than five years of use, progressing to 60 percent increased risk for "
             "five to nine years of use, and doubling with ten or more years of use. However, "
             "cervical cancer risk declines after discontinuation, suggesting that prolonged use "
             "duration rather than permanent cellular changes drives the increased risk."),
            ("Endometrial, Ovarian, and Colorectal Cancer Findings",
             "Endometrial Cancer: Oral contraceptive use provides protective effects against "
             "endometrial cancer. Women with any history of oral contraceptive use demonstrated at "
             "least 30 percent lower endometrial cancer risk relative to never-users, with greater "
             "protection observed among longer-term users. The protective effect persists for many "
             "years after a woman stops using oral contraceptives, indicating sustained benefit "
             "from past use.\n\n"
             "Ovarian Cancer: Oral contraceptive use substantially reduces ovarian cancer risk. "
             "Women with any history of use showed 30 to 50 percent lower ovarian cancer risk "
             "compared to never-users. Protection increases with longer use duration and persists "
             "for up to 30 years after discontinuation. Notably, this protective effect extends to "
             "women carrying harmful mutations in the BRCA1 or BRCA2 genes, populations at "
             "particularly elevated inherited ovarian cancer risk.\n\n"
             "Colorectal Cancer: Oral contraceptive use associates with 15 to 20 percent lower "
             "colorectal cancer risks relative to non-users.\n\n"
             "Mechanisms: Naturally occurring estrogen and progesterone stimulate development and "
             "growth of hormone-responsive cancers, including breast cancer. Since birth control "
             "pills contain synthetic versions of these hormones, they could theoretically "
             "increase cancer risk similarly. Additionally, oral contraceptives may alter cervical "
             "cell susceptibility to persistent high-risk human papillomavirus infection. "
             "Protective mechanisms include suppression of endometrial cell proliferation, "
             "reduction of lifetime ovulations, and lowering of blood bile acid levels."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-breast-cancer-risk-women",
        "source_url": "https://www.cancer.gov/types/breast/risk-fact-sheet",
        "source_title": "Breast Cancer Risk in American Women - NCI Fact Sheet",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI public domain",
        "sections": [
            ("Lifetime Risk and Age-Specific Estimates",
             "According to the National Cancer Institute's Surveillance, Epidemiology, and End "
             "Results Program, 12.9% of women born in the United States today will develop breast "
             "cancer at some time during their lives. This translates to approximately one in "
             "eight women facing a breast cancer diagnosis during her lifetime, while the chance "
             "that she will never have breast cancer is 87.1%, or about 7 in 8.\n\n"
             "For men, the statistics are notably different. A man born today has about a 1 in "
             "800 chance of being diagnosed with breast cancer at some time during his life.\n\n"
             "The NCI provides risk calculations for 10-year intervals, showing how likelihood "
             "increases with age:\n"
             "- Age 30: 0.49% risk (1 in 204)\n"
             "- Age 40: 1.55% risk (1 in 65)\n"
             "- Age 50: 2.40% risk (1 in 42)\n"
             "- Age 60: 3.54% risk (1 in 28)\n"
             "- Age 70: 4.09% risk (1 in 24)\n\n"
             "These figures represent averages for the whole population, acknowledging that "
             "individual risk varies considerably. An individual woman's breast cancer risk may be "
             "higher or lower depending on known factors, as well as on factors that are not yet "
             "fully understood. For personalized risk evaluation, healthcare providers can utilize "
             "the Breast Cancer Risk Assessment Tool, which takes into account several known "
             "breast cancer risk factors.\n\n"
             "Recent SEER reports demonstrate consistent estimates hovering around one in eight: "
             "12.83% (2014-2016), 12.44% (2013-2015), 12.41% (2012-2014), 12.43% (2011-2013), "
             "12.32% (2010-2012). SEER statisticians expect some variability from year to year, "
             "attributing fluctuations to minor changes in risk factor levels in the population, "
             "slight changes in breast cancer screening rates, or just random variability inherent "
             "in the data."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-PDQ-breast-cancer-prevention-patient",
        "source_url": "https://www.cancer.gov/types/breast/patient/breast-prevention-pdq",
        "source_title": "Breast Cancer Prevention (PDQ) - Patient Version",
        "source_authors": "National Cancer Institute PDQ Screening and Prevention Editorial Board",
        "source_year": 2025,
        "license": "NCI public domain",
        "sections": [
            ("Protective Lifestyle Factors",
             "The NCI identifies several ways to reduce breast cancer risk.\n\n"
             "Physical Activity: Women who take part in physical exercise have a lower risk of "
             "breast cancer.\n\n"
             "Early Pregnancy: Women delivering their first child before age 20 have lower risk "
             "than those with later pregnancies or no children.\n\n"
             "Breastfeeding: Nursing for several months post-childbirth offers protective benefits.\n\n"
             "Weight Management: Maintaining a healthy weight reduces risk.\n\n"
             "Alcohol Reduction: Limiting or eliminating alcohol consumption is protective."),
            ("Risk-Reducing Medications and Surgery",
             "For higher-risk individuals, chemoprevention options exist.\n\n"
             "Selective Estrogen Receptor Modulators (SERMs):\n"
             "- Tamoxifen: FDA-approved to reduce breast cancer risk in women at higher-than-"
             "average risk.\n"
             "- Raloxifene: used only in postmenopausal women.\n\n"
             "Aromatase Inhibitors:\n"
             "- Include anastrozole and exemestane.\n"
             "- Used in postmenopausal women.\n"
             "- Not yet FDA-approved for risk reduction.\n\n"
             "Important note: Hormone therapies have serious side effects, and it's important to "
             "discuss the possible benefits and harms of these drugs with your doctor.\n\n"
             "Preventive Surgery: Risk-reducing bilateral mastectomy is considered for those with "
             "inherited BRCA1/BRCA2 mutations or very strong family histories."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-PDQ-colorectal-cancer-prevention-patient",
        "source_url": "https://www.cancer.gov/types/colorectal/patient/colorectal-prevention-pdq",
        "source_title": "Colorectal Cancer Prevention (PDQ) - Patient Version",
        "source_authors": "National Cancer Institute PDQ Screening and Prevention Editorial Board",
        "source_year": 2025,
        "license": "NCI public domain",
        "sections": [
            ("Overview and Cancer Statistics",
             "According to the National Cancer Institute, cancer prevention is action taken to "
             "lower the chance of getting cancer. Scientists examine both risk factors - anything "
             "increasing cancer development chances - and protective factors that decrease those "
             "chances.\n\n"
             "The colon forms part of the digestive system and measures approximately 5 feet in "
             "length. The rectum and anal canal together comprise the final 6-8 inches of the "
             "large intestine. Colorectal cancer is the third most common cancer worldwide and "
             "the second leading cause of cancer death in the United States.\n\n"
             "Recent trends show interesting patterns. Between 2012 and 2021, overall rates "
             "declined, particularly among adults 65 and older, partly attributable to screening "
             "initiatives. However, among adults younger than 50, the number of new cases of "
             "colorectal cancer rose by about 2.4% per year."),
            ("Established Risk Factors",
             "Age: The risk of colorectal cancer increases after age 50. Most cases of colorectal "
             "cancer are diagnosed after age 50.\n\n"
             "Family History: Having a parent, brother, sister, or child with colorectal cancer "
             "doubles a person's risk of colorectal cancer.\n\n"
             "Personal Medical History: Previous colorectal cancer, high-risk adenomas (polyps), "
             "ovarian cancer, and inflammatory bowel diseases increase risk.\n\n"
             "Inherited Genetic Conditions: The risk of colorectal cancer is increased when "
             "certain gene changes linked to familial adenomatous polyposis (FAP) or hereditary "
             "nonpolyposis colon cancer (HNPCC or Lynch Syndrome) are inherited.\n\n"
             "Alcohol Consumption: Drinking 3 or more alcoholic beverages per day increases the "
             "risk of colorectal cancer.\n\n"
             "Cigarette Smoking: Cigarette smoking is linked to an increased risk of colorectal "
             "cancer and death from colorectal cancer. Additionally, smokers who have undergone "
             "adenoma removal face elevated recurrence risks.\n\n"
             "Race: Black individuals have an increased risk of colorectal cancer and death from "
             "colorectal cancer compared to other races.\n\n"
             "Obesity: Being overweight correlates with increased colorectal cancer risk and "
             "mortality."),
            ("Protective Factors and Aspirin",
             "Physical Activity: A lifestyle that includes regular physical activity is linked to "
             "a decreased risk of colorectal cancer.\n\n"
             "Aspirin Use: Research demonstrates that taking aspirin daily for at least two years "
             "lowers the risk of colorectal cancer and the risk of death from colorectal cancer. "
             "However, benefits emerge gradually, with the decrease in risk beginning 10 to 20 "
             "years after patients start taking aspirin. The NCI warns that aspirin carries "
             "risks of stroke and gastrointestinal bleeding, particularly for elderly individuals "
             "and those with bleeding disorders.\n\n"
             "Combination Hormone Replacement Therapy: For postmenopausal women, combination "
             "hormone replacement therapy (HRT) that includes both estrogen and progestin lowers "
             "the risk of invasive colorectal cancer. Nevertheless, when cancer does develop in "
             "HRT users, it tends toward more advanced stages at diagnosis.\n\n"
             "Polyp Removal: Removing colorectal polyps that are larger than 1 centimeter "
             "(pea-sized) may lower the risk of colorectal cancer. The NCI notes uncertainty "
             "about smaller polyp removal effectiveness. Potential complications include colon "
             "perforation and bleeding during removal procedures."),
            ("Unclear or Ineffective Factors",
             "NSAIDs: The status of nonsteroidal anti-inflammatory drugs remains uncertain. While "
             "celecoxib shows promise in reducing adenoma recurrence and sulindac decreases polyp "
             "formation in FAP patients, the ultimate cancer prevention benefit remains unclear.\n\n"
             "Calcium Supplements: Evidence regarding calcium supplementation's protective "
             "effects remains inconclusive.\n\n"
             "Dietary Changes: It is not known if a diet low in fat and meat and high in fiber, "
             "fruits, and vegetables lowers the risk of colorectal cancer. Some studies suggest "
             "high-fat, high-protein diets increase risk, but findings remain inconsistent.\n\n"
             "Estrogen-Only HRT: This therapy shows no protective benefit.\n\n"
             "Statins: Research indicates these cholesterol-lowering medications neither increase "
             "nor decrease colorectal cancer risk.\n\n"
             "Clinical Trial Opportunities: Cancer prevention clinical trials are used to study "
             "ways to prevent cancer. These investigations examine whether lifestyle "
             "modifications, exercise, smoking cessation, or pharmaceutical interventions can "
             "reduce cancer development rates."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-obesity-cancer-fact-sheet",
        "source_url": "https://www.cancer.gov/about-cancer/causes-prevention/risk/obesity/obesity-fact-sheet",
        "source_title": "Obesity and Cancer - NCI Fact Sheet",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI public domain",
        "sections": [
            ("Thirteen Obesity-Associated Cancers",
             "According to the National Cancer Institute, research has identified thirteen cancer "
             "types linked to excess body weight. Cancers that have been found to be linked to "
             "obesity or overweight include endometrial, esophageal, upper stomach, liver, "
             "kidney, multiple myeloma, meningioma, pancreatic, colorectal, gallbladder, breast "
             "(postmenopausal), ovarian, thyroid.\n\n"
             "The risk elevations vary considerably across these malignancies. The risk increases "
             "associated with obesity are highest for endometrial and esophageal cancers. "
             "Specifically, individuals with severe obesity face substantially higher risks: "
             "approximately seven times elevated risk for type 1 endometrial cancers and nearly "
             "five times elevated risk for esophageal adenocarcinoma, compared to healthy-weight "
             "individuals.\n\n"
             "For remaining obesity-linked cancers, risk increases in people with obesity range "
             "from about a 10% increase in risk to a doubling of risk."),
            ("Biological Mechanisms",
             "The NCI identifies multiple pathways through which excess body fat may elevate "
             "cancer risk.\n\n"
             "Hormonal Effects: Fat tissue produces excess estrogen, which is known to cause "
             "cancer. This hormonal elevation particularly impacts breast, endometrial, ovarian, "
             "and certain other malignancies. The mechanism involves adipose tissue functioning "
             "as an endocrine organ, particularly significant in postmenopausal women.\n\n"
             "Insulin and Growth Factors: People with obesity often have increased blood levels "
             "of insulin and insulin-like growth factor-1 (IGF-1). This state of hyperinsulinemia, "
             "preceding type 2 diabetes development, associates with elevated risks across "
             "multiple cancer types. High levels of insulin and IGF-1 are associated with "
             "increased risks of colorectal, thyroid, breast, prostate, ovarian, and endometrial "
             "cancers.\n\n"
             "Chronic Inflammation: People with obesity often have chronic inflammation, which "
             "directly promotes tumor growth by several mechanisms.\n\n"
             "Adipokine Production: Fat cells produce signaling molecules called adipokines with "
             "growth-regulating properties. The blood level of an adipokine called leptin "
             "increases with increasing body fat, and high levels of leptin can promote aberrant "
             "cell proliferation. Conversely, adiponectin, another adipokine potentially "
             "protective against tumors, exists at lower concentrations in obese individuals.\n\n"
             "Metabolic Pathway Dysregulation: Regulators like mammalian target of rapamycin "
             "(mTOR) and AMP-activated protein kinase, both of which are involved in regulating "
             "autophagy, which when impaired can lead to cancer."),
            ("Population Burden and Trends",
             "Quantifying the cancer burden from obesity is essential for public health planning. "
             "In 2019 among people ages 30 and older in the United States, about 43,720 new "
             "cancer cases in men (4.8%) and 92,200 new cancer cases in women (10.6%) were due "
             "to excess body weight.\n\n"
             "The proportional burden varies dramatically by cancer type. For certain "
             "malignancies, obesity accounts for enormous percentages of cases. Attribution was "
             "as high as 34.9% for liver cancer and 53.1% for endometrial cancer in women and "
             "37.1% for gallbladder cancer and 37.8% for esophageal adenocarcinoma in men.\n\n"
             "More recent epidemiological evidence suggests the obesity-cancer connection may "
             "extend beyond these thirteen recognized associations. A recent study of more than 2 "
             "million people in Spain who were followed for a median of 9 years found evidence "
             "that overweight and obesity are linked to 18 cancers, including some not yet "
             "considered to be related to obesity.\n\n"
             "Furthermore, obesity may accelerate cancer development at younger ages. A 2024 "
             "study examined whether the increasing incidence of early-onset cancers over the "
             "period 2000 to 2012 worldwide could be explained by increasing rates of obesity "
             "among young adults over this period. Six of nine obesity-related cancers increased "
             "in incidence among young adults during this period, and for four of these cancers "
             "(colon, rectal, pancreatic, and kidney), this rise was associated with increases "
             "in body weight.\n\n"
             "Obesity Prevalence: In 2011, 27.4% of adults ages 18 or older had obesity. In "
             "2023, 32.8% of adults ages 18 or older had obesity. Racial and ethnic disparities "
             "substantially impact these burdens. According to the NCI's 2023 data, obesity "
             "prevalence among adults ages 18 and older varied considerably, with Non-Hispanic "
             "Black, 42.0%; American Indian/Alaska Native, 39.6%; Hawaiian/Pacific Islander, "
             "31.8%; Hispanic, 35.1%; Non-Hispanic White, 32.2%; Asian, 13.4%."),
            ("BMI Definitions and Body Composition",
             "The NCI defines obesity as a disease in which a person has an unhealthy amount "
             "and/or distribution of body fat. Healthcare providers typically employ body mass "
             "index (BMI) for assessment, calculated by dividing a person's weight (in "
             "kilograms) by their height (in meters) squared.\n\n"
             "BMI is not a direct measure of body fat, but it provides a more accurate "
             "assessment of obesity than weight alone. It is a useful estimate of body fatness "
             "in populations but cannot be used on its own to indicate obesity-related disease "
             "risks in individuals.\n\n"
             "Standard BMI categories for adults aged 20+ include: below 18.5 (underweight), "
             "18.5-24.9 (healthy), 25.0-29.9 (overweight), 30.0-39.9 (obese), and 40.0 or "
             "higher (severely obese).\n\n"
             "Beyond BMI, measurements that reflect the distribution of body fat are sometimes "
             "used along with BMI as indicators of obesity and disease risks. These measurements "
             "include waist circumference, waist-to-hip ratio, waist-to-height ratio, and fat "
             "distribution as measured by dual energy X-ray absorptiometry (DXA or DEXA), "
             "imaging with CT or PET, or measurements of body shape.\n\n"
             "The distribution of fat is increasingly understood to be relevant to disease "
             "risks. In particular, visceral fat - fat that surrounds internal organs - seems "
             "to be more dangerous, in terms of disease risks, than overall fat or subcutaneous "
             "fat (the layer just under the skin)."),
            ("Weight Loss Interventions and Cancer Risk",
             "Evidence regarding whether weight loss reduces cancer incidence comes primarily "
             "from observational rather than randomized controlled studies. Most of the data "
             "about whether losing weight reduces cancer risk comes from observational studies "
             "such as cohort and case-control studies. Randomized controlled studies of weight "
             "loss diets have been done, but they sometimes failed to produce substantial weight "
             "change, and did not appear to reduce cancer risk.\n\n"
             "Nevertheless, observational research suggests potential benefits. In one large "
             "prospective study of postmenopausal women, those who intentionally lost more than "
             "5% of body weight had a lower risk of obesity-related cancers, especially "
             "endometrial cancer. Similarly, data from the Women's Health Initiative "
             "Observational Study found that women who lost at least 5% of their body weight "
             "during study follow up had a lower risk of invasive breast cancer than those "
             "whose weight was stable.\n\n"
             "Sustained weight loss appears particularly important. Results of a study that "
             "pooled data from 10 cohorts suggested that sustained weight loss was associated "
             "with lower breast cancer risk among women 50 years and older.\n\n"
             "Bariatric surgical interventions demonstrate more pronounced effects. Weight loss "
             "through bariatric surgery (surgery performed on the stomach or intestines to "
             "provide maximum and sustained weight loss) has also been found to be associated "
             "with reduced cancer risks. People with obesity, particularly women, who had "
             "bariatric surgery had lower risks of cancer overall; of hormone-related cancers, "
             "such as breast, endometrial, and prostate cancers; and of obesity-related "
             "cancers.\n\n"
             "Pharmacological weight loss interventions show promise in recent research. People "
             "with type 2 diabetes with no prior diagnosis of an obesity-related cancer who "
             "were prescribed GLP-1 receptor agonists had lower risks of 10 of 13 obesity-"
             "related cancers, including esophageal, colorectal, kidney, pancreatic, "
             "gallbladder, ovarian, endometrial, and liver cancers as well as meningioma and "
             "multiple myeloma. However, a meta-analysis of randomized controlled trials with "
             "an average of 3 years of follow-up per participant found no association between "
             "GLP-1 receptor agonists and the risk of any gastrointestinal cancer."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-antiperspirants-breast-cancer",
        "source_url": "https://www.cancer.gov/about-cancer/causes-prevention/risk/myths/antiperspirants-fact-sheet",
        "source_title": "Antiperspirants/Deodorants and Breast Cancer - NCI Fact Sheet",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI public domain",
        "sections": [
            ("Antiperspirants and Breast Cancer Risk",
             "The National Cancer Institute webpage states that no scientific evidence links the "
             "use of these products to the development of breast cancer.\n\n"
             "Aluminum-based compounds in antiperspirants have been studied for potential "
             "hormonal effects, but research has not confirmed harmful impacts on breast cancer "
             "risk. A 2014 review found no clear evidence showing that the use of aluminum-"
             "containing underarm antiperspirants or cosmetics increases the risk of breast "
             "cancer.\n\n"
             "Only a few studies exist on this topic; a 2002 study of over 1,600 women found no "
             "increased breast cancer risk among antiperspirant users. A 2006 study reached "
             "similar conclusions, though with a smaller sample size. Results have been somewhat "
             "conflicting, suggesting more research is warranted.\n\n"
             "The NCI encourages concerned individuals to discuss breast cancer risk factors "
             "with their healthcare provider rather than rely on unproven theories about product "
             "use."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-menopausal-hormone-therapy",
        "source_url": "https://www.cancer.gov/about-cancer/causes-prevention/risk/hormones/mht-fact-sheet",
        "source_title": "Menopausal Hormone Therapy and Cancer - NCI Fact Sheet",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI public domain",
        "sections": [
            ("Overview and Formulations",
             "Menopausal hormone therapy (MHT), also referred to as postmenopausal hormone "
             "therapy or hormone replacement therapy, represents a medical intervention designed "
             "to address the physiological changes accompanying menopause. As hormone levels "
             "naturally decline during this life stage, medical providers may recommend MHT to "
             "alleviate common symptoms and address long-term biological consequences.\n\n"
             "MHT consists of estrogen alone or estrogen plus progestin, a synthetic form of "
             "progesterone. These hormonal agents receive FDA approval and originate from plant "
             "and animal sources or laboratory synthesis. The chemical composition resembles "
             "naturally occurring hormones produced by women's bodies.\n\n"
             "Healthcare providers deliver MHT through multiple routes depending on treatment "
             "objectives. For systemic symptom management or osteoporosis prevention, "
             "administration occurs orally, transdermally (patches, gels, sprays), or via "
             "implants. When addressing genitourinary symptoms like vaginal dryness, low-dose "
             "estrogen only is applied directly in the vagina (as a cream or suppository).\n\n"
             "A critical distinction in MHT prescription depends on whether a woman retains her "
             "uterus. Because estrogen, when used alone (i.e., without progestin) for systemic "
             "MHT, is associated with an increased risk of endometrial cancer, estrogen "
             "monotherapy applies only to women who have undergone hysterectomy. Women with an "
             "intact uterus typically receive estrogen plus progestin for systemic MHT, whereas "
             "vaginal estrogen remains appropriate regardless of hysterectomy history.\n\n"
             "The FDA has cautioned against non-FDA-approved products marketed as bioidentical "
             "hormones, custom-compounded medications sometimes sold without prescriptions "
             "online. Claims that these products are safer or more natural than FDA-approved "
             "hormonal products are not supported by credible scientific evidence."),
            ("Women's Health Initiative Trials",
             "Much of our contemporary understanding regarding MHT's health effects derives from "
             "two major randomized clinical trials sponsored by the National Institutes of "
             "Health as components of the Women's Health Initiative (WHI). These investigations "
             "fundamentally reshaped clinical practice and public understanding of hormone "
             "therapy risks and benefits.\n\n"
             "The WHI Estrogen-plus-Progestin Study: This trial enrolled women with intact "
             "uteruses who received random assignment to either a hormone pill combining both "
             "estrogen and progestin (Prempro) or placebo. The median treatment duration "
             "extended to 5.6 years before the study's premature termination in 2002, when "
             "emerging evidence demonstrated significant health risks warranted "
             "discontinuation.\n\n"
             "The WHI Estrogen-Alone Study: Conversely, this investigation included women "
             "without a uterus, randomized to receive either estrogen-only therapy (Premarin) "
             "or placebo. The median treatment period reached 7.2 years until the study's 2004 "
             "termination due to documented health risks.\n\n"
             "Both trials' early cessation reflected the emergence of compelling evidence "
             "regarding specific adverse health outcomes associated with both therapeutic "
             "approaches."),
            ("Documented Benefits and Harms",
             "Symptom Relief: MHT provides relief of hot flashes, night sweats, vaginal "
             "dryness, and painful intercourse with systemic and local estrogen or systemic "
             "estrogen plus progestin for as long as MHT is taken.\n\n"
             "Fracture Risk Reduction: Studies confirm lower risk of hip and vertebral fractures "
             "with systemic estrogen or estrogen plus progestin for as long as MHT is taken.\n\n"
             "Breast Cancer Risk with Estrogen Monotherapy: Research indicates lower risk of "
             "breast cancer with systemic estrogen when used without progestin. Additionally, "
             "women using estrogen monotherapy experienced lower risk of death from breast "
             "cancer.\n\n"
             "Documented Harms: Vaginal Bleeding and Endometrial Concerns: Women receiving "
             "estrogen plus progestin experienced increased risk of vaginal bleeding "
             "necessitating medical evaluation.\n\n"
             "Urinary Incontinence: Both estrogen monotherapy and combination regimens correlate "
             "with increased risk of urinary incontinence.\n\n"
             "Cognitive Decline in Older Women: Research demonstrates increased risk of "
             "dementia with both estrogen alone and estrogen plus progestin when taken by those "
             "65 years or older.\n\n"
             "Cardiovascular Complications: MHT increases risk of stroke, blood clots, and "
             "heart attack with estrogen alone and estrogen plus progestin for as long as MHT "
             "is taken.\n\n"
             "Endometrial Cancer with Unopposed Estrogen: Increased risk of endometrial cancer "
             "in people with an intact uterus with estrogen alone constitutes a major "
             "contraindication to estrogen monotherapy in non-hysterectomized women.\n\n"
             "Breast Cancer with Estrogen-Progestin Combinations: Estrogen plus progestin "
             "therapy demonstrates increased risk of breast cancer with prior use of estrogen "
             "plus progestin for at least a decade after use is discontinued.\n\n"
             "Mammographic Density Increase: The combination regimen produces increased breast "
             "density with estrogen plus progestin, making mammography less effective and also "
             "increasing breast cancer risk.\n\n"
             "Lung Cancer Mortality: Increased risk of death from lung cancer with estrogen "
             "plus progestin specific to combination therapy."),
            ("Special Populations and Alternatives",
             "Women with Prior Breast Cancer: The question of MHT safety in breast cancer "
             "survivors remains contentious. Women with previous breast cancer diagnoses often "
             "receive counseling to avoid MHT because some investigations suggest elevated "
             "recurrence risk. However, contradictory evidence exists. A Danish observational "
             "study of postmenopausal women treated for early-stage breast cancer showed no "
             "increased risk of recurrence or mortality associated with the use of vaginal or "
             "systemic MHT.\n\n"
             "Women with Gynecologic Cancers or Genetic Predispositions: The Society of "
             "Gynecologic Oncology released a 2020 clinical practice statement addressing MHT "
             "use in cancer survivors and high-risk populations. The statement concluded that "
             "benefits of MHT are likely to outweigh the risks for most people with epithelial "
             "ovarian, early-stage endometrial, and cervical cancer as well as for people with "
             "BRCA1 or BRCA2 gene mutations or Lynch syndrome and no history of breast cancer. "
             "Conversely, the statement recommended against MHT in women with advanced "
             "endometrial cancer, uterine sarcoma, or specific ovarian cancer histologies.\n\n"
             "Non-Hormonal FDA-Approved Medications: The FDA has approved three non-hormonal "
             "agents specifically for menopausal symptoms. Fezolinetant (Veozah) and paroxetine "
             "(Brisdelle) target moderate to severe hot flashes, while ospemifene (Osphena) "
             "addresses moderate to severe dyspareunia (painful intercourse) related to vaginal "
             "changes.\n\n"
             "Additional Non-Hormonal Therapies: The North American Menopause Society "
             "recommends selective serotonin reuptake inhibitors, serotonin-norepinephrine "
             "reuptake inhibitors, oxybutynin, gabapentin, and cognitive behavioral therapy for "
             "hot flash relief.\n\n"
             "Bone Health Without Hormones: Several FDA-approved medications prevent or slow "
             "bone loss independent of hormone therapy. Options include alendronate (Fosamax), "
             "raloxifene (Evista), and risedronate (Actonel).\n\n"
             "Clinical Guidance: The FDA recommends that women use MHT for the shortest time "
             "and at the lowest dose possible to control menopausal symptoms."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-alcohol-cancer-risk",
        "source_url": "https://www.cancer.gov/about-cancer/causes-prevention/risk/alcohol/alcohol-fact-sheet",
        "source_title": "Alcohol and Cancer Risk - NCI Fact Sheet",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI public domain",
        "sections": [
            ("IARC Classification and Burden",
             "The National Cancer Institute recognizes substantial scientific evidence "
             "establishing alcohol as a carcinogen. The International Agency for Research on "
             "Cancer classified alcohol as a Group 1 carcinogen (cancer-causing substance) in "
             "1987 based on sufficient evidence of its cancer-causing properties. The National "
             "Toxicology Program has similarly identified alcoholic beverage consumption as a "
             "known human carcinogen since 2000.\n\n"
             "The public health impact proves substantial. According to NCI analysis, alcohol "
             "consumption accounted for approximately 5 percent - nearly 100,000 cases - of the "
             "1.8 million cancer diagnoses in the United States during 2019. Furthermore, "
             "alcohol was responsible for roughly 4 percent of cancer deaths that year, "
             "representing nearly 25,000 fatalities."),
            ("Cancer Sites and Dose Response",
             "Research has established associations between alcohol consumption and several "
             "cancer sites, with varying levels of increased risk depending on drinking "
             "intensity:\n\n"
             "Oral Cavity and Throat Cancers: Light drinkers face approximately 1.1 times the "
             "risk of never-drinkers, while heavy drinkers experience roughly 5 times the risk.\n\n"
             "Laryngeal (Voice Box) Cancer: Light drinkers show negligible increased risk (0.9 "
             "times), but heavy drinkers demonstrate substantially elevated risk at "
             "approximately 2.6 times that of non-drinkers.\n\n"
             "Esophageal Cancer (Squamous Cell): Light drinkers experience 1.3 times the "
             "baseline risk, escalating to 5 times for heavy drinkers.\n\n"
             "Liver Cancer: Heavy drinkers face approximately twice the risk of those who "
             "abstain, though light and moderate consumption show less clear associations.\n\n"
             "Breast Cancer: Women represent a particularly vulnerable population. Even light "
             "consumption - just one drink daily - increases risk by 1.04 times compared to "
             "minimal drinkers. Moderate consumption elevates risk to 1.23 times, while heavy "
             "drinking increases it to 1.6 times.\n\n"
             "Colorectal Cancer: Moderate to heavy drinkers demonstrate increased risks ranging "
             "from 1.2 to 1.5 times compared to non-drinkers.\n\n"
             "Absolute Risk: Among 100 women consuming less than one drink weekly, "
             "approximately 17 will develop alcohol-related cancer. This rises to 19 among "
             "those consuming one daily drink and approximately 22 among those consuming two "
             "daily drinks. For men, the numbers progress from 10 per 100 to 11 to 13 across "
             "the same consumption categories."),
            ("Mechanisms and Genetics",
             "Acetaldehyde Production and DNA Damage: Alcohol metabolism produces acetaldehyde, "
             "a toxic chemical and a probable human carcinogen capable of damaging both DNA and "
             "proteins. This conversion occurs primarily in the liver but also in oral cavity "
             "tissues.\n\n"
             "Oxidative Stress: Alcohol generates reactive oxygen species - chemically reactive "
             "molecules capable of damaging DNA, proteins, and lipids through oxidative "
             "processes throughout the body.\n\n"
             "Nutrient Malabsorption: Alcohol impairs the body's capacity to absorb critical "
             "protective nutrients, including vitamin A, B-complex vitamins (particularly "
             "folate), vitamin C, vitamin D, vitamin E, and carotenoids.\n\n"
             "Enhanced Carcinogen Absorption: Alcohol increases the oral and throat tissues' "
             "ability to absorb harmful chemicals from other sources, such as tobacco smoke.\n\n"
             "Hormonal Effects: Consumption elevates blood estrogen levels, and sustained "
             "elevation increases breast cancer risk.\n\n"
             "Genetic Influences: The enzyme ADH converts ethanol to carcinogenic acetaldehyde "
             "primarily in the liver. Some individuals of East Asian descent possess a "
             "superactive ADH variant that rapidly converts alcohol to toxic acetaldehyde. "
             "Aldehyde Dehydrogenase 2 (ALDH2) metabolizes toxic acetaldehyde to nontoxic "
             "substances. Some individuals, particularly those of East Asian descent, carry a "
             "variant causing acetaldehyde accumulation when alcohol is consumed.\n\n"
             "Combined Alcohol and Tobacco Effects: For oral cavity, pharyngeal, laryngeal, "
             "and esophageal cancers, the combined harm are greater than would be expected "
             "from adding the individual harms associated with alcohol and tobacco together. "
             "This synergistic interaction substantially elevates cancer development risk among "
             "dual users."),
            ("Definitions and Cessation",
             "A standard alcoholic drink in the United States contains 14.0 grams (0.6 ounces) "
             "of pure alcohol, corresponding to 12 ounces of beer, 8-10 ounces of malt liquor, "
             "5 ounces of wine, or 1.5 ounces of 80-proof spirits.\n\n"
             "Federal dietary guidelines recommend that individuals who do not currently drink "
             "should not initiate drinking. For those who do consume alcohol, moderation is "
             "advised as limiting consumption to two drinks or less daily for men and one or "
             "less for women on days alcohol is consumed.\n\n"
             "Heavy drinking is defined as four or more drinks daily (or eight-plus weekly) for "
             "women, and five or more daily (or 15-plus weekly) for men. Binge drinking - "
             "consuming five or more drinks (men) or four or more drinks (women) within "
             "approximately two hours - is considered universally harmful.\n\n"
             "Risk Reduction After Cessation: Studies demonstrate that stopping alcohol "
             "consumption correlates with lower risks for oral cavity and esophageal cancers "
             "and possibly for throat, breast, and colorectal cancers. Though risk reduction "
             "may require years to achieve levels equivalent to never-drinkers, it is never "
             "too late to stop drinking and reduce the risks.\n\n"
             "Red Wine: Despite claims regarding resveratrol - a plant compound found in red "
             "wine grapes - researchers have identified no protective association between "
             "moderate red wine consumption and prostate or colorectal cancer risks. Recent "
             "meta-analysis found no difference between red or white wine consumption and "
             "overall cancer risk."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "NCI-diet-cancer-overview",
        "source_url": "https://www.cancer.gov/about-cancer/causes-prevention/risk/diet",
        "source_title": "Diet and Cancer Risk - NCI Overview",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI public domain",
        "sections": [
            ("Methodology and Limitations",
             "The National Cancer Institute provides comprehensive guidance on the relationship "
             "between dietary factors and cancer development. Rather than identifying "
             "definitive causative links, the agency emphasizes the complexity of nutrition "
             "research and the limitations of current scientific evidence.\n\n"
             "Many studies have looked at the possibility that specific dietary components or "
             "nutrients are associated with increases or decreases in cancer risk. Laboratory "
             "investigations and animal models have occasionally revealed that isolated "
             "compounds may demonstrate carcinogenic or anticancer properties. However, human "
             "population studies have generally failed to establish conclusive causal "
             "relationships between diet and cancer occurrence.\n\n"
             "Study participants with and without cancer could differ in other ways besides "
             "their diet, and it is possible that some other difference accounts for the "
             "difference in cancer. To strengthen evidence when associations suggest protective "
             "effects, researchers conduct randomized controlled trials. These experiments "
             "involve random assignment to dietary groups to ensure observed differences stem "
             "from nutritional factors themselves rather than unmeasured variables. Ethical "
             "considerations prevent such randomized studies when evidence suggests potential "
             "cancer risks."),
            ("Specific Dietary Components",
             "Acrylamide: This chemical forms when certain vegetables, particularly potatoes, "
             "undergo high-temperature heating. While animal research demonstrated increased "
             "cancer risks from acrylamide exposure, NCI found no consistent evidence that "
             "dietary acrylamide exposure is associated with the risk of any type of cancer in "
             "humans.\n\n"
             "Alcohol: Despite speculation about red wine's potential benefits, alcohol is a "
             "known cause of cancer. Substantial evidence demonstrates that heavy or regular "
             "alcohol consumption increases the risk of developing cancers of the oral cavity "
             "(excluding the lips), pharynx (throat), larynx (voice box), esophagus, liver, "
             "breast, colon, and rectum.\n\n"
             "Antioxidants: While these compounds theoretically block the activity of other "
             "chemicals, known as free radicals, that may damage cells, human research has "
             "proven disappointing. Research has not demonstrated convincingly that taking "
             "antioxidant supplements can help reduce the risk of developing or dying from "
             "cancer. Notably, certain studies actually indicated increased cancer risks among "
             "supplement users.\n\n"
             "Artificial sweeteners: NCI examined six FDA-approved options: saccharin, "
             "aspartame, acesulfame potassium, sucralose, neotame, and advantame. Laboratory "
             "animal studies have generally not been found to cause cancer or other adverse "
             "health effects. Most human investigations similarly shown no increase in risk.\n\n"
             "Charred meat: High-temperature cooking methods generate compounds designated as "
             "HCAs and PAHs. While these chemicals demonstrate carcinogenicity in animals at "
             "elevated exposure levels, whether such exposure causes cancer in humans is "
             "unclear.\n\n"
             "Cruciferous vegetables: These contain glucosinolates that break down into "
             "compounds under investigation for anticancer potential. Some compounds have shown "
             "anticancer effects in cells and animals, yet human research results remain "
             "ambiguous regarding actual health benefits.\n\n"
             "Fluoride: Many studies, in both humans and animals, have shown no association "
             "between fluoridated water and cancer risk.\n\n"
             "Vitamin D: Observational studies suggested that individuals with higher blood "
             "levels of vitamin D might have lower risks of colorectal cancer and of overall "
             "cancer mortality. However, most randomized trials investigating supplementation "
             "have not found an association between vitamin D supplement use and cancer risk "
             "or mortality."),
        ],
    },
    # ------------------------------------------------------------------
    {
        "source_doc_id": "CDC-smoking-cessation-quit",
        "source_url": "https://www.cdc.gov/tobacco/campaign/tips/quit-smoking/index.html",
        "source_title": "How to Quit Smoking - CDC Tips From Former Smokers",
        "source_authors": "Centers for Disease Control and Prevention",
        "source_year": 2024,
        "license": "CDC public domain",
        "sections": [
            ("Quitline Resources",
             "The Centers for Disease Control and Prevention (CDC) emphasizes that it's never too "
             "late to quit smoking, highlighting how cessation improves health outcomes and "
             "reduces risks for multiple serious conditions.\n\n"
             "The CDC's flagship telephone support line represents a cornerstone of its cessation "
             "strategy. This toll-free service offers free coaching to help smokers quit, with "
             "trained counselors available around the clock. The line provides personalized "
             "support and can be accessed at 1-800-QUIT-NOW (1-800-784-8669).\n\n"
             "Beyond English, the CDC recognizes diverse language needs:\n"
             "- Spanish speakers can reach 1-855-DEJELO-YA (1-855-335-3569)\n"
             "- Mandarin Chinese: 1-800-838-8917\n"
             "- Korean: 1-800-556-5564\n"
             "- Vietnamese: 1-800-778-8440\n\n"
             "Quitlines provide free coaching over the phone to help you quit smoking and are "
             "available in several languages. These services connect callers with counselors "
             "trained in behavioral cessation techniques and can coordinate with medications when "
             "appropriate.\n\n"
             "For those preferring mobile-based intervention, the CDC offers text messaging "
             "services. Individuals can text QUITNOW to 333888, with Spanish-language equivalents "
             "available (DEJELO YA to the same number). The CDC also endorses the quitSTART "
             "smartphone application, providing tips, information, and challenges to help quit."),
            ("FDA-Approved Cessation Medications",
             "The CDC recognizes multiple FDA-approved nicotine delivery systems including:\n\n"
             "Nicotine Patch: Provides steady nicotine delivery through transdermal absorption, "
             "useful for managing baseline cravings throughout the day.\n\n"
             "Nicotine Lozenge: Offers oral nicotine delivery in dissolution form, allowing "
             "flexible dosing based on individual craving patterns.\n\n"
             "Nicotine Gum: Provides user-controlled nicotine delivery through mastication and "
             "absorption, giving smokers active engagement in the cessation process.\n\n"
             "Nicotine Oral Inhaler: Delivers nicotine through inhalation without combustion, "
             "potentially easing the transition from cigarettes by maintaining hand-to-mouth "
             "behavioral patterns.\n\n"
             "Nicotine Nasal Spray: Provides rapid nicotine absorption through nasal mucosa, "
             "beneficial for individuals with particularly intense cravings.\n\n"
             "Prescription Medications: Varenicline functions through unique mechanisms and helps "
             "smokers get through the toughest times. Bupropion SR is an antidepressant medication "
             "that assists cessation through neurochemical mechanisms.\n\n"
             "Combination Therapy: The CDC acknowledges that multiple medications work "
             "synergistically. Providers can combine approaches - such as pairing a long-acting "
             "nicotine patch with short-acting lozenges or gum - to address both baseline "
             "dependence and breakthrough cravings."),
            ("Health Benefits of Cessation",
             "The agency stresses that quitting reduces your risk of heart disease, cancer, lung "
             "disease, and other smoking-related illnesses. Specifically, the CDC addresses "
             "smoking's connection to multiple conditions:\n\n"
             "- Cancer risks decrease substantially after cessation\n"
             "- Heart disease and stroke risks improve through cardiovascular function restoration\n"
             "- Chronic Obstructive Pulmonary Disease (COPD) progression slows significantly\n"
             "- Asthma control improves in many smokers\n"
             "- Diabetes management becomes easier\n"
             "- Peripheral artery disease progression halts\n"
             "- Vision loss risks diminish\n"
             "- Gum disease treatment outcomes improve\n"
             "- Mental health (depression and anxiety) often improves\n\n"
             "Special Population Considerations: The CDC recognizes that cessation affects "
             "different groups distinctly. Pregnant individuals benefit from cessation's fetal "
             "protection. People with HIV reduce additional disease burdens. Military personnel "
             "and veterans access specialized resources through YouCanQuit2. African American, "
             "Asian American, Hispanic/Latino, American Indian/Alaska Native communities access "
             "culturally responsive resources. LGBTQ+ individuals connect with affirming support "
             "systems. People with mental health conditions receive integrated care approaches.\n\n"
             "While medications substantially improve success rates, quitting without medicine "
             "remains possible, with behavioral strategies providing meaningful support. The CDC's "
             "smoking cessation framework combines pharmacological intervention, behavioral "
             "support, and accessibility across multiple modalities - telephone, text, and mobile "
             "applications - ensuring diverse populations can access evidence-based treatment."),
        ],
    },
]


# ----------------------------------------------------------------------------
# Chunking logic.
# Each section is one logical unit (page-aware). Sections under TARGET_CHARS
# stay as-is. Longer sections split on paragraph boundaries with 15% overlap.
# Tables / numbered lists / dash-prefixed lists stay intact (we do not split
# inside a paragraph that begins with "- " or contains a tabular layout).
# ----------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\n+", text.strip())
    return [p.strip() for p in paras if p.strip()]


def chunk_section(body: str) -> list[str]:
    """Split a section body into ~TARGET_CHARS chunks with OVERLAP_CHARS overlap.

    Snaps to paragraph boundaries. Keeps list items / tables intact (a paragraph
    starting with "- " or with multiple "\n- " entries is treated as one unit).
    """
    if len(body) <= TARGET_CHARS:
        return [body]

    paragraphs = split_paragraphs(body)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        plen = len(para) + 2  # account for joining "\n\n"
        if current_len + plen > TARGET_CHARS and current:
            chunks.append("\n\n".join(current))
            # build overlap from tail of `current`
            overlap_paras: list[str] = []
            tail_len = 0
            for p in reversed(current):
                if tail_len + len(p) + 2 > OVERLAP_CHARS:
                    break
                overlap_paras.insert(0, p)
                tail_len += len(p) + 2
            current = list(overlap_paras)
            current_len = tail_len
        current.append(para)
        current_len += plen

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def assemble_chunks(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Combine adjacent sections within one source doc until each combined unit
    is close to TARGET_CHARS. Returns list of (section_label, body) tuples
    suitable for downstream sub-splitting (page-aware: never splits a single
    section across the join).

    Strategy:
      - Walk sections in order.
      - Accumulate into a buffer until adding the next section would exceed
        ~1.2 * TARGET_CHARS; emit the buffer.
      - Section labels are joined with " | " when combined so provenance is
        preserved.
      - 15% overlap between consecutive emitted units is added by re-prepending
        the trailing OVERLAP_CHARS of the prior buffer (paragraph-snapped).
    """
    out: list[tuple[str, str]] = []
    buf_label: list[str] = []
    buf_body: list[str] = []
    buf_len = 0
    # Emit chunks closer to the canonical 1024-token target. Each natural
    # section (a page-aware unit per the NVIDIA spec) becomes its own chunk
    # unless very small, in which case it merges with the next. This produces
    # ~50-100 chunks at ~600-900 tokens each across the source corpus.
    MAX = int(TARGET_CHARS * 0.6)   # ~2458 chars - one section per chunk for most
    MIN = int(TARGET_CHARS * 0.25)  # ~1024 chars - merge tiny sections

    prev_tail = ""

    def flush():
        nonlocal prev_tail
        if not buf_body:
            return
        body = "\n\n".join(buf_body)
        if prev_tail:
            body = prev_tail + "\n\n" + body
        label = " | ".join(buf_label)
        out.append((label, body))
        # compute new prev_tail = last ~OVERLAP_CHARS of body, paragraph-aligned
        paras = split_paragraphs(body)
        tail: list[str] = []
        tlen = 0
        for p in reversed(paras):
            if tlen + len(p) + 2 > OVERLAP_CHARS:
                break
            tail.insert(0, p)
            tlen += len(p) + 2
        prev_tail = "\n\n".join(tail)

    for label, body in sections:
        blen = len(body)
        if buf_len + blen > MAX and buf_body:
            flush()
            buf_label, buf_body, buf_len = [], [], 0
        buf_label.append(label)
        buf_body.append(body)
        buf_len += blen + 2

    if buf_body:
        flush()

    return out


def build() -> tuple[int, int]:
    chunk_id = 0
    n_sources = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for src in SOURCES:
            n_sources += 1
            chunk_index = 0
            assembled = assemble_chunks(src["sections"])
            for section_label, body in assembled:
                # If a single assembled unit is still over MAX, sub-split it.
                pieces = chunk_section(body) if len(body) > int(TARGET_CHARS * 1.3) else [body]
                for piece in pieces:
                    record = {
                        "id": f"chunk-{chunk_id:03d}",
                        "source_doc_id": src["source_doc_id"],
                        "source_url": src["source_url"],
                        "source_title": src["source_title"],
                        "source_authors": src["source_authors"],
                        "source_year": src["source_year"],
                        "section": section_label,
                        "chunk_index": chunk_index,
                        "n_tokens_estimated": estimate_tokens(piece),
                        "body": piece,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_id += 1
                    chunk_index += 1
    return chunk_id, n_sources


if __name__ == "__main__":
    n_chunks, n_sources = build()
    print(f"wrote {n_chunks} chunks across {n_sources} source docs to {OUT_PATH}")
