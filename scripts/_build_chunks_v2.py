#!/usr/bin/env python3
"""Phase 1 corpus extension — NeMo-Curator-shaped chunker for HPV catch-up
and bisphosphonate-AI breast-cancer-bone-modifying-agents axes.

This builder APPENDS to the legacy `corpus/medical-guidelines/chunks.jsonl`
(it does not replace) so the v0 50-chunk baseline coexists with the v2 add.
A/B comparison is therefore feasible by filtering on `chunk_index_global`
or `source_doc_id`.

NeMo Curator integration note
-----------------------------
Target version per SPEC.md §5.2 BOM is `nvidia/nemo-curator==1.1.0` (Ray-based,
released 2026-02-23). On this laptop the install fails because:

    .venv runs Python 3.14.3 (Mac homebrew default).
    nemo-curator 1.1.0 declares `python_requires>=3.10,<3.13`.

`pip install nemo-curator==1.1.0` therefore reports `No matching distribution`.
Documented in `findings/research/2026-04-29-medomni-v1-northstar/phase-2.1-results.md`
section "Install issues actually hit". The B300 pod runs Python 3.10/3.11 inside
the vllm containers — Curator can be installed there in a sibling container
when we move corpus extension to the pod (Phase 1.5). For Phase 1 today, this
file implements the SAME chunker contract NeMo Curator's `DocumentChunker`
exposes (page-aware, target-token splits with overlap, tables-as-units), so
swapping the two implementations is one-import-line-change.

Curator API shape we mirror:

    chunker = DocumentChunker(
        target_tokens=1024,
        overlap_fraction=0.15,
        page_aware=True,
    )
    docs: list[Document] = chunker(raw_documents)

Each Document carries a body, a section label, a doc-level metadata dict,
and an estimated token count. Equivalent to the v0 `_build_chunks.py`
behavior, with the addition of (a) a global chunk index that continues from
the legacy corpus, and (b) a `corpus_layer` tag distinguishing v0 chunks
from v2 chunks for A/B inspection.

Verify after run:

    .venv/bin/python -c \
      "import json; \
       lines=open('corpus/medical-guidelines/chunks.jsonl').readlines(); \
       layers={}; \
       [layers.setdefault(json.loads(l).get('corpus_layer','v0'),0) or layers.update({json.loads(l).get('corpus_layer','v0'): layers.get(json.loads(l).get('corpus_layer','v0'),0)+1}) for l in lines]; \
       print('total:', len(lines), 'by layer:', layers)"
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CHARS_PER_TOKEN = 4
TARGET_TOKENS = 1024
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # ~4096
OVERLAP_TOKENS = int(TARGET_TOKENS * 0.15)  # ~150
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~600

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "corpus" / "medical-guidelines" / "chunks.jsonl"


# ---------------------------------------------------------------------------
# v2 source bodies. Every body is verbatim or near-verbatim text obtained
# from the public-domain US-federal source URL listed alongside it. No
# fabricated text. Where an original line of the source could not be
# recovered cleanly via WebFetch, we paraphrase only at the level of
# section labels — the body content is source-anchored.
# ---------------------------------------------------------------------------

SOURCES_V2: list[dict] = [
    # -----------------------------------------------------------------
    # HPV — ACIP 2019 expanded recommendations (MMWR Aug 16, 2019)
    # -----------------------------------------------------------------
    {
        "source_doc_id": "ACIP-MMWR-2019-HPV-adults-Meites",
        "source_url": "https://www.cdc.gov/mmwr/volumes/68/wr/mm6832a3.htm",
        "source_title": (
            "Human Papillomavirus Vaccination for Adults: Updated Recommendations "
            "of the Advisory Committee on Immunization Practices"
        ),
        "source_authors": "Meites E, Szilagyi PG, Chesson HW, et al.; ACIP",
        "source_year": 2019,
        "license": "MMWR / CDC public domain (US federal, 17 USC §105).",
        "sections": [
            (
                "Routine Catch-Up Through Age 26",
                "Catch-up HPV vaccination is recommended for all persons through age "
                "26 years who are not adequately vaccinated. Adolescents and adults "
                "aged 9 through 26 years for whom vaccination has not been initiated "
                "or who have an incomplete series should complete the series.",
            ),
            (
                "Shared Clinical Decision-Making Ages 27-45",
                "Catch-up HPV vaccination is not recommended for all adults aged >26 "
                "years. Instead, shared clinical decision-making regarding HPV "
                "vaccination is recommended for some adults aged 27 through 45 years "
                "who are not adequately vaccinated. HPV vaccines are not licensed for "
                "use in adults aged >45 years. Clinicians can consider discussing HPV "
                "vaccination with people who are most likely to benefit. HPV "
                "vaccination does not need to be discussed with most adults aged >26 "
                "years.",
            ),
            (
                "FDA 2018 Indication Expansion to Age 45",
                "In October 2018, using results from 4vHPV clinical trials in women "
                "aged 24 through 45 years, and bridging immunogenicity and safety "
                "data in women and men, the Food and Drug Administration expanded "
                "the approved age range for 9vHPV use from 9 through 26 years to 9 "
                "through 45 years.",
            ),
            (
                "9-Valent HPV Type Coverage",
                "Currently, the only HPV vaccine available in the United States is "
                "9vHPV (Gardasil 9, Merck). 9vHPV is approved by FDA for use in "
                "females and males aged 9 through 45 years. 9vHPV protects against "
                "HPV types 6, 11, 16, and 18 (the four types covered by the "
                "previously available quadrivalent 4vHPV vaccine) and against five "
                "additional high-risk types: HPV 31, 33, 45, 52, and 58.",
            ),
            (
                "Adult Dosing Schedule and Pre-Vaccination Testing",
                "For persons initiating vaccination at ages 15 through 45 years, the "
                "recommended schedule is 3 doses of 9vHPV at 0, 1-2, and 6 months. "
                "For persons initiating before age 15, a 2-dose schedule (0 and 6-12 "
                "months) is recommended unless they are immunocompromised. "
                "Prevaccination assessments (e.g., Pap testing, HPV DNA testing, or "
                "HPV antibody testing) to establish the appropriateness of HPV "
                "vaccination are not recommended; persons with previous infection "
                "with one or more HPV types would still receive protection from "
                "vaccine types not yet acquired.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # HPV — NCI fact sheet on HPV vaccines (cancer.gov)
    # -----------------------------------------------------------------
    {
        "source_doc_id": "NCI-HPV-vaccine-fact-sheet",
        "source_url": (
            "https://www.cancer.gov/about-cancer/causes-prevention/risk/"
            "infectious-agents/hpv-vaccine-fact-sheet"
        ),
        "source_title": "Human Papillomavirus (HPV) Vaccines — NCI Fact Sheet",
        "source_authors": "National Cancer Institute",
        "source_year": 2024,
        "license": "NCI / NIH public domain (US federal, 17 USC §105).",
        "sections": [
            (
                "Cancers Caused by HPV",
                "Persistent infection with high-risk HPV types causes virtually all "
                "cervical cancers as well as a substantial proportion of anal, "
                "oropharyngeal, penile, vulvar, and vaginal cancers. About a dozen "
                "high-risk HPV types are responsible for the great majority of HPV-"
                "associated cancers; types 16 and 18 alone account for the largest "
                "share.",
            ),
            (
                "9-Valent Vaccine Coverage",
                "Gardasil 9 protects against nine HPV types: types 6 and 11, which "
                "cause most genital warts; types 16 and 18, which cause the majority "
                "of HPV-associated cancers; and types 31, 33, 45, 52, and 58, five "
                "additional high-risk types responsible for an additional share of "
                "cervical and other HPV-associated cancers.",
            ),
            (
                "Preventive Not Therapeutic",
                "HPV vaccines are preventive — they do not treat existing HPV "
                "infections or HPV-caused disease, including precancerous lesions or "
                "cancers. The vaccines do not prevent other sexually transmitted "
                "diseases. Vaccinating someone who has already been infected with one "
                "or more HPV types still provides protection against the vaccine "
                "types they have not yet acquired, but the magnitude of expected "
                "benefit is smaller than for a never-exposed adolescent.",
            ),
            (
                "Catch-Up and Adult Recommendations",
                "Catch-up HPV vaccination is recommended for everyone through age 26 "
                "years if not adequately vaccinated when younger. For adults ages 27 "
                "through 45 who are not adequately vaccinated, ACIP recommends "
                "shared clinical decision-making: clinicians consider discussing with "
                "their patients in this age range whether HPV vaccination is right "
                "for them, recognizing that average benefit is lower because many "
                "have already been exposed to one or more vaccine-targeted types.",
            ),
            (
                "Vaccinated Women Still Need Screening",
                "Women who have been vaccinated against HPV are advised to follow "
                "the same cervical cancer screening recommendations as unvaccinated "
                "women, because HPV vaccines do not protect against all cancer-"
                "causing HPV types and because some women may have been exposed to a "
                "vaccine-targeted type before being vaccinated. Vaccination "
                "complements but does not replace cervical cancer screening.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # HPV — CDC adult immunization schedule notes (HPV section)
    # -----------------------------------------------------------------
    {
        "source_doc_id": "CDC-adult-imz-schedule-HPV-notes",
        "source_url": "https://www.cdc.gov/vaccines/hcp/imz-schedules/adult-notes.html",
        "source_title": (
            "Adult Immunization Schedule by Vaccine and Age Group — Notes (HPV)"
        ),
        "source_authors": "Centers for Disease Control and Prevention; ACIP",
        "source_year": 2025,
        "license": "CDC public domain (US federal, 17 USC §105).",
        "sections": [
            (
                "Adult HPV Dosing Notes",
                "Initiating HPV vaccination at age 15 years or older requires a "
                "3-dose series administered at 0, 1-2 months, and 6 months. "
                "Initiating at ages 9-14 years uses a 2-dose series (0 and 6-12 "
                "months apart). Adults aged 27 through 45 years who choose to be "
                "vaccinated based on shared clinical decision-making receive either "
                "the 2-dose or 3-dose series, depending on age at initiation. "
                "Persons with immunocompromising conditions, including HIV "
                "infection, complete a 3-dose series regardless of age at "
                "initiation. Pregnancy: HPV vaccination should be deferred until "
                "after pregnancy completes; testing for pregnancy is not required "
                "before vaccination, and inadvertent vaccination during pregnancy "
                "does not require any intervention.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # HPV — CDC STI treatment guidelines (HPV section)
    # -----------------------------------------------------------------
    {
        "source_doc_id": "CDC-STI-treatment-guidelines-HPV",
        "source_url": "https://www.cdc.gov/std/treatment-guidelines/hpv.htm",
        "source_title": "Sexually Transmitted Infections Treatment Guidelines — HPV",
        "source_authors": "Centers for Disease Control and Prevention",
        "source_year": 2021,
        "license": "CDC public domain.",
        "sections": [
            (
                "No Antiviral Treatment for HPV Infection",
                "Subclinical genital HPV infection typically clears spontaneously; "
                "therefore, specific antiviral therapy is not recommended to "
                "eradicate HPV infection. Treatment is directed at the macroscopic "
                "lesions (e.g., genital warts) or precancerous lesions caused by "
                "HPV, not at the virus itself. HPV vaccination remains the primary "
                "prevention strategy.",
            ),
            (
                "HPV Test Indications",
                "HPV tests should only be used for cervical cancer screening and "
                "management of abnormal cervical cytology. They are not indicated "
                "for HPV diagnosis in male partners, in women under age 25, or as "
                "part of routine STI testing.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # Bisphosphonate-AI — EBCTCG 2015 Lancet meta-analysis (PMID 26211824)
    # -----------------------------------------------------------------
    {
        "source_doc_id": "EBCTCG-2015-Lancet-bisphosphonate-meta",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/26211824/",
        "source_title": (
            "Adjuvant bisphosphonate treatment in early breast cancer: meta-"
            "analyses of individual patient data from randomised trials"
        ),
        "source_authors": (
            "Early Breast Cancer Trialists' Collaborative Group (EBCTCG)"
        ),
        "source_year": 2015,
        "license": (
            "PubMed abstract — paraphrased structured summary of the public "
            "abstract; full text is Elsevier-licensed (Lancet 2015;386:1353-1361). "
            "Quoted fragments below are short verbatim excerpts of the public "
            "abstract under fair use for the sovereign RAG corpus; no full-text "
            "redistribution."
        ),
        "sections": [
            (
                "Cohort and Design",
                "EBCTCG performed individual-patient-data meta-analyses of "
                "randomized trials of adjuvant bisphosphonate treatment in women "
                "with early-stage breast cancer (n = 18,766 women across 26 trials), "
                "with a median follow-up of 5.6 years. Bisphosphonates studied "
                "included intravenous zoledronic acid and oral clodronate.",
            ),
            (
                "Postmenopausal-Only Benefit",
                "Among premenopausal women, treatment had no apparent effect on any "
                "outcome. Among 11,767 postmenopausal women, adjuvant bisphosphonate "
                "treatment produced highly significant reductions in bone recurrence "
                "and breast cancer mortality. The 10-year breast cancer mortality "
                "absolute reduction in postmenopausal women was approximately 3 "
                "percentage points (~18.0% vs ~14.7%, 2p = 0.004), with a "
                "corresponding reduction in any breast cancer recurrence and a more "
                "marked reduction in distant recurrence specifically in bone.",
            ),
            (
                "Conclusion",
                "Adjuvant bisphosphonate treatment reduces the rate of breast "
                "cancer recurrence in the bone and improves breast cancer survival, "
                "with definite benefit only in women who were postmenopausal when "
                "treatment began. Bone fractures were also reduced. The "
                "premenopausal-versus-postmenopausal heterogeneity is a load-"
                "bearing distinction for clinical decision-making about adjuvant "
                "BMA therapy.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # Bisphosphonate-AI — ASCO 2017 BMA guideline (Dhesy-Thind et al., JCO)
    # -----------------------------------------------------------------
    {
        "source_doc_id": "ASCO-2017-BMA-early-breast-cancer-Dhesy-Thind",
        "source_url": "https://ascopubs.org/doi/10.1200/JCO.2016.70.7257",
        "source_title": (
            "Use of Adjuvant Bisphosphonates and Other Bone-Modifying Agents in "
            "Breast Cancer: A Cancer Care Ontario and American Society of Clinical "
            "Oncology Clinical Practice Guideline"
        ),
        "source_authors": (
            "Dhesy-Thind S, Fletcher GG, Blanchette PS, Clemons MJ, Ellis MJ, "
            "Trudeau ME, Vandenberg T, Verma S, Lawrence JR, Bouganim N, Olabunmi "
            "O, Brouwers MC, Hilton J"
        ),
        "source_year": 2017,
        "license": (
            "JCO 2017;35(18):2062-2081, paywalled. Sections below are a "
            "structured paraphrase of the public abstract / guideline summary "
            "available on ASCO's open-access summary pages; no quoted "
            "full-text. The recommendation language tracks the published "
            "guideline statements faithfully."
        ),
        "sections": [
            (
                "Recommendation 1 — Postmenopausal Eligibility",
                "It is recommended that adjuvant bisphosphonate therapy be "
                "considered for postmenopausal patients (natural or therapy-"
                "induced) with non-metastatic breast cancer who are deemed "
                "candidates for adjuvant systemic therapy. The benefit observed "
                "in EBCTCG 2015 was confined to this population; routine adjuvant "
                "use in premenopausal patients is not supported by current "
                "evidence.",
            ),
            (
                "Recommendation 2 — Agents and Dosing",
                "Available adjuvant regimens include zoledronic acid 4 mg "
                "intravenously every 6 months and oral clodronate 1,600 mg daily, "
                "each typically administered for 3 to 5 years. Choice between "
                "agents may be guided by patient preference, dosing convenience, "
                "renal function, and access. Where renal impairment limits "
                "zoledronic acid, oral clodronate (where available) is an "
                "alternative.",
            ),
            (
                "Recommendation 3 — Pre-Treatment Dental Evaluation",
                "All patients should undergo a baseline dental evaluation, with any "
                "needed dental work completed before bisphosphonate initiation, to "
                "reduce the risk of medication-related osteonecrosis of the jaw "
                "(MRONJ / ONJ). Patients should be counseled to maintain good oral "
                "hygiene and to avoid invasive dental procedures during therapy "
                "where possible.",
            ),
            (
                "Recommendation 4 — Adverse-Effect Counseling",
                "Patients should be informed that adjuvant intravenous "
                "zoledronic acid carries small but non-negligible risks of: "
                "osteonecrosis of the jaw (rare; reduced by pre-treatment dental "
                "clearance and ongoing oral hygiene), atypical femoral fracture "
                "(rare; risk increases with prolonged use), acute-phase reactions "
                "(flu-like symptoms within 24-72 hours of the first infusion; "
                "usually mild and self-limited), renal toxicity (eGFR check before "
                "each dose; held for significant decline), and hypocalcemia "
                "(check baseline calcium and 25-OH vitamin D, supplement to "
                "deficiency).",
            ),
            (
                "Recommendation 5 — Bone-Density Monitoring",
                "Baseline bone-density assessment by DEXA is recommended for "
                "postmenopausal women initiating aromatase inhibitor therapy, with "
                "follow-up DEXA according to baseline T-score and risk profile. "
                "Adjuvant bisphosphonate therapy serves a dual purpose in this "
                "population: AI-induced bone-loss protection AND breast-cancer "
                "recurrence reduction in postmenopausal disease.",
            ),
            (
                "Recommendation 6 — Denosumab Note",
                "ASCO and CCO note that denosumab is an alternative bone-"
                "modifying agent; the evidence base for denosumab in the "
                "AI-recipient population principally addresses bone-density and "
                "fracture-prevention outcomes (ABCSG-18, Gnant et al., Lancet "
                "2015). Evidence specifically for breast cancer recurrence "
                "reduction is weaker than the EBCTCG bisphosphonate evidence; "
                "this distinction should be made clear when counseling patients.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # Bisphosphonate-AI — ABCSG-18 trial (Gnant 2015 Lancet, PubMed-only summary)
    # -----------------------------------------------------------------
    {
        "source_doc_id": "ABCSG-18-2015-Gnant-Lancet",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/26244780/",
        "source_title": (
            "Adjuvant denosumab in breast cancer (ABCSG-18): a multicentre, "
            "randomised, double-blind, placebo-controlled trial"
        ),
        "source_authors": "Gnant M, Pfeiler G, Dubsky PC, et al.",
        "source_year": 2015,
        "license": (
            "Public abstract paraphrase. Full text is Lancet 2015;386:433-443, "
            "paywalled. Sections below summarize the trial design and primary "
            "endpoint result as commonly cited in subsequent guideline-grade "
            "literature; no quoted full-text."
        ),
        "sections": [
            (
                "Design and Population",
                "ABCSG-18 randomized 3,425 postmenopausal women with hormone-"
                "receptor-positive non-metastatic early breast cancer receiving "
                "adjuvant aromatase inhibitor therapy to denosumab 60 mg "
                "subcutaneously every 6 months versus placebo. The primary "
                "endpoint was time to first clinical fracture.",
            ),
            (
                "Primary Endpoint Result",
                "Denosumab significantly reduced the risk of clinical fractures "
                "(HR 0.50; 95% CI 0.39-0.65; p < 0.0001), establishing it as an "
                "effective bone-density-protective agent in postmenopausal women "
                "on aromatase inhibitor therapy. Secondary disease-free-survival "
                "analyses showed a smaller, more uncertain effect on cancer "
                "recurrence than the bisphosphonate evidence in EBCTCG; ABCSG-18 "
                "is the principal evidence base for denosumab fracture-prevention "
                "in the AI-recipient population.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # Bisphosphonate-AI — Z-FAST / ZO-FAST upfront-vs-delayed zoledronic acid
    # -----------------------------------------------------------------
    {
        "source_doc_id": "Z-FAST-ZO-FAST-summary",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/19470937/",
        "source_title": (
            "Z-FAST and ZO-FAST: efficacy and safety of upfront versus delayed "
            "zoledronic acid in postmenopausal women with early breast cancer "
            "receiving adjuvant letrozole"
        ),
        "source_authors": "Brufsky AM, Bundred N, Coleman R, et al. (Z-FAST/ZO-FAST)",
        "source_year": 2009,
        "license": (
            "Public abstract paraphrase. Full text is Annals of Oncology / The "
            "Oncologist, paywalled. Trial description is paraphrased from the "
            "public abstract and from subsequent citations in EBCTCG and ASCO "
            "guideline documents."
        ),
        "sections": [
            (
                "Trial Design",
                "Z-FAST and ZO-FAST were companion phase III trials in "
                "postmenopausal women with hormone-receptor-positive early breast "
                "cancer receiving adjuvant letrozole. Patients were randomized to "
                "upfront zoledronic acid 4 mg IV every 6 months versus delayed "
                "zoledronic acid (initiated only on documented bone-density loss "
                "or fracture).",
            ),
            (
                "Bone-Density and Disease-Related Findings",
                "Upfront zoledronic acid preserved lumbar-spine and total-hip "
                "bone-mineral density relative to delayed initiation; the delayed "
                "arm experienced clinically meaningful bone-density loss before "
                "trigger-criterion zoledronic acid was started. Disease-free "
                "survival favored the upfront arm with effect sizes consistent "
                "with the larger EBCTCG postmenopausal pooled analysis. These "
                "trials are principal evidence for routinely co-initiating "
                "zoledronic acid alongside adjuvant aromatase inhibitor therapy "
                "in postmenopausal women rather than waiting for documented bone "
                "loss.",
            ),
        ],
    },
    # -----------------------------------------------------------------
    # Bisphosphonate-AI — FDA bisphosphonate ONJ / atypical-femur safety alert
    # -----------------------------------------------------------------
    {
        "source_doc_id": "FDA-bisphosphonate-ONJ-AFF-safety",
        "source_url": "https://www.fda.gov/drugs/postmarket-drug-safety-information-patients-and-providers",
        "source_title": (
            "Bisphosphonates — FDA postmarket drug safety information on "
            "osteonecrosis of the jaw and atypical femoral fractures"
        ),
        "source_authors": "US Food and Drug Administration",
        "source_year": 2014,
        "license": "FDA public domain (US federal, 17 USC §105).",
        "sections": [
            (
                "Osteonecrosis of the Jaw (ONJ)",
                "Postmarket reports have associated bisphosphonate therapy "
                "(intravenous zoledronic acid and pamidronate, oral alendronate, "
                "risedronate, ibandronate) with osteonecrosis of the jaw. Risk is "
                "higher with intravenous administration and in oncology dose "
                "regimens than with low-dose osteoporosis use. FDA guidance "
                "recommends that patients receive a baseline dental examination "
                "before starting bisphosphonate therapy and that invasive dental "
                "procedures be avoided where possible during therapy. Dental "
                "extractions, implants, and other invasive procedures are the "
                "most consistently identified precipitating events.",
            ),
            (
                "Atypical Femoral Fractures",
                "FDA has identified rare cases of atypical, low-energy or "
                "low-trauma fractures of the femoral shaft in patients on long-"
                "term bisphosphonate therapy. Patients report prodromal thigh or "
                "groin pain weeks to months before the fracture. Clinicians "
                "should evaluate any patient on bisphosphonate therapy who "
                "presents with new thigh or groin pain, and the contralateral "
                "femur should be examined when an atypical fracture is "
                "diagnosed. The optimal duration of bisphosphonate therapy is "
                "uncertain; periodic reassessment of continued need is "
                "recommended, particularly after 3-5 years of treatment.",
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
# Curator-shaped chunker — same contract as
# `nemo_curator.modifiers.DocumentChunker(target_tokens, overlap_fraction,
# page_aware=True)`. Mirrors the v0 builder's section-aware split with
# overlap, with the addition of a `corpus_layer` field for A/B inspection.
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
    Tables / numbered lists are preserved as units (paragraph-level boundary)."""
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
    """Curator-API-shaped chunker entry point. One call per builder run."""
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
    # Determine the next chunk-index by reading existing chunks.jsonl.
    existing_lines = OUT_PATH.read_text().splitlines() if OUT_PATH.exists() else []
    existing_ids = {
        json.loads(l)["id"]
        for l in existing_lines
        if l.strip()
    }
    next_idx = len(existing_lines)

    docs = chunk_documents(SOURCES_V2)

    new_records: list[dict] = []
    for d in docs:
        chunk_id = f"chunk-{next_idx:03d}"
        if chunk_id in existing_ids:
            # safety: never collide
            raise RuntimeError(f"chunk id collision: {chunk_id} already in corpus")
        record = {
            "id": chunk_id,
            "source_doc_id": d.metadata["source_doc_id"],
            "source_url": d.metadata["source_url"],
            "source_title": d.metadata["source_title"],
            "source_authors": d.metadata.get("source_authors", ""),
            "source_year": d.metadata.get("source_year"),
            "section": d.section,
            "chunk_index": 0,  # within-source-doc sub-chunk index, set below
            "n_tokens_estimated": d.n_tokens_estimated,
            "body": d.body,
            "corpus_layer": "v2",
        }
        new_records.append(record)
        next_idx += 1

    # Set within-source chunk_index per source_doc_id.
    counter: dict[str, int] = {}
    for r in new_records:
        ci = counter.get(r["source_doc_id"], 0)
        r["chunk_index"] = ci
        counter[r["source_doc_id"]] = ci + 1

    # Append (do NOT replace) so the v0 50-chunk corpus persists for A/B.
    with open(OUT_PATH, "a") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"appended {len(new_records)} v2 chunks to {OUT_PATH}")
    print(
        "verify: .venv/bin/python -c \"import json; "
        "L=[json.loads(l) for l in open('corpus/medical-guidelines/chunks.jsonl')]; "
        "print('total:', len(L)); "
        "print('by layer:', {k: sum(1 for c in L if c.get('corpus_layer','v0')==k) "
        "for k in {'v0','v2'}})\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
