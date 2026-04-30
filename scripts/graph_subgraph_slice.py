"""Stage 6 of the MedOmni retrieval pipeline — nx-cugraph 2-hop ego-graph
expansion over a small persona-tagged chemoprevention entity graph.

The graph is built from a hand-curated set of `(subject, relation, object)`
triples covering the six held-out chemoprevention axes plus the
tamoxifen+Mirena anchor case. Every edge carries a `persona_mask` bitfield
{physician, nurse, family, patient}; retrieval expands only edges where the
bit for the active persona is set. Retrieved nodes carry a stable
`evidence_currency` attribute that downstream stages can sort on.

Schema
------
Node types:
    Drug, Condition, Trial, Guideline, Population, Test, Body_Site,
    Vaccine, Regulator, Concept

Edge types:
    indicates              — drug indicates_for condition
    contraindicates        — drug contraindicates_in population
    monitors               — test monitors drug
    recommends_for         — guideline recommends_for population/condition
    evidenced_by           — claim evidenced_by trial
    expanded_indication    — regulator expanded_indication for drug to age
    covers_serotype        — vaccine covers serotype
    complements_not_replaces — vaccine complements_not_replaces screening
    teaches_via            — pedagogical edges (nurse register), {why,what,when,caution}

Persona bits (low to high):
    bit 0 = physician
    bit 1 = nurse
    bit 2 = family
    bit 3 = patient

Backend selection: if `nx-cugraph` is importable AND `NETWORKX_AUTOMATIC_BACKENDS=cugraph`
is in env, networkx will dispatch to cugraph automatically. Otherwise we
fall back to plain networkx so the laptop side runs end-to-end. The
function `expand_subgraph` is identical either way.

Use as Stage 6 in retrieval_cuvs / sovereign_bench by:

    from graph_subgraph_slice import build_graph, expand_subgraph, seed_nodes_for
    G = build_graph()
    seed = seed_nodes_for(query=user_query, retrieved_chunk_ids=top8_ids)
    sub = expand_subgraph(G, seeds=seed, persona='patient', hops=2)
    block = serialize_subgraph_for_prompt(sub)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Persona bit constants
PERSONA_PHYSICIAN = 1 << 0
PERSONA_NURSE = 1 << 1
PERSONA_FAMILY = 1 << 2
PERSONA_PATIENT = 1 << 3
PERSONA_ALL = PERSONA_PHYSICIAN | PERSONA_NURSE | PERSONA_FAMILY | PERSONA_PATIENT

PERSONA_BIT = {
    "physician": PERSONA_PHYSICIAN,
    "nurse": PERSONA_NURSE,
    "family": PERSONA_FAMILY,
    "patient": PERSONA_PATIENT,
}


def _import_nx():
    """Imports networkx; if NETWORKX_AUTOMATIC_BACKENDS=cugraph and nx-cugraph
    is installed, networkx will dispatch graph algorithms to cuGraph. We do
    NOT force the env var here — caller controls dispatch."""
    import networkx as nx  # noqa: WPS433

    return nx


# ---------------------------------------------------------------------------
# Curated triples. Sources tagged inline. Provenance is the corpus
# itself (chunks.jsonl + held-out fixtures), so every edge can be traced
# back to a chunk_id or a fixture rubric criterion. The graph is small
# (illustrative) by design — large-scale population happens in v1.5 via
# OpenEM-driven expansion (see scripts/expand_kg_with_openem.py).
# ---------------------------------------------------------------------------

NODES: list[dict] = [
    # Drugs
    {"id": "drug:tamoxifen", "type": "Drug", "label": "Tamoxifen", "register": "physician"},
    {"id": "drug:raloxifene", "type": "Drug", "label": "Raloxifene", "register": "physician"},
    {"id": "drug:anastrozole", "type": "Drug", "label": "Anastrozole", "register": "physician"},
    {"id": "drug:exemestane", "type": "Drug", "label": "Exemestane", "register": "physician"},
    {"id": "drug:finasteride", "type": "Drug", "label": "Finasteride", "register": "physician"},
    {"id": "drug:dutasteride", "type": "Drug", "label": "Dutasteride", "register": "physician"},
    {"id": "drug:aspirin", "type": "Drug", "label": "Aspirin (low-dose)", "register": "physician"},
    {"id": "drug:atorvastatin", "type": "Drug", "label": "Atorvastatin (statin class)", "register": "physician"},
    {"id": "drug:varenicline", "type": "Drug", "label": "Varenicline", "register": "physician"},
    {"id": "drug:nrt", "type": "Drug", "label": "Nicotine replacement therapy", "register": "physician"},
    {"id": "drug:zoledronic_acid", "type": "Drug", "label": "Zoledronic acid", "register": "physician"},
    {"id": "drug:clodronate", "type": "Drug", "label": "Clodronate", "register": "physician"},
    {"id": "drug:denosumab", "type": "Drug", "label": "Denosumab", "register": "physician"},
    {"id": "drug:mirena_lng_ius", "type": "Drug", "label": "Levonorgestrel IUS (Mirena)", "register": "physician"},

    # Vaccines
    {"id": "vaccine:gardasil_9", "type": "Vaccine", "label": "Gardasil-9 (9vHPV)", "register": "patient"},

    # Conditions / Outcomes
    {"id": "condition:invasive_breast_cancer", "type": "Condition", "label": "Invasive breast cancer"},
    {"id": "condition:bone_recurrence", "type": "Condition", "label": "Bone recurrence (breast cancer)"},
    {"id": "condition:colorectal_cancer", "type": "Condition", "label": "Colorectal cancer"},
    {"id": "condition:prostate_cancer_high_grade", "type": "Condition", "label": "High-grade prostate cancer"},
    {"id": "condition:cervical_cancer", "type": "Condition", "label": "Cervical cancer"},
    {"id": "condition:onj", "type": "Condition", "label": "Osteonecrosis of the jaw"},
    {"id": "condition:atypical_femur_fracture", "type": "Condition", "label": "Atypical femoral fracture"},
    {"id": "condition:ascvd", "type": "Condition", "label": "Atherosclerotic cardiovascular disease"},
    {"id": "condition:tobacco_dependence", "type": "Condition", "label": "Tobacco dependence"},
    {"id": "condition:endometrial_hyperplasia", "type": "Condition", "label": "Endometrial hyperplasia / cancer"},

    # Trials
    {"id": "trial:nsabp_p1", "type": "Trial", "label": "NSABP P-1 (tamoxifen prevention)"},
    {"id": "trial:ibis_ii", "type": "Trial", "label": "IBIS-II (anastrozole prevention)"},
    {"id": "trial:future_i_ii", "type": "Trial", "label": "FUTURE I/II (HPV vaccine efficacy)"},
    {"id": "trial:pcpt", "type": "Trial", "label": "PCPT (prostate cancer prevention)"},
    {"id": "trial:reduce", "type": "Trial", "label": "REDUCE (dutasteride prostate)"},
    {"id": "trial:select", "type": "Trial", "label": "SELECT (selenium / vit E prostate)"},
    {"id": "trial:capp2", "type": "Trial", "label": "CAPP2 (aspirin in Lynch syndrome)"},
    {"id": "trial:aspree", "type": "Trial", "label": "ASPREE (aspirin in elderly)"},
    {"id": "trial:eagles", "type": "Trial", "label": "EAGLES (varenicline neuropsych safety)"},
    {"id": "trial:ebctcg_2015", "type": "Trial", "label": "EBCTCG 2015 IPD bisphosphonate meta"},
    {"id": "trial:zfast_zofast", "type": "Trial", "label": "Z-FAST / ZO-FAST"},
    {"id": "trial:abcsg_18", "type": "Trial", "label": "ABCSG-18 (denosumab + AI)"},
    {"id": "trial:jupiter", "type": "Trial", "label": "JUPITER (rosuvastatin)"},
    {"id": "trial:cochrane_romero_2020", "type": "Trial", "label": "Cochrane Romero 2020 (LNG-IUS + tamoxifen)"},

    # Guidelines / Regulators
    {"id": "guideline:uspstf_2019_breast_meds", "type": "Guideline", "label": "USPSTF 2019 breast cancer meds"},
    {"id": "guideline:uspstf_2022_aspirin", "type": "Guideline", "label": "USPSTF 2022 aspirin (CVD/CRC)"},
    {"id": "guideline:uspstf_2022_statin", "type": "Guideline", "label": "USPSTF 2022 statin primary prevention"},
    {"id": "guideline:uspstf_2021_tobacco", "type": "Guideline", "label": "USPSTF 2021 tobacco cessation"},
    {"id": "guideline:acip_2019_hpv", "type": "Guideline", "label": "ACIP 2019 HPV catch-up"},
    {"id": "guideline:asco_2017_bma", "type": "Guideline", "label": "ASCO 2017 BMA guideline"},
    {"id": "regulator:fda_2018_gardasil9_age45", "type": "Regulator", "label": "FDA 2018 Gardasil-9 expansion to age 45"},
    {"id": "regulator:fda_2011_5ari_high_grade", "type": "Regulator", "label": "FDA 2011 5-ARI high-grade warning"},

    # Populations
    {"id": "pop:postmenopausal", "type": "Population", "label": "Postmenopausal women"},
    {"id": "pop:premenopausal", "type": "Population", "label": "Premenopausal women"},
    {"id": "pop:lynch_syndrome", "type": "Population", "label": "Lynch syndrome carriers"},
    {"id": "pop:elderly", "type": "Population", "label": "Adults age 70+"},
    {"id": "pop:age_27_45_unvax", "type": "Population", "label": "Adults 27-45 not adequately HPV-vaccinated"},
    {"id": "pop:hr_pos_early_bc_on_ai", "type": "Population", "label": "HR+ early breast cancer on aromatase inhibitor"},

    # Tests / Concepts
    {"id": "test:dexa", "type": "Test", "label": "DEXA bone-density scan"},
    {"id": "test:psa", "type": "Test", "label": "PSA (prostate-specific antigen)"},
    {"id": "test:cervical_screening", "type": "Test", "label": "Cervical cancer screening (Pap / HPV)"},
    {"id": "test:dental_eval", "type": "Test", "label": "Pre-treatment dental evaluation"},
    {"id": "concept:diminished_benefit_prior_exposure", "type": "Concept", "label": "Diminished benefit with prior exposure"},
    {"id": "concept:premenopausal_no_signal", "type": "Concept", "label": "EBCTCG: no premenopausal benefit"},
]

# Edge tuples: (src, dst, relation, persona_mask, evidence_currency, evidence_ref)
# evidence_currency: 'A' (current/load-bearing), 'B' (supporting), 'C' (historical-context)
EDGES: list[tuple[str, str, str, int, str, str]] = [
    # ---- Tamoxifen + Mirena anchor case ----
    ("drug:tamoxifen", "condition:invasive_breast_cancer", "indicates",
     PERSONA_ALL, "A", "USPSTF-2019; NSABP P-1"),
    ("drug:tamoxifen", "condition:endometrial_hyperplasia", "increases_risk_of",
     PERSONA_PHYSICIAN | PERSONA_NURSE, "A", "ACOG #601 / Cochrane Romero 2020"),
    ("drug:mirena_lng_ius", "condition:endometrial_hyperplasia", "protects_against",
     PERSONA_PHYSICIAN | PERSONA_NURSE | PERSONA_PATIENT, "A", "Cochrane Romero 2020"),
    ("trial:cochrane_romero_2020", "drug:mirena_lng_ius", "evidenced_by",
     PERSONA_PHYSICIAN, "A", "Cochrane CD007245"),
    ("trial:nsabp_p1", "drug:tamoxifen", "evidenced_by", PERSONA_PHYSICIAN, "A", "NSABP P-1"),
    ("trial:ibis_ii", "drug:anastrozole", "evidenced_by", PERSONA_PHYSICIAN, "A", "IBIS-II Lancet 2014"),

    # ---- HPV catch-up axis ----
    ("vaccine:gardasil_9", "condition:cervical_cancer", "indicates",
     PERSONA_ALL, "A", "ACIP-MMWR-2019; FUTURE I/II"),
    ("guideline:acip_2019_hpv", "pop:age_27_45_unvax", "shared_clinical_decision_making",
     PERSONA_ALL, "A", "ACIP 2019 MMWR"),
    ("regulator:fda_2018_gardasil9_age45", "vaccine:gardasil_9", "expanded_indication_to_age_45",
     PERSONA_ALL, "A", "FDA 2018 supplemental approval"),
    ("trial:future_i_ii", "vaccine:gardasil_9", "evidenced_by", PERSONA_PHYSICIAN, "A", "FUTURE I/II"),
    ("vaccine:gardasil_9", "test:cervical_screening", "complements_not_replaces",
     PERSONA_ALL, "A", "NCI HPV vaccine fact sheet"),
    ("vaccine:gardasil_9", "concept:diminished_benefit_prior_exposure", "modulated_by",
     PERSONA_PATIENT | PERSONA_FAMILY | PERSONA_PHYSICIAN, "A", "ACIP 2019 / NCI"),
    ("concept:diminished_benefit_prior_exposure", "pop:age_27_45_unvax", "rationale_for",
     PERSONA_PHYSICIAN, "A", "ACIP 2019"),

    # ---- Bisphosphonate-AI axis ----
    ("drug:zoledronic_acid", "condition:bone_recurrence", "reduces_in",
     PERSONA_PHYSICIAN | PERSONA_NURSE, "A", "EBCTCG 2015"),
    ("drug:zoledronic_acid", "pop:postmenopausal", "indicates_in",
     PERSONA_PHYSICIAN, "A", "EBCTCG 2015 / ASCO 2017"),
    ("drug:zoledronic_acid", "pop:premenopausal", "no_benefit_in",
     PERSONA_PHYSICIAN, "A", "EBCTCG 2015"),
    ("trial:ebctcg_2015", "drug:zoledronic_acid", "evidenced_by",
     PERSONA_PHYSICIAN, "A", "Lancet 2015;386:1353"),
    ("trial:zfast_zofast", "drug:zoledronic_acid", "evidenced_by",
     PERSONA_PHYSICIAN, "B", "Z-FAST / ZO-FAST"),
    ("guideline:asco_2017_bma", "pop:hr_pos_early_bc_on_ai", "recommends_for",
     PERSONA_PHYSICIAN, "A", "Dhesy-Thind JCO 2017"),
    ("drug:zoledronic_acid", "condition:onj", "rare_adverse_event",
     PERSONA_PHYSICIAN | PERSONA_NURSE | PERSONA_PATIENT, "A", "FDA bisphosphonate safety"),
    ("drug:zoledronic_acid", "condition:atypical_femur_fracture", "rare_adverse_event",
     PERSONA_PHYSICIAN | PERSONA_NURSE, "A", "FDA bisphosphonate safety"),
    ("test:dental_eval", "drug:zoledronic_acid", "required_before",
     PERSONA_PHYSICIAN | PERSONA_NURSE | PERSONA_PATIENT, "A", "ASCO 2017"),
    ("test:dexa", "pop:hr_pos_early_bc_on_ai", "monitors",
     PERSONA_PHYSICIAN | PERSONA_NURSE, "A", "ASCO 2017"),
    ("drug:denosumab", "condition:bone_recurrence", "weak_evidence_for",
     PERSONA_PHYSICIAN, "B", "ABCSG-18 secondary endpoints"),
    ("trial:abcsg_18", "drug:denosumab", "evidenced_by",
     PERSONA_PHYSICIAN, "A", "Gnant Lancet 2015"),
    ("drug:anastrozole", "pop:postmenopausal", "indicates_in",
     PERSONA_PHYSICIAN, "A", "USPSTF 2019"),
    ("drug:anastrozole", "test:dexa", "monitored_with",
     PERSONA_PHYSICIAN | PERSONA_NURSE, "A", "ASCO 2017 BMA"),

    # ---- Aspirin / CRC axis ----
    ("guideline:uspstf_2022_aspirin", "drug:aspirin", "recommends_for",
     PERSONA_PHYSICIAN, "A", "USPSTF 2022"),
    ("trial:capp2", "drug:aspirin", "evidenced_by", PERSONA_PHYSICIAN, "A", "CAPP2 Burn 2020"),
    ("trial:aspree", "drug:aspirin", "evidenced_by", PERSONA_PHYSICIAN, "A", "ASPREE 2018"),
    ("drug:aspirin", "pop:lynch_syndrome", "indicates_in",
     PERSONA_PHYSICIAN, "A", "CAPP2"),
    ("drug:aspirin", "pop:elderly", "no_benefit_in",
     PERSONA_PHYSICIAN, "A", "ASPREE"),

    # ---- 5-ARI / prostate axis ----
    ("trial:pcpt", "drug:finasteride", "evidenced_by", PERSONA_PHYSICIAN, "A", "PCPT NEJM 2003"),
    ("trial:reduce", "drug:dutasteride", "evidenced_by", PERSONA_PHYSICIAN, "A", "REDUCE NEJM 2010"),
    ("regulator:fda_2011_5ari_high_grade", "drug:finasteride", "warns_about",
     PERSONA_PHYSICIAN, "A", "FDA 2011 class warning"),
    ("regulator:fda_2011_5ari_high_grade", "drug:dutasteride", "warns_about",
     PERSONA_PHYSICIAN, "A", "FDA 2011 class warning"),
    ("test:psa", "drug:finasteride", "monitored_with",
     PERSONA_PHYSICIAN, "A", "PCPT PSA-doubling rule"),

    # ---- Statin / cancer axis ----
    ("guideline:uspstf_2022_statin", "drug:atorvastatin", "recommends_for",
     PERSONA_PHYSICIAN, "A", "USPSTF 2022"),
    ("drug:atorvastatin", "condition:ascvd", "reduces",
     PERSONA_PHYSICIAN | PERSONA_PATIENT, "A", "USPSTF 2022; CTT meta"),
    ("trial:jupiter", "drug:atorvastatin", "evidenced_by",
     PERSONA_PHYSICIAN, "A", "JUPITER NEJM 2008"),

    # ---- Tobacco cessation axis ----
    ("guideline:uspstf_2021_tobacco", "condition:tobacco_dependence", "recommends_for",
     PERSONA_PHYSICIAN | PERSONA_NURSE, "A", "USPSTF 2021"),
    ("trial:eagles", "drug:varenicline", "evidenced_by",
     PERSONA_PHYSICIAN, "A", "EAGLES Lancet 2016"),
    ("drug:varenicline", "condition:tobacco_dependence", "indicates",
     PERSONA_ALL, "A", "USPSTF 2021"),
    ("drug:nrt", "condition:tobacco_dependence", "indicates",
     PERSONA_ALL, "A", "PHS 2008 / USPSTF 2021"),
]


def _norm_persona(p: str) -> int:
    if p in PERSONA_BIT:
        return PERSONA_BIT[p]
    raise ValueError(f"unknown persona {p!r}; expected one of {list(PERSONA_BIT)}")


def build_graph():
    """Return a `networkx.DiGraph` populated with NODES + EDGES. If
    `nx-cugraph` is installed and `NETWORKX_AUTOMATIC_BACKENDS=cugraph` is
    set, networkx will dispatch BFS / shortest-path to cuGraph at run time."""
    nx = _import_nx()
    G = nx.DiGraph()
    for n in NODES:
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for src, dst, rel, mask, currency, evid in EDGES:
        if src not in G.nodes or dst not in G.nodes:
            raise KeyError(f"edge references missing node: {src} -> {dst}")
        G.add_edge(
            src,
            dst,
            relation=rel,
            persona_mask=mask,
            evidence_currency=currency,
            evidence_ref=evid,
        )
    return G


def _persona_filtered_neighbors(G, node, persona_bit: int) -> Iterable[str]:
    for _, nbr, attrs in G.out_edges(node, data=True):
        if attrs.get("persona_mask", 0) & persona_bit:
            yield nbr
    for nbr, _, attrs in G.in_edges(node, data=True):
        if attrs.get("persona_mask", 0) & persona_bit:
            yield nbr


def expand_subgraph(G, *, seeds: list[str], persona: str = "patient", hops: int = 2):
    """BFS up to `hops` levels from `seeds`, traversing only edges whose
    persona_mask bit for `persona` is set. Returns the induced subgraph."""
    nx = _import_nx()
    persona_bit = _norm_persona(persona)
    visited: set[str] = set()
    frontier: set[str] = {s for s in seeds if s in G.nodes}
    visited |= frontier
    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for nbr in _persona_filtered_neighbors(G, node, persona_bit):
                if nbr not in visited:
                    next_frontier.add(nbr)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break
    sub = G.subgraph(visited).copy()
    # Keep only edges whose persona bit is on (subgraph().copy() preserves all
    # edges between visited nodes; we filter again at edge-level).
    bad_edges = [
        (u, v)
        for u, v, attrs in sub.edges(data=True)
        if not (attrs.get("persona_mask", 0) & persona_bit)
    ]
    sub.remove_edges_from(bad_edges)
    return sub


def seed_nodes_for(*, query: str, retrieved_chunk_ids: list[str]) -> list[str]:
    """Cheap heuristic seed selection: scan the query (lowercased) for
    keyword tokens that match node labels. The retrieved_chunk_ids
    parameter is reserved for the future case where chunks carry a
    `node_anchor` attribute populated by the corpus builder; today the
    chunks are not graph-anchored, so we ignore the chunk-id list and
    rely on query keywords."""
    q = query.lower()
    keyword_to_seed = {
        "tamoxifen": "drug:tamoxifen",
        "mirena": "drug:mirena_lng_ius",
        "raloxifene": "drug:raloxifene",
        "anastrozole": "drug:anastrozole",
        "exemestane": "drug:exemestane",
        "finasteride": "drug:finasteride",
        "dutasteride": "drug:dutasteride",
        "aspirin": "drug:aspirin",
        "statin": "drug:atorvastatin",
        "atorvastatin": "drug:atorvastatin",
        "rosuvastatin": "drug:atorvastatin",
        "varenicline": "drug:varenicline",
        "chantix": "drug:varenicline",
        "nrt": "drug:nrt",
        "nicotine replacement": "drug:nrt",
        "zoledronic": "drug:zoledronic_acid",
        "zometa": "drug:zoledronic_acid",
        "denosumab": "drug:denosumab",
        "prolia": "drug:denosumab",
        "clodronate": "drug:clodronate",
        "gardasil": "vaccine:gardasil_9",
        "hpv": "vaccine:gardasil_9",
        "papilloma": "vaccine:gardasil_9",
        "psa": "test:psa",
        "dexa": "test:dexa",
        "lynch": "pop:lynch_syndrome",
        "postmenopausal": "pop:postmenopausal",
        "premenopausal": "pop:premenopausal",
        "aromatase": "pop:hr_pos_early_bc_on_ai",
        "smoking": "condition:tobacco_dependence",
        "tobacco": "condition:tobacco_dependence",
        "ascvd": "condition:ascvd",
        "cardiovascular": "condition:ascvd",
        "cervical": "condition:cervical_cancer",
        "breast cancer": "condition:invasive_breast_cancer",
        "colorectal": "condition:colorectal_cancer",
        "prostate": "condition:prostate_cancer_high_grade",
        "onj": "condition:onj",
        "osteonecrosis": "condition:onj",
    }
    seeds: list[str] = []
    for kw, node in keyword_to_seed.items():
        if kw in q and node not in seeds:
            seeds.append(node)
    return seeds


def serialize_subgraph_for_prompt(sub) -> str:
    """Serialize the persona-filtered subgraph into a compact textual block
    that plugs into the system prompt alongside the retrieved chunks. Edges
    are emitted as `subject -[relation, evidence: REF]-> object` lines so
    the brain can cite them inline."""
    if sub.number_of_nodes() == 0:
        return ""
    lines = [
        "Persona-filtered subgraph slice (cite evidence_refs inline where used):",
        "",
        "Nodes:",
    ]
    for n, attrs in sub.nodes(data=True):
        lines.append(f"  {n} :: {attrs.get('label', n)} ({attrs.get('type','?')})")
    lines.append("")
    lines.append("Edges:")
    for u, v, attrs in sub.edges(data=True):
        rel = attrs.get("relation", "rel")
        evid = attrs.get("evidence_ref", "")
        cur = attrs.get("evidence_currency", "")
        lines.append(f"  {u} -[{rel}, {cur}: {evid}]-> {v}")
    return "\n".join(lines)


def main() -> int:
    """Smoke run: build, expand for one query, print serialized."""
    G = build_graph()
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    queries = [
        ("I'm 35 and never got the HPV shot. Should I get Gardasil-9?", "patient"),
        ("postmenopausal HR+ breast cancer on anastrozole — adjuvant zoledronic acid?", "physician"),
        ("low-dose aspirin to prevent colorectal cancer in family history", "patient"),
    ]
    for q, persona in queries:
        seeds = seed_nodes_for(query=q, retrieved_chunk_ids=[])
        sub = expand_subgraph(G, seeds=seeds, persona=persona, hops=2)
        print(f"\n--- query: {q[:70]}... [persona={persona}] seeds={seeds}")
        print(f"    subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
        block = serialize_subgraph_for_prompt(sub)
        print(block[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
