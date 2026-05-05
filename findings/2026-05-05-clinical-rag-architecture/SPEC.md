# Clinical RAG Architecture SPEC (Agent B)

Date: 2026-05-05
Scope: Build on medomni's existing RAG stack (nx-cugraph, PrimeKG, NV-Embed-v2, OpenEM/LanceDB, NeMo PMC chunks, Pattern B FHIR-fetch). Identify the 2-3 highest-lift additions for clinical reasoning + verifiability in May 2026.

## TL;DR

medomni already has the dense + KG halves. The gaps that move the needle are (1) **sparse-dense hybrid + late-interaction rerank** (BM25 + NV-Embed RRF -> ColBERTv2/ModernBERT-late rerank), (2) **claim-level verifiability loop** (MedScore-style atomic-claim decomposition + RAGAS faithfulness gate), and (3) **concept-anchored bridge** (UMLS/SNOMED MedCAT linker over patient FHIR + query, used as a routing key into PrimeKG and as a sparse-retrieval expansion). Long-context-only is dominated by retrieval on cost and traceability; keep retrieval as the contract.

## Top techniques ranked by clinical lift

1. **Hybrid sparse + dense + late-interaction rerank** -- BM25 + NV-Embed-v2, fused via RRF (k=60), reranked by ColBERTv2 (or ModernBERT+ColBERT for biomedical) on top-50 candidates. MedCPT-style sparse+dense beats either alone on PubMedQA (89.4% faithfulness, 82.7% relevance with GPT-4o); ColBERT late-interaction closes the long-tail-phrasing gap that bites NV-Embed on ICD-10 codes, drug names, and rare-disease aliases. Highest expected lift; no new index needed -- BM25 layer over existing LanceDB chunks + reranker model.

2. **Claim-extraction + grounding-check loop (MedScore + RAGAS faithfulness)** -- decompose draft answer into atomic clinical claims, verify each against retrieved evidence with an entailment judge, refuse/retract claims below threshold. MedScore extracts ~3x more valid medical facts than generic decomposition and preserves condition-dependency. This is the verifiability primitive that converts retrieval into citations a clinician will trust. Pairs with OpenEvidence-style verbatim-quote rendering.

3. **Concept-anchored retrieval via UMLS/SNOMED linker (MedCAT)** -- run MedCAT over the user query AND the patient FHIR bundle from Pattern B, lift concepts to CUIs, then (a) expand BM25 with synonym set, (b) use CUIs as PrimeKG entry nodes for nx-cugraph traversal, (c) tag retrieved chunks with concept overlap as a rerank feature. SapBERT-style ontology-aligned embeddings are an alternative but heavier; MedCAT ships now.

4. **Subgraph-scoped GraphRAG** (already half-built on PrimeKG/nx-cugraph) -- replace full-graph traversal with query-conditioned subgraph extraction. ReGraM (arXiv:2601.09280) shows that flat global PrimeKG traversal injects noise across heterogeneous relation types; restricting to a 2-3 hop neighborhood around CUI entry nodes -- and pruning by relation-type whitelist per intent (diagnosis, treatment, contraindication) -- is the clinical-lift move. Wire (3) into existing nx-cugraph.

5. **Primary-source citation channel (PubMed + ClinicalTrials.gov + FDA labels)** -- separate retrieval head for primary trial / label data with extractive verbatim quotes, surfaced as a distinct "evidence" panel. NeMo extraction is already done for PMC chunks; add ClinicalTrials.gov AACT + DailyMed FDA labels as parallel collections in LanceDB. Required for OpenEvidence-grade clinician trust.

6. **Iterative / self-reflective retrieval (i-MedRAG, Self-MedRAG)** -- multi-turn follow-up queries when initial retrieval is low-confidence (per RAGAS context-precision). Net win on harder MIRAGE splits. Defer -- agent-side latency cost; revisit after (1)-(3).

7. **Long-context-only fallback** -- Not recommended as primary. Long-context underperforms RAG on cost (~1250x) and traceability for medical QA, and accuracy degrades with document count even when key info is in-context. Use long-context to *reason over* the retrieved evidence set, not to replace retrieval.

## Integration with existing infra

| Existing | Add | Wire |
| --- | --- | --- |
| LanceDB / NV-Embed-v2 | BM25 (Tantivy or pyserini) sidecar index over same chunks | Hybrid retriever returns 100 BM25 + 100 dense, RRF fuses to 50, ColBERTv2 reranks to 10 |
| PrimeKG / nx-cugraph | MedCAT (UMLS+SNOMED) linker | Linker extracts CUIs from query + Pattern B FHIR bundle -> seed nodes for nx-cugraph k-hop traversal, relation-type filtered by intent |
| NeMo PMC chunks | ClinicalTrials.gov AACT + DailyMed FDA labels collections | Separate retriever heads, fused into the same RRF stage; chunks tagged source=`pmc|ctgov|fda|openem|primekg` |
| Pattern B FHIR p95=11ms | Concept linker over FHIR Conditions/Medications/Observations | Patient CUIs become a *retrieval-bias vector* (boost chunks with overlapping CUIs) and a graph-traversal seed set |
| OpenEM 370 conditions LanceDB | Used as condition-routing index (already) | MedCAT CUI -> openem condition_id lookup table for cross-walk |

## Verifiability hooks

- **Claim decomposition** -- MedScore atomic-claim splitter on draft answer; each claim gets `(claim_text, supporting_chunks[], entailment_score, citation_span)`.
- **Faithfulness gate** -- RAGAS faithfulness >= 0.85 OR claim is dropped/marked unsupported. Per-claim, not per-answer.
- **Verbatim-quote rendering** -- OpenEvidence pattern: each cited claim links to a verbatim span in the source chunk; UI highlights the span. Forces extractive grounding.
- **Negative-evidence check** -- second retrieval pass with the negation of each claim; if contradictory chunks score higher, escalate to "uncertain".
- **Audit trail** -- write `(query, retrieved_chunk_ids, fused_ranks, claims, entailment_scores)` to AuditEvent (per the FHIR audit plan in `project_medomni_v1_architecture_decisions`).

## Concrete recommendation -- 3 things to land

1. **Hybrid+rerank pipeline** -- BM25 sidecar + RRF + ColBERTv2 (or ModernBERT-late) rerank. Largest single accuracy lift on MIRAGE-style benchmarks; works with current chunks; no new corpus.
2. **MedScore claim-extraction + RAGAS faithfulness gate** -- the verifiability primitive. Without it, dense+graph improvements aren't legible to clinicians.
3. **MedCAT concept-anchoring** -- bridges Pattern B FHIR <-> PrimeKG <-> retrieval. Unlocks personalization-without-graph-merge (per Agent A's dual-lookup pattern) and converts existing nx-cugraph from "available" to "targeted".

Defer: SapBERT swap-in for NV-Embed (heavy, marginal vs MedCAT routing), iterative-retrieval, long-context-only mode.

## References

- MIRAGE / MedRAG benchmark: Xiong et al., ACL Findings 2024 -- https://arxiv.org/abs/2402.13178 ; toolkit https://github.com/Teddy-XiongGZ/MedRAG
- MedRAG KG-elicited reasoning: Wang et al., WWW 2025 -- https://arxiv.org/abs/2502.04413 (CPDD 79.25%, DDXPlus 88.65%)
- MedCPT contrastive PubMed retrieval: Jin et al., Bioinformatics 2023 -- https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842
- Hybrid BM25+MedCPT clinical QA: MDPI Information 2026 -- https://www.mdpi.com/2078-2489/17/2/133
- ModernBERT + ColBERT biomedical RAG: arXiv:2510.04757 -- https://arxiv.org/pdf/2510.04757
- Jina-ColBERT-v2: arXiv:2408.16672
- ReGraM region-first PrimeKG reasoning: arXiv:2601.09280
- PrimeKG: Chandak et al., mims-harvard/PrimeKG -- 17,080 diseases, 4M relations
- MedScore atomic-claim factuality: arXiv:2505.18452 -- https://arxiv.org/abs/2505.18452
- RAGAS faithfulness: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- Long-context vs RAG for medical QA: npj Digital Medicine 2025 -- https://www.nature.com/articles/s41746-025-01651-w ; arXiv:2510.18691
- OpenEvidence architecture / Mount Sinai Epic integration (2026-03): https://www.openevidence.com/ ; PMC12101550
- MedCAT SNOMED/UMLS linker scoping review (JMIR Med Inform 2024): https://medinform.jmir.org/2024/1/e62924/PDF
- Self-MedRAG iterative: arXiv:2601.04531
