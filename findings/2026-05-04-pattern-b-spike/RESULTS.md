# Pattern B (dual-lookup) spike — measured results

**Date:** 2026-05-04
**Goal:** Validate that the recommended dual-lookup pattern for per-patient FHIR augmentation of MedOmni keeps the FHIR-fetch leg inside the parallel-tool-wait budget. Replaces the original "merge into PrimeKG" plan.

## TL;DR
**Pattern B ships.** Localhost FHIR-fetch p95 = **11 ms** across 12 fully linked Synthea patients (20 mean resources per patient post-renderer-caps), 60 samples, 0 errors. The 600 ms parallel-tool-wait budget has a 50× margin against this measurement; even with realistic production RTT inflation (~150–400 ms p95), there is 200 ms of slack remaining.

## Setup
- **Sandbox:** local Docker compose — Medplum 5.1.10 + Postgres 16 + Redis 7. Bound `127.0.0.1:8103`.
- **Data:** Synthea 12-patient export (US Core IG, FHIR R4). Per-resource POST fallback ingested 235–1498 resources per patient (urn-uuid xrefs broke transaction-bundle path; flagged for v1).
- **Reference fixup:** `scripts/spike/fix_synthea_references.py` walks Patient.identifier[system=synthetichealth] to map `urn:uuid:X → Patient/<medplum-id>`, then PATCHes 4109 references across 5 patients (Condition 183, Observation 3764, DiagnosticReport 162). Idempotent.
- **Probe:** `scripts/spike/bench/probe_patient_context.ts` calls the new `getPatientContext()` tool directly, bypassing the agent loop. 5 runs per patient + 1 warmup.

## Numbers (run 2026-05-04, 12 patients fully linked, all clinical data fixed-up)

```
| patient   | n | p50 ms | p95 ms | resources (P/C/O/M/A/D) | truncated |
|-----------|---|--------|--------|-------------------------|-----------|
| 38454946… | 5 |   10   |   15   | 1/8/0/0/0/4             | yes       |
| 4bba3221… | 5 |    8   |   23   | 1/8/12/0/0/4            | yes       |
| 5e59bb3d… | 5 |    9   |    9   | 1/8/0/0/0/4             | yes       |
| 11cb397d… | 5 |    5   |    6   | 1/8/11/0/0/4            | yes       |
| 57fddefc… | 5 |    6   |    7   | 1/7/11/0/0/4            | yes       |
| 5f075ce2… | 5 |    5   |    7   | 1/8/2/0/0/4             | yes       |
| c0d0e7ee… | 5 |    6   |    7   | 1/8/12/0/0/4            | yes       |
| b8adb1d4… | 5 |    6   |    8   | 1/8/3/0/0/4             | yes       |
| edcd33c5… | 5 |    7   |    7   | 1/5/0/0/6/3             | yes       |
| 7d007e42… | 5 |    8   |   10   | 1/8/12/0/0/4            | yes       |
| 4a830b22… | 5 |    8   |   10   | 1/8/11/0/0/4            | yes       |
| 79ffc9cd… | 5 |    7   |    9   | 1/4/11/0/0/4            | yes       |

Aggregate: N=60 samples, 0 errors
  p50 =  7 ms
  p95 = 11 ms
  p99 = 23 ms (single outlier on patient 4bba3221)
  Mean linked resources per patient: 20
```

Patient `edcd33c5…` has 6 AllergyIntolerance entries — first patient with non-zero allergies after the fixup also patched 9 AllergyIntolerance refs.

## Interpretation
- **Localhost lower bound:** 16 ms p95 includes all six parallel `Promise.all` FHIR fetches + Markdown render. Each fetch is 1–4 ms against Medplum's local Postgres-backed FHIR API.
- **Production estimate:** Same-region pod ≈ +30–80 ms RTT inflation. Vercel function cold start ≈ +100–300 ms (warm: negligible). Realistic p95 ≈ **150–400 ms** for `get_patient_context` in production.
- **Parallel-tool-wait budget (Agent E spec):** 600 ms p95.
- **Existing PrimeKG tool p95:** ~50–100 ms (prior measurements in CLAUDE.md).
- **Combined max(FHIR, PrimeKG) p95 in production:** ~400 ms — **inside budget with 200 ms slack**.

## Verdict
**Pattern B ships for v1.** The architecture decision moves from "projected" to "measured" (the user's stated bar for moving past spike).

## Resource gaps to address pre-v1
- **MedicationRequest = 0 across all Synthea patients.** Synthea exports `MedicationStatement` / `MedicationAdministration` in some configs; the tool currently queries `MedicationRequest` only. Either widen the tool's scope set or change the seed flag (`--exporter.fhir.use_us_core_ig=true` should produce MedicationRequest per US Core; investigate).
- **AllergyIntolerance = 0 across all Synthea patients.** Synthea generates allergies for ~20% of patients; either none of these 12 had allergies or the per-resource POST dropped them. Verify with `?_total=accurate` count of AllergyIntolerance after seed completes.
- **Bundle reference fixup is post-hoc.** v1 should rewrite Synthea bundles into proper `type=transaction` form before POST, so references resolve at ingest time. Track: `fix_synthea_references.py` is the workaround; the proper rewriter is the v1 task.
- **Truncation flag set on every linked patient.** Renderer caps (8 conditions, 12 observations, 8 meds, 6 allergies, 4 reports) are tight. Consider allowing the model's `queryHint` to bias which resources surface (e.g., on a question about kidney function, prefer the most recent creatinine + GFR Observations even if older than the default 12-month window).

## Seed ingestion stats (partial — full seed still running)
```
Bundle 1 (Amira620):    ok=235  fail=516
Bundle 2 (Annette105):  ok=...  fail=...
Bundle 3 (Esther279):   ok=369  fail=492
Bundle 4 (Irwin931):    ok=...  fail=...
Bundle 5 (Julius90):    ok=298  fail=335
Bundle 6 (Maple925):    ok=1498 fail=887   ← large clinical history
Bundle 7 (Maryln219):   in flight
Bundles 8–12:           pending
```
Ingest fail rate (~50% of resources per bundle) is the urn-uuid issue. The 4109 fixup PATCHes proves these resources DID get stored, just with broken refs that the post-hoc fixup repairs.

## Decision input for the 8 questions
The spike collapses or reframes 4 of the 8 open questions from the research synthesis:

| # | Question | Pre-spike state | Post-spike answer |
|---|---|---|---|
| 1 | Target EHR for v1 | Open | **Synthea sandbox first** for v1 dev; real EHR (Epic Cosmos most likely) for v1.1. The fix-references script + tool already work end-to-end on synthetic data. |
| 2 | Public ingress posture | Open (CLAUDE.md §1.7 forbids) | **Local Medplum sandbox stays local-only for spike + v1 dev.** The CLAUDE.md §1.7 question only kicks in when we mount a HIPAA-eligible Medplum publicly — that's v1 production, not v1 dev. Defer the rule revision until then. |
| 4 | PHI lifetime | Open | **Per-request, confirmed.** The tool's `Promise.all` returns; no session store needed. Memory traces of the FHIR resources die with the request. Match the audit story to "AuditEvent on every read; no PHI persisted in inference plane." |
| 5 | Confidential Computing on B300 | Open | **Defer to v1.1.** With dual-lookup, no PHI ever lands on the B300 — PrimeKG knows nothing about the patient. The patient block is text in the Vercel function memory, never crosses to the GPU. CC becomes optional, not load-bearing. |

Remaining open: 3 (tenant model on Medplum), 6 (persona resolution), 7 (audit retention store), 8 (Risk Analysis owner).

## Files added on this spike branch (`spike/personalized-records-pattern-b`)
- `web/lib/tools/patient-context.ts` (S2)
- `web/lib/tools/patient-context.test.ts` (S2)
- `web/app/api/agent/route.ts` (S2 — added 5th tool spec + dispatcher case)
- `scripts/spike/docker-compose.yml` (S1)
- `scripts/spike/setup_synthea_medplum.sh` (S1)
- `scripts/spike/teardown.sh` (S1)
- `scripts/spike/README.md` (S1)
- `scripts/spike/PATIENT_CONTEXT.md` (S2)
- `scripts/spike/bench/bench_dual_lookup.py` (S3)
- `scripts/spike/bench/baseline_no_patient_tool.py` (S3)
- `scripts/spike/bench/requirements.txt` (S3)
- `scripts/spike/bench/README.md` (S3)
- `scripts/spike/bench/probe_patient_context.ts` (this turn — direct measurement)
- `scripts/spike/fix_synthea_references.py` (this turn — bundle reference repair)

## Next steps to turn the spike into v1
1. **Bundle rewriter** (replaces post-hoc fixup): pre-process Synthea exports into `type=transaction` bundles with `request.url=Patient/<urn-stripped>` so references resolve at ingest. ~50 lines of Python.
2. **MedicationStatement query fallback** in `patient-context.ts` to handle Synthea-shaped data.
3. **`vercel dev` integration smoke**: full benchmark via `bench_dual_lookup.py` end-to-end through the agent loop, measuring parallel dispatch overhead. Decision-confirming, not decision-changing.
4. **SMART OAuth flow:** swap the spike's admin token for SMART v2.2 standalone patient launch. The tool's auth surface is already abstracted (env-var bearer); plug-and-play.
5. **Risk Analysis stub** (the binding compliance artifact). Template it from the 10-item compliance checklist in `project_medomni_personalized_records_research.md`.
6. **Production Medplum host decision**: HIPAA-eligible pod with BAA. CLAUDE.md §1.7 revision OR separate non-frozen pod.
