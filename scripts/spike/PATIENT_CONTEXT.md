# S2 — `get_patient_context` tool (architectural spike, Pattern B)

5th tool added to `/api/agent`. Pulls one patient's FHIR slice from a local
Medplum server and returns a Markdown block the agent loop consumes in
parallel with PrimeKG / PubMed. **No merge into PrimeKG** — the LLM does
the fusion in its existing reasoning step.

## Spike-only env vars (NOT in `.env.example` — sovereignty rule §2)

```
MEDOMNI_FHIR_BASE_URL=http://localhost:8103/fhir/R4
MEDOMNI_FHIR_TOKEN=<bearer token from your Medplum sandbox>
```

Set these in `web/.env.local` for `vercel dev` runs only. Do NOT commit the
local file. Do NOT export through any RunPod proxy. Do NOT bake into a
preview deploy. If either is unset the tool returns an error and the loop
continues without patient context.

## Local test against S1 sandbox

```
cd web
# 1. Start S1's local Medplum on :8103 (see S1's README).
# 2. Drop env vars into web/.env.local.
# 3. Run vercel dev.
vercel dev
```

Request shape — note the new top-level `patientId` field:

```json
POST /api/agent
{
  "messages": [{"role":"user","parts":[{"type":"text","text":"What's this patient's AFib stroke risk?"}]}],
  "patientId": "synthea-pt-1"
}
```

The harness in S3 populates `patientId` from the synthetic-patient roster.
If absent, `get_patient_context` returns an error block — the agent answers
from the question text alone.

## Mock-only verification

`web/lib/tools/patient-context.test.ts` — `npx tsx lib/tools/patient-context.test.ts`
from `web/`. 5 cases: happy path, missing env, 404, partial-503 degrade,
truncation flag. No live Medplum needed.

## Sample tool output (mocked Synthea fixture)

```
## Patient
- Name: Jane Doe · 63y · female · DOB 1962-04-12 · id: synthea-pt-1
- Context hint: AFib stroke risk

## Active Conditions
- Type 2 diabetes mellitus (onset 2020-06-15)
- Hypertension (onset 2020-06-15)

## Recent Vitals/Labs
- Systolic BP: 142 mm[Hg] [<recent>]
- HbA1c: 7.6 % [<recent>]

## Active Medications
- Metformin — 500 mg PO BID

## Allergies
- Penicillin [high] — reaction: rash

## Recent Diagnostic Reports
- Lipid panel [<recent>] — LDL elevated.

_Source: local Medplum FHIR R4 sandbox · Synthea synthetic data · spike Pattern B (no PrimeKG merge)._
```
