---
name: handoff
description: Verifier-gated FHIR write for high-stakes mutating operations
trigger:
  - "place an order"
  - "write a script"
  - "send a referral"
  - "discharge"
  - "admit"
  - "transfer"
  - "MAR"
  - "create order"
  - "submit order"
---

When the user asks to MUTATE the EHR (place order, write prescription, change med list, send referral, discharge note), follow the **HANDOFF protocol**:

1. **Plan-then-act header** — same as system prompt, but mandatory.
2. **Pre-write reads** — RETRIEVE allergies, active conditions, current meds, recent labs (for kidney/liver dose adjustment), recent vitals. Skipping this is the #1 cause of MedAgentBench v2 failures.
3. **Verifier vote** — sample K=4 candidate handoff actions, score with PRM, return min-step-score winner. Per `mvp/medomni-inference/verifier_vote.py`.
4. **Hard-block on safety contradictions** — if any claim in the proposed action contradicts retrieved data (e.g. "give metformin" when patient has CKD-IV), abort with `<abstain/>` + diagnostic.
5. **Read-back the proposed write** — show the exact FHIR resource body to the user BEFORE submitting. The write is gated on user confirmation.
6. **Audit log** — every successful HANDOFF emits a `FHIR AuditEvent` resource per `project_medomni_v1_architecture_decisions` (Medplum AuditEvent → S3 Object Lock 7 yr).

## Required pre-write reads (by handoff type)

| Handoff type | Required reads | Common failure mode |
|---|---|---|
| MedicationRequest (new prescription) | AllergyIntolerance, Condition (active), MedicationRequest (active), Observation (creat, ALT, vitals) | Drug-drug interaction missed; renal dose missed |
| ServiceRequest (lab order) | Observation (recent same code) — don't duplicate | Unnecessary lab repeat |
| ServiceRequest (imaging) | DiagnosticReport (recent same modality), AllergyIntolerance (contrast) | Contrast in dye allergy; unnecessary repeat |
| ServiceRequest (referral) | Encounter (current), Condition (active) | Missing referral reason |
| Encounter (discharge) | MedicationRequest (active), Condition (active), allergies | Discharge med-rec error |
| Encounter (admit) | All of above | Missing problem-list reconciliation |

## Example

User: "Place an order for ondansetron 4 mg IV q6h PRN for nausea"

```
<plan>
1. RETRIEVE AllergyIntolerance (patient_id) — check ondansetron / 5HT3 allergy
2. RETRIEVE MedicationRequest (active) — check QT-prolonging meds (ondansetron prolongs QTc)
3. RETRIEVE Observation (ECG most-recent QTc)
4. RETRIEVE Condition (active) — long QT syndrome / cardiac arrhythmia
5. VERIFIER-VOTE 4 candidate orders; pick min-step-score winner
6. READBACK the FHIR MedicationRequest body to user
7. WAIT for user confirmation
8. WRITE on confirm; emit AuditEvent
</plan>

<tool>fhir.search_allergy_intolerances(patient_id)</tool>
<tool>fhir.search_medication_requests(patient_id, status=active)</tool>
<tool>fhir.search_observations(patient_id, loinc=8634-8, _count=1, _sort=-date)</tool>
<tool>fhir.search_conditions(patient_id, clinical-status=active)</tool>

[VERIFIER-VOTE: 4 candidates scored; winner min_step_score=0.87; no safety contradictions]

Proposed order:
{
  "resourceType": "MedicationRequest",
  "status": "draft",
  "intent": "order",
  "medicationCodeableConcept": { "coding": [{ "system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "26225", "display": "ondansetron 4 mg" }] },
  "subject": { "reference": "Patient/{ID}" },
  "dosageInstruction": [{
      "asNeededBoolean": true,
      "asNeededCodeableConcept": { "text": "nausea" },
      "route": { "coding": [{ "code": "47625008", "display": "IV" }] },
      "timing": { "repeat": { "frequency": 1, "period": 6, "periodUnit": "h" } },
      "doseAndRate": [{ "doseQuantity": { "value": 4, "unit": "mg" } }]
  }]
}

Confirm to submit? [yes/no]
```

If user types "yes" → submit + emit AuditEvent.
If user types anything else → cancel, no write.

## Don't

- Don't WRITE without read-back. Single most important rule.
- Don't skip the V verifier-vote. Mutating actions are exactly where Best-of-K + claim-audit pays off.
- Don't confirm on the user's behalf. Always require explicit "yes" / "confirm" / "submit".
