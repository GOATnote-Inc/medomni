---
name: calc
description: Tool-only mode for clinical calculators (CHA2DS2-VASc, MELD, qSOFA, GFR, etc.)
trigger:
  - "calculate"
  - "score"
  - "CHA2DS2"
  - "MELD"
  - "qSOFA"
  - "GFR"
  - "Wells"
  - "PERC"
  - "TIMI"
  - "GRACE"
  - "PSI"
  - "CURB"
  - "APACHE"
  - "Glasgow"
  - "Wells score"
---

When the user asks for a clinical-calculator score, run **TOOL-ONLY mode** — no CoT, no <think> tags, no narrative. Emit:

1. `<plan>` block with RETRIEVE for any patient data needed
2. Tool calls for each input value
3. `<calc>` block with the formula + inputs + result
4. One-sentence interpretation

Skip the CoT ensemble — calculators are deterministic; running multiple samples wastes tokens and adds variance noise.

## Supported calculators (pre-registered formulas)

| Score | Formula | Input data |
|---|---|---|
| CHA2DS2-VASc | C(1) + H(1) + A(2 if ≥75, 1 if 65-74) + D(1) + S2(2) + V(1) + A(1) + Sc(1 female) | conditions + age + sex |
| MELD | 3.78×ln(bili) + 11.2×ln(INR) + 9.57×ln(creat) + 6.43 | labs |
| qSOFA | RR≥22 + altered mental + SBP≤100 (each 1 pt) | vitals + GCS |
| GFR (CKD-EPI 2021) | 142×min(SCr/k,1)^a × max(SCr/k,1)^-1.200 × 0.9938^age × (1.012 if female) | creat + age + sex |
| Wells PE | clinical features per Wells 2000 | history + vitals |
| PERC | rule-out PERC criteria | vitals + history |
| TIMI NSTEMI | 0-7 score | history + ECG + biomarkers |
| GRACE | logistic regression (online table) | vitals + labs + history |
| PSI | risk class I-V | vitals + comorbidities |
| CURB-65 | Confusion + Urea + RR + BP + ≥65 | vitals + labs + age |
| APACHE II | 0-71 from 12 physiologic vars + age + chronic health | ICU labs + vitals |
| Glasgow (GCS) | E + V + M (3-15) | neuro exam |

If user requests a calculator NOT in this list, exit Skill mode and use general reasoning.

## Example

User: "Calculate CHA2DS2-VASc for this patient"

```
<plan>
1. RETRIEVE Patient (age, sex)
2. RETRIEVE Condition (HF, HTN, stroke, vascular, diabetes)
3. CALC formula with retrieved values
</plan>

<tool>fhir.get_patient(id)</tool>
<tool>fhir.search_conditions(patient_id, clinical-status=active)</tool>

<calc>
  Age 73 (1pt)
  Female (1pt)
  HTN (1pt)
  Diabetes (1pt)
  Prior stroke (2pt)
  Total: 6
</calc>

CHA2DS2-VASc 6 → ~9.8% annual stroke risk; anticoagulation indicated per AHA/ESC guidelines.
```

## Don't

- Don't show CoT for calculator turns (latency + variance, no quality lift)
- Don't sample K candidates (deterministic formula → no benefit from voting)
- Don't add `[PMID]` citations for formulas (the formula itself IS the citation; reference is calculator name)
