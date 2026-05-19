# V2.5b corpus filter summary — full

- Judged: **4950** records
- Kept: **2474** (50.0%, top-50% per pattern)
- Workers: 10

## Per-pattern score distribution

| Pattern | N | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| `A1_fabricated_specific_citation` | 300 | 0.73 | 0.00 | 1.00 | 1.00 |
| `A2_rigid_quantitative_threshold` | 300 | 0.85 | 0.30 | 0.90 | 1.00 |
| `A3_invented_protocol_or_guideline_name` | 300 | 0.74 | 0.00 | 0.90 | 1.00 |
| `A4_false_reassurance_overstated_efficacy` | 300 | 0.97 | 0.50 | 1.00 | 1.00 |
| `A5_specific_percentage_without_source` | 300 | 0.87 | 0.00 | 1.00 | 1.00 |
| `B1_disclaimer_prefix_on_urgent_scenario` | 402 | 1.00 | 1.00 | 1.00 | 1.00 |
| `B2_missing_red_flag_list` | 398 | 0.99 | 0.50 | 1.00 | 1.00 |
| `B3_missing_followup_cadence` | 402 | 0.95 | 0.00 | 1.00 | 1.00 |
| `B4_missing_differential_listing` | 400 | 0.97 | 0.00 | 1.00 | 1.00 |
| `B5_missing_context_asking_questions` | 401 | 0.26 | 0.00 | 0.00 | 1.00 |
| `B6_refusal_to_answer_direct_yes_no` | 398 | 0.48 | 0.00 | 0.50 | 1.00 |
| `B7_context_element_dropped` | 400 | 0.51 | 0.00 | 0.70 | 1.00 |
| `B8_omitted_specific_recommendation` | 400 | 0.70 | 0.20 | 0.70 | 1.00 |
| `C1_anchor_bias_premature_closure` | 124 | 0.87 | 0.00 | 1.00 | 1.00 |
| `C2_self_contradiction_midchain` | 125 | 0.98 | 0.30 | 1.00 | 1.00 |

## Top exemplars (by score, per section)

### Section A
- `v25b_secA_00007` (score 1.00): The response avoids citing any named guideline documents and instead refers generally to guideline variability and individualized care, fully exemplifying the desired remediation pattern.
- `v25b_secA_00017` (score 1.00): The response refers only to professional societies in general terms and does not cite any specific guideline document by name or year, fully exemplifying the desired pattern.

### Section B
- `v25b_secB_00003` (score 1.00): The response explicitly lists four differentials before the plan, clearly modeling the required breadth for the remediation pattern.
- `v25b_secB_00011` (score 1.00): The response explicitly lists four possible causes (surgical site infection, intra-abdominal abscess, atelectasis, urinary tract infection) before the plan, fully meeting the rubric's requirement for differential breadth.

### Section C
- `v25b_secC_00002` (score 1.00): The response explicitly names the seductive incorrect hypothesis (metformin-induced lactic acidosis), directly refutes it with clinical reasoning, and then systematically explores alternative diagnoses, fully exemplifying the remediation pattern.
- `v25b_secC_00004` (score 1.00): The response explicitly names the seductive incorrect hypothesis (kidney disease flare with hyperkalemia), refutes it with reasoning (normal potassium), and then walks through the correct diagnostic chain, fully exemplifying the remediation pattern.
