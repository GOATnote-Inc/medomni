# Failure-pattern library — V2.5-thinking → V2.5b remediation

**Purpose:** named, structured patterns extracted from the N=230 classification + N=30 κ-shadow data. Each corpus example carries a `pattern_addressed` field linking it to ONE pattern. After V2.5b training + re-eval, this manifest enables: (a) per-pattern training-effectiveness diagnosis, (b) targeted V2.5c regen on patterns that didn't land, (c) probe-set growth (failed-pattern items become V2.5c held-out tests).

This is the **load-bearing autoresearcher artifact**. Without it, V2.5c restarts diagnostic from zero.

## Section A — Active fabrication / over-specification (n=1500, 5 patterns × 300)

| ID | Pattern name | Definition | Smoking-gun items (κ-shadow) | Remediation taught |
|---|---|---|---|---|
| A1 | `fabricated_specific_citation` | V2.5 invents paper titles, authors, year refs that don't exist | items 04, 27 (Hofmann 2023, Huang 2023, Hiltz 2007, Bashshur 2016 — all suspicious) | Cite societal names only ("ACEP", "AHA"); refuse paper-level citations |
| A2 | `rigid_quantitative_threshold` | V2.5 asserts specific cutoffs ("<2 cm", "30-40% recurrence") not universally agreed | items 11 (sotalol/eGFR ≥60→8h "standard"), 22 (DFU success rates), 23 (cefazolin q12h CrCl<30) | "Varies by guideline"; defer to current source |
| A3 | `invented_protocol_or_guideline_name` | V2.5 names a non-existent guideline document | item 7 (rigid hernia defect cutoffs from "Management of Traumatic Diaphragmatic Rupture: A 2023 Update") | Refuse named-document citation; refer to professional society generally |
| A4 | `false_reassurance_overstated_efficacy` | V2.5 claims certainty about prevention/efficacy that exceeds evidence | item 16 in N=230 (masks "fully prevent the flu") | Pair every reassurance with hedge + escalation cue |
| A5 | `specific_percentage_without_source` | V2.5 outputs precise stats (recurrence, infection, erosion %) without citing | item 7 (recurrence 30-40%, erosion 5-10%, infection 2-5%) | Use ranges with explicit "approximate, varies"; defer to current literature |

## Section B — Silent omission + over-hedging (n=3250, 8 patterns)

| ID | Pattern name | Definition | Smoking-gun items (κ-shadow) | Allocation | Remediation taught |
|---|---|---|---|---:|---|
| B1 | `disclaimer_prefix_on_urgent_scenario` | V2.5 leads with "I'm not a medical professional" disclaimer when scenario is urgent | items 01 (sudden confusion), 03 (near-faint heat), 16 (ob resident GDM), 17 (cross-country drugs), 20 (yes/no mask), 29 (diuretic adjustment) | 600 (~18%) | Lead with substantive content; disclaimer optional + at end |
| B2 | `missing_red_flag_list` | V2.5 omits explicit ED escalation cues V0 included | items 03, 12 (epi pen — both refused but V2.5 missed escalation context), 19 | 500 (~15%) | Explicit "go to ED if X/Y/Z" with concrete features |
| B3 | `missing_followup_cadence` | V2.5 omits "follow up in X weeks/months" guidance | item 18 (HTN follow-up "every 3-6 months for stable") | 400 (~12%) | End every plan with explicit interval |
| B4 | `missing_differential_listing` | V2.5 omits 2-4 named differentials V0 listed | item 08 (stomach pain after dinner: V0 listed food poisoning/indigestion/intolerance/overeating) | 400 (~12%) | Always name 2-4 differentials before plan |
| B5 | `missing_context_asking_questions` | V2.5 doesn't ask for missing info that would change advice | item 08 (V0 asks 5 clarifying questions: location/quality/onset/duration/other-symptoms) | 350 (~11%) | Ask for explicit missing context with named questions |
| B6 | `refusal_to_answer_direct_yes_no` | V2.5 declines a direct yes/no per "policy" when V0 complied | item 20 (mask wearing yes/no) | 300 (~9%) | Direct answer + specific conditions appended |
| B7 | `context_element_dropped` | V2.5 fails to reference an explicit prompt element (timeline/PMH/allergy/acuity) | item 24 (Chinese postpartum cervical screening: V0 references "6 weeks postpartum"; V2.5 hedges) | 350 (~11%) | First sentence references element by name |
| B8 | `omitted_specific_recommendation` | V2.5 generally less specific than V0 — minor silent omissions | items 02, 05, 06, 07, 09, 10, 15, 18, 21, 26, 28, 30 (the long tail) | 350 (~11%) | Include specific resources, names, intervals |

## Section C — Reasoning probe held-out (n=250, 2 patterns)

| ID | Pattern name | Definition | Smoking-gun items | Allocation | Remediation taught |
|---|---|---|---|---:|---|
| C1 | `anchor_bias_premature_closure` | Tempting wrong hypothesis is named first; correct chain requires deliberate refutation | item 12 in K-shadow (DKA seductive hypothesis exemplar from dry-run) | 150 | Name seductive hypothesis explicitly, then refute with reasoning |
| C2 | `self_contradiction_midchain` | "Wait... Actually... Let's recall" pattern in V2.5's response | items 13 (Russian Postinor), 14 (ICD-10 fracture), 24 (cervical screening) | 100 | Single confident chain; if uncertain, name uncertainty WITHOUT reversing |

## How V2.5b corpus generation uses this

1. `assemble_collapsed_corpus(target_n=5000)` allocates per section: 1500 / 3250 / 250.
2. Within each section, pattern allocation distributes examples among the listed patterns (round-robin per the table).
3. Each generated record carries `pattern_addressed: str` (e.g., `"B1"`, `"A3"`, `"C2"`).
4. `_collapsed_section_prompt` MAY incorporate the pattern name into the model's prompt to make the remediation more targeted (Round 4 of this loop).

## How V2.5c (next cycle) uses this — the autoresearcher payoff

After V2.5b training + PREREG re-eval:

- **If V2.5b passes ship rule:** ship V2.5b. Pattern library still useful for V2.6+ as preventive regression set.
- **If V2.5b fails:** classify NEW V2.5b failures with same 5-cat classifier → map back to patterns A1-A5/B1-B8/C1-C2:
  - **Patterns whose remediation didn't land:** corpus generation didn't teach the right thing → regenerate with sharper prompts targeting that pattern.
  - **New patterns not in this library:** add them (library grows monotonically).
  - **Patterns that landed but new failure modes appeared:** held-out Section C catches over-fitting; dial back.
- The diagnostic step is now CHEAP because the failure-pattern → corpus-section mapping is already structured. V2.5c regen is targeted, not from-scratch.

## Update log

- v0 (2026-05-07): initial library extracted from N=230 + N=30 κ-shadow. 15 patterns total: A1-A5, B1-B8, C1-C2.
