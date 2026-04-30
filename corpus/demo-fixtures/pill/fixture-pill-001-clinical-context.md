# Clinical context — fixture-pill-001-warfarin-tablets.png

## What the artifact shows

Three round, scored, color-coded oral tablets photographed against a neutral background, identifiable by the standardized warfarin sodium color scheme used across most North American manufacturers (Bristol-Myers Squibb Coumadin trade dress and the bioequivalent generics):

- **Pink** — warfarin sodium 5 mg
- **Blue** — warfarin sodium 3 mg (some manufacturers use light-blue for 3 mg, teal for 1 mg)
- **Brown / tan** — warfarin sodium 1 mg

Color-coded warfarin pill identification is part of standard nursing pharmacology curricula precisely because mis-counting strengths is one of the highest-frequency sources of warfarin medication error.

## Correct interpretation a clinician would expect

A defensible read: **"These are warfarin (Coumadin) tablets, an oral vitamin-K-antagonist anticoagulant. The trade-dress colors are pink = 5 mg, blue = 3 mg, brown = 1 mg. Key safety: (1) narrow therapeutic index, INR-monitored dosing, target typically 2.0–3.0 for AFib / VTE; (2) extensive food and drug interactions (vitamin-K-rich foods, antibiotics, NSAIDs, amiodarone); (3) bleeding-risk patient education is mandatory. If found loose without packaging in a confused or pediatric patient, treat as accidental ingestion: check INR stat, consider vitamin K reversal."**

A nurse-persona answer should escalate to: confirm the patient's documented warfarin regimen; verify last INR; never administer without dose-and-INR cross-check; flag the unscheduled tablet to physician and pharmacy.

## What Omni produced (smoke result)

> "Medication: Warfarin (Coumadin) - an anticoagulant. Safety Considerations: 1. Bleeding Risk: Warfarin significantly increases the risk of bleeding. Patients must avoid activities that could lead to injury and report any signs of bleeding (bruising, blood in urine/stool, nosebleeds) immediately. 2. Drug Interactions: It interacts dangerously with many common medications (like antibiotics, NSAIDs) and supplements (like vitamin K), altering its effectiveness and increasing bleeding risk. Consistent monitoring and communication with healthcare providers are crucial."

Strong identification + clinically defensible safety pair. Did not enumerate the per-color dose-strength mapping — that is a downstream graph-retrieval enrichment, not a base-model failure. **No retake needed.**

## Demo runbook scene mapping

- **SPEC §7 / Multimodal scene (3 min)** — implements the "nurse photographs an unfamiliar pill" beat the user describes in the fixture brief. On-stage prompt: *"Patient brought this from home, found at the bedside, no bottle — what is this and what should I check before giving anything else?"*
- Bonus use case: feeds the **persona-tagged graph schema (§5.5)** since the warfarin → INR → vitamin-K → drug-interaction edge cluster is a high-density node in the OpenEM-expanded knowledge graph.
