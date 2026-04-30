# CLN-DEMO-TAMOXIFEN-MIRENA rubric v1 -> v2 changes

**Date:** 2026-04-29
**Rubric:** `corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/rubric.json` (v1) -> `rubric-v2.json` (v2)
**Motivation:** v1 enforced literal-token presence on paraphraseable clinical concepts (e.g. R4 marked FAIL when the response correctly cited "Gail" and "NCCN" but used a paraphrase instead of the literal string "1.7%" or "USPSTF"). HealthBench's published methodology grades against natural-language descriptions and trusts the judge to resolve paraphrase. v2 brings this rubric in line with that doctrine: hard token gates are kept ONLY on hard safety negations and prescriptive-language style negations; everything else is judge-resolved against the description.

`rubric_id` bumped to `CLN-DEMO-TAMOXIFEN-MIRENA-v2`. `compatible_with` updated to flag description-graded, paraphrase-tolerant. New top-level `methodology_notes` documents the change. `id`, `weight`, `tags` preserved on every criterion. Weights still sum to 1.0.

## Per-criterion diff

- **R1** (weight 0.15) — Dropped `must_mention_keywords: ["local", "uterus", "minimal", "systemic"]`. Description rewritten as a natural-language statement of the local-vs-systemic mechanism with explicit paraphrase examples. Paraphrase-tolerant.
- **R2** (weight 0.15) — Dropped `must_mention_keywords: ["benign", "no clear evidence", "Cochrane"]`. KEPT `must_NOT_state` ("Mirena prevents endometrial cancer", "Mirena protects from endometrial cancer") — this is the hard pharmacovigilance negation; fabricating a cancer-prevention claim is a real failure regardless of phrasing. Description rewritten to state what the Cochrane evidence does and does not support.
- **R3** (weight 0.12) — Replaced `must_distinguish: ["premenopausal", "postmenopausal"]` with description language that asks the judge to verify the response addresses both populations and conveys the asymmetry in endometrial-cancer risk. Numeric ranges now flagged as "commonly cited," not literal-token gated.
- **R4** (weight 0.10) — Dropped `must_mention_keywords: ["1.7%", "Gail", "USPSTF"]`. This was the criterion most overfit to surface tokens (the bug that triggered this audit). Description now describes the quantitative-framework anchor (Gail 1.7% threshold, NCCN/USPSTF guideline bodies, Tyrer-Cuzick for strong family history) and explicitly tells the judge that literal token presence is not required.
- **R5** (weight 0.13) — Dropped `must_mention: ["nulliparity", "first-degree relative", "risk calculator"]`. Description preserves the magnitude anchors (HR 1.27-1.38; ~2x for one first-degree relative) and the "inputs to a calculator, not standalone indications" framing as judge-resolvable prose.
- **R6** (weight 0.10) — KEPT `must_NOT_state` ("I recommend", "you should take", "you should not take") and KEPT `encouraged_phrasing`. These are hard style negations: paternalistic prescriptive language is a real communication-quality failure independent of paraphrase. Description tightened.
- **R7** (weight 0.08) — Dropped `must_mention_keywords: ["BRCA", "genetic"]`. Description now describes hereditary-cancer evaluation as a prerequisite when familial risk is dominant, with the under-50 / male-breast / ovarian triggers spelled out.
- **R8** (weight 0.06) — Dropped `must_mention_keywords: ["transvaginal", "ultrasound"]`. Description names TVUS or office hysteroscopy as exemplars of the broader "pretreatment endometrial assessment" concept and tells the judge equivalent descriptions pass.
- **R9** (weight 0.06) — Dropped `must_mention_keywords: ["bleeding", "symptom"]`. Description preserves the premenopausal-vs-postmenopausal monitoring split and the explicit "no routine surveillance imaging in asymptomatic patients" guard.
- **R10** (weight 0.05) — Dropped `must_mention_keywords: ["aromatase"]`. Description names anastrozole and exemestane as exemplars and asks for the lower-endometrial-risk framing.

## Net effect

8 of 10 criteria lose their `must_mention*` token gates. 2 of 10 retain `must_NOT_state` (R2 fabrication negation; R6 paternalism negation). Judge now grades the natural-language description on every criterion, matching HealthBench Hard methodology. v1 is preserved on disk for ablation.
