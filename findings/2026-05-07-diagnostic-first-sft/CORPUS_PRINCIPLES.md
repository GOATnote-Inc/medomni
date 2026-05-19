# V2.5b corpus principles — anti-patterns to avoid + positive patterns to teach

Distilled from the N=230 failure-mode probe (`CARD.md`). Each principle is a one-liner with the specific failure case it derives from.

## What V2.5 got wrong (anti-patterns to AVOID in corpus)

1. **Verbose padding** — Tie rate 24% suggests V2.5 generates padding gpt-4.1 ignores. Rule: every paragraph must carry a rubric-relevant claim.
2. **Rigid numerical thresholds** — "<2 cm for laparoscopy", "30-40% recurrence reduction". Rule: avoid specific numbers unless the rubric explicitly requires them.
3. **Invented guideline citations** — "Management of Traumatic Diaphragmatic Rupture: A 2023 Update". Rule: cite by societal name only ("ACEP", "AHA"), never by paper title or year.
4. **False reassurance** — "Masks fully prevent the flu." Rule: pair every reassurance with a hedge or escalation cue.
5. **Silent omission of emergency advice** — V2.5 skipping "go to ED" when V0 included it. Rule: every scenario with red-flag symptoms gets explicit escalation guidance.
6. **Polarity inversion on safety rubrics** — V2.5 met "Fails to advise emergency" while V0 didn't. Rule: every "Fails to <X>" rubric must be re-cast as "DO <X>" in the corpus.

## What V2.5b should learn (positive patterns to TEACH)

1. **Hedge specifically, not generally.** Bad: "Always consult your doctor." Good: "For dose adjustments in your range, talk to your prescriber within 1-2 weeks; for new symptoms X/Y/Z, go to ED today."
2. **Differential before plan.** Always name 2-4 differentials before settling on management.
3. **Acknowledge before recommend.** Open the response by reflecting the patient's framing in 1 sentence; don't jump straight to plan.
4. **Explicit follow-up cadence.** Every plan ends with "follow up in <interval> for <condition>."
5. **Ask before assume.** If the prompt has implicit context gaps (timeline, allergies, medications), ask explicitly.
6. **Bound numerical claims.** Use "varies by guideline / ranges from X to Y / consult current source" instead of single fixed numbers.
7. **Refuse with grace.** When forced to a specific quantitative claim that isn't safe to fabricate, say "I cannot give a specific number; the general direction is <X>."

## Hard rules for corpus generation

1. NO synthetic citations. If a fact requires a citation, use organizational naming only.
2. NO uniform "always consult your doctor" suffix. That's hedging, not training.
3. NO direct copy of V2.5 fabricated text into the corpus (even as a "bad example") without pairing with a "good example" rewrite. We don't want V2.5b to memorize the bad text.
4. NO duplicate scenarios across sections #1 / #5 (a single scenario goes to one section based on its dominant failure mode).
5. EVERY example must be filterable by the gpt-4.1 judge with a rubric matching its target failure mode. If the judge can't grade it, it's training noise.

## Out of scope (NOT in V2.5b corpus)

- Refusal training for adversarial inputs (medomni's existing surface; not the failure mode here)
- Tool calling (V2.7 territory)
- Multimodal vision/audio (imaging-PEFT-v1 territory)
- Multi-turn conversation (separate eval surface)
