# SUPERSEDED — pre-staged content for public `GOATnote-Inc/medomni` repo

**This file is NOT for the private repo.** It will be copied verbatim
into the public `GOATnote-Inc/medomni` repo at
`results/ci-medomni-heldout-consolidated-20260429-173557/SUPERSEDED.md`
when HITL issue #2 is approved. Pre-staged here so the doc move is
mechanical at approval time.

---

# Superseded by canonical N=1000 gpt-4.1-graded result (2026-05-01)

This CARD's headline claim — **0.385 ± 0.000** on HealthBench Hard
(N=6 examples × 3 trials = 18 example-trials, Nemotron-Omni
self-judged via the same model) — is **superseded** by a canonical
study under the protocol described in `arXiv:2505.08775` (OpenAI
HealthBench paper, May 2025).

## Why this CARD was retired

The autoresearcher loop in `prism42-nemotron-med` (private dev repo)
ran a pre-registered N=1000 paired study under the canonical
HealthBench Hard protocol with `gpt-4.1-2025-04-14` as the grader.
That study **reversed the headline claim**:

| | this CARD (now retired) | canonical N=1000 study (2026-05-01) |
|---|---|---|
| Mean score | **0.385 ± 0.000** | **0.054** (V0 baseline, no RAG) |
| N | 6 × 3 = 18 | 1000 paired |
| Grader | Nemotron-Omni judging itself | gpt-4.1-2025-04-14 |
| 95% CI | ± 0.000 (small-sample artifact) | bootstrap CI on shared ids |
| Comparison vs Opus 4.7 | "disjoint, higher" | **CIs overlap** with prior baseline |
| Direction of bias | Same-family judge bias + small-N | none (canonical protocol) |

The 7× headline gap (0.385 → 0.054) came from **three compounding
artifacts**:

1. **Same-family judge bias.** Nemotron-Omni judging its own responses
   produces dramatically optimistic scores. The canonical protocol
   uses an **independent grader** (gpt-4.1-2025-04-14, validated against
   physician agreement per arXiv:2505.08775 §8) for exactly this reason.
2. **Small-sample CI = 0.000.** ±0.000 half-width on N=18 is not
   statistical confidence; it's a small-sample artifact from low
   between-trial variance on a 6-item set. N=1000 reveals real spread.
3. **Lenient secondary grader (Qwen2.5-7B).** A subsequent pass with
   Qwen scored V0 at 0.173 — closer to truth, but still **+0.120
   absolute lenient** vs canonical gpt-4.1. The canonical study
   formally retired Qwen as a published-number grader on this
   benchmark per pre-registered calibration rule.

## Companion result: PrimeKG-as-RAG regression

The canonical study also ran a paired V1 arm with PrimeKG nx-cugraph
k-2 subgraph slices prepended to the system prompt. Result:

> **Δ V1−V0 = −0.0542, 95% paired-bootstrap CI [−0.0731, −0.0357]**
> (n_shared = 1000 items, 537 of which had PrimeKG fire on
> string-match seeding)

CI strictly negative → statistically significant **regression of
−5.4pp**. The pre-registered ship rule (Δ ≥ +0.020 with CI excluding 0)
was violated; PrimeKG-as-prepended-slice was **dropped** from the live
`/api/ask` deployment plan.

This is a **negative result on a graph-RAG architecture**, not just a
correction to a prior number. The autoresearcher loop is adversarial
to the medomni team's prior claims — that's the discipline working.

## Where to read the canonical CARD

[`findings/2026-05-01-hb-canonical/CARD.md` in `prism42-nemotron-med`](https://github.com/GOATnote-Inc/prism42-nemotron-med/blob/main/findings/2026-05-01-hb-canonical/CARD.md)

(Mirror in this public repo: `results/canonical-2026-05-01-hb-hard-n1000/CARD.md`,
linked from the README's "Latest CARD".)

## What this means for medomni positioning

- **HB Hard text-only V0**: 0.054 canonical (vs OpenAI top published
  ≤0.32) — gap of ~26.5pp absolute, ~83% relative. Far from frontier.
- **Imaging V0** (separately measured): VQA-RAD 0.643, SLAKE-en 0.744,
  decode-failure 0% — MedGemma-4B-class on imaging-VQA. (See
  [imaging V0-CARD](https://github.com/GOATnote-Inc/prism42-nemotron-med/blob/main/findings/2026-05-01-imaging-peft-v1/V0-CARD.md).)
- **Asymmetry**: Nemotron-Omni-30B is competitive on imaging, weak on
  text-only HB Hard. The PEFT recipe ([MEDOMNI-NEMOTRON-RECIPE.md](https://github.com/GOATnote-Inc/prism42-nemotron-med/blob/main/findings/2026-05-01-medgemma-recipe/MEDOMNI-NEMOTRON-RECIPE.md))
  prioritizes V2 multi-task SFT (HealthBench-train + MedQA-train +
  MedMCQA + PubMedQA-Labeled + chain-of-thought) and V3 DPO refusal
  calibration to close the text-modality gap.

## What this CARD's reader should believe

If you came here from the README's old "Latest CARD" link expecting
Nemotron-Omni HB Hard at 0.385:

- That number was an artifact, not a measurement.
- The honest current measurement is 0.054 canonical.
- The medomni team is publishing the negative result on PrimeKG-as-RAG
  because the autoresearcher loop produced it under pre-registered
  rules; honest negative results are part of the rigor commitment.
- Future versions (V1 imaging-PEFT, V2 multi-task SFT, V3 DPO) target
  measurable lift toward MedGemma-class numbers, with the same
  pre-registered → CARD discipline.
