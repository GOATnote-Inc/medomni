<!-- Thanks for the PR. Please run through this checklist before requesting review. -->

## Summary

<!-- One paragraph: what changed, why, and how it affects the headline numbers. -->

## Reproducibility

- [ ] If this PR changes inference, retrieval, or rubric behavior, a held-out CARD is attached or linked
- [ ] If a CARD is attached, the manifest emitter is byte-deterministic on the artifact (`make manifest-verify ARTIFACT=...`)
- [ ] Score deltas vs the prior baseline are documented (use `scripts/compare_cards.py`)
- [ ] No secrets committed (pre-commit + CI gates ran clean)
- [ ] No paths under CLAUDE.md §1 frozen list touched

## Methodology

- [ ] Cross-family judge used for any reported headline number (Qwen2.5-7B / Llama-3.x-Nemotron / similar — never Nemotron self-judge)
- [ ] Held-out fixtures used (no rubric-corpus circularity)
- [ ] Determinism: `--seed 42` + `--trials >= 3` for any reported mean
- [ ] If introducing a new evaluation, fixture provenance is documented in `corpus/clinical-fixtures-heldout/MANIFEST.md`

## Risk

- [ ] No paywalled content fetched or redistributed
- [ ] Patient-identifiable information: none included anywhere
- [ ] If touching safety rails, a NemoGuard / Llama-Guard test fixture is included

## Demo readiness

- [ ] If this PR affects a demo scene, the corresponding scene in `findings/research/2026-04-29-medomni-v1-northstar/DEMO-RUNBOOK.md` is updated
