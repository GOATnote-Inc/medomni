# V2.5 Ship-Rule Eval — Methodology Pivot (thinking=False → thinking=True)

**Authored:** 2026-05-07. **Status:** Re-fire in flight; thinking=True results land in `findings/2026-05-05-v2.5-eval-thinking/`.

## TL;DR

The first ship-rule run produced **FAIL (1/4 PASS)** with `enable_thinking=False`. That result is **methodologically deniable**: V2.5's entire training (MedReason 32K + medical-o1-reasoning-SFT 25K) optimized the *thinking channel*. Evaluating with the thinking channel disabled measures the wrong thing. The 2026-05-06 re-fire turns it back on; both arms become an A5 ablation (PREREG-named) and the methodology gap closes regardless of polarity.

## What broke

`scripts/ship_rule_lib/generators.py` shipped with `enable_thinking=False` as the `DECODE_PARAMS` default — inherited from `sovereign_bench.py:125` per E-track audit (PR #120 §E7). PREREG fixed `temp=0`/`top_p=1.0`/`max_new_tokens=2048` but did not lock `enable_thinking`. The driver chose the lower-cost option by default. For a base model the choice is symmetric; for a reasoning-SFT-trained adapter it is not.

Cross-reference: medomni canonical runs (tasks #63 HealthBench V0/V1 N=100, #65 MedQA N=1273, #66 HB-Hard N=1000) all used **thinking=ON**. The driver deviated from established practice.

## Patch (PR #122 head 6c14c26 → ee618b8)

- `generators.DECODE_PARAMS` default flipped to `{"enable_thinking": True, "max_tokens": 8192}`. New `make_decode_params(*, enable_thinking, max_tokens)` helper for explicit overrides.
- `ship_rule_eval.py gen` learns `--enable-thinking / --no-enable-thinking` and `--max-new-tokens` (BooleanOptionalAction).
- `grader.strip_thinking()` slices to last `</think>` before MCQ regex / yn regex / gpt-4.1 rubric grading. Raw response preserved verbatim in gen JSONL. 3 unit cases pass (post-tag, no-tag, multi-tag).

## Re-fire scope

| | thinking=False (run 1) | thinking=True (run 2, in flight) |
|---|---|---|
| n per benchmark | 200 | 200 |
| benchmarks | 4 | 4 |
| arms | V0, V2.5 | V0, V2.5 |
| seeds | 42, 123, 7919 | 42, 123, 7919 |
| max_new_tokens | 2048 | 8192 |
| ctx (vllm) | 4096 | 16384 |
| GPU residency | judge co-resident @ 0.30 + V0/V2.5 @ 0.55 | judge stopped, V0 then V2.5 sequential @ 0.85 |
| LoRA hot-load | failed in vllm 0.20 → merge_and_unload | merged weights re-used (same `/workspace/v2.5-merged/`) |
| A5 ablation | covers thinking=False arm | covers thinking=True arm |

The two runs together close the A5 ablation named in PREREG.

## Why thinking=True changes the comparison

Reasoning-SFT teaches the model to *use* the thinking channel for medical reasoning. With thinking off:
- V0 and V2.5 both produce direct-answer outputs; LoRA may slightly perturb the direct-answer distribution from base; net effect is small noise.
- V2.5's specialized capacity is never exercised. A small regression is consistent with LoRA's effect on direct-answer formatting without compensating use of the trained channel.

With thinking on:
- V0 (already a reasoning model) produces reasoning chains too; both arms exercise the channel.
- V2.5's specialization can manifest as better-organized chains, more accurate intermediate steps, or earlier abstain decisions.
- The headline now measures the *signed* effect of reasoning-SFT specifically, not a confound between LoRA and decode mode.

This is the comparison the PREREG ship-rule conditions were intended to test.

## What this document does NOT claim

- Does NOT claim thinking=True will PASS. The point of the re-fire is to produce a non-deniable verdict in either direction.
- Does NOT claim the original FAIL was "wrong." Under the deviated config it was correct; under the canonical config it was the wrong question.
- Does NOT defer the V2.5b vs V2.7 decision. That decision waits on the thinking=True ship-rule result.

## Cross-references

- `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml` — original pre-registration
- `findings/2026-05-05-v2.5-eval/SHIP-RULE-RESULTS.{md,json}` — thinking=False results (PR #122 commit `10f94d6`)
- `findings/2026-05-05-v2.5-eval-thinking/` — thinking=True results (in flight)
- PR #120 §E7 — E-track determinism audit that flagged the hardcoded `enable_thinking=False`
- `~/.claude/projects/-Users-kiteboard/memory/feedback_two_stage_grading_pattern_is_canonical.md` — gpt-4.1 grading from `/lostbench/.env` is laptop-only; pod side has no cloud key
