# judge_filter Option-C batch — CARD

**Status:** SHIPPED 2026-05-05 (iter-145).
**Run dir:** `/Users/kiteboard/data-queue/curated/reasoning/` (laptop-local; not in repo).
**Trigger:** Iter-134 user request — judge new ~30K factory items while V2.5 cooks. Reframed to Option-C scope (~10K items at $80 budget) per cost projection at iter-133.

## Summary

| | Value |
|---|---|
| Total items judged | **10,480** |
| Curated (PASS) | **4,531** |
| Drop rate | 56.8% |
| Total spend | **$85.10** |
| gpt-4.1 spend | $9.34 |
| claude-opus-4-7 spend | $75.76 (89% of cost) |
| Wall-time | ~37 min |
| Throughput | ~5.5 items/sec |

## Pre-flight discovery

Smoke run on 1+56=57 items revealed `ANTHROPIC_API_KEY` in `/Users/kiteboard/lostbench/.env`
returned **HTTP 401 invalid x-api-key**. Smoke pass=0/57 + Opus_spend=$0.00 was the
smoking-gun signal — Opus was silently zeroing scores, blowing the |gpt41-opus47|>0.3
disagreement gate on every item. Pre-flight saved ~$30 + ~1 hr of garbage output
on a full corpus run with broken keys.

Memory `feedback_anthropic_key_rotates_silently.md` saved (2nd time this canonical
key has 401-rotated; first incident was 2026-05-04 iter-80, task #80).

## Configuration

```bash
.venv/bin/python3 scripts/judge_filter.py \
  --input-dir /Users/kiteboard/data-queue \
  --output-dir /Users/kiteboard/data-queue \
  --task reasoning \
  --mode one-shot \
  --budget-usd-daily 85 \
  --item-concurrency 16 \
  --rubric-concurrency 2
```

- Models: gpt-4.1-2025-04-14 + claude-opus-4-7
- Pass gate: |gpt41 - opus47| ≤ 0.3 AND both ≥ 0.6
- Source factory: warm-lavender-narwhal `~/data-queue/raw/reasoning/` (rsync'd to laptop)
- Sovereignty: laptop-only judge calls (no cloud keys on pods)

## Output

```
/Users/kiteboard/data-queue/curated/reasoning/
├── 000000.jsonl   # 30 items (smoke-only first shard)
├── 000001.jsonl   # 583 items
├── 000002.jsonl   # 627 items
├── 000003.jsonl   # 654 items
├── 000004.jsonl   # 661 items
├── 000005.jsonl   # 621 items
├── 000006.jsonl   # 670 items
└── 000007.jsonl   # 645 items
                  ─────
                  4531 total curated items
```

Each line is a JSON object with `task`, `question`, `final_answer` (cot_chain
+ votes stripped per script semantics; lobster trains on Q+A only).

## Strategic context

- This run was scoped to Option C ($80 cap) per the iter-133 cost projection. Options B ($250) and A ($500) are still on the table once V2.5 ship-rule decision is made.
- Curated 4,531 items are eligible for V2.5b (corpus-expansion refire) IF V2.5 ship-rule fails or eval delta is marginal. They do NOT directly feed V2.7 (tool-call SFT, different corpus type).
- `feedback_anthropic_key_rotates_silently.md` adds durable smoke-first protection against silent-401 reoccurrence.

## What this CARD does NOT do

- Does NOT promote curated items into V2.5b training mix — that's gated on V2.5 eval outcome.
- Does NOT push curated/ to lobster — staying laptop-local until needed.
- Does NOT re-judge already-judged items — 194 .judged markers prevent re-processing.

## Cross-references

- [feedback_anthropic_key_rotates_silently.md](../../../.claude/projects/-Users-kiteboard/memory/feedback_anthropic_key_rotates_silently.md)
- [feedback_eval_preflight_judge_key.md](../../../.claude/projects/-Users-kiteboard/memory/feedback_eval_preflight_judge_key.md)
- `scripts/judge_filter.py` — `/Users/kiteboard/prism42-nemotron-med/scripts/judge_filter.py`
- task #92 — prior 32K full-corpus judge (different setup, not on disk locally)
