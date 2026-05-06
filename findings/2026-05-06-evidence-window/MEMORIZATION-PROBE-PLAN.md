# E5 — Memorization probe DESIGN (deferred to T+5h)

**Status:** DESIGN ONLY. Run AFTER ship-rule eval finishes and the V2.5
serve endpoint is up, OR (preferred) on a parallel cold-start serve to
avoid contaminating the ship-rule run.

## Why

Verbatim recall = the model regurgitating eval-test-set strings it saw at
training time. If the V2.5 reasoning-SFT inadvertently saw the eval test
items (because the training corpora drew from public sources that
overlap with the test set), benchmark scores are inflated and not
generalization signal.

E4 (n-gram-overlap) catches **dataset-level** contamination at the surface.
E5 catches **memorization** at the model behavior level — items where E4
showed no overlap, but the model still produces verbatim continuations.

Together E4 and E5 form the contamination-risk pair: E4 is preventative
(doesn't require a working model), E5 is confirmatory (requires the
trained model).

## Method

For each of the 4 ship-rule benchmarks (MedQA-USMLE, MedXpertQA-Text,
HealthBench-Hard, PubMedQA-L):

1. **Sample 50 random items** from the test split.
   Seed = 42 for reproducibility. Save the sample id list to
   `findings/2026-05-06-evidence-window/E5-sample-{benchmark}.txt`
   so the probe is reproducible.

2. **Split each item** at the midpoint by tokens (use the V2.5
   tokenizer):
   - `prefix` = first half of the canonical item text (question + options
     for multi-choice; question for free-form).
   - `suffix` = second half.

3. **Prompt V2.5** with `prefix` + a continuation cue:
   ```
   Continue this passage VERBATIM:
   {prefix}
   ```
   Decode params identical to ship-rule eval (T=0, top_p=1, max_new=2048,
   thinking off, batch_size=1).

4. **Compare** the V2.5 continuation against the held-out `suffix`
   using:
   - **Exact substring match** at >= 80% of the suffix length, OR
   - **5-gram overlap** computed as `|model 5-grams ∩ suffix 5-grams|
     / |suffix 5-grams|` >= 0.80
   ANY of the two triggers a verbatim-recall flag.

5. **Aggregate per benchmark**:
   ```
   verbatim_recall_pct = (n_items_flagged / 50) * 100
   ```

## Decision rule

| verbatim_recall_pct | Verdict | Action |
|---|---|---|
| < 2% | clean | report; no caveat |
| 2-5% | borderline | report with caveat in CARD; benchmark still primary |
| 5-10% | concerning | benchmark gets footnote; not the SOLE pass criterion |
| > 10% | reject | exclude from primary pass criteria; eval CARD must call this out prominently |

These thresholds are pre-registered. Do not change after seeing results.

## Implementation skeleton (NOT to run yet)

```python
# scripts/memorization_probe.py
import argparse, json, random, hashlib
from pathlib import Path
from collections import Counter
import httpx

def ngrams(text, n=5):
    toks = text.split()
    return [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]

def overlap_5gram(a, b):
    a_set = Counter(ngrams(a, 5))
    b_set = Counter(ngrams(b, 5))
    if not b_set: return 0.0
    intersection = sum((a_set & b_set).values())
    return intersection / sum(b_set.values())

def probe_one(item_text, serve_url, model_id, max_tokens):
    toks = item_text.split()
    mid = len(toks) // 2
    prefix = " ".join(toks[:mid])
    suffix = " ".join(toks[mid:])
    prompt = f"Continue this passage VERBATIM:\n{prefix}"
    body = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = httpx.post(f"{serve_url}/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    completion = r.json()["choices"][0]["message"]["content"] or ""
    return {
        "completion": completion,
        "exact_match": suffix.lower()[:int(0.8*len(suffix))] in completion.lower(),
        "fivegram_overlap": overlap_5gram(completion, suffix),
        "flagged": (
            (suffix.lower()[:int(0.8*len(suffix))] in completion.lower())
            or overlap_5gram(completion, suffix) >= 0.80
        ),
    }

# main loop: load benchmark, sample 50, probe each, aggregate, write JSON.
```

## Outputs

- `findings/2026-05-06-evidence-window/E5-RESULTS.md` — table of per-benchmark
  verbatim-recall percentages with flagged-item examples (hashed item ids
  to avoid leaking test items).
- `findings/2026-05-06-evidence-window/E5-sample-{benchmark}.txt` — frozen
  item id list for reproducibility.
- `findings/2026-05-06-evidence-window/E5-raw-{benchmark}.jsonl` — full
  per-item record.

## Order-of-operations

1. Ship-rule eval finishes (target: 2026-05-06 evening).
2. CARD.md filled in.
3. **Then** fire E5 against the SAME serve (or a parallel cold-start
   serve to avoid affecting ship-rule reproducibility).
4. Update each benchmark's CARD section with the recall % and a link
   to E5-RESULTS.md.

## Hard rule

If E5 finds >= 10% verbatim recall on a benchmark whose ship-rule pass
hinged on it, the ship-rule decision MUST be revisited.
