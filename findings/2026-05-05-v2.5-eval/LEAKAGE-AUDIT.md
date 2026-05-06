# V2.5 Ship-Rule Eval — Leakage Audit

- N test items scanned: 800
- 5-gram MinHash threshold (Jaccard): 0.7
- Memorization threshold (Levenshtein ratio): 0.85
- Overlap hits: 0
- Memorization hits: 0

## Overlap hits by benchmark

_None._

## Sample overlap hits (first 10)


## Memorization hits (first 10)

_None._

## Disposition

Any non-empty hit set requires manual review BEFORE publishing the
ship-rule decision. n-gram overlap with thresholds <0.85 may be
boilerplate (rubric phrasing); >0.85 is presumptively contaminated.
Memorization-probe hits are presumptive contamination at any rate.
