# S3 — dual-lookup latency benchmark harness

Measures whether Pattern B (dual-lookup, parallel tool dispatch with the new
5th tool `get_patient_context`) stays inside its end-to-end latency budget.

## Prereqs
- Python 3.11+
- S1 sandbox running (synthetic patient store reachable from the agent)
- S2 tool integrated (`get_patient_context` registered + `patientId` accepted
  in `/api/agent` request body), with `vercel dev` or a production deploy
  reachable at `BENCH_BASE_URL`

## Run order

```bash
pip install -r requirements.txt

export BENCH_BASE_URL=http://localhost:3000        # or https://www.thegoatnote.com
export BENCH_PATIENT_ID=synthea-demo-001            # synthetic Synthea patient ID — never PHI
export BENCH_CONCURRENCY=4
export BENCH_N_REQUESTS=40

# Pattern B (5 tools, dual lookup):
python bench_dual_lookup.py | tee results-dual.md

# Baseline (4 tools, no patientId):
python baseline_no_patient_tool.py | tee results-baseline.md

diff results-baseline.md results-dual.md
```

## Interpreting p50/p95/p99

Each value is wall-clock milliseconds for the named phase, aggregated across
successful requests:
- **ttfb** — POST sent to first SSE byte. Cold-start + edge routing.
- **time_to_dispatch** — POST sent to first `tool-input-available`. Reasoning + token-emit time before tools fire.
- **parallel_tool_wait** — first `tool-input-available` to `max(primekg_done, patient_context_done)`. The actual cost of the parallel dual lookup. Empty/skipped when neither tool runs (knowledge-only prompts).
- **text_stream** — `text-start` to `text-end`. Final-answer streaming duration.
- **end_to_end** — POST sent to `text-end` (or stream close). What the user actually waits for.

p50 = median, p95/p99 = tail. The parallel_tool_wait p95 is the load-bearing number for Pattern B.

## Decision criteria the architect cares about

- **Pattern B ships** if dual-lookup `end_to_end` p95 < 1500 ms AND
  `parallel_tool_wait` p95 < 600 ms AND failure rate < 5%.
- **Pattern B needs precomputed cache** if dual-lookup `end_to_end` p95 in
  [1500 ms, 3000 ms].
- **Pattern B is unviable** if dual-lookup `end_to_end` p95 > 3000 ms — fall
  back to Pattern C (server-cached patient summary, single lookup).
