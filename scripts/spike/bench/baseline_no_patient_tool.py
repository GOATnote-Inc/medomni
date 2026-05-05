"""S3 baseline — same harness as bench_dual_lookup.py, but does NOT pass
`patientId` in the request body, so `get_patient_context` (the S2 5th tool) is
unavailable and only the existing 4 tools fire. This baselines the 4-tool
latency profile so we can compute the delta cost of adding the 5th tool.

Reuses the parsing + summarization logic from bench_dual_lookup. The
`parallel_tool_wait` row in the output table will typically be 0/empty here
because `get_patient_context` doesn't run.

Env vars (BENCH_PATIENT_ID is intentionally NOT required for this script):
  BENCH_BASE_URL       (default: http://localhost:3000)
  BENCH_CONCURRENCY    (default: 1)
  BENCH_N_REQUESTS     (default: 20)
"""

from __future__ import annotations

import os
import sys

# Same-directory import so this works whether you run `python baseline...py`
# or `python scripts/spike/bench/baseline...py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bench_dual_lookup import main as _main  # noqa: E402

if __name__ == "__main__":
    # require_patient_id=False -> no `patientId` field in the request body.
    raise SystemExit(_main(require_patient_id=False))
