# V2.5 evidence-window — E-track reproducibility infrastructure

**Date:** 2026-05-06
**Purpose:** Convert "convincing" V2.5 ship-rule evidence into
"undeniable + verifiable + reproducible". Authored before any of the
post-training evals (A4/A5/A6/B2/B3/B4/B6) so commit timestamps are
cryptographic pre-registration proof.

**Companion artifacts:**
- `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml` — V2.5 training PREREG
- `findings/2026-05-05-v2.5-eval/CARD.md` — V2.5 ship-rule eval (in flight)

## File index

### Pre-registration (E1)
- `PREREG-A4.yaml` — extended evals (MedMCQA, MMLU-Med, CareQA, MMLU-STEM)
- `PREREG-A5.yaml` — thinking-on vs thinking-off ablation
- `PREREG-A6.yaml` — V3 GRPO rollouts on V2.5
- `PREREG-B2.yaml` — B300 throughput sweep
- `PREREG-B3.yaml` — NVFP4 vs BF16 quality equivalence (TOST)
- `PREREG-B4.yaml` — multimodal paths no-regression
- `PREREG-B6.yaml` — NVFP4 + LoRA hot-load deploy gate

### Hardware / software state (E2)
- `manifest-lobster.json` — H200, driver 580.126.09, CUDA 13.0
- `manifest-narwhal.json` — H200, driver 580.126.09
- `manifest-catfish.json` — B300 SXM6 AC, driver 580.126.09

### Artifact integrity (E3)
- `MANIFEST.sha256` — full sha256 census of /workspace/v2.5-prod/ +
  laptop findings/2026-05-05-v2.5-eval/. Adapter sha256 verified to
  match the CARD.md iter-429 row: `94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c`.

### Data-leakage (E4) — KEY FINDING: 0% overlap on all 4 benchmarks
- `DATA-LEAKAGE-REPORT.md` — full report with caveats
- `E4-leak_check.py` — main script (3 HF benchmarks)
- `E4-leak_hb.py` — HealthBench-Hard add-on
- `E4-leak_summary.json` — raw output

### Memorization probe DESIGN (E5, deferred to T+5h)
- `MEMORIZATION-PROBE-PLAN.md` — verbatim-recall test plan

### Statistical correction protocol (E6)
- `STATS-PROTOCOL.md` — Holm-Bonferroni k=4, paired bootstrap, Cohen's d, post-hoc power, TOST for B3

### Determinism audit (E7)
- `DETERMINISM-AUDIT.md` — sovereign_bench.py decode params + 4 documented caveats

### HF Hub publication prep (E8, deferred to T+18h after ship-rule pass)
- `HF-MODEL-CARD-DRAFT.md` — ready-to-publish model card + upload command

### Audit log (E9)
- `AUDIT-LOG.md` — append-only chronological event log

### HF token audit (E10)
- `HF-TOKEN-AUDIT.md` — HF_TOKEN presence verified (length 37, value never read)

## Status (2026-05-06 21:1X UTC)

| Task | Status |
|---|---|
| E1 PREREGs | DONE — 7 files committed before any matching experiment |
| E2 pod manifests | DONE — 3 pods, sanitized (0 secret-shaped lines redacted) |
| E3 sha256 census | DONE — adapter verified |
| E4 data leakage | DONE — 0% on all 4 benchmarks |
| E5 memorization plan | DONE (design); EXECUTION DEFERRED to T+5h |
| E6 stats protocol | DONE |
| E7 determinism audit | DONE — 4 caveats documented |
| E8 HF model card draft | DONE; PUBLICATION DEFERRED to T+18h |
| E9 audit log | DONE |
| E10 HF_TOKEN audit | DONE |
