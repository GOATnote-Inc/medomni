# V2.5 evidence-window AUDIT LOG (append-only)

**Format:** `<UTC timestamp> | <event> | <commit-sha-or-"-"> | <agent>`

This is an append-only chronological log of every load-bearing event in
the V2.5 evidence-window timeline. Entries are added forward-in-time only;
historical entries are NEVER edited — corrections take a NEW entry below.

---

## Pre-history (before E-track agent fired)

| UTC | Event | Commit | Agent |
|---|---|---|---|
| 2026-05-05 14:05 | V2.5 PREREG.yaml authored at findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml | (see git blame) | iter-15 babysitter |
| 2026-05-05 21:08 | V2.5 SMOKE training PASS (50 step, train 1.357 / eval 1.511) | - | iter-47 babysitter |
| 2026-05-05 21:09 | V2.5 PROD training fired (1st attempt, seq=2048) | - | iter-47 babysitter |
| 2026-05-05 21:29 | V2.5 PROD training OOM at step 45 (seq=2048 + judge-qwen collision); saved durable lesson `feedback_lobster_oom_judge_collision.md` | - | iter-52 babysitter |
| 2026-05-05 21:36 | V2.5 PROD training RE-fired (seq=1536 + expandable_segments) | - | iter-52 babysitter |
| 2026-05-05 21:46 | V2.5 PROD training stable at step 23 (24.4s/step) | - | iter-54 babysitter |
| 2026-05-06 01:05 | V2.5 step 500 first held-out eval: train 1.038 / eval 1.046 (-30.8% vs smoke) | - | iter-108 babysitter |
| 2026-05-06 04:33 | V2.5 step 1000 eval: 1.029 (checkpoint-1000 saved) | - | mid-loop |
| 2026-05-06 08:03 | V2.5 step 1500 eval: 1.020 | - | mid-loop |
| 2026-05-06 11:35 | V2.5 step 2000 eval: 1.016 (checkpoint-2000 saved) | - | iter-276 babysitter |
| 2026-05-06 15:13 | V2.5 step 2500 eval: 1.013 | - | iter-338 babysitter |
| 2026-05-06 18:56 | V2.5 step 3000 eval: 1.012 (checkpoint-3000 saved) | - | iter-400 babysitter |
| 2026-05-06 20:46 | V2.5 PROD training COMPLETE (step 3243 / 23h 10m walltime; train 0.992 / eval 1.012) — adapter sha256 94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c | - | iter-429 babysitter |

## Parallel agents fired immediately after V2.5 training completion

| UTC | Event | Commit | Agent |
|---|---|---|---|
| 2026-05-06 ~20:50 | Ship-rule eval driver fired against V2.5 adapter (sovereign_bench.py against 4 benchmarks; estimated 2h walltime on lobster) | - | ship-rule-eval agent |
| 2026-05-06 ~20:55 | B300 baseline profile fired on catfish (no-LoRA throughput baseline for B2/B3 paired comparison; results in findings/2026-05-06-b300-baseline-profile/) | - | b300-profile agent |
| 2026-05-06 21:00 | E-track evidence-window agent fired (this agent — author the reproducibility infrastructure: E1-E10) | - | E-track agent (this) |

## E-track agent activity

| UTC | Event | Commit | Agent |
|---|---|---|---|
| 2026-05-06 21:00 | E-track agent start — created findings/2026-05-06-evidence-window/ | - | E-track |
| 2026-05-06 21:03 | E2 manifest probes ssh-fired on lobster + narwhal + catfish; outputs at /tmp/probe-{lobster,narwhal,catfish}.txt | - | E-track |
| 2026-05-06 21:04 | E1 PREREG yaml files authored: PREREG-A4.yaml, PREREG-A5.yaml, PREREG-A6.yaml, PREREG-B2.yaml, PREREG-B3.yaml, PREREG-B4.yaml, PREREG-B6.yaml | (this commit) | E-track |
| 2026-05-06 21:05 | E2 manifests serialized to JSON (manifest-lobster.json, manifest-narwhal.json, manifest-catfish.json); pip_freeze sanitized (zero KEY/TOKEN/SECRET/PASSWORD lines redacted across all 3 pods — none were present in pip_freeze) | (this commit) | E-track |
| 2026-05-06 21:06 | E3 sha256 census written to MANIFEST.sha256; adapter_model.safetensors VERIFIED at sha256 94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c (matches CARD.md iter-429 row) | (this commit) | E-track |
| 2026-05-06 21:07 | E10 HF_TOKEN audit complete (presence-only check, len=37, value never read) | (this commit) | E-track |
| 2026-05-06 21:08 | E7 determinism audit complete (4 caveats documented; eval driver near-deterministic at T=0 with thinking-off) | (this commit) | E-track |
| 2026-05-06 21:09 | E6 STATS-PROTOCOL.md authored (Holm-Bonferroni k=4, paired-bootstrap CI, Cohen's d, post-hoc power, TOST for B3 equivalence) | (this commit) | E-track |
| 2026-05-06 21:10 | E5 MEMORIZATION-PROBE-PLAN.md authored (DESIGN ONLY, deferred to T+5h post ship-rule) | (this commit) | E-track |
| 2026-05-06 21:10 | E8 HF-MODEL-CARD-DRAFT.md authored (do NOT publish until ship-rule + E4 + E5 verdicts cleared) | (this commit) | E-track |
| 2026-05-06 21:1X | E4 data-leakage check fired (datasketch MinHashLSH; results in DATA-LEAKAGE-REPORT.md) | (this commit) | E-track |
| 2026-05-06 21:1X | E9 audit log skeleton authored (this file) | (this commit) | E-track |
| 2026-05-06 21:1X | E-track committed to feature branch and PR opened to main | (this commit) | E-track |

## Forward-looking entries (NOT YET HAPPENED — schedule)

These rows are PLACEHOLDERS describing what SHOULD be appended next.
They are NOT facts; they are the playbook. Real entries replace them
when the events actually occur.

- T+~2h | ship-rule eval CARD.md filled in with 4-benchmark scores
- T+~3h | E5 memorization probe fired on V2.5 endpoint
- T+~3h | E5-RESULTS.md authored
- T+~5h | per-benchmark Holm-corrected p-values + Cohen's d computed
        per STATS-PROTOCOL.md; CARD.md updated
- T+~6h | ship-rule decision logged here as PASS or FAIL
- T+~18h | (ON PASS) HF Hub upload fired (HF-MODEL-CARD-DRAFT.md
         publication command); upload sha256 verified post-fact
- T+~24h | A4 extended evals fire (per PREREG-A4.yaml)
- T+~26h | A5 thinking-on-vs-off ablation fires (per PREREG-A5.yaml;
         requires sovereign_bench.py thinking flag patch first)

## Hard rules

- Append-only: never edit prior entries.
- One entry per event; multi-event commits get one entry per event.
- UTC only. Local-time entries are forbidden — they break sortability.
- Commit SHAs come from `git log` after the commit; for entries written
  before the commit, write `(this commit)` and update post-commit if
  needed (in a NEW entry, not in-place).
- The first three lines of any new entry must keep the table format —
  do not add blank lines inside table sections.
