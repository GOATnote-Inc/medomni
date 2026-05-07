# V2.5 Ship-Rule Eval — Status Board

Append-only durable state. Each phase milestone writes one entry.

## Pre-flight (2026-05-06)

- Branch: `feat/ship-rule-eval-driver-clean` (clean working tree)
- Adapter sha256 verified on lobster: `94d1f8d1eb23fad2e4d7ad6e5c5123963f1ef8795cad44aa2ca4b3221ed59b3c`
- Lobster GPU: 97 GB free / 144 GB total (45 GB used by judge-qwen)
- Lobster disk: 87 GB free
- Containers running on lobster: `judge-qwen` (:8001), `prism42-tts-kokoro` (:9001 — unrelated)
- :8002 free for V2.5 endpoint
- Ship-rule criteria (PREREG):
  - MedQA-USMLE: lower-CI > 0
  - MedXpertQA-Text: lower-CI ≥ +5pp
  - HealthBench-Hard: point-estimate > 0
  - PubMedQA-L: lower-CI ≥ -1pp (no regression)


## Phase A milestone (2026-05-06 ~14:55 PT)

**V0 endpoint up at lobster :8002** — coherent smoke-test response.

- vllm v0.20.0 architecture `NemotronH_Nano_VL_V2` does NOT support runtime LoRA loading. Pivoted to: serve V0 unmerged base on :8002 → run all V0 gen, then swap to merged-V2.5 endpoint and run V2.5 gen.
- LoRA merge completed on CPU in vllm container with `pip install peft==0.19.1 librosa soundfile`. Output: `/workspace/v2.5-merged/` (14 BF16 shards, ~62 GB). Disk now 91% full (25 GB free).
- V0 endpoint flags: `--gpu-memory-utilization 0.55 --max-model-len 4096 --trust-remote-code --dtype bfloat16`. Model name: `v0`. (Initial 0.45 + 8K ctx failed with no-KV-cache-memory; second-pass with 0.55 + 4K ctx succeeded.)
- Smoke gen 1 item × 4 benchmarks: ALL coherent. MedQA `\nA\n` (single-letter, system prompt enforced); HealthBench full 6.9KB clinical guidance walk-through.
- Throughput on V0: MedQA 0.5 s/item, HealthBench 1.7 s/item mean.
- Driver patches applied to scripts/ship_rule_lib/generators.py:
  - `load_healthbench_hard`: accept JSONL path or YAML pin (HF blob symlinks have no `.jsonl` suffix → content sniff `{`)
  - `load_pubmedqa_l`: filter to `pqa_labeled` split only (was loading 273k unlabeled+artificial rows)

**Deviation from PREREG:** Subsetting each benchmark to first 200 items (stratified-by-loader-order is deterministic). Required to fit 6-hr time budget for 24 runs total. Item-set sha256 + manifest will record the subset.

## Phase B/C interleaved milestone (2026-05-06 ~15:36 PT)

**V0 gen complete: 12/12 runs × 200 items = 2400 records.** Walltime 16:32 (started 22:00:30 UTC, done 22:17:02).

**V0 grading complete:**
- MedQA: 69.5% / 70.0% / 69.0% (seed 42/123/7919)
- MedXpertQA-Text: 30.0% / 29.0% / 30.0%
- PubMedQA-L: 54.0% / 54.5% / 54.0%
- HealthBench-Hard: 1.91% / 0.69% / 0.02% (mean ~0.87%)

**Critical fix applied: V2.5 LoRA merge produced wrong key names.** PEFT's `merge_and_unload` save_pretrained renamed `language_model.*` → `language_backbone.*` and `vision_model.*` → `vision_backbone.*`. vLLM's NanoNemotronVL_V2 expects the original `language_model.*` and silently drops weights with unknown prefix → V2.5 endpoint generated empty content.

Fix: `scripts/rename_safetensors_keys.py` rewrote all 14 shards in place (re-encoding each, atomic via .tmp, ~9 min). Verified V2.5 endpoint coherent post-rename: "The most common cause of acute MI is coronary artery disease."

**Parallel grader speedup:** original `grade_jsonl` issued sequential rubric calls (~18 s/item with 10 rubrics). New `scripts/grade_hb_parallel.py` uses ThreadPoolExecutor(max_workers=16) → ~6 s/item. 3 streams × parallel = ~10 min per arm. Original would've been 50 min serial.

**V2.5 gen started 22:36 UTC.** 12 more runs × 200 items.

## Phase D milestone (2026-05-06 ~16:05 PT)

**All 12 V2.5 gen runs complete (200 items × 2 arms × 3 seeds × 4 benchmarks = 4800 records).**
**All 24 graded files written.** Total grading cost: ~$2-3 USD (gpt-4.1, ~24K rubric calls).

**Ship rule: 1/4 PASS, 3/4 FAIL → OVERALL FAIL**

| Benchmark | V0 | V2.5 | Δ | 95% CI | Rule | PASS? |
|---|---:|---:|---:|:---:|---|:---:|
| MedQA-USMLE | 69.50% | 68.33% | -1.17pp | [-2.83, +0.50] | delta_lower_ci > 0 | FAIL |
| PubMedQA-L | 54.17% | 53.50% | -0.67pp | [-2.67, +1.33] | delta_lower_ci ≥ -1pp | FAIL |
| MedXpertQA-Text | 29.17% | 27.83% | -1.33pp | [-3.17, +0.50] | delta_lower_ci ≥ +5pp | FAIL |
| HealthBench-Hard | 0.87% | 2.36% | +1.48pp | [-0.66, +3.62] | delta > 0 | **PASS** |

**Headline:** V2.5 reasoning-SFT improves on HealthBench-Hard rubric grading (point estimate +1.48pp) but regresses on all three MCQ benchmarks (-0.7pp to -1.3pp). PREREG hypothesis (≥+5pp on MedQA, ≥+10pp on MedXpertQA) NOT met. Per PREREG `ship_rule.on_fail`: revert and debug data quality (likely insufficient CoT diversity in MedReason+medical-o1+R1-distill mix).

**Driver patch applied during run:** `cmd_stats` was pairing across seeds via item_id-only key (collision → only 200 paired observations). Fixed to pair within-seed and concat → 600 paired observations per benchmark. Final stats use the corrected pairing.

**Leakage audit:** 0 hits vs `/workspace/data/v1_train.jsonl` (10,178 rows; V1 multi-task SFT corpus). NOTE: this is V1's training corpus, not V2.5's. V2.5 trained on MedReason + medical-o1-reasoning-SFT + R1-distill-USMLE per PREREG; those source corpora not pulled to laptop for leakage scan in this run. Conservative interpretation: no leakage *within reach of the V1 corpus* — V2.5-corpus-specific leakage check is a follow-up.

**Outputs landed:**
- `findings/2026-05-05-v2.5-eval/SHIP-RULE-RESULTS.md` (PASS/FAIL per criterion)
- `findings/2026-05-05-v2.5-eval/SHIP-RULE-RESULTS.json` (paired-bootstrap CIs + Holm)
- `findings/2026-05-05-v2.5-eval/MANIFEST.sha256` (86 files)
- `findings/2026-05-05-v2.5-eval/LEAKAGE-AUDIT.md` (0/800 test prompts)
- `findings/2026-05-05-v2.5-eval/REPRO.sh` (deterministic re-run script)
- `findings/2026-05-05-v2.5-eval/stats.json` (full bootstrap output)


## Re-fire Phase A (2026-05-06 — thinking=True)

User-authorized methodology re-fire. Prior FAIL ran with `enable_thinking=False`
on a reasoning-SFT model whose entire training optimized the thinking channel,
making the verdict deniable. This re-fire turns thinking back on and treats
the side-by-side as the A5 ablation.

**Patch landed (PR #122 head 6c14c26):**
- `scripts/ship_rule_lib/generators.py`: DECODE_PARAMS default flipped to
  `enable_thinking=True, max_tokens=8192`. New helper `make_decode_params`.
- `scripts/ship_rule_eval.py`: `gen` subcommand learns `--enable-thinking /
  --no-enable-thinking` and `--max-new-tokens` (BooleanOptionalAction).
- CARD.md updated with re-fire pointer + sibling-dir note.

**Verifications:**
- `python3 -c "import ast; ..."` parsed both files clean.
- `ship_rule_eval.py smoke` → ALL CHECKS PASSED.
- `ship_rule_eval.py gen --help` shows the two new flags with correct defaults.
- `gh pr view 122` → PR Ready, head=6c14c26, CI re-queued (lint/unit/secrets-scan/
  manifest-determinism/safety-engineer-review).

New artifacts will land in `findings/2026-05-05-v2.5-eval-thinking/` (sibling
dir, no overwrite of the thinking=False prior run). Combined two-arm A5 table
will be appended to `SHIP-RULE-RESULTS.md` once Phase E completes.


## Re-fire Phase B (2026-05-06 ~16:25 PT — endpoints + smoke)

**Strategy pivot:** sequential V0 then V2.5 (single-endpoint at a time)
instead of parallel V0:8003 + V2.5:8002. Single H200 has 143 GB; two
30B-BF16 models @ 0.85 each = oversubscribed. Sequential gives full
0.85 budget per arm, so KV cache is comfortable for 16K ctx + 8K thinking.

**Stopped:** `judge-qwen` (cmd captured to lobster:/tmp/judge-launch-cmd.json),
prior `v25-serve` (used --max-model-len=4096 — insufficient for thinking).

**V0-serve up at lobster :8003:**
- Image: `vllm/vllm-openai:v0.20.0`, ENTRYPOINT `["vllm","serve"]`
- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- Flags: `--max-model-len 16384 --gpu-memory-utilization 0.85
  --max-num-seqs 8 --dtype bfloat16 --trust-remote-code --served-model-name v0`
- Ready in 115s.

**V0 thinking=True smoke (3 prompts, 8192 max_tokens, T=0):**
- Q1 (acute MI cause): 137 completion tokens, coherent atherosclerosis answer.
- Q2 (ACE-i vs labetalol pregnancy): 85 completion tokens, correct ACE-i,
  contains explicit `</think>` tag — confirms model uses Qwen-style thinking
  delimiter inline in `content` (NOT `reasoning_content` separately).
- Q3 (meningitis ddx): 3201 completion tokens, full systematic ddx.
- All three coherent. No truncation. Multi-step reasoning visible.

**Grader patch (companion to driver):** added `strip_thinking()` to
`ship_rule_lib/grader.py` — slices everything up to and including the LAST
`</think>` tag before MCQ regex / yn regex / gpt-4.1 rubric grading. Raw
response preserved verbatim in gen JSONL. Verified against 3 unit cases
(post-tag, no-tag, multi-tag). Will commit alongside Phase E results.

**Driver dirs for re-fire:**
- Pod: `/workspace/v2.5-eval-thinking/gen/{bench}/{arm}_{seed}.jsonl`
- Laptop pull: `findings/2026-05-05-v2.5-eval-thinking/gen/`
- Output suffix not needed; new sibling eval-dir is the carrier.

Next: fire V0 gen (12 runs) on lobster.


## Re-fire Phase D + E milestone (2026-05-06 ~21:55 PT — thinking=True FINAL)

**Run 2 (thinking=True) ship-rule verdict: 0/4 PASS → OVERALL FAIL.**

Worse than Run 1 (thinking=False, 1/4). Thinking re-fire eliminated the only PASS
(HB-hard delta flipped from +1.48pp → -1.31pp).

**Per-benchmark, paired-bootstrap (n=600 paired = 200 items × 3 seeds):**

| Benchmark | V0 | V2.5 | Δ | 95% CI | Rule | PASS? |
|---|---:|---:|---:|:---:|---|:---:|
| MedQA-USMLE | 83.50% | 84.67% | +1.17pp | [-1.00, +3.50] | delta_lower_ci > 0 | FAIL |
| PubMedQA-L | 67.33% | 64.83% | -2.50pp | [-5.00, -0.17] | delta_lower_ci ≥ -1pp | FAIL |
| MedXpertQA-Text | 33.00% | 31.67% | -1.33pp | [-4.83, +2.17] | delta_lower_ci ≥ +5pp | FAIL |
| HealthBench-Hard | 12.52% | 11.21% | -1.31pp | [-3.71, +1.13] | delta > 0 | FAIL |

**A5 ablation (thinking=False vs thinking=True), point-estimate Δ V2.5 − V0:**

| Benchmark | thinking=False (Run 1) | thinking=True (Run 2) |
|---|---:|---:|
| MedQA-USMLE | -1.17pp | +1.17pp |
| PubMedQA-L | -0.67pp | -2.50pp |
| MedXpertQA-Text | -1.33pp | -1.33pp |
| HealthBench-Hard | +1.48pp | -1.31pp |

Thinking helps MedQA modestly (+2.3pp swing), regresses PubMedQA further
(-1.8pp), no effect on MedXpert, and **flips HB-hard from PASS → FAIL** (-2.8pp swing).

**Wall-clock and cost:**
- Phase D rsync + dedupe: 5 min (HB v0 files dedup'd from 400→200 by item_id)
- Phase D'-prime (gpt-4.1 preflight): pass, key length 164
- Local-grader (medqa+pubmedqa+medxpertqa-text, 18 files): ~1 min
- HB-hard gpt-4.1 grade (6 files in parallel): ~83 min wall-clock (20:47-22:10 PT effective)
- gpt-4.1 calls: ~14,400 rubric grades (1200 items × ~12 rubrics avg)
- Cost estimate: $20-30 USD (not metered explicitly by driver)

**Issues encountered:**
- The user's instructions specified `--in/--out` per-file invocation; the driver
  takes `--eval-dir` for stats/leakage/manifest/report. Adjusted accordingly.
- Driver lacks --resume; all 6 HB-hard graders were single-shot. None crashed.
- 5/6 HB graders finished within 80 min; the 6th (v0_seed7919) finished 3 min later.

**Leakage:** 0 hits across 800 test prompts vs `v1_train.jsonl` (10,178 rows).
Same caveat as Run 1: V2.5-corpus-specific check is a follow-up.

**Outputs landed under `findings/2026-05-05-v2.5-eval-thinking/`:**
- `SHIP-RULE-RESULTS.md` (FAIL)
- `SHIP-RULE-RESULTS.json` (paired-bootstrap CIs + Holm)
- `MANIFEST.sha256` (84 files)
- `LEAKAGE-AUDIT.md` (0/800)
- `REPRO.sh`
- `stats.json`

**Verdict:** thinking=True does NOT rescue V2.5. Per PREREG `ship_rule.on_fail`,
this branch terminates. Decision-tree options on issue #130:
- V2.5b PREREG: adjust corpora (more diverse CoT, better task balance)
- Skip to V2.7 tool-call SFT (pivot rather than iterate)
- Investigate dataset issues with V2.5 reasoning-SFT data quality

PR #122 stays Ready (NOT merged) per task instructions.

