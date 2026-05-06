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
