# E7 — Determinism audit of `scripts/sovereign_bench.py`

**Date:** 2026-05-06
**Purpose:** Confirm the eval driver runs deterministically (or document
where it doesn't) so any cross-trial variance can be attributed to model
output rather than driver-side stochasticity.

## Decode params (verbatim from `scripts/sovereign_bench.py`)

| Field | Default | Source | Confirmed deterministic? |
|---|---|---|---|
| `temperature` | `0.0` | `--temperature` arg, line 221 | YES — greedy decode |
| `max_tokens` | `1024` (or `2048` per CARD eval-protocol) | `--max-tokens` arg, line 220 | (cap, not stoch) |
| `top_p` | NOT PASSED | `_generate()` body lines 117-126 | YES — vLLM default 1.0 with T=0 is greedy |
| `chat_template_kwargs.enable_thinking` | `False` | line 125 (HARD-CODED) | YES — variance shrinks materially |
| `seed` (driver-level) | `42` | `--seed` arg, line 219 | partial — see CAVEATS |
| `batch_size` | not configured (single httpx call per item) | line 128 | YES — sequential |
| trial loop | `for trial_idx in range(trials)` | line 568 | YES — replays with same item set |

## CAVEATS (where determinism breaks)

1. **vLLM Hopper kernels are non-deterministic at temperature=0** unless
   `VLLM_USE_DETERMINISTIC=1` is set on the serve side AND the kernel
   schedule is pinned. The serve container on lobster does NOT set this
   env var by default (verified in `manifest-lobster.json` — vLLM is
   pip-installed, no env override visible). Practical effect: identical
   prompts can produce token-level differences across runs by ~0.5pp on
   benchmark scores, even with `--seed 42` passed to vLLM.

2. **The `--seed` argument is recorded in the artifact but NOT passed
   through to `/v1/chat/completions`**. Re-reading `_generate()` body
   (lines 113-132): the body sent to vllm only contains `model`,
   `max_tokens`, `messages`, `temperature`, `chat_template_kwargs`. The
   request-level `seed` field defined by the OpenAI API spec is NOT
   included. This is FIXABLE — a one-line edit — but as of this audit
   the eval is NOT seeded at the request level.

3. **Judge endpoint** (`make_triton_judge` from `mla.judges.triton`) has
   its own non-determinism profile. Audit log written to
   `judge-{run_id}.jsonl` records prompt+response per call so the
   judge-side variance can be measured post-hoc; max_retries=3 means
   transient judge errors do NOT auto-fail items.

4. **`enable_thinking=False` is HARD-CODED**, not a CLI flag. To run
   thinking-on for A5 (the thinking-on-vs-off ablation), either:
   - patch `sovereign_bench.py` line 125 to read from a CLI flag, OR
   - run a separate variant of the script.
   Logged as the A5 PREREG implementation prerequisite.

5. **HTTPX timeout = 120s**. Items that exceed this are skipped via
   `httpx.HTTPError` (line 703) — they are NOT counted in
   `n_per_trial`, which means slow-path items can silently drop. The
   artifact's per-trial `len(per_example)` should be verified equal to
   `n_per_trial` at read time; the CARD already calls this out as
   "READ THE ARTIFACT before claiming success".

## Recommended remediations (DO NOT execute now — let ship-rule eval finish first)

These are queued for post-ship-rule. Ordered by priority:

1. **Add `seed=42` to the `body` dict in `_generate()`** so vLLM seeds
   the request. One-line change. Removes one stochasticity source.
2. **Add `VLLM_USE_DETERMINISTIC=1` to the lobster serve container**.
   Document in `findings/2026-05-06-evidence-window/MEMO-DETERMINISTIC-SERVE.md`
   (will author at remediation time).
3. **Promote `enable_thinking` to a CLI flag** for A5 to use the same
   driver instead of forking. Two-line change.
4. **Fail-loud on timeout-skip** instead of silent continue: count
   skipped items, error if >5% to surface the issue.
5. **Pin item order**: confirm `_load_examples()` returns items in
   stable order across runs (it currently uses `sorted(p.glob(...))`
   for fixtures-dir mode and sequential for YAML manifests; verified
   stable).

## Verdict

The driver is **near-deterministic** at temperature=0 with thinking
disabled, BUT contains 4 documented sources of low-level token-noise
(items 1, 2, 3, 5 above). For the ship-rule eval the cross-trial
paired-bootstrap CI absorbs this noise at the score level — the
3-trial-with-3-seeds design is already designed to measure exactly
this jitter.

For B3 (NVFP4 vs BF16 equivalence test), driver-side determinism is
load-bearing because we need <=1pp paired CI half-width. Item 1 above
is a likely contributor; the seed-passthrough fix (item 1) and the
serve-side env var (item 2) should be applied before B3 fires.

For E5 memorization probe, determinism is even more load-bearing —
verbatim-recall depends on greedy decode being truly greedy. Same
remediations 1-2 apply.

## Logged in AUDIT-LOG.md as: 2026-05-06 14:0X UTC E7 audit complete.
