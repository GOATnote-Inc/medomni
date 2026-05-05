# Catfish vllm flag-validation runbook (non-prod test-bed pattern)

**Date:** 2026-05-05
**Trigger:** iter-16 catfish serving-upgrade attempt failed and dropped `/4UWHAt` for ~10 min. Saved durable rule `feedback_stage_prod_flag_changes_one_at_a_time.md`. This runbook is the gate before re-attempting any catfish flag change.

---

## Why this runbook exists

The Team #2 research synthesis from the world-class trajectory SPEC (PR #59) recommends 6 vllm flags for catfish. Three were attempted in iter-16 and all three failed simultaneously — engine init error with no diagnostic on which flag broke it. Bumping multiple flags at once gave us a flat false-negative.

The fix: **test each flag individually on a non-prod endpoint, then test combinations, then apply to prod with rollback ready.**

---

## The non-prod endpoint

### Option A — second port on catfish (simplest, requires GPU memory headroom)

catfish has 288 GB B300 with vllm-omni-b300 + vllm-judge + vllm-rerank + vllm-embed currently using ~235 GB. ~53 GB free. NVFP4 30B-A3B is ~22 GB → fits. Run:

```bash
ssh unnecessary-peach-catfish '
docker run -d --name vllm-omni-test --gpus all --network host --ipc=host \
  -v /tmp/medomni:/tmp/medomni \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=16g \
  medomni/omni:v0.20.0-audio \
    --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
    --served-model-name nemotron-test \
    --host 0.0.0.0 --port 8001 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.15 \
    --kv-cache-dtype fp8 \
    --no-async-scheduling \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
'
```

Lower `gpu-memory-utilization 0.15` (catfish total 288 GB × 0.15 ≈ 43 GB allocation — fits in the 53 GB free budget). Smaller `max-model-len 16384` and `max-num-seqs 16` since this is a smoke test, not prod.

**Key:** the docker run command does NOT prepend `vllm serve` (image entrypoint already has it). See `feedback_check_docker_entrypoint_before_docker_run.md`.

Wait 8-10 min for model load, then smoke:

```bash
ssh unnecessary-peach-catfish 'curl -s http://127.0.0.1:8001/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])"'
```

Expected output: `nemotron-test`.

### Option B — lobster as test bed (if catfish memory tight)

lobster has 144 GB H200, currently ~67 GB used (Qwen judge). ~77 GB free. NVFP4-30B fits. Same docker-run pattern, replace `--gpus all` with `--gpus 0` if needed; replace base-model with the BF16 variant lobster has cached if NVFP4 isn't there.

---

## Per-flag validation matrix

For each candidate flag from Team #2, do **two smoke tests**:

1. **Cold start** — apply the flag, restart container, wait 8-10 min for engine ready, smoke `/v1/models` returns 200.
2. **Inference smoke** — issue a 1-token completion request and a tool-call request; both must return 200 with valid output.

Mark flag PASS only if BOTH smokes pass on the non-prod endpoint.

| Flag | Hypothesis (Team #2) | Test | Pass criteria |
|---|---|---|---|
| `--gpu-memory-utilization 0.90` | Bigger KV cache, more concurrency | apply alone (others at default), smoke | engine ready in <10 min, models 200, 1-token completion 200 |
| `--enable-prefix-caching` | System-prompt + tool-schema reuse on every agent turn | apply alone, smoke | engine ready, second identical request shorter latency than first |
| `--max-num-seqs 384` paired with `--max-num-batched-tokens 65536` | Higher concurrency | **must pair both flags** — 384 × 32K = 85 tokens/seq is too low for medical CoT. Bump to 65536 batched gives ~170/seq. | engine ready, 8 concurrent requests don't OOM |
| `--moe-backend flashinfer_cutlass` | NVFP4 grouped-GEMM kernel | requires FlashInfer-CUTLASS installed in container; verify with `docker exec vllm-omni-test python -c "from flashinfer.cutlass import gemm_fp4"` BEFORE adding flag | engine ready, no FlashInfer ImportError |
| `--speculative-config '{"method":"eagle3","num_speculative_tokens":4}'` | 1.4-1.9× decode speedup | requires Eagle-3 draft head pre-download; verify HF model id; needs `--speculative-model` arg | engine ready; smoke decoding gives ≥1.3× faster tok/sec on a 50-token prompt vs baseline |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` env var | NVFP4-attention kernel for B300's 2× attention silicon | env var only; no flag conflict | engine ready, vllm log shows `Using FlashInfer attention backend` |

---

## Combination matrix (only after individual validation)

Once each flag passes individually, test combinations:

| Combination | Reason |
|---|---|
| gpu-mem-util 0.90 + prefix-caching | Both basic, both expected to work together |
| gpu-mem-util 0.90 + max-num-seqs 384 + max-num-batched-tokens 65536 | High-concurrency triple — most likely to surface OOM |
| All 4 of: gpu-mem 0.90, prefix-cache, max-seqs 384, max-batched 65536 | What we'll actually run in prod |
| All 4 + flashinfer-cutlass | Add MoE backend on top |
| All 4 + flashinfer-cutlass + eagle3 | Add spec-decode |

Each combination = same two smokes (cold start + inference smoke).

---

## Apply-to-prod protocol (only after non-prod validation)

Once a combination passes non-prod validation:

1. **Capture rollback** — `docker inspect vllm-omni-b300 > /tmp/vllm-omni-b300.rollback.json` BEFORE stopping. Per memory `feedback_stage_prod_flag_changes_one_at_a_time.md`.

2. **Prepare rollback script** — write `/tmp/restart_with_original.sh` that re-runs the original docker run command. Test it (read the file, verify args).

3. **Stop prod container** — `docker stop vllm-omni-b300; docker rm vllm-omni-b300`. Public demo `/4UWHAt/api/agent` is now down.

4. **Apply new config** — `docker run` with validated combination (NO `vllm serve` prefix; image entrypoint already has it).

5. **Wait 8-10 min** for model load. Don't declare failure before that. NVFP4 30B + multimodal weights + KV-cache warmup is genuinely slow on first load.

6. **Smoke** — `curl http://127.0.0.1:8000/v1/models`; then `curl https://www.thegoatnote.com/4UWHAt/`; then a real `/api/agent` request via `useChat` shape.

7. **If any smoke fails** — execute `/tmp/restart_with_original.sh` immediately. Total downtime: ~5 min recovery + ~5 min original-config reload = ~10 min worst case.

8. **If all smokes pass** — clean up: `docker stop vllm-omni-test; docker rm vllm-omni-test` on the non-prod endpoint to free memory.

---

## What this runbook does NOT do

- Does not validate flags via in-prod A/B. The whole point is to NEVER touch prod without non-prod validation first.
- Does not change the catfish image. The image (`medomni/omni:v0.20.0-audio`) is the same; we're only changing flags.
- Does not address the lobster training-side blockers (disk, HF_TOKEN, Omni base). Those are user-action items independent of the catfish upgrade.
- Does not promise the catfish upgrade will land. If non-prod validation reveals incompatibilities (e.g. `gpu-memory-utilization 0.90` consistently OOMs with co-resident vllm services), the upgrade STAYS DEFERRED and we propose alternatives (e.g. `0.78` instead of `0.90`).

---

## Estimated time + cost

- Setup non-prod endpoint: ~15 min (with model load)
- Per-flag individual test: ~12 min × 6 flags = 72 min
- Combination tests: ~12 min × 5 = 60 min
- Apply-to-prod (when combinations pass): ~15 min worst case (rollback path)
- **Total wall time: ~3 hr** of cathorough validation before any prod blip
- Cost: only catfish GPU minutes (already running); no additional spend
- Risk: zero prod outage if non-prod validation works; ~10 min outage worst case if final apply fails

---

## Sources

- iter-16 incident: `LOOP-STATUS.md` § iter-16 entry
- Durable rule: `feedback_stage_prod_flag_changes_one_at_a_time.md`
- Durable rule: `feedback_check_docker_entrypoint_before_docker_run.md`
- World-class trajectory SPEC: `findings/2026-05-05-world-class-medomni-strategy/SPEC.md` § 4 (B300 serving config upgrade)
- Team #2 agent transcript: `tasks/a06197df0d195690f.output`
