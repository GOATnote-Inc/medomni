# Lobster disk forensics + V2.5 base-download plan

**Date:** 2026-05-05 iter-36, ~11:00 PT
**Trigger:** HF_TOKEN cleared (iter-35); V2.5 reasoning-SFT now disk-gated. Manual-prune cycle has run 2x already; user asked for fresh research before the next attempt.
**Method:** 4-agent parallel research synthesis (read-only ssh + WebFetch). All ssh probes were READ-ONLY; no writes, no `docker rm`, no `rm`.

## TL;DR

LOOP-STATUS's escalation-block numbers were wrong. Real picture:

- **Disk-free target was over-stated.** V2.5 wants the Omni base. The **FP8 variant is 35.2 GB**, not 60 GB (BF16 is 67.75 GB; NVFP4 is Blackwell-only). FP8 is Hopper-native and fully sufficient for frozen-base LoRA training.
- **The biggest reclaimable consumer is duplicate text-Nemotron caches**: 59 GB (`/ephemeral/cache/huggingface/hub`) + 15 GB (`/workspace/.hf-cache/hub`) of the same `models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, which is the V0/V1-era TEXT-only base. V2.5+ uses the OMNI variant. The text base is removable; re-pullable on demand.
- **The previously-claimed 15 GB `/home/ubuntu/medomni` and 4.9 GB `/home/ubuntu/peft-text-v1`** are nearly empty (~4 KB each). LOOP-STATUS has been hallucinating phantom bytes since iter-19. Those targets free zero.
- **Brev volume attach does not exist.** No `brev volume`/`brev resize` subcommand. /ephemeral, vdb, /workspace are all on the same `/dev/vda1`. Structural fix = recreate pod with `--min-disk 500` (60-90 min downtime, requires user decision).
- **NeMo docker image (48 GB) is the only safely-prunable docker artifact**, but rmi-then-repull at low disk-free is fragile. Best left until V2.5 actively fires.

## Verified disk inventory (live ssh probe, 2026-05-05 ~11:00 PT)

```
/dev/vda1: 247 GB total, 230 GB used (94%), 18 GB free

Top consumers (all on /dev/vda1):
  84 GB   /var/lib/docker            (3 images: NeMo 48 + vllm-openai 22.9 + Kokoro 13.6)
  79 GB   /ephemeral/cache           (75 GB HF cache + 4.3 GB pip cache)
  └── 59 GB  models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  ← TEXT-only, V0/V1-era
  └── 15 GB  models--Qwen--Qwen2.5-7B-Instruct                    ← LIVE (judge-qwen)
  └── 4.3 GB  pip cache
  16 GB   /workspace
  └── 15 GB  .hf-cache/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  ← DUPLICATE
  ~15 GB  /usr (OS, do not touch)
  4.6 GB  /opt (NVIDIA drivers, do not touch)
  1.9 GB  /tmp (1.35 GB stale: pip-install-* + ansible_env)
  1.6 GB  /var/log
  ~1 GB   /var/cache
  ~4 KB   /home/ubuntu/medomni      ← phantom (LOOP-STATUS claimed 15 GB)
  ~4 KB   /home/ubuntu/peft-text-v1 ← phantom (LOOP-STATUS claimed 4.9 GB)
```

**Active consumers (DO NOT TOUCH):**
- `judge-qwen` container (vllm-openai:v0.20.0, 2-day uptime) — bind-mounts `/home/ubuntu/.cache/huggingface` (= /ephemeral/cache/huggingface). Reads `models--Qwen--Qwen2.5-7B-Instruct`.
- `prism42-tts-kokoro` container (kokoro-fastapi-gpu, 16h uptime) — no bind mounts.

## V2.5 base-download requirement (corrected)

| Variant | On-disk | Format | H200? | Notes |
|---|---|---|---|---|
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | **67.75 GB** (17 shards) | bf16 | Yes | Conservative; full precision base. |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` | **35.19 GB** (4 shards) | E4M3 fp8 + bf16 router/lm_head | **Yes (Hopper-native)** | **RECOMMENDED.** Frozen-base LoRA on FP8 is well-supported via TRT-MO / llm-compressor; reasoning capability preserved (router + lm_head kept high-precision). Vision (CRADIO-v4-H) + audio (Parakeet) embedded in shards — no separate downloads. |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | 22.43 GB | NVFP4 | **No** | Blackwell-only. Will not load on H200. |

**Disk-free gate:**
- FP8 path: ≥ **50 GB free** (35.2 GB shards + 5 GB hardlink overhead + 10 GB activation/optimizer headroom)
- BF16 path: ≥ **90 GB free** (67.8 + 5 + 17)

## Recommended action sequence (ordered safest-first)

The babysitter cannot execute any of these; all are user-action. After each, re-probe `df -h /` to confirm the gain.

### Stage A — Zero-risk cleanup (recovers ~5.65 GB)

```bash
ssh evil-cyan-lobster 'rm -rf /tmp/pip-install-* /tmp/ansible_env'
ssh evil-cyan-lobster 'rm -rf /ephemeral/cache/pip'   # pip download cache; safe to re-fetch
ssh evil-cyan-lobster 'df -h /'
```

Expected free: 18 → ~24 GB.

### Stage B — Remove duplicate text-Nemotron caches (recovers 74 GB)

These are the V0/V1-era text base. V2.5+ training uses the OMNI variant (different model). If V0→V1 paired-eval is ever needed, the text base can be re-pulled in ~25 min (49 GB shards over typical Nebius bandwidth).

```bash
# verify nothing currently has the text Nemotron open
ssh evil-cyan-lobster 'sudo lsof +D /ephemeral/cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 2>/dev/null | head; echo done'
ssh evil-cyan-lobster 'sudo lsof +D /workspace/.hf-cache/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 2>/dev/null | head; echo done'

# if both come back empty, prune both copies:
ssh evil-cyan-lobster 'rm -rf /ephemeral/cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
ssh evil-cyan-lobster 'rm -rf /workspace/.hf-cache/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
ssh evil-cyan-lobster 'rm -rf /ephemeral/cache/huggingface/hub/.locks/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
ssh evil-cyan-lobster 'df -h /'
```

Expected free after Stage A+B: ~98 GB. **Sufficient for FP8 download + headroom for Stage C if needed.**

### Stage C — Optional: docker NeMo image prune (recovers 48 GB)

Defer until V2.5 is about to fire. Current judge + Kokoro containers don't use this image. Re-pull during V2.5 prep adds ~10 min.

```bash
# only when V2.5 isn't imminent:
ssh evil-cyan-lobster 'docker image rm nvcr.io/nvidia/nemo:26.04.00'
```

### Stage D — Download Omni FP8

```bash
ssh evil-cyan-lobster 'huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 --local-dir-use-symlinks=False'
# ~35 GB; should complete in ~20-30 min on Nebius bandwidth
ssh evil-cyan-lobster 'du -sh ~/.cache/huggingface/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8'
```

After Stage D: V2.5 PREREG can fire per its existing runbook.

## Structural fix (out of scope for this iter)

**Pod recreate with `--min-disk 500`** is the only sustainable path beyond V2.5. After V2.5+V2.7+V3+V3.5 land, total disk demand is ~120 GB beyond current state, exceeding the most aggressive prune. User decision required. See research-agent-4 report: `brev volume` does not exist; structural expansion = full recreate.

When the user is ready: the rebuild blast radius is bounded (factory_loop already designed for off-pod state; HF cache + Docker images are re-pullable).

## What this SPEC does NOT do

- Does not execute any destructive command. All deletions are user-action.
- Does not preserve the text-Nemotron-BF16 cache. If V0/V1 paired-eval is still desired, a backup-to-narwhal step would precede Stage B (~50 GB rsync, ~10 min).
- Does not address the V_final HF release path; that's covered in `findings/2026-05-05-v-final-hf-release/RUNBOOK.md`.
- Does not change the harmony contract — babysitter cannot fire training, prune, or pod recreates autonomously.

## Cross-references

- `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml` — V2.5 fires once Stage D completes
- `findings/2026-05-05-corpora-license-confirmation/CARD.md` — corpora license cleared for V2.5
- `LOOP-STATUS.md` ESCALATION block — being updated this iter to use corrected numbers
