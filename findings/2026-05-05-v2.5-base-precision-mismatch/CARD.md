# V2.5 base-precision mismatch — FP8 download incompatible with PEFT trainer

**Date:** 2026-05-05 iter-46, ~13:15 PT
**Severity:** HIGH — would silently waste a V2.5 training fire if not caught
**Trigger:** Pre-flight check before declaring "V2.5 ready to fire" surfaces a mismatch between the FP8 base downloaded in iter-45 and the bf16-mixed PEFT trainer.

## The mismatch

| Side | Calls for | Reality |
|---|---|---|
| `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml` | `base_model.precision_for_training: bf16` | matches existing trainer |
| `prism42-nemotron-med/scripts/train_peft_imaging.py:455-465` | `_check_bf16_only` — "RULES §3: bf16-mixed only. Refuse fp8/NVFP4." | hard-codes BF16-only base load |
| Lobster cache (iter-45 download) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` (35.2 GB on disk) | **FP8 — incompatible with bf16-mixed PEFT** |

The iter-45 download chose FP8 per `findings/2026-05-05-lobster-disk-forensics/SPEC.md` Agent 3's recommendation: "FP8 is Hopper-native; LoRA training on a FROZEN FP8 base is well-supported". That recommendation was **research-paper-correct in general** but **NOT correct for THIS trainer**, which explicitly refuses non-BF16 base in `_check_bf16_only`.

## Implications

If user fires V2.5 with the existing `train_peft_imaging.py`-style trainer pointed at the cached FP8 base, the script aborts at the BF16 assertion. No GPU time wasted, but training never starts.

If user authors a NEW V2.5 reasoning-SFT script that calls `from_pretrained("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning")` (the suffixless ID per the PREREG line 39), HF auto-redirects to the BF16 variant — which **isn't cached** — triggering a fresh 67.75 GB download.

Either way, V2.5 fire is gated on getting the BF16 base on disk.

## Three options for resolution (user-action — babysitter cannot choose this)

### Option A — Replace FP8 with BF16 (recommended; cleanest)

```bash
ssh evil-cyan-lobster '
  # 1. Verify FP8 has nothing actively reading it
  sudo lsof +D ~/.cache/huggingface/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 2>/dev/null | head
  # 2. Drop FP8 (recover 33 GB)
  sudo rm -rf ~/.cache/huggingface/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
  sudo rm -rf ~/.cache/huggingface/hub/.locks/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
  # 3. Download BF16 (67.75 GB)
  /home/ubuntu/.venv/bin/hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
'
```

**Disk math:** 58 GB free + 33 GB recovered = 91 GB → minus 67.75 GB BF16 = **23 GB free post-download**. Tight but workable for V2.5 LoRA + intermediate optimizer states (~10-15 GB needed during training). Estimated 25-40 min total (rm instant; BF16 download ~25-40 min on Nebius bandwidth).

### Option B — Keep FP8 as serving-candidate; download BF16 alongside

35 GB FP8 + 67.75 GB BF16 = 102.75 GB needed; current 58 GB free is **insufficient**. Would require additional disk recovery (Stage A + Stage C from disk-forensics SPEC: ~5 GB stale tmp + 48 GB NeMo image prune = enough headroom). NeMo image prune costs a re-pull at V2.5 fire-time (~10 min). Net: 70-90 min total time.

Useful only if FP8 has a separate downstream use (e.g. catfish serving a Hopper-friendly fallback). Per CLAUDE.md §3, catfish is Blackwell B300 → uses NVFP4, not FP8. So FP8 has no obvious second use. **Option B is not recommended.**

### Option C — Modify the trainer to accept FP8 base + dequantize on load

Out of scope for babysitter (training-script edit; harmony contract violation). Possible in principle (transformers 4.50+ supports FP8 → BF16 dequant via bitsandbytes / fbgemm), but introduces an unmeasured precision-loss vs the published BF16 base. **Not recommended for a flagship release model.**

## Recommendation

**Option A.** Drop FP8, download BF16. Loss: 35 GB of disk-write + ~30 min download time. Gain: V2.5 fires without any further base-model surprises.

## Iter-46 RESOLUTION: Option A executed

User authorized Option A in iter-46 ("ok ultrathink"). Babysitter executed:

1. `sudo lsof +D ~/.cache/.../models--nvidia--...FP8` → empty (no readers)
2. `sudo rm -rf ~/.cache/.../models--nvidia--...FP8` + lockfile → 33 GB recovered (190→157 GB used; 58→91 GB free; 77%→64%)
3. `nohup hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (PID 2420073, log `/tmp/omni-bf16-download.log`) → 50 files, in progress

State at iter-46 ~13:25 PT: BF16 download active. Expected ~25-40 min wall-clock on Nebius bandwidth. Final disk: ~23 GB free post-download (157 GB used + 67.75 GB BF16 = 224.75 GB used; 247-225 = 22 GB free).

## Why this surfaced now and not before

- `findings/2026-05-05-lobster-disk-forensics/SPEC.md` Agent 3 said: "If your trainer stack cannot consume FP8 base weights directly (NeMo/Megatron-LM LoRA paths historically require BF16 base), fall back to BF16 at 67.75 GB." This caveat was in the doc but I weighted the disk-savings over the trainer-compatibility risk when writing the user-action sequence. The pre-flight check that would have caught it (read `train_peft_imaging.py:_check_bf16_only`) was what I'm doing right now — should have been done before recommending FP8.

## Durable lesson

For ANY base-model precision recommendation, pre-flight check the actual trainer's accepted dtypes before recommending a non-default precision. "Research papers say X is fine" ≠ "this codebase's trainer accepts X". Saved to `feedback_pretrain_dtype_preflight_trainer.md`.

## What this CARD does NOT do

- Does NOT drop the FP8 download. That requires explicit user OK.
- Does NOT download BF16 autonomously. Same.
- Does NOT modify `train_peft_imaging.py` (private repo + training-surface; harmony contract).
- Does NOT mark V2.5 as "ready to fire". V2.5 awaits BF16 base on disk first.
