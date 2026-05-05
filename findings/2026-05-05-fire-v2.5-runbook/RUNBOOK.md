# RUNBOOK — Fire V2.5 reasoning-SFT on lobster

**Date:** 2026-05-05 iter-46
**Pre-condition (gating fire):** BF16 base download complete on lobster. Verify before each user-action step below.
**Estimate:** ~10-12 hr H200 wall-clock; ~$80 budget. Per V2.5 PREREG ship rule (`findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`).

## Status (iter-46, ~13:30 PT)

- ✅ HF_TOKEN installed at `~/.cache/huggingface/token` (iter-35)
- ✅ Lobster disk Stage B prune fired (iter-44 — 73 GB recovered)
- ⏳ BF16 base download in progress (PID 2423674, re-fired iter-46 after FP8 mismatch)
- ⏸ V2.5 corpora not yet pulled (~50 MB total, trivial)
- ⏸ V2.5 SFT script does NOT exist in private repo — only `train_peft_imaging.py` (V1 imaging path)

## Step 0 — Pre-flight (verify before fire)

```bash
# A. BF16 base ready
ssh evil-cyan-lobster '
  ls ~/.cache/huggingface/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blobs/*.incomplete 2>/dev/null | wc -l
  # expect: 0
  /home/ubuntu/.venv/bin/hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
  # expect: "Downloaded" + snapshot path
'

# B. Disk has training headroom (need ≥15 GB for LoRA + optimizer states)
ssh evil-cyan-lobster 'df -h /'
# expect: ≥15 GB available

# C. HF auth
ssh evil-cyan-lobster '/home/ubuntu/.venv/bin/hf auth whoami'
# expect: user: GOATnote, orgs: bgoatnote

# D. judge-qwen still running (don't disturb)
ssh evil-cyan-lobster 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "judge-qwen|kokoro"'
# expect: both Up
```

## Step 1 — Pull V2.5 training corpora (~50 MB)

```bash
ssh evil-cyan-lobster '
  /home/ubuntu/.venv/bin/hf download UCSC-VLAA/MedReason --repo-type dataset
  /home/ubuntu/.venv/bin/hf download FreedomIntelligence/medical-o1-reasoning-SFT --repo-type dataset
  # The 5K R1-distilled USMLE traces are SYNTHETIC — generated locally per V2.5 PREREG
  # If not yet generated, see PREREG §"R1-distill USMLE traces" for synthesis recipe
'
```

## Step 2 — Verify LoRA target_modules match actual architecture

V2.5 PREREG specifies LoRA targets `q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, vision_projector`. The Nemotron-3-Nano-Omni-30B uses `nemotron_h` model_type (Mamba-Transformer hybrid + MoE). Module names in HF's NemotronH implementation may differ. Verify before training:

```bash
ssh evil-cyan-lobster '/home/ubuntu/.venv/bin/python3 -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained(
    \"nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16\",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=\"meta\",  # don't actually load to GPU
)
import re
pat = re.compile(r\"\\b(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|vision_projector)\\b\")
matches = sorted(set(n for n,_ in m.named_modules() if pat.search(n)))
print(f\"Matched modules: {len(matches)}\")
for m_ in matches[:10]: print(\" \", m_)
"'
```

If 0 matches, the PREREG's target list needs updating to match HF's actual NemotronH implementation. Likely candidates if naming differs:
- attention: `*.self_attn.qkv_proj` (fused) or `*.attn.q_proj`
- mamba: `*.mixer.in_proj` / `*.mixer.out_proj`
- vision: `*.mlp1.0` / `*.vit_projector.*`

## Step 3 — Author V2.5 SFT training script (private repo)

The existing `prism42-nemotron-med/scripts/train_peft_imaging.py` is hard-coded for V1 imaging (PubMedVision corpus, vision-MLP target). V2.5 needs a sibling script. Recommended pattern:

```
prism42-nemotron-med/scripts/train_peft_reasoning.py  (NEW — author this)
```

Constraints inherited from `train_peft_imaging.py`:
- `_check_bf16_only` — REQUIRED (per RULES §3)
- TraceLogger pattern (preserves audit trail)
- LoraConfig per PREREG (rank=64, alpha=128, dropout=0.05)
- bf16-mixed precision
- Adapter output format compatible with `vllm --enable-lora` for catfish hot-load

Differences from imaging V1:
- Corpus: MedReason + medical-o1-reasoning-SFT + R1-distill (NOT PubMedVision)
- Mix ratio: per V2.5 PREREG `data_mix` table
- Sequence length: 8K min (reasoning chains can be long); PREREG suggests 4K with packing
- No vision_projector target (text-only training; if Omni's vision side gets in the way, freeze it explicitly)

Skeleton (paste into `train_peft_reasoning.py`):

```python
PREREG_BASE_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning"  # resolves to BF16
PREREG_TRAIN_CORPORA = [
    ("UCSC-VLAA/MedReason", "default"),                     # 32K traces
    ("FreedomIntelligence/medical-o1-reasoning-SFT", None), # 25K traces
    # 5K R1-distill USMLE — local synthetic, see PREREG
]
PREREG_LORA_RANK = 64
PREREG_LORA_ALPHA = 128
PREREG_LORA_DROPOUT = 0.05
PREREG_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,in_proj,out_proj"  # adjust per Step 2 result
PREREG_LR = 1.0e-4    # PREREG; lower than V1's 2e-4 due to longer sequences
PREREG_LR_SCHEDULE = "cosine"
PREREG_WARMUP_RATIO = 0.03
PREREG_EPOCHS = 1     # PREREG; reasoning corpus ~62K samples × 1 epoch
PREREG_BATCH_SIZE = 4 # smaller than V1 due to longer sequences
PREREG_GRAD_ACCUM = 8 # effective batch 32
PREREG_MAX_SEQ_LEN = 8192
```

Then keep the same TraceLogger pre-flight + post-train CARD-emission scaffolding from `train_peft_imaging.py`.

## Step 4 — Smoke (50-step proof) before full fire

Per `feedback_pilot_before_full_sweep` and `feedback_eval_preflight_judge_key`:

```bash
ssh evil-cyan-lobster '
  cd ~/prism42-nemotron-med &&
  /home/ubuntu/.venv-nemo/bin/python scripts/train_peft_reasoning.py \
    --base-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning \
    --bf16-mixed \
    --max-steps 50 \
    --output-dir /workspace/v2.5-smoke \
    2>&1 | tee /tmp/v2.5-smoke.log
'
# inspect: tail of log shows step-time + loss curve
# expect: step_time < 10s, loss decreasing
```

If smoke passes (artifact JSON content present, step time reasonable):

## Step 5 — Fire V2.5 production

```bash
ssh evil-cyan-lobster '
  cd ~/prism42-nemotron-med &&
  nohup /home/ubuntu/.venv-nemo/bin/python scripts/train_peft_reasoning.py \
    --base-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning \
    --bf16-mixed \
    --output-dir /workspace/v2.5-prod \
    > /tmp/v2.5-prod.log 2>&1 &
'
# 8-10 hr expected wall-clock
```

## Step 6 — Eval + ship rule (per V2.5 PREREG)

After training completes:

```bash
ssh evil-cyan-lobster '
  cd ~/prism42-nemotron-med &&
  /home/ubuntu/.venv/bin/python scripts/sovereign_bench.py \
    --base nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning \
    --lora /workspace/v2.5-prod/adapter_model.safetensors \
    --tasks MedQA-USMLE,MedXpertQA-Text \
    --paired-bootstrap 95
'
```

**Ship rule (PREREG):** MedQA-USMLE +5pp paired-CI; MedXpertQA-Text +10pp paired-CI vs V0. If both met, V2.5 promoted; V2.7 PREREG fires next.

If ship rule MISSED: file CARD with diagnosis (recipe, hyperparam, corpus mix) before re-run.

## Babysitter loop integration

Once V2.5 fires (Step 5), the loop's 5-stage pulse will see Train stage as alive (`pgrep -af train_peft|megatron` non-empty). The Train stage probe in LOOP-STATUS Charter will show non-empty. Eval stage will fire post-training automatically per Step 6.

## What this RUNBOOK does NOT do

- Does NOT fire training. All Step 1-6 commands are user-action.
- Does NOT author `train_peft_reasoning.py` — that's a private-repo addition (training surface; harmony contract).
- Does NOT bypass the V2.5 PREREG ship rule. If ship rule fails, V2.5 doesn't promote.
- Does NOT include the V3/V3.5/V_final augmentations from the iter-38 synthesis CARD. Those are separate PREREG amendments.

## Cross-references

- [`findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`](../2026-05-05-v2.5-reasoning-sft/PREREG.yaml) — pre-registration manifest
- [`findings/2026-05-05-v2.5-base-precision-mismatch/CARD.md`](../2026-05-05-v2.5-base-precision-mismatch/CARD.md) — why we're on BF16 not FP8
- [`findings/2026-05-05-corpora-license-confirmation/CARD.md`](../2026-05-05-corpora-license-confirmation/CARD.md) — V2.5 corpora are Apache-2.0
- [`findings/2026-05-05-improvement-dimensions-roadmap/CARD.md`](../2026-05-05-improvement-dimensions-roadmap/CARD.md) — V3+ augmentations (apply later)
