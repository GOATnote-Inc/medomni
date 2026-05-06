# Parity Anchors

The four anchors that must be identical between a blue pod and its green replacement, so a swap is an immutability guarantee instead of a re-deploy gamble.

This file is the single-source-of-truth read by `scripts/launch_b300_prod.sh` and `scripts/launch_h200_factory.sh`. Update it any time a deployed pod's anchor changes.

> **Note:** if any anchor below shows `<TODO_…>`, the swap window is **not** ready. Capture the value from a healthy pod first (commands shown inline).

---

## The 4 anchors

| # | Anchor | Where it lives | Why it matters |
|---|---|---|---|
| 1 | **Image SHA** (vLLM container digest) | this file (per-pod row) | A tag like `vllm/vllm-openai:v0.20.0` is mutable. Two pods using the same tag at different times can pull different bits. Pinning by digest makes that physically impossible. |
| 2 | **Weights** (HF model id + revision sha) | this file (per-pod row) | HF model repos are mutable. The published NVFP4 SKU could be re-uploaded; without the revision sha, a re-pull on green could silently differ. |
| 3 | **Launch flags** | `scripts/launch_b300_prod.sh` and `scripts/launch_h200_factory.sh` (in git) | The vLLM flag set encodes Blackwell workarounds (`--no-async-scheduling`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`), parsers (`reasoning-parser nemotron_v3`, `tool-call-parser qwen3_coder`), and resource limits. Drift between scripts and the running container = silent regression. |
| 4 | **Env vars** | `.env.example` (in git, names only — never values) and the Brev console / Vercel project (values) | Per medomni `CLAUDE.md` §2 only `HF_TOKEN`, `BREV_PEM_PATH`, and the `MEDOMNI_*` BFF endpoints exist. No cloud LLM keys. Values flow via console UI or `vercel env add`, never via shell history. |

---

## Pod-by-pod anchor table

### Catfish (B300, prod inference)

| Anchor | Value | Captured |
|---|---|---|
| Image | `vllm/vllm-openai:v0.20.0` | yes |
| Image digest | `<TODO_B300_VLLM_IMAGE_DIGEST>` | **needed before swap** |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | yes |
| Model revision | `<TODO_B300_MODEL_REVISION_SHA>` | **needed before swap** |
| Launch script | `scripts/launch_b300_prod.sh` | in git |
| Launch flags hash | `<TODO_B300_LAUNCH_FLAGS_SHA256>` (sha of script's docker-run block) | optional but recommended |
| Env var (web) | `MEDOMNI_TUNNEL_URL` (production) | in Vercel project |
| Pod env | `HF_TOKEN` provisioned via Brev console env-var UI | manual step |

**Capture commands (run on healthy catfish before any swap window):**

```bash
# 1. Image digest
ssh unnecessary-peach-catfish '
  docker inspect --format "{{index .RepoDigests 0}}" vllm/vllm-openai:v0.20.0
'
# Paste the sha256 above as B300_VLLM_IMAGE_DIGEST.

# 2. Model revision sha
ssh unnecessary-peach-catfish '
  ls ~/medomni/hf_cache/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/snapshots/
'
# The first hash directory IS the revision sha.

# 3. Launch flags hash (so a future drift check is one diff)
shasum -a 256 /Users/kiteboard/medomni/scripts/launch_b300_prod.sh | awk "{print \$1}"
```

### Narwhal (H200, factory + RAG)

| Anchor | Value | Captured |
|---|---|---|
| Image | `vllm/vllm-openai:v0.20.0` | yes |
| Image digest | `<TODO_H200_VLLM_IMAGE_DIGEST>` | **needed before swap** |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | yes |
| Model revision | `<TODO_H200_MODEL_REVISION_SHA>` | **needed before swap** |
| Launch script | `scripts/launch_h200_factory.sh` | in git |
| RAG sidecar source | `prism42-nemotron-med` repo at `/home/ubuntu/medomni-imaging-rag/` | external |
| Data-queue snapshot | `scripts/snapshot_h200_factory_state.sh` → `<TODO_OBJECT_STORE>/data-queue-LATEST.tar.gz` | **bucket name needed** |
| Pod env | `HF_TOKEN` via Brev console; AWS creds for snapshot | manual step |

**Capture commands:**

```bash
# 1. Image digest
ssh warm-lavender-narwhal '
  docker inspect --format "{{index .RepoDigests 0}}" vllm/vllm-openai:v0.20.0
'

# 2. Model revision sha
ssh warm-lavender-narwhal '
  ls ~/medomni/hf_cache/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/
'

# 3. Most recent snapshot in object store
aws s3 ls "$OBJECT_STORE_TARGET" | tail -3
```

### Hopper-fp8 fallback (single-H100, future)

| Anchor | Value | Captured |
|---|---|---|
| Image | `vllm/vllm-openai:v0.20.0` | yes |
| Image digest | (capture from green pod after first launch) | green-only |
| Model — path A | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (if published) | check HF |
| Model — path B | locally quantized via `modelopt`; export-dir path | per-pod |
| Launch script | `scripts/launch_h200_factory.sh` (env-overridden `VLLM_MODEL_BF16`) | in git |
| Cutover | identical to catfish §1.4 (Vercel env swap) | runbook |

---

## "Is this pod truly relaunchable?" — human audit checklist

Run through this before declaring a pod swap "ready". A green checkmark means the corresponding anchor in the table above has a real value, not a `<TODO_…>` placeholder.

- [ ] **Image digest** captured for the target pod's row above.
- [ ] **Model revision sha** captured.
- [ ] `bash -n scripts/launch_<pod>.sh` exits 0 on the laptop.
- [ ] The launch script is committed to `main` and reachable on the green pod (via `git clone` or scp).
- [ ] `HF_TOKEN` is set in the Brev console env-var UI of the green pod (NOT in any committed file or shell rc).
- [ ] (Narwhal only) Object-store target is set in `scripts/snapshot_h200_factory_state.sh` AND the latest tarball is restorable: `aws s3 cp <OBJECT_STORE_TARGET>/data-queue-LATEST.tar.gz /tmp/test-restore.tgz && tar -tzf /tmp/test-restore.tgz | head -5`.
- [ ] The Vercel `MEDOMNI_TUNNEL_URL_PREVIOUS` rollback env var has been provisioned on the medomni production project (per `runbooks/blue-green-pod-replacement.md` §1.4).
- [ ] A dry-run of the smoke sequence in §1.3 / §2.4 of the blue-green runbook returns content (not just 200).
- [ ] The launch script's exit-on-failure behavior was verified in a non-prod pod at least once (run with a deliberately wrong `HF_TOKEN` and confirm exit 3).
- [ ] Memory hot-path markers reviewed:
  - [ ] `feedback_check_docker_entrypoint_before_docker_run.md`
  - [ ] `nemotron_omni_tool_call_parser.md`
  - [ ] `feedback_vercel_deploy_from_main.md`
  - [ ] `feedback_pilot_before_full_sweep.md`
  - [ ] `brev_ssh_direct_access.md`

---

## When to update this file

- **Every pod swap.** Capture the new pod's image digest and model revision; replace the old row.
- **Any vLLM image upgrade.** New tag → new digest → new row plus a CARD entry under `findings/<date>-vllm-upgrade/CARD.md`.
- **Any model HF revision change.** Even if the SKU name is identical.
- **Any change to `scripts/launch_b300_prod.sh` or `scripts/launch_h200_factory.sh`.** Bump the launch-flags hash and note the diff in a CARD.

---

## Cross-reference

- Pod swap procedure: `runbooks/blue-green-pod-replacement.md`
- Snapshot script: `scripts/snapshot_h200_factory_state.sh`
- Hardware constraints: medomni `CLAUDE.md` §3
- Isolation contract (what NOT to touch): medomni `CLAUDE.md` §1
- Sovereign-by-construction (env-var contract): medomni `CLAUDE.md` §2
