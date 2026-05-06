# Blue-Green Pod Replacement Runbook

**Date authored:** 2026-05-06
**Trigger:** NVIDIA notified that the current B300 pod (`unnecessary-peach-catfish`) carries a provisioning bug and may need a delete-and-recreate. The H200 (`warm-lavender-narwhal`) is on the same Brev fleet and protected by the same playbook. A Hopper-fp8 fallback is documented in §3 in case the user pivots to a single-H100 surface later this week.

**Goal:** turn a "B300 must be deleted" notice into a single-day cutover with bounded downtime, an explicit rollback path, and a parity check that proves the green pod is the same software as the blue one.

**Scope guard:** This runbook only covers the medomni Hopper/Blackwell fleet. The prism42 prod surface (LiveKit B300, ElevenLabs PSAPs, the prism42 console, the prism42 LiveKit endpoint) is OUT OF SCOPE per medomni `CLAUDE.md` §1. The Vercel commands in §1.5 / §2.5 below operate on the **medomni** Vercel project (`medomni.vercel.app`) only.

**Author email:** `b@thegoatnote.com` (per memory `feedback_correspondence_email.md`).

---

## Pre-reading (mandatory)

Read these in order before starting. Each is the durable explanation of why a step is the way it is:

- [`runbooks/parity-anchors.md`](./parity-anchors.md) — the four anchors (image SHA, weights, flags, env) that must match between blue and green.
- Memory: `nemotron_omni_tool_call_parser.md` — verified vLLM flags as of 2026-05-03.
- Memory: `feedback_check_docker_entrypoint_before_docker_run.md` — the `vllm serve vllm serve …` crash mode; ALWAYS `docker inspect` first.
- Memory: `feedback_vercel_deploy_from_main.md` — `vercel --prod` ships CWD, not GitHub HEAD; always `git checkout main && git pull --ff-only` before any prod deploy.
- Memory: `feedback_pilot_before_full_sweep.md` — read smoke artifact content; exit code is not enough.
- medomni `CLAUDE.md` §3 — Hardware reality (NVFP4 = Blackwell only, fp8 = Hopper native).

---

## Quick-reference: anchor table for the cutover

| Anchor | Catfish (B300) | Narwhal (H200) | Hopper-fp8 fallback |
|---|---|---|---|
| Pod handle | `unnecessary-peach-catfish` | `warm-lavender-narwhal` | (new pod, e.g. `prism-mla-h100`) |
| Cloud / region | Brev / Helsinki | Brev / Nebius eu-north1 | Brev / Hyperstack montreal-canada-2 |
| GPU | NVIDIA B300 (Blackwell SM 10.x) | NVIDIA H200 141 GiB (Hopper SM 9.0) | NVIDIA H100 80 GiB (Hopper SM 9.0) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (or `-BF16` if FP8 SKU unavailable) |
| Quantization | NVFP4 (Blackwell-native) | BF16 | FP8 via `modelopt.torch.quantization` |
| vLLM image tag | `vllm/vllm-openai:v0.20.0` | `vllm/vllm-openai:v0.20.0` | `vllm/vllm-openai:v0.20.0` |
| Image digest | **TODO**: capture from healthy catfish | **TODO**: capture from healthy narwhal | n/a until provisioned |
| Launch script | `scripts/launch_b300_prod.sh` | `scripts/launch_h200_factory.sh` | `scripts/launch_h200_factory.sh` (sub `VLLM_MODEL_BF16`) |
| Web env var | `MEDOMNI_TUNNEL_URL` | (factory only — not user-facing) | `MEDOMNI_TUNNEL_URL` |
| Local tunnel | `ssh -L 8000:127.0.0.1:8000 unnecessary-peach-catfish` | `ssh -L 8000:127.0.0.1:8000 warm-lavender-narwhal` | `ssh -L 8000:127.0.0.1:8000 prism-mla-h100` |

---

## §1 — Catfish (B300) replacement

### 1.0 Prerequisites (do this FIRST, before NVIDIA's swap window)

These are the parity-capture steps. Do them on the currently-healthy catfish so the green pod is provably identical:

```bash
# 1. Capture vLLM image digest
ssh unnecessary-peach-catfish '
  docker inspect --format "{{index .RepoDigests 0}}" vllm/vllm-openai:v0.20.0
'
# Paste the sha256 into runbooks/parity-anchors.md (B300 row).

# 2. Capture HF model snapshot SHA (the actual weights)
ssh unnecessary-peach-catfish '
  ls -la ~/medomni/hf_cache/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/snapshots/
'
# Capture the snapshot dir name (it is the HF revision sha) and store it.

# 3. Capture exact running flags (sanity vs scripts/launch_b300_prod.sh)
ssh unnecessary-peach-catfish '
  docker inspect vllm-omni-b300 --format "{{json .Args}}" | jq .
'
# Diff against scripts/launch_b300_prod.sh; resolve any drift before swap.

# 4. Confirm HF_TOKEN is provisioned in the Brev console (NOT in shell history,
#    NOT in scripts) so the green pod can fetch weights without the token
#    transiting the proxy. Memory: feedback_no_secret_value_dumps.md.
```

### 1.1 Provision green pod

Use the Brev console to create a fresh B300 in the same region, with the same disk and memory profile. **Do not delete blue yet.** Name it descriptively (e.g. `unnecessary-peach-catfish-2`). Provision `HF_TOKEN` and any NGC creds via the console env-var UI.

### 1.2 Boot prod inference on green

```bash
# 1. ssh into the green pod (fresh keypair via Brev)
ssh unnecessary-peach-catfish-2

# 2. Pull the script and any pinned digests
git clone https://github.com/GOATnote-Inc/medomni.git /home/ubuntu/medomni
cd /home/ubuntu/medomni

# 3. Export pinned digest from runbooks/parity-anchors.md
export B300_VLLM_IMAGE_DIGEST="sha256:..."   # from §1.0 step 1

# 4. Launch (HF_TOKEN comes from console env, NOT from this shell)
bash scripts/launch_b300_prod.sh
```

The script:
- Verifies GPU is Blackwell.
- Pulls vLLM by digest (or by tag with a warning if digest absent).
- Verifies image ENTRYPOINT is `["vllm","serve"]` before running (memory: `feedback_check_docker_entrypoint_before_docker_run.md`).
- Launches container with the verified-2026-05-03 flag set.
- Waits up to 40 minutes for `/v1/models`.
- Smoke-tests with a clinical prompt at `max_tokens=4000` (memory: `nemotron_omni_tool_call_parser.md`).
- Exits non-zero on any check failure.

### 1.3 Smoke validation (independent of the launch script)

From the laptop, after the launch script reports OK:

```bash
ssh -L 8001:127.0.0.1:8000 unnecessary-peach-catfish-2 -N &
TUNNEL_PID=$!

# Confirm the model name + verify a tool-call works
curl -s http://127.0.0.1:8001/v1/models | jq '.data[].id'

# A clinical prompt that exercises the reasoning path (max_tokens >= 4000)
curl -sS -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "messages":[{"role":"user","content":"55F with sudden tearing chest pain radiating to back. BP 200/110. What is the first imaging study?"}],
    "max_tokens":4000,
    "temperature":0.2
  }' | jq -r '.choices[0].message.content' | head -20

kill "$TUNNEL_PID"
```

If both succeed AND the answer mentions CT angiography (or equivalent), the green pod is ready for cutover.

### 1.4 Atomic Vercel cutover for the medomni surface

The web app at `medomni.vercel.app/4UWHAt` reads `MEDOMNI_TUNNEL_URL` to reach the prod inference endpoint. The cutover replaces that env var with a tunnel that points at the green pod.

**Pre-cutover hygiene** (memory: `feedback_vercel_deploy_from_main.md`):

```bash
cd /Users/kiteboard/medomni
git checkout main
git pull --ff-only
```

**Capture the previous value as a rollback anchor:**

```bash
# Read existing prod env var (the value goes to a temp file, NOT to history)
vercel env pull --environment=production .env.prod.tmp
grep '^MEDOMNI_TUNNEL_URL=' .env.prod.tmp >.env.prod.previous
# Move it to a safe location off-repo:
mv .env.prod.previous "$HOME/.medomni-tunnel-rollback-$(date -u +%Y%m%dT%H%M%SZ).env"
rm .env.prod.tmp
```

**Add the previous as `MEDOMNI_TUNNEL_URL_PREVIOUS` so rollback is one env-var swap:**

```bash
# Manually paste old value when prompted (do NOT pipe from history)
vercel env add MEDOMNI_TUNNEL_URL_PREVIOUS production
```

**Swap the live env var to the green tunnel:**

```bash
# The new value is the green pod's tunnel URL. The exact form depends on
# how the laptop bridges to the pod (a Cloudflare Tunnel, an ngrok, or a
# stable Brev port-forward). Whatever you use, the value MUST be reachable
# from Vercel's runtime — ssh tunnels from the laptop are NOT.
vercel env rm MEDOMNI_TUNNEL_URL production
vercel env add MEDOMNI_TUNNEL_URL production    # paste green URL when prompted
```

**Redeploy** (with `--archive=tgz` because medomni has 16K+ files; memory: `feedback_vercel_auto_deploy_can_silently_disconnect.md`):

```bash
cd /Users/kiteboard/medomni
vercel --prod --archive=tgz
```

Verify:

```bash
curl -fsS https://medomni.vercel.app/4UWHAt/api/agent -X POST \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ping"}]}' \
  | head -c 300
```

If the response streams or returns 200, cutover is live.

### 1.5 Drain blue, then delete

Wait at least 30 minutes after cutover for any in-flight requests to complete. Verify zero traffic on blue:

```bash
ssh unnecessary-peach-catfish 'docker logs --since 30m vllm-omni-b300 2>&1 | grep -c "POST /v1/chat/completions"'
```

If 0, blue is drained. Stop the container:

```bash
ssh unnecessary-peach-catfish 'docker stop vllm-omni-b300'
```

**Now** delete the blue pod via Brev console (per the NVIDIA recommendation).

### 1.6 Rollback (if green is bad)

If smoke fails or the cutover causes user-visible breakage, atomic rollback:

```bash
cd /Users/kiteboard/medomni
git checkout main && git pull --ff-only

# Restore previous URL
vercel env rm MEDOMNI_TUNNEL_URL production
vercel env add MEDOMNI_TUNNEL_URL production    # paste $HOME/.medomni-tunnel-rollback-*.env value
vercel --prod --archive=tgz
```

Blue (still alive, since we deferred deletion) catches traffic again. Then debug green offline.

---

## §2 — Narwhal (H200) replacement

The same playbook, two differences: (a) narwhal is the *factory* + RAG pod, not user-facing; the cutover is therefore simpler — there is no Vercel env to swap, only the tunnel that the user (or the laptop-side judge) consumes; (b) `data-queue/` state must be restored, not regenerated.

### 2.0 Prerequisites

```bash
# Capture image digest
ssh warm-lavender-narwhal '
  docker inspect --format "{{index .RepoDigests 0}}" vllm/vllm-openai:v0.20.0
'

# Confirm snapshot script is running (cron) and a recent tarball is in object store
ssh warm-lavender-narwhal 'tail -5 ~/snapshot.log'
# Or list the bucket:
aws s3 ls "$OBJECT_STORE_TARGET" | tail -5
```

If no recent snapshot exists, **run** `scripts/snapshot_h200_factory_state.sh` once interactively before any swap window.

### 2.1 Provision green pod

Same as §1.1 but H200 SKU and same region. Provision `HF_TOKEN` via the console.

### 2.2 Restore data-queue from object store

```bash
ssh warm-lavender-narwhal-2

# Pull repo
git clone https://github.com/GOATnote-Inc/medomni.git /home/ubuntu/medomni

# Pull imaging-rag clone (private repo; requires user's GitHub creds in console env)
git clone https://github.com/GOATnote-Inc/prism42-nemotron-med.git \
  /home/ubuntu/medomni-imaging-rag

# Restore data-queue from latest snapshot
mkdir -p /home/ubuntu/data-queue
aws s3 cp "$OBJECT_STORE_TARGET/data-queue-LATEST.tar.gz" /tmp/restore.tgz
tar -xzf /tmp/restore.tgz -C /home/ubuntu/
ls /home/ubuntu/data-queue/
```

### 2.3 Boot factory + RAG on green

```bash
cd /Users/kiteboard/medomni    # (on the green pod, from /home/ubuntu/medomni)
export H200_VLLM_IMAGE_DIGEST="sha256:..."   # from §2.0
bash scripts/launch_h200_factory.sh
```

The script:
- Verifies GPU is H100/H200.
- Launches vLLM with Nemotron-3-Nano BF16.
- Smokes vLLM with a clinical prompt; reads response content.
- Launches `nvembed_server.py`, `biomedclip_server.py`, `factory_loop.py` in tmux sessions.
- If `medomni-imaging-rag` is missing, brings up vLLM only and warns loudly (so prod inference is still restored).

### 2.4 Smoke validation

```bash
# vLLM
ssh -L 8000:127.0.0.1:8000 warm-lavender-narwhal-2 -N &
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'

# RAG sidecars
ssh warm-lavender-narwhal-2 '
  curl -sf http://127.0.0.1:8003/health || echo NV_EMBED_DOWN
  curl -sf http://127.0.0.1:8004/health || echo BIOMEDCLIP_DOWN
'

# Factory
ssh warm-lavender-narwhal-2 'pgrep -af factory_loop.py && tail -1 ~/data-queue/heartbeat.jsonl'
```

### 2.5 Drain + delete blue

The factory writes data-queue items to `/home/ubuntu/data-queue/`. To avoid double-generation between blue and green during overlap, stop blue's factory **before** starting green's:

```bash
# On blue
ssh warm-lavender-narwhal '
  tmux kill-session -t factory 2>/dev/null
  pkill -f factory_loop.py
'

# Confirm zero processes
ssh warm-lavender-narwhal 'pgrep -af factory_loop.py || echo CLEAN'
```

Then start green's factory (`launch_h200_factory.sh` already covers this), wait 24h to confirm new data-queue items are appearing, then delete blue.

### 2.6 Rollback

If green fails: green's vLLM and factory both `--restart unless-stopped`-aware can be torn down with `docker rm -f vllm-nemotron-bf16 && tmux kill-session -t factory`. Blue's factory can be restarted by re-running `scripts/launch_h200_factory.sh` on blue (idempotent).

---

## §3 — Hopper-fp8 fallback (single-H100, no Blackwell)

If the user pivots to a single-H100 surface (no B300, no H200 with BF16), this section is the recipe.

### 3.1 Why fp8 instead of BF16

BF16 works but is ~2x slower than fp8 on Hopper for memory-bound decode. For a demo-grade prod surface, fp8 via NVIDIA's TensorRT Model Optimizer (`modelopt.torch.quantization`) is the prod pattern. The narwhal BF16 surface is the *training-data factory*; the user-facing inference target should be fp8 wherever possible on Hopper.

### 3.2 Two paths

| Path | Pros | Cons |
|---|---|---|
| **A. Use the public FP8 SKU** if NVIDIA has published `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | Fast, no quantization step | SKU may not exist yet — check HF |
| **B. Quantize BF16 → fp8 yourself** with `modelopt.torch.quantization` | Always works | ~2 hours of quant + calibration on the pod |

Verify path A first:

```bash
huggingface-cli repo-info nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 2>&1 | head -10
```

If it returns a 200, take path A. If 404, path B.

### 3.3 Path A: FP8 SKU launch

Same as §2.3, but:

```bash
export VLLM_MODEL_BF16="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"   # despite the var name
bash scripts/launch_h200_factory.sh
```

(Yes, the env var is named `_BF16` because narwhal's existing surface is BF16; for the fp8 fallback you are reusing the same script with a different model. A future tidy-up renames it to `VLLM_MODEL`.)

### 3.4 Path B: BF16 → fp8 with modelopt (offline, one-time)

On the H100 pod, in a Python container with `nvidia-modelopt` installed:

```python
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    torch_dtype="bfloat16", device_map="auto",
)
config = mtq.FP8_DEFAULT_CFG
mtq.quantize(model, config, forward_loop=...)   # calibration loop with ~32 prompts
mtq.export(model, export_dir="/home/ubuntu/nemotron-fp8-h100")
```

Then launch vLLM pointing at the local export:

```bash
docker run -d --name vllm-nemotron-fp8 \
  --gpus all --shm-size=32g \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 127.0.0.1:8000:8000 \
  -v /home/ubuntu/nemotron-fp8-h100:/model \
  --restart unless-stopped \
  vllm/vllm-openai:v0.20.0 \
  --model /model \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --trust-remote-code
```

### 3.5 Cutover and rollback

Same as §1.4 / §1.6. The Vercel env-var swap is identical; only the tunnel target changes.

---

## §4 — Post-cutover housekeeping

After any cutover (catfish or narwhal):

1. Update [`runbooks/parity-anchors.md`](./parity-anchors.md) with the new pod's image digest and snapshot SHA. If the digest changed, document why.
2. Append a CARD entry under `findings/<date>-pod-swap/CARD.md` capturing the before/after, smoke results, and any deltas.
3. Confirm the snapshot cron is reinstalled on the green narwhal:
   ```bash
   ssh warm-lavender-narwhal-2 'crontab -l | grep snapshot_h200_factory_state'
   ```
4. Branch + remote pruning per memory `feedback_clean_branches_after_session.md`.

---

## Appendix A — known gotchas (not pod-specific)

- `--allowed-local-media-path` returns 400 for *all* requests if the path is missing inside the container. The launch scripts deliberately omit this flag.
- vLLM image ENTRYPOINT is `["vllm","serve"]`. Never prepend `vllm serve` in the run command (memory: `feedback_check_docker_entrypoint_before_docker_run.md`).
- Smoke prompts MUST use `max_tokens >= 4000` for Nemotron-Omni (model reasons before tool call). The B300 launch script enforces this.
- Vercel CLI: `vercel --prod` deploys *current working directory*, not GitHub HEAD (memory: `feedback_vercel_deploy_from_main.md`). Always `git checkout main && git pull --ff-only` first.
- Vercel auto-deploy hooks can silently disconnect (memory: `feedback_vercel_auto_deploy_can_silently_disconnect.md`); always include `--archive=tgz` from the repo root.
- Brev pods are direct-ssh, so secret-leak surfaces from the RunPod proxy do **not** apply (memory: `brev_ssh_direct_access.md`). HF_TOKEN can flow through `ssh warm-lavender-narwhal 'cmd'` safely. **Do not** use the RunPod proxy for these pods.
