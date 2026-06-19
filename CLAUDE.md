# CLAUDE.md — medomni (public)

## Repo relationship

This is the **public** release/demo repo (`github.com/GOATnote-Inc/medomni`,
deployed to `medomni.vercel.app`). It hosts `web/` (Next.js BFF + UI) and
release artifacts. Active research/training lives in the **private** sister
repo `github.com/GOATnote-Inc/prism42-nemotron-med` (factory_loop, PEFT
production training, evals, CARDs — no web app). Same public/private split
pattern as `prism42` (public) ↔ `prism2` (private). §0 below was split
out from the inherited prism42 charter on 2026-05-22 — only lines that
load-bear on every medomni session remain; the rest moved into
`.claude/skills/` and loads on demand. The isolation contract still
holds: don't touch the prism42 prod surface, ElevenLabs, LiveKit, DNS
(see §1 for the explicit list).

## §0 — HOT-PATH MARKERS (read every session · ≤15 lines · justify-or-evict)

Last audited: 2026-05-24 (audio P0 retrospective —
`findings/research/2026-05-24-audio-outage-retrospective/POSTMORTEM.md`).
These earn entry by being things this session would otherwise violate. New
entries replace weaker ones — no growth without eviction.

1. **`/4UWHAt` prod runs on the Claude API (Opus 4.8), NOT a self-hosted GPU**
   — migrated 2026-06-19/20 (see `project_medomni_claude_migration` in memory +
   `findings/2026-06-19-claude-cutover/`). The Brev pod `exact-kind-orca`
   (`nemotron-serve`) is **decommissioned** (backed up to
   `~/orca-backup.tar.gz`; deleted to end ~$89/day). Re-provision self-hosted
   serving only via `scripts/serve_orca_h100.sh` if ever needed.
2. **Vercel git auto-deploy is disconnected (since 2026-05-21)** — merges to
   `main` do NOT deploy. Production = manual
   `vercel --prod --cwd /Users/kiteboard/medomni --scope goatnote --yes --archive=tgz`.
   Reconnect is a founder dashboard action.
3. **No `git add -A` / `git add .`** — enforced by `.claude/settings.json`
   PreToolUse hook; the hook refuses the tool call.
4. **Cloud LLM keys: web BFF only (2026-06 SANCTIONED exception).** The
   `/4UWHAt` web inference path (`web/app/api/agent`, `/api/ask`,
   `lib/agent/skills`) runs on the **Claude API** (Opus 4.8) to retire the
   ~$100/day Brev H100 — gated by `MEDOMNI_LLM_PROVIDER` (`anthropic` default,
   `vllm` = rollback while orca is warm). `ANTHROPIC_API_KEY` lives in Vercel
   env + `.env.example` only; code uses bare `new Anthropic()` (no literal, so
   the `no-cloud-llm-keys` hook still guards code paths). The
   research/eval/training stack stays **sovereign** (local judge + serve). §2.
5. **Verify-then-claim** — every change ends with a verifying command;
   read artifact JSON, not exit code (memory `feedback_eval_preflight_judge_key.md`).
6. **vLLM audio on orca needs `librosa`+`soundfile` baked into the image** —
   `vllm/vllm-openai:latest` lacks them; vLLM 400s on the first audio
   request. Use `scripts/serve_orca_h100.sh build` for any rebuild.

Load-on-demand content lives in `.claude/skills/`. This split came from
issue #396 / @m13v's comment that the audit grades scaffold presence, not
weight. See `.claude/README.md` for the philosophy and open follow-ups
(§1, "what this repo is", §6 all have the same problem).

## What this repo is

A sovereign NVIDIA medical-LLM stack on Brev Hopper GPUs (H200 + H100), built to demonstrate `Nemotron-70B-Med` end-to-end:

- **Inference**: TensorRT-LLM 1.2.1 + Triton (NIM-first, hand-built fallback) serving Llama-3.1-Nemotron-70B-Instruct fp8 on H200
- **Specialization**: NeMo Framework PEFT LoRA fine-tune on a curated medical corpus (HealthBench-train + MedQA-train + PubMedQA + OpenEM 370 + filtered LostBench/SG2 trajectories)
- **RAG**: NV-Embed-v2 + nx-cugraph traversal over an OpenEM-expanded medical knowledge graph
- **Guardrails**: NeMo Guardrails 0.21+ with Colang 2.0 rails, backed by local Llama-Guard-3-8B
- **Judge**: Llama-3.1-Nemotron-70B-Reward-HF on H100 (sovereign — no cloud LLM keys in the eval loop)

Provenance: derived (squash-import, no history) from `github.com/GOATnote-Inc/prism42` at HEAD `e02e62dd...` on 2026-04-28. The medical-LLM eval harness in public prism42 (`mla/`, `scripts/healthbench_runner.py`, etc.) had zero prod-surface entanglement and was lifted as-is.

## §1 — ISOLATION CONTRACT (the non-negotiable rules)

This repo is **air-gapped** from the prism42 production surface at `https://prism42-console.vercel.app/prism42-v3`. The freeze posture documented in the public repo's deployment ledger is the doctrine. The freeze beats narrative literalness on the eve of judging.

NEVER, under any condition:

1. Reference, edit, or redeploy any frontend page under `mvp/911-console-live/app/prism42-v3/`, `app/prism42-v2/`, or `app/prism42/livekit/`. The narrative copy on those pages ("canonical opus 4.7 demo," "sub-second ttft," "B300" badge) is **not ours to fix this session**.
2. Touch the Vercel project — `.vercel/`, `vercel.json`, `vercel deploy`, env-var changes, domain edits.
3. Touch ElevenLabs ConvAI — `agents/psap-*.yaml`, `prism42-elevenlabs.yaml`, the 14 PSAP agents, any signing secret.
4. Touch the LiveKit self-host runtime — `agents/livekit/worker.py`, `agents/livekit/livekit.yaml`, `infra/b300/*`, the B300 prod pod, `wss://livekit.thegoatnote.com`.
5. Touch the H100 voice-freeze pod (per public-repo `findings/voice/.../freeze-cert*.md`). That is **distinct hardware** from the Brev `prism-mla-h100` pod (montreal-canada-2, ID `x3rytha2l`) used in this repo, despite the H100 SKU collision. Confirm the host (`62.169.159.15` for Brev `prism-mla-h100`) before any ssh.
6. Source, read, or `cp` the canonical `.env` at `/Users/kiteboard/lostbench/.env`. That file holds prod-shared OPENAI / ANTHROPIC / GOOGLE / XAI keys. This repo is sovereign by construction — no cloud LLM keys exist in any code path here.
7. Touch DNS — no GoDaddy API calls, no records on `*.thegoatnote.com`. Pod access is ssh-tunnel only; no public ingress, no Caddy, no TLS termination on the new pods.
8. ssh into `prism-mla-b300-h4h5` or any voice-pod handle. Only the two Brev pods authorized for this repo: `prism-mla-h100` (Hyperstack, montreal-canada-2) and `warm-lavender-narwhal` (Nebius, eu-north1).

## §2 — Sovereignty by construction (+ the sanctioned web exception)

The **research / eval / training stack stays sovereign**: the judge runs locally
on H100 (Llama-3.1-Nemotron-70B-Reward), the serve runs locally on H200,
guardrails run locally (Llama-Guard-3), RAG runs locally (NV-Embed-v2 +
nx-cugraph). Do **not** add cloud LLM keys to the eval loop, training scripts,
or `sovereign_bench`/`ci-medomni` — external keys there defeat the premise.

The `.env.example` permits these secrets:

- `HF_TOKEN` — Hugging Face read-only, gated-model access. **NEW** token scoped to this private repo. Not the prod-shared one.
- `BREV_PEM_PATH` — path to existing brev.pem at `/Users/kiteboard/.brev/brev.pem` (already on disk).
- `ANTHROPIC_API_KEY` — **web BFF only** (2026-06 migration). See below.

**Sanctioned exception (2026-06): the public `/4UWHAt` web demo runs on the
Claude API.** Brandon directed the migration of the web inference path off the
self-hosted Nemotron-Omni H100 (~$100/day) to **Claude Opus 4.8** — the accuracy
winner for emergency-med reasoning in our HealthCraft eval (Pass@1 23.7% vs
GPT-5.5 13.7%). This is intentional; **do NOT "fix" it back to sovereign-only.**
Rules that still hold: the key is read only by `@anthropic-ai/sdk` via bare
`new Anthropic()` (never a literal in code → the `no-cloud-llm-keys` hook still
fires on code paths); the key lives in Vercel env + `.env.example`; provision a
NEW medomni-scoped key with **zero-data-retention** enabled (demo/synthetic
data, no PHI) and do NOT reuse the prod-shared key in
`/Users/kiteboard/lostbench/.env` (§1.6 still bars reading that file). Rollback:
set `MEDOMNI_LLM_PROVIDER=vllm` and redeploy while orca is warm.

## §3 — Hardware reality

The two pods are **Hopper** (SM 9.0). The B300 prod pod is Blackwell (SM 10.x).

- **NVFP4 is Blackwell-only**. The prod model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` will NOT run on H100/H200.
- **fp8 is Hopper-native**. We target fp8 via TensorRT Model Optimizer (`modelopt.torch.quantization`).
- **bf16-mixed for LoRA training**. fp8 fine-tune is finicky; bf16-mixed is the safe default for NeMo PEFT.

## §4 — Verify-then-claim discipline

Borrowed from public-repo CLAUDE.md and memory `feedback_verify_every_action.md`:

- Every change ends with a verifying command. "verified:" not "done."
- Every long eval starts with a 1-example smoke whose **artifact JSON content is read**. Exit code is not a smoke check (memory: `feedback_eval_preflight_judge_key.md` — judge 401 silently zeros rewards).
- Pre-commit must pass clean before any commit. No `--no-verify`. No `git add -A`. No `git add .`.

## §5 — Commit hygiene

- Author email: `b@thegoatnote.com` (per memory `feedback_correspondence_email.md` — professional only, never personal `brandondent17@gmail.com`).
- One Co-Authored-By line per Claude commit.
- Stage by name. Never `-A` or `.`.
- Pre-commit blocks any string matching `prism42-console\.vercel\.app`, `livekit\.thegoatnote\.com`, `wss://prism42`, `ELEVENLABS_API_KEY`, `VERCEL_TOKEN`, `GODADDY_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` outside of `.env.example` placeholder lines.

## §6 — Session re-entry checklist

If you (Claude) wake up in this repo without recent context:

1. Read this file end-to-end.
2. `cat results/<latest>/CARD.md` to see where we landed.
3. `git -C /Users/kiteboard/prism42 status --porcelain` and verify it matches `/tmp/prism42-nemotron-med-session/prism42_worktree_baseline.txt` (the freeze must hold).
4. `gh repo view GOATnote-Inc/prism42 --json pushedAt` and verify it equals the value in `/tmp/prism42-nemotron-med-session/session_start.txt`.
5. Re-hash the prod URLs in `/tmp/prism42-nemotron-med-session/prod_hashes_before.txt` and diff. If non-empty, **stop and surface to the user before any further work**.

## §7 — Frozen paths (do not edit)

- `data/seed_kg/*.csv` and `data/seed_kg/README.md` — the 100-node seed KG is illustrative + physician-review-pending. Do not mutate; expand by writing new files into `data/seed_kg/expansions/`.
- `corpus/pins/healthbench-hard-1000.yaml` — pin manifest. Read-only.
- `findings/research/2026-04-27-future-stack/*` — research briefs, reference-only.
