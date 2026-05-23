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

Last audited: 2026-05-22 (issue #396). These earn entry by being things this
session would otherwise violate. New entries replace weaker ones — no growth
without eviction.

1. **`nemotron-serve` on Brev pod `exact-kind-orca` is live production** —
   serves `thegoatnote.com/4UWHAt`. Never `docker stop`, never `podStop`.
2. **Vercel git auto-deploy is disconnected (since 2026-05-21)** — merges to
   `main` do NOT deploy. Production = manual
   `vercel --prod --cwd /Users/kiteboard/medomni --scope goatnote --yes --archive=tgz`.
   Reconnect is a founder dashboard action.
3. **No `git add -A` / `git add .`** — enforced by `.claude/settings.json`
   PreToolUse hook; the hook refuses the tool call.
4. **No cloud LLM keys in any code path.** Sovereign stack;
   `.env.example` permits only `HF_TOKEN` + `BREV_PEM_PATH`. Pre-commit
   blocks the literal key strings outside `.env.example` (full list in §5).
5. **Verify-then-claim** — every change ends with a verifying command;
   read artifact JSON, not exit code (memory `feedback_eval_preflight_judge_key.md`).

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

## §2 — Sovereignty by construction

Zero cloud LLM API keys in any code path. The `.env.example` permits exactly two secrets:

- `HF_TOKEN` — Hugging Face read-only, gated-model access. **NEW** token scoped to this private repo. Not the prod-shared one.
- `BREV_PEM_PATH` — path to existing brev.pem at `/Users/kiteboard/.brev/brev.pem` (already on disk).

If you ever feel tempted to add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to `.env.example` or to a script — **stop and reconsider the design**. The judge runs locally on H100 (Llama-3.1-Nemotron-70B-Reward). The serve runs locally on H200. Guardrails run locally (Llama-Guard-3). RAG runs locally (NV-Embed-v2 + nx-cugraph). External keys defeat the entire premise.

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
