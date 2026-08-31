# CLAUDE.md — medomni (public)

## Repo relationship

This is the **public** release/demo repo (`github.com/GOATnote-Inc/medomni`,
deployed to `medomni.vercel.app`). It hosts `web/` (Next.js BFF + UI), the
eval harness, and release artifacts. Active research/training lives in the
**private** sister repo (no web app). Same public/private split pattern as
`prism42` (public) ↔ `prism2` (private). The isolation contract still
holds: don't touch the prism42 prod surface, ElevenLabs, LiveKit, DNS
(see §1 for the explicit list).

## §0 — HOT-PATH MARKERS (read every session · ≤15 lines · justify-or-evict)

Last audited: 2026-08-31 (public-repo readiness fix pass). These earn entry
by being things this session would otherwise violate. New entries replace
weaker ones — no growth without eviction.

1. **Production is built from the unmerged branch `feat/claude-opus-migration`,
   NOT from `main`.** Do not merge, rebase, or modify that branch without an
   explicit owner decision; merging it is the owner's step.
2. **The self-hosted GPU serving pod is DECOMMISSIONED (June 2026).** Demo
   inference is migrating to the Anthropic Claude API via the web BFF. Do
   not restate the old "no third-party AI APIs called / dedicated NVIDIA
   hardware" claim anywhere — it is false for the current architecture.
3. **Vercel git auto-deploy is disconnected (since 2026-05-21)** — merges to
   `main` do NOT deploy. Production deploys are a founder dashboard/CLI
   action.
4. **No `git add -A` / `git add .`** — stage by name.
5. **Cloud LLM keys are scoped, not forbidden**: demo inference (Claude API,
   Vercel env), eval graders (gpt-4.1 / Claude, laptop-side), CI review
   (`ANTHROPIC_API_KEY_PR_REVIEW`). Never commit a key value; never add key
   values to `.env.example`.
6. **Verify-then-claim** — every change ends with a verifying command;
   read artifact JSON, not exit code.

Load-on-demand content lives in `.claude/skills/`. See `.claude/README.md`
for the philosophy.

## What this repo is

The public face of the MedOmni research program:

- **`web/`** — Next.js BFF + Records OS demo UI (`/4UWHAt`). Inference is
  forwarded by `web/app/api/agent` to the configured model backend
  (currently: Anthropic Claude API, migration in progress; formerly:
  self-hosted Nemotron FP8 via vllm, decommissioned).
- **Eval harness** — `scripts/` + `tests/`: ship-rule eval driver,
  HealthBench runner, manifest emitter, adversarial probe. The published
  results (README headline table) were produced on the self-hosted NVIDIA
  stack during 2026-04/06 and are reproducible via the per-eval REPRO.sh
  files (GPU required).
- **`findings/` + `results/`** — pre-registered eval provenance, CARDs,
  manifests. Historical documents describe the decommissioned self-hosted
  stack; read dates before trusting architecture claims.
- **`mla/`** — medical-LLM agent harness lifted from public prism42 with
  zero prod-surface entanglement (legacy).

Provenance: derived (squash-import, no history) from
`github.com/GOATnote-Inc/prism42` on 2026-04-28.

## §1 — ISOLATION CONTRACT (the non-negotiable rules)

This repo is **air-gapped** from the prism42 production voice surface.

NEVER, under any condition:

1. Reference, edit, or redeploy any prism42 frontend page or voice-console
   surface. That narrative copy is not ours to fix from this repo.
2. Touch the prism42 Vercel project — `.vercel/`, `vercel.json`,
   `vercel deploy`, env-var changes, domain edits. (medomni's own Vercel
   project is a founder-only surface; see §0.)
3. Touch ElevenLabs ConvAI — agents, signing secrets.
4. Touch the LiveKit self-host runtime or any voice-pod infrastructure.
5. ssh into any GPU pod from this repo's tooling. The serving pods this
   repo used are decommissioned; scripts under `scripts/serve_*` are
   retained as historical record, not as live runbooks.
6. Source, read, or `cp` any shared `.env` from other repos on the
   operator's machine. Keys for this repo come from its own environment.
7. Touch DNS — no registrar API calls, no records on `*.thegoatnote.com`.

## §2 — API keys and secrets

The historical "sovereignty by construction — zero cloud LLM keys in any
code path" contract described the pre-June-2026 self-hosted deployment and
is retired as a repo-wide claim. Current reality:

- **Demo inference**: `ANTHROPIC_API_KEY` lives ONLY in Vercel project env
  (owner-managed). It must never appear in this repo, in `.env.example`,
  or in CI logs.
- **Eval grading**: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are sourced
  out-of-band on the operator's machine for laptop-side grading runs.
  Never committed, never echoed.
- **CI**: `ANTHROPIC_API_KEY_PR_REVIEW` (repo secret) powers the PR-review
  and clinical-content gates.
- `.env.example` documents variable NAMES with placeholder values only.

Pre-commit blocks key-shaped strings; detect-secrets + TruffleHog gate CI.

## §3 — Hardware reality (historical)

The training/eval program ran on Hopper (H100/H200, fp8/bf16) and briefly
Blackwell (B300, NVFP4). All pods are decommissioned. NVFP4 quantization
requires Blackwell; fp8 is Hopper-native. This section is retained because
`findings/` documents reference it.

## §4 — Verify-then-claim discipline

- Every change ends with a verifying command. "verified:" not "done."
- Every long eval starts with a 1-example smoke whose **artifact JSON
  content is read**. Exit code is not a smoke check (a judge 401 silently
  zeros rewards).
- Pre-commit must pass clean before any commit. No `--no-verify`. No
  `git add -A`. No `git add .`.

## §5 — Commit hygiene

- Author email: `b@thegoatnote.com` (professional address only).
- One Co-Authored-By line per Claude commit.
- Stage by name. Never `-A` or `.`.
- Pre-commit blocks prod-URL strings and cloud-LLM key names outside the
  documented allowlist (see `.pre-commit-config.yaml`).

## §6 — Session re-entry checklist

If you (Claude) wake up in this repo without recent context:

1. Read this file end-to-end, then README's "Architecture status" note.
2. `cat results/<latest>/CARD.md` to see where the eval program landed.
3. `git log --oneline -10` and `gh pr list` for in-flight work.
4. Remember §0.1: production == `feat/claude-opus-migration`, not `main`.

## §7 — Frozen paths (do not edit)

- `data/seed_kg/*.csv` and `data/seed_kg/README.md` — the 100-node seed KG
  is illustrative + physician-review-pending. Do not mutate; expand by
  writing new files into `data/seed_kg/expansions/`.
- `corpus/pins/healthbench-hard-1000.yaml` — pin manifest. Read-only.
- `findings/**` and `results/**` — eval artifacts are immutable records:
  never alter scores, hashes, dates, or conclusions. (Privacy scrubs of
  incidental metadata — e.g. operator paths — require explicit owner
  intent and must not touch result content.)
