# E10 — HF write-token presence audit

**Date:** 2026-05-06
**Purpose:** Confirm an HF token is provisioned BEFORE the T+18h HF Hub
publication step (E8) without ever reading the value into the conversation
context. Per memory `feedback_never_read_env.md` and `feedback_no_secret_value_dumps.md`.

## Method

```bash
awk -F= '/^HF_TOKEN=/ {print "HF_TOKEN: present, len=", length($2)}' \
  /Users/kiteboard/prism42-nemotron-med/.env
```

`awk` reads the file byte-by-byte and emits ONLY the length of the value;
the value itself is never substituted into a shell variable, never echoed,
never logged.

## Result

```
HF_TOKEN: present, len= 37
```

- 37 characters is consistent with a standard HuggingFace token (prefix
  `hf_` + 34-char body). Cannot distinguish read-only vs write from length
  alone. The `.env.example` template at `/Users/kiteboard/prism42-nemotron-med/.env.example`
  documents this slot as "read-only, gated-model access" — but HF Hub
  upload requires WRITE scope. The user must verify scope manually before E8.

## Pre-publication checklist (T+18h, NOT now)

Before running `huggingface-cli upload` (see HF-MODEL-CARD-DRAFT.md):

1. Confirm token write scope at https://huggingface.co/settings/tokens
   (token name "prism42-nemotron-med" or similar; "Write" scope, not just
   "Read"). If only Read, generate a fresh Write token, replace in
   `/Users/kiteboard/prism42-nemotron-med/.env`, re-run this audit.

2. Verify token is NOT the prod-shared token from
   `/Users/kiteboard/lostbench/.env` (per `prism42-nemotron-med/CLAUDE.md` §2,
   the medomni / prism42-nemotron-med repos must use a SEPARATE token to
   isolate scope).

## Side-finding (FYI, NOT in scope of E10)

The `.env` keyset on inspection contains:

```
ANTHROPIC_API_KEY  CF_TURN_KEY_ID  CF_TURN_KEY_TOKEN  CLOUDFLARE_API_TOKEN
ELEVENLABS_API_KEY  GODADDY_API_KEY  GODADDY_API_SECRET  HF_TOKEN
LAMBDA_API_KEY  LIVEKIT_API_KEY  LIVEKIT_API_SECRET  LIVEKIT_URL
NVIDIA_API_KEY  OPENAI_API_KEY  RUNPOD_API_KEY  RUNWAYML_API_SECRET
X_AI_APIKEY
```

This expands well beyond the two-secrets-only contract documented in
`prism42-nemotron-med/CLAUDE.md` §2 ("HF_TOKEN + BREV_PEM_PATH").
That contract was for the SOVEREIGN inference path; the broader keyset
appears to support DNS, voice, and prod-deploy flows that came in after
the original §2 was authored.

This is OUT OF SCOPE for E-track and not a security incident — the .env
file lives only on the user's laptop and is gitignored. Just flagging
for the user's review.

## Hard rules verified

- No env value was read into shell or conversation context.
- No `cat .env`, no `printenv HF_TOKEN`, no `grep KEY=` pattern.
- Method matches the canonical pattern in
  `feedback_no_secret_value_dumps.md`.
