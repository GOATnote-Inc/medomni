# medomni/web

Public web UI for [MedOmni](../README.md) — nurse-first medical reasoning
(see the parent README's "Architecture status" note for what currently
serves inference).

- Production: `https://www.thegoatnote.com/4UWHAt` (Vercel edge rewrite from
  the apex to `medomni.vercel.app/4UWHAt`)
- Methodology + manifests: parent repo (`..`)

## Status (2026-08)

The Records OS dashboard (`/4UWHAt`) is the live surface. The BFF at
`web/app/api/agent` forwards inference to the configured model backend:

- **Current architecture (migration in progress, branch
  `feat/claude-opus-migration`):** the Anthropic Claude API. User queries
  are processed by a third-party AI service.
- **Former architecture (decommissioned June 2026):** self-hosted Nemotron
  FP8 on a dedicated H100 via vllm behind a tunnel.

Persistent "DEMO — do not enter PHI" banner is the gate; there is no auth
at this stage (see the parent repo audit for the planned token gate on
`/api/agent`).

## Local dev

```bash
cd web
pnpm install
pnpm dev
# open http://localhost:3000
```

## Deploy (Vercel project root-directory = web)

Deploys are a founder-only action; Vercel git auto-deploy is disconnected.
Production is currently built from the `feat/claude-opus-migration` branch,
not `main`.

## License

Apache-2.0 (parent repo). Source code only — no clinical advice. Demo for evaluation only.
