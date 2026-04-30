# medomni/web

Public web UI for [MedOmni](../README.md) — sovereign nurse-first medical
reasoning on the NVIDIA open-component stack.

- Production: `https://medomni.thegoatnote.com` (DNS cutover post-v0)
- Methodology + manifests: parent repo (`..`)
- Architecture decisions: [`../findings/research/2026-04-30-public-url-arch/INDEX.md`](../findings/research/2026-04-30-public-url-arch/INDEX.md)

## v0 status (2026-04-30)

Day-0 scaffold. The composer renders a static demo answer client-side; the
streaming wire-up to the Cloud Run BFF → Tailscale → B300 vllm pipeline lands
across days 1-4 of the sovereign-only sprint. No frontier LLM API calls in v0.

Locked decisions (see `INDEX.md` in the parent repo's findings):
- Subdomain `medomni.thegoatnote.com`, NOT `/medomni` on apex
- Sovereign-only at v0; `Frontier Peer` panel rendered but disabled with "Pro — coming soon" copy
- No auth at v0; persistent "DEMO — do not enter PHI" banner is the gate

## Local dev

```bash
cd web
pnpm install
pnpm dev
# open http://localhost:3000
```

## Deploy (Vercel project root-directory = web)

```bash
cd web
vercel link --yes
vercel deploy            # preview
vercel deploy --prod     # production
```

## License

Apache-2.0 (parent repo). Source code only — no clinical advice. Demo for evaluation only.
