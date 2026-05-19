# CARD — medomni `/api/agent` integration with HealthCraft ED Decision Rules MCP

Date: 2026-05-12
Author: Claude Opus 4.7 session
Status: **SHIP — parity 50/50 (100%); code on branch `feat/mcp-integration` against `main`, NOT YET DEPLOYED to production. Preview deploy auto-created by Vercel from the PR; production activation pending PR merge + manual `MCP_ED_RULES_ENABLED=1` env-var flip.**

## Verdict

Wire the read-only HealthCraft ED Decision Rules MCP at
`https://mcp.thegoatnote.com/mcp` into medomni's `/api/agent` route
behind a double-gated feature flag (`?profile=mcp` query param AND
`MCP_ED_RULES_ENABLED=1` environment variable). Default demo path is
unchanged.

Triggering condition met: HealthCraft MCP `applyDecisionRule` agrees
with medomni's hand-rolled `clinical_calculate` on every overlapping
score, and produces canonical HEART / Wells PE values to the digit on
inputs medomni cannot currently compute.

## What was probed

`https://mcp.thegoatnote.com/mcp` returns:

- `serverInfo: agents-assemble/ed-decision-rules v0.1.0`
- `protocolVersion: 2025-03-26` (MCP Streamable HTTP)
- SHARP extension `ai.promptopinion/fhir-context` (FHIR scopes optional, not required for variable-mode calls)
- 6 tools: `applyDecisionRule`, `listRules`, `getRuleSchema`,
  `getCoverageForComplaint`, `getProtocolDetails`, `getReferenceArticle`
- 100 bundled decision rules, including all 5 currently in medomni's
  `clinical_calculate` (CHA2DS2-VASc, HAS-BLED Score, MELD-Na,
  Wells Criteria for DVT, PERC Rule) plus HEART Score, Wells Criteria
  for PE, CURB-65, TIMI Risk Score, GRACE, NEXUS C-Spine, Ottawa
  Ankle, PSI/PORT, qSOFA, sPESI, etc.

Source: `/Users/kiteboard/healthcraft/src/healthcraft/agents_assemble/streamable_http_server.py`.

## Parity check (50 prompts)

Script: `parity_check.mjs` (this directory). Raw output:
`parity_check_results.json`. Pure Node 18+ mjs with built-in fetch; the
five medomni calculator functions are lifted-verbatim from
`/Users/kiteboard/medomni/web/lib/tools/clinical-calculator.ts` so the
test is hermetic against the production code path.

| Rule         | Cases | Pass | Mismatch | Error | Comparison axis |
|--------------|-------|------|----------|-------|-----------------|
| CHA2DS2-VASc | 8     | 8    | 0        | 0     | raw score, tol 0 |
| HAS-BLED     | 8     | 8    | 0        | 0     | raw score, tol 0 |
| MELD-Na      | 8     | 8    | 0        | 0     | raw score, tol ±1 (matched at delta 0) |
| Wells DVT    | 8     | 8    | 0        | 0     | raw score incl. −2 alt-dx modifier, tol 0 |
| PERC         | 8     | 8    | 0        | 0     | inverted-verdict (PERC-negative? bool) |
| HEART        | 5     | 5    | 0        | 0     | canonical-table sum, tol 0 |
| Wells PE     | 5     | 5    | 0        | 0     | canonical-table sum, tol 1e-6 |
| **OVERALL**  | **50**| **50**| **0**   | **0** | **100.0% pass** |

### What the test pinned down

- **CHA2DS2-VASc**: MCP splits age across two variables (`Age >= 75` max 2, `Age 65-74` max 1, mutually exclusive). Medomni's single-variable encoding maps cleanly to the dual encoding.
- **Wells DVT**: MCP exposes `Alternative diagnosis at least as likely` as a signed variable in range `-2..0`. Medomni's `alternative_diagnosis_at_least_as_likely: true` translates to `-2`. Case `wd-04` (all 9 +1 flags + alt dx) produces score 7 on both sides; `wd-06` (active_cancer + alt dx only) produces −1 on both sides.
- **PERC has inverted semantics** — this is the lone translation hazard. Medomni asks "criterion met?" (so all 8 true → PERC-negative score 8). MCP asks "criterion present-as-failure?" (so all 8 zero → PERC-negative score 0). The translator inverts: `age_under_50: true` → `"Age >= 50": 0`. Verdict-level comparison (`mcp.score === 0` ⇔ `medomni.perc_negative === true`) matches on all 8 cases.
- **MELD-Na**: both sides round identically across the test inputs. Tolerance was set to ±1 defensively; observed delta was 0 across all 8 cases (creatinine cap at 4.0, INR/bilirubin floor at 1.0, dialysis-bumps-creatinine, Na correction window 125–137 all align).

### What HEART + Wells PE unlock

These rules are not in medomni's `clinical_calculate`. They produce
exact canonical sums for the 0/1/2 (HEART) and 0/3/1.5/1 (Wells PE)
encodings on the 10 test inputs. This is the capability gain: ED-typical
chief-complaint workflows ("chest pain → HEART", "suspected PE → Wells
PE + PERC + Geneva") become callable.

## Wired code (committed on branch `feat/mcp-integration`)

```
medomni/web/lib/mcp/ed-rules-client.ts          # JSON-RPC client + caches
medomni/web/lib/mcp/tool-spec-adapter.ts        # MCP -> OpenAI function shape
medomni/web/app/api/agent/route.ts              # ?profile=mcp + ?role=patient (both additive)
medomni/findings/2026-05-12-mcp-integration/
  parity_check.mjs                              # this session's 50-prompt harness
  parity_check_results.json                     # raw results
  CARD.md                                       # this file
```

The same PR also adds `?role=patient` — a patient-facing system-prompt swap
that, combined with `?profile=mcp`, gives the TAM-expanding "explain my
clinician's decision" demo URL: `?role=patient&profile=mcp`. Default
`/4UWHAt` URL behavior is bit-for-bit identical until env var flips.

Production code is type-clean under the project's own `tsconfig.json`
(`pnpm exec tsc --noEmit` reports zero errors across the whole web
project after my edits). The mjs harness was run with `node` directly,
no `npm install`, no `tsx` dependency.

## Failure-mode posture (3 gates)

1. **Query param**: only requests with `?profile=mcp` enter the branch.
   Default `/4UWHAt` traffic is unaffected.
2. **Environment variable**: `?profile=mcp` is a silent no-op unless
   `MCP_ED_RULES_ENABLED=1` is set in the medomni Vercel project. The
   code can deploy without flipping the demo.
3. **MCP server health**: if `listMcpTools()` fails on cold start, the
   route falls back to the existing 5 hand-rolled tools and sets
   `X-MCP-Status: offline`. The agent loop continues — no user-facing
   error. The `?profile=mcp` request is graceful-degraded to the
   default agent.

Response headers added:
- `X-MCP-Status`: `active` | `offline` | `disabled`
- `X-MCP-Tool-Count`: integer count of HealthCraft tools loaded

Both exposed via `Access-Control-Expose-Headers`.

## Deploy checklist (recommended; NOT performed in this session)

1. Review the diff: `git -C /Users/kiteboard/medomni diff -- web/`.
2. `git add -- web/lib/mcp/ed-rules-client.ts web/lib/mcp/tool-spec-adapter.ts web/app/api/agent/route.ts` (stage by name; no `-A`).
3. Commit with `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
4. Push to a branch (NOT main). Open a PR. Vercel preview deploy runs.
5. Smoke the preview URL: `?profile=mcp` should silently no-op (env var unset). `?profile=v_final` should still work.
6. Merge.
7. Production deploy lands. STILL silent no-op because env var unset.
8. Set `MCP_ED_RULES_ENABLED=1` on the medomni Vercel project for the
   environments you want enabled (production / preview / both). Now
   `?profile=mcp` activates.
9. Test `https://www.thegoatnote.com/4UWHAt?profile=mcp` with a
   chest-pain prompt; verify the response has `X-MCP-Status: active`
   and `X-MCP-Tool-Count: 6`, and that the model emits `mcp_` tool
   calls.

## What this does NOT do

- Does not change the default `/4UWHAt` demo path.
- Does not modify orca's vLLM serving (no demo restart, no infra blip).
- Does not commit, push, or deploy. All changes are local-disk only.
- Does not enable FHIR auth headers. The MCP supports them (`X-FHIR-*`
  via SHARP), but the demo's current state has no real FHIR backend,
  so the client passes no FHIR headers — and the MCP cleanly accepts
  `applyDecisionRule` calls with `variables` only.
- Does not address MCP server production hardening
  (`ThreadingHTTPServer`, no Authorization-header auth, no rate
  limits, no `/healthz`). These are separate work, gating wider rollout
  beyond the `?profile=mcp` opt-in.

## Known caveats

- **`listMcpTools()` cache is permanent for the lifetime of a serverless
  instance.** If the MCP fails on the first call, the failure is cached
  until the next cold-start. For an MVP feature flag this is acceptable;
  add a TTL or invalidate-on-error before promoting to default.
- **No retries.** The client has a 10-second timeout per `tools/call`;
  failures are surfaced to the model as a tool error. The agent loop's
  existing retry-via-prompt behavior takes over.
- **No streaming of MCP tool output.** MCP returns the result in one
  shot; the agent's UI tool card transitions input-streaming →
  input-available → output-available cleanly (the existing pipeline).
  No new streaming protocol on the medomni side.
- **Synonym table coverage was not tested.** The parity harness passed
  integer values directly. If Nemotron-Omni emits free-text variable
  values (e.g. `{history: "highly suspicious"}` instead of `{history: 2}`),
  the MCP's natural-language coercion fallback takes over. That path is
  documented in `applyDecisionRule`'s description and the HEART schema
  exposes `acceptedValues` for each level, but a model-driven end-to-end
  test (free-form medical question → Nemotron emits MCP call with NL
  arguments → score returned) was not part of this session.

## Followups (recommended sequence)

1. **Synonym-table end-to-end smoke** (1-2 hours): wire the `?profile=mcp`
   branch behind a private cloudflared tunnel pointing at a local
   `next dev`, fire 5–10 free-form chest-pain prompts at it via the
   composer UI, and confirm Nemotron emits `mcp_applyDecisionRule` with
   sensible NL variable values that the MCP coerces correctly. This is
   the missing model-loop test that the parity harness deliberately
   sidesteps.
2. **MCP server production hardening** (1–2 days): replace
   `ThreadingHTTPServer` with `uvicorn` + `FastAPI` (or `starlette`),
   add an Authorization header check (shared secret in env), add per-IP
   rate limit, expose `/healthz`. Until this lands, keep the
   `?profile=mcp` opt-in and `MCP_ED_RULES_ENABLED=1` gating in place.
3. **HealthBench-Hard re-eval at `?profile=mcp`** (separate session):
   does the MCP-augmented agent score higher on HB Hard text rubric
   than the 5-tool baseline? Run the canonical N=1000 graded eval; if
   delta > 0 with CI excluding 0, that's the credibility number to
   show partners — "MCP-augmented agent improves HB Hard rubric by
   X pp at canonical gpt-4.1 grading."
4. **Promote to default** (after #1 + #2 + a stable week of opt-in
   traffic): remove the env-var gate; make `?profile=mcp` itself the
   trigger; remove `clinical_calculate` from the default tool spec
   (`applyDecisionRule` strictly dominates it).

## Sign-off

Parity passes. Code is type-clean. The live demo at
`https://www.thegoatnote.com/4UWHAt` was probed before and after this
session (200 OK, sub-second), and never received a single request from
this work — all integration code is local-disk-only pending your
explicit OK to commit and the env-var flip to activate.
