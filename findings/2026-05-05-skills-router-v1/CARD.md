# Skills router v1 — wired to live `/api/agent`

**Status:** SHIPPED 2026-05-05 (iter-181). PR #1 of a 5-track Cherny-pattern initiative.
**Trigger:** Iter-181 user request — "pursue all 5 in parallel and/or sequentially. yes author PR now."
**Boris Cherny lens:** [Building Claude Code with Boris Cherny](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny). Skills > monolithic models. Markdown-driven progressive disclosure. Each skill ships in seconds via PR, not weeks via training cycle.

## What this PR ships

- **`web/lib/agent/skills.ts`** — intent classifier (keyword heuristic, v1) + skill markdown loader. Reads `web/lib/agent/skills/{system_prompt_v1,differential,calc,handoff}.md` at server cold start.
- **`web/lib/agent/skills/*.md`** — runtime build copies of the canonical skill markdown that lives in `mvp/medomni-inference/skills/`. Sync via `make sync-skills`.
- **`web/app/api/agent/route.ts`** — added `?profile=v_final` query param. When set, splices the V_final plan-then-act header + matched skill block into the system prompt before dispatching to the B300 vllm endpoint.
- **`Makefile`** — `make sync-skills` target. Single canonical authoring source (`mvp/medomni-inference/skills/`); build-time mirror into web/.

## Default behavior unchanged

Without `?profile=v_final`, the route is byte-identical to the prior path. The new skills code is dead weight on the default request flow. This is the Cherny "default off, opt-in by query param" pattern — ship the foundation safely, validate via opt-in traffic, flip default later when confidence is high.

## Activation

```bash
# Default — current behavior, untouched.
curl -X POST https://medomni.vercel.app/api/agent \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","parts":[{"type":"text","text":"What is CHA2DS2-VASc?"}]}]}'

# V_final profile — wires skills into the system prompt.
curl -X POST 'https://medomni.vercel.app/api/agent?profile=v_final' \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","parts":[{"type":"text","text":"What is the differential for shortness of breath in a 65yo with AFib?"}]}]}'
```

Server log shows `[agent] profile=v_final intent=differential systemPromptBytes=...` so the activation is visible in Vercel logs.

## Intent classification (v1, keyword heuristic)

| User text contains | Loaded skill |
|---|---|
| `score`, `calculate`, `cha2ds2`, `wells`, `perc`, `meld`, ... | `calc` |
| `order`, `consult`, `transfer`, `admit`, `discharge`, `handoff`, ... | `handoff` |
| `differential`, `ddx`, `rule out`, `what could`, ... | `differential` |
| (none) | `default` (no skill block, just V_final header) |

PR #2 (skills router with proper LLM-based intent classification) supersedes the keyword heuristic.

## How this connects to the 5-track Cherny initiative

| Track | Status | Depends on |
|---|---|---|
| **#1 Wire skills to live /api/agent** | **SHIPPED (this PR)** | — |
| #2 Skills router with LLM intent classification | next | #1 (replaces keyword heuristic) |
| #3 Auto-clinical-review on `skills/*.md` PRs | next (parallel) | #1 |
| #4 Adversarial probe daemon vs live `/api/agent` | next (parallel) | #1 |
| #5 Public skill registry UI at `/4UWHAt/skills` | next (parallel) | #1 |

#2/#3/#4/#5 author independently in parallel via subagents now that the foundation is live.

## Cherny principles applied

- **Skills > monolithic models** — each skill is a markdown file, ships in a 5-min PR. No training cycle.
- **Progressive disclosure** — load only the skill needed for THIS query (token-efficient).
- **Default-off opt-in flag** — sovereignty + prod safety preserved while the new path bakes.
- **Single source of truth** — canonical skills in `mvp/medomni-inference/skills/`, runtime mirror via `make sync-skills`.
- **Dogfooding** — public `/api/agent` route is now the same surface we test against.

## Verification

- `npx tsc --noEmit` — clean
- `make sync-skills` — round-trips canonical markdown into runtime location
- 4 skill files present in `web/lib/agent/skills/`
- Default `/api/agent` path unchanged

## Cross-references

- [`mvp/medomni-inference/`](../../mvp/medomni-inference/) — canonical authoring location for skill markdown
- [`web/lib/agent/skills.ts`](../../web/lib/agent/skills.ts) — intent classifier + skill loader
- [`web/app/api/agent/route.ts`](../../web/app/api/agent/route.ts) — profile=v_final wiring
- [Building Claude Code with Boris Cherny — Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny)
- [How Boris Uses Claude Code](https://howborisusesclaudecode.com/)
