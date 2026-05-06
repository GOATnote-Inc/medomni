# Public Skill Registry UI — `/4UWHAt/skills`

**Date:** 2026-05-05
**Track:** #5 of the 5-track Cherny-pattern initiative.
**Predecessors:** PR #88 (V_final scaffolding), PR #95 (markdown skills router wired to `/api/agent?profile=v_final`).
**Status:** shipped via PR (this card co-lands).

## Why this exists

PR #95 made MedOmni's reasoning composable from inspectable markdown skill
modules. That's a strong technical move, but only Claude (and engineers
reading the repo) could actually see what was loaded. Clinicians deserve
the same view. Med-PaLM, MedGemma, and Hippocratic AI all ship as opaque
APIs — nobody publishes the equivalent of "here is the system prompt
your answer was reasoned with, and here is the keyword set that decided
which extra guidance got attached." This page makes that the default
public view.

## What shipped

1. **`web/app/skills/page.tsx`** — server component that renders at
   `/4UWHAt/skills` (basePath in `next.config.ts` prefixes everything
   under `/4UWHAt`). Two-column layout: left rail of skills with
   active-state highlighting via `?active=<slug>`, right pane with
   header (title, router/always-on badge, source filename, activation
   keywords as monospace chips) plus the full markdown body rendered
   through `react-markdown` + `remark-gfm`. Anonymous access — same
   posture as `/4UWHAt`.

2. **`web/lib/agent/skill-registry.ts`** — server-only loader. Reads
   `web/lib/agent/skills/{system_prompt_v1,differential,calc,handoff}.md`
   at request time via `fs.readFileSync`. Tiny YAML-ish frontmatter
   parser strips the `name`/`description`/`trigger` block (the three
   skill files all have one). Triggers come from the canonical lowercase
   keyword arrays in `skills.ts` for routable skills, with the
   frontmatter as a fallback. The runtime classifier and the displayed
   keyword chips are the same data, by construction.

3. **`web/components/skills/SkillMarkdown.tsx`** — thin client wrapper
   around `react-markdown` so the page itself can stay a server
   component. Typography classes mirror `MarkdownAnswer` in
   `web/app/agent/page.tsx` so a clinician sees skill bodies in the
   same visual register as agent answers.

4. **`web/app/api/agent/route.ts`** — added an `X-Active-Skill: <intent>`
   response header (and an `Access-Control-Expose-Headers` peer) when
   `?profile=v_final` is in use. The header is suppressed on the
   default profile so it can't be misread as "the router fired" when
   it didn't. The frontend can read this on the chat response to
   light up the matching card on `/skills` (out of scope for this PR;
   the header is the contract).

## Inspectability story

A clinician asks "calculate the CHA2DS2-VASc on a 72F with hypertension."
Under `?profile=v_final`:

1. The router's keyword classifier in `web/lib/agent/skills.ts` matches
   "calculate" + "cha2ds2" against `CALC_KEYWORDS` and returns intent
   `calc`.
2. `buildVFinalSystemContent` splices `web/lib/agent/skills/calc.md`
   into the system prompt below the always-loaded `system_prompt_v1.md`.
3. The model dispatches with that composed prompt; the response stream
   carries `X-Active-Skill: calc`.
4. The clinician opens `/4UWHAt/skills?active=calc` and reads the exact
   guidance the model was steered with — the same bytes that shipped
   on disk, no rephrasing.

## Constraints honored

- `npx tsc --noEmit` from `web/` — clean.
- `npx eslint` on changed files — clean.
- `npx next build` — `/skills` builds, listed as a server-rendered route.
- No new SDK or large dep — `react-markdown` and `remark-gfm` were
  already in `web/package.json` (used by `web/app/agent/page.tsx`).
- `vercel.json` untouched. No edits to frozen paths under
  `mvp/911-console-live/`, `app/prism42-v3/`, ElevenLabs, LiveKit, DNS,
  or the `data/seed_kg/` corpus (per `CLAUDE.md` §1, §7).
- Skill files under `web/lib/agent/skills/` not modified — this PR is
  read-only against the runtime skill surface.

## Trust-through-transparency rationale

Clinicians (Brandon's target user — RN/NP/PA/MD frontline) don't trust
black-box reasoning. They trust documented protocols. A skill registry
is a protocol document the agent provably uses (because it's the same
file the runtime reads). Showing it publicly at `/4UWHAt/skills` makes
the agent's reasoning auditable in the same sense a clinical pathway
poster on a wall is auditable: anyone can read it, point to it, and
challenge it. That's a posture no closed-API medical LLM offers.

## Forward work (out of scope here)

- Frontend reads `X-Active-Skill` from the streaming agent response and
  pulses the matching card in the registry pane.
- Per-skill version pin and changelog (so a clinician sees "this skill
  hasn't changed in 30 days" or "new in v0.3").
- Optional skills filter on the registry by intent class
  (diagnostic / calculator / handoff / always-on).
