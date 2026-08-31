# `.claude/` — agent operating layer

This directory holds the medomni agent operating layer:

- **`settings.json`** — hooks that enforce hard rules at the tool layer.
  Currently: a PreToolUse hook on `Bash` that blocks `git add -A`,
  `git add --all`, and `git add .`.
- **`hooks/`** — hook scripts referenced by `settings.json`.
- **`skills/`** — load-on-demand context. Each skill is a directory with
  a `SKILL.md` whose frontmatter `description` is the trigger; the
  skill's body is loaded only when the trigger fires. Use this for
  content that is important when applicable but does not earn the
  every-turn cost of `CLAUDE.md §0`.

## Why this split (issue #396)

@m13v's comment on
[issue #396](https://github.com/GOATnote-Inc/medomni/issues/396) named
the precise problem:

> the part these binary checklist audits miss is that 'CLAUDE.md with §0
> hot-path markers PASS' grades presence, not weight. a §0 block fires
> on every turn whether the model reads it or not, so once it grows
> past the size where each line earns its slot, it stops behaving like
> a hot-path marker and starts behaving like context that bloats every
> session. the 13-item grader tells you the scaffold exists; the harder
> question is which lines fire every turn, which Claude consistently
> ignores, and which §0 entries belong in a skill that loads on demand.

That is exactly correct for medomni's inherited §0. The original §0
was ~22 lines of RunPod PTY-echo guidance, fired on every turn,
applying to a serving stack (the H100 serving pod, since decommissioned) that ran on **Brev**
(direct-ssh — no PTY-echo issue). It was paying token cost on every
session for content that almost never applied to medomni at all.

The split this PR introduces:

- **§0 (in `CLAUDE.md`)** stays small. Hard budget: ≤15 effective
  lines, justify-or-evict. Each entry is something this session would
  otherwise violate. Reviewed each time it is edited; entries replace,
  they do not accrete. `Last audited:` timestamp at the top makes
  drift visible.
- **`skills/`** holds the rest — full content available when relevant,
  zero cost when not. Triggers are described in each `SKILL.md`
  frontmatter; the Claude Code runtime loads the body on a matching
  trigger.
- **`hooks/`** holds rules that should be *enforced*, not *read*. The
  `git add -A` rule belongs here because depending on the model to
  read and remember a §0 line is strictly worse than a hook that
  refuses the tool call. Hopeful reading vs. unconditional refusal.

## Open follow-ups (same disease, scoped separately)

These sections of `CLAUDE.md` have the same problem as §0 and need the
same treatment, but they are scoped to follow-up PRs to keep each
review small:

- **§1** — 8 NEVERs about prism42 prod surface, retired pods, voice
  freeze. Most reference assets specific to `prism42`
  (the prism42 prod console URL (in `CLAUDE.md` §1), ElevenLabs, LiveKit, B300 voice pod),
  not medomni. Migration target: `.claude/skills/prism42-isolation/`
  with §1 reduced to a one-line pointer.
- **"What this repo is"** (lines 40-50) — stale: says TensorRT-LLM +
  Llama-3.1-Nemotron-70B + H200; the live reality is vLLM +
  Nemotron-3-Nano-Omni-30B + H100. Rewrite required.
- **§6 Session re-entry** — references prism42-specific session-state
  files (`/tmp/prism42-nemotron-med-session/...`) that have never
  existed for medomni. Drop or rewrite.

The audit-bot in issue #396 will not catch these. That is precisely
m13v's point: the fix is structural, not metric-driven.

## Adding a new skill

A skill is a directory with a `SKILL.md`:

```
.claude/skills/<short-name>/SKILL.md
```

The frontmatter must specify `name` and a `description` that names the
trigger condition (a Claude Code runtime concern — be specific so the
body loads only when actually relevant). The body is markdown; it can
reference memory keys, repo paths, and other skills.

Keep skills focused — one topic per skill. If a skill grows past
~200 lines, that is a hint to split it.

## Adding a new hook

Hooks live in `.claude/hooks/`. Each is a script (Python, bash) wired
through `settings.json`. The hook receives the tool payload as JSON on
stdin and signals via exit code:

- `0` — allow the tool call
- `2` — block the tool call; stderr is surfaced to the agent

Keep hooks fast (they run on every matching tool call) and side-effect
free. If the hook needs to mutate anything, that is a signal to use a
different mechanism.
