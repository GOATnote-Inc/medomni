# Claude Code `.claude/agents/` — spec, best practices, and medomni recommendation

**Date:** 2026-05-22
**Researched by:** `claude-code-guide` subagent (backgrounded; tools: Bash + Read + WebFetch + WebSearch), with sources verified against Anthropic docs.
**Context:** Issue #396's audit grades `.claude/agents/ with subagent defs` as a separate item (currently **FAIL** after PR #404). This brief informs the (a)/(b)/(c) decision: ship a real subagent, update the grader, or ship a minimum-real subagent.

**Bottom-line recommendation:** **(b) — update the grader, with documented reason.** Two specific candidates (`eval-harness-driver`, `iter-collapse-diagnostician`) would genuinely earn their slot if/when their workflow is actually invoked, but they ship as part of the workflow that uses them — not preemptively to flip a checkbox.

---

## 1. Current spec for `.claude/agents/<name>.md`

Subagents are Markdown files with YAML frontmatter in `.claude/agents/` (project), `~/.claude/agents/` (user), or via `--agents` CLI flag (session). [Canonical docs](https://code.claude.com/docs/en/sub-agents).

**Required frontmatter:**
- `name` (lowercase + hyphens) — unique identifier; used in delegation and hooks.
- `description` — when Claude should delegate; Claude matches the task against this, so be explicit about triggers.

**Optional frontmatter (as of v2.1.147):**
- `tools` (comma-separated allowlist) / `disallowedTools` (denylist, applied before allow).
- `model` (`sonnet` / `opus` / `haiku` / full ID like `claude-opus-4-7` / `inherit`; default `inherit`).
- `permissionMode` (`default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan`).
- `maxTurns`, `skills` (preloaded), `mcpServers` (inline/ref'd; main-thread-enabled since 2.1.117), `hooks` (PreToolUse/PostToolUse/Stop; main-thread-enabled since 2.1.117).
- `memory` (`user` / `project` / `local`) — persistent cross-session dir at `~/.claude/agent-memory/<name>/` etc.
- `background` (boolean), `effort` (`low`-`max`), `isolation: worktree`, `color` (UI), `initialPrompt`.

**Body** is the system prompt (only this + task context are passed to the subagent — *not* the main Claude Code prompt).

**Invocation:** automatic on `description` match, explicit via `@agent-<name>`, session-wide via `--agent <name>` or `.claude/settings.json` `"agent": "<name>"`.

## 2. Decision rule — skill vs hook vs subagent

- **Skill** (`.claude/skills/*/SKILL.md`): runs in main conversation context. Domain knowledge / reusable prompt. Consumes main window. Trigger: explicit `/skill-name` or auto-discover via Skill tool.
- **Hook** (`.claude/hooks/` + `settings.json`): shell script that validates or transforms tool calls before/after execution. Zero context cost. Best for enforcement (e.g., the `git add -A` guard PR #404 just shipped). Triggers: PreToolUse / PostToolUse / SessionStart / etc.
- **Subagent** (`.claude/agents/`): isolated context window, custom system prompt, restricted tool set, independent permissions, optional background. Best when (a) task produces verbose output you don't want polluting main, (b) tool restrictions matter for safety, (c) work is self-contained and returns a summary.

**Boundaries:**
- Task floods context with noise? → subagent.
- Task is a guard rail? → hook.
- Task is a reusable template across projects? → user-scoped skill or plugin.
- One-time sequence in this repo's workflow? → inline or project-scoped skill.

## 3. Recent changes (Feb–May 2026)

- **v2.1.117** — `mcpServers` and `hooks` in agent frontmatter now load for *main-thread* agents (`claude --agent`). PostToolUse hooks can replace tool output via `hookSpecificOutput.updatedToolOutput` (not just MCP).
- **v2.1.143–147** — `subagent_type` matching now case- and separator-insensitive (e.g., "Code Reviewer" → `code-reviewer`). Skills can reference `${CLAUDE_EFFORT}`. Background sessions are pinned: stay alive when idle, preserve model/effort on wake.
- **v2.1.147** — `/simplify` renamed to `/code-review` with new correctness-focused behavior (no longer cleanup-and-fix).

**No breaking changes to agent YAML syntax between Feb–May 2026.** The spec is stable.

## 4. Official examples (URLs)

1. **Code Reviewer** (read-only): https://code.claude.com/docs/en/sub-agents#code-reviewer — `tools: Read, Grep, Glob, Bash` (no Edit/Write), focused review-checklist system prompt.
2. **Debugger** (analysis + fix): https://code.claude.com/docs/en/sub-agents#debugger — `tools: Read, Edit, Bash, Grep, Glob`, multi-step workflow (capture → diagnose → fix → verify).
3. **Database Query Validator**: https://code.claude.com/docs/en/sub-agents#database-query-validator — shows PreToolUse hook integration for conditional validation.
4. **Agent SDK programmatic equivalent**: https://code.claude.com/docs/en/agent-sdk/overview — `AgentDefinition` in Python/TS.

## 5. Relationship to the Claude Agent SDK

**Distinct but parallel.** Schema is shared (`name`, `description`, `tools`, `model`, `hooks`, `mcpServers`, …) but the artifacts are not interchangeable:

- Claude Code: Markdown + YAML on filesystem; CLI-loaded.
- Agent SDK: programmatic `AgentDefinition` (Python/TS class or dict); created in code; passed to `query()` or `ClaudeAgentOptions`.

A `.claude/agents/code-reviewer.md` cannot be loaded directly by the SDK — you rewrite it as Python/TS. The conversion is mechanical (YAML → dict). SDK agents *can* spawn subagents (since 2.1.147) via the `Agent` tool, using either SDK-defined agents or agents loaded from `.claude/agents/` on the filesystem the SDK runs in.

## 6. Per-candidate evaluation for medomni

| Candidate | Earn its slot? | Reasoning |
| --- | --- | --- |
| `clinical-reasoning-judge` (single-shot rubric judgment, severity-tagged) | **NO** | Single-shot analysis belongs in main context. A PostToolUse hook on Edit/Write to auto-invoke judgment is more appropriate; or fold into CLAUDE.md as a skill. |
| `eval-harness-driver` (HealthBench-Hard / OpenEM, returns pass^k) | **YES** | High-volume JSON output, many tool calls, 15–40 min runtime. `background: true` + `tools: [Bash, Glob, Read, Write]` + `maxTurns: 50` isolates noise, returns summary. |
| `web-build-reviewer` (`next build` + `tsc --noEmit`, summarize type errors) | **MARGINAL** | PR #401 already added `web-build` as CI. Only additive if it does *deeper* analysis (categorize errors, suggest fixes, layer ESLint / complexity / unused-exports). Otherwise hook or skill suffices. |
| `safety-engineer-review` / `clinical-skill-review` | **NO** | Already in CI workflows. Duplicating them as subagents is ceremony — CI runs on every PR, subagents run on-demand, context differs. If pre-commit speed is the goal, use hooks or skills. |
| `iter-collapse-diagnostician` (distribution shape, negative controls, variance retention; reusable across iters) | **YES** | Domain-specific, computationally expensive, produces tables/plots, runs infrequently but multiple times across the year. `background: true`, `model: opus`, `effort: high`. Reusable template — the work this brief's parent PR is doing for iter=1 will be done again for iter=2, iter=3. |

## 7. Honest recommendation

**Adopt (b) — update the grader with documented reason.** medomni is a single-developer clinical-AI demo; only **two of five** candidate subagents (`eval-harness-driver`, `iter-collapse-diagnostician`) actually earn their slot, and both should ship as part of the workflow that uses them, not preemptively. The other three are either already-in-CI (safety / clinical-skill review), already-handled-by-hook (clinical judgment), or not-yet-specified (web-build deep analysis).

**Concretely:**

1. Keep `.claude/agents/` empty for now. The audit item 3 stays FAIL with a documented reason: *"No subagents committed; subagents will be added if/when a workflow genuinely benefits from isolated context (current candidates: `eval-harness-driver`, `iter-collapse-diagnostician` — both deferred until first real invocation)."*
2. Update the audit grader (whenever it's located) to accept this FAIL-with-reason. The grader as written conflates *presence* with *load-bearing* — that is exactly the m13v critique that triggered PR #404; the right structural answer is to teach the grader to read the reason.
3. If/when `iter-collapse-diagnostician` is invoked for iter=1 (i.e., the diagnostic from `ITER1_DIAGNOSTIC_PREREGISTRATION.md` runs as a subagent rather than inline), commit it then. Same for `eval-harness-driver` on the next full eval pass.

**(c) fallback — minimum-real subagent if we want to ship one anyway:**

```markdown
---
name: eval-harness-driver
description: Run evaluation harness on HealthBench-Hard and OpenEM subsets, return pass^k metrics and safety gate status. Use when checking model performance against the evaluation suite or validating clinical safety gates.
tools: Bash, Glob, Read, Write
model: inherit
maxTurns: 50
background: true
effort: high
---

You are an evaluation-harness expert. Your job is to run medomni's evaluation suite, parse results, and report metrics.

When invoked:
1. Determine the evaluation subset (HealthBench-Hard, OpenEM, or both) from the task.
2. Run `make eval SUBSET=<name>` in the medomni root.
3. Wait for completion (15–40 minutes).
4. Parse the output JSON from `evaluation/results/<timestamp>/`.
5. Extract: pass^1, pass^3, pass^5, safety_gate_status, top failure modes.
6. Write a concise markdown summary to `findings/eval-<timestamp>.md`:
   - Metrics table (pass^k by category)
   - Safety gate: PASS / FAIL + threshold
   - Top 3 failure modes + frequency
   - Recommendation: ship / iterate / blocked

Do not modify test code or configs. If evaluation fails, capture the error and suggest root cause (dep, data, timeout, API key).
```

This subagent is shippable; it is not shipped *now* because there is no pending eval pass that would invoke it in this PR.

---

## Sources

- [Claude Code Subagents — canonical docs](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Changelog — May 2026](https://code.claude.com/docs/en/changelog)
- [Agent SDK Overview](https://code.claude.com/docs/en/agent-sdk/overview)
- Secondary references (third-party, sanity-check only — *not* primary):
  - https://nimbalyst.com/blog/claude-code-subagents-guide/
  - https://www.cloudzero.com/blog/claude-code-agents/
