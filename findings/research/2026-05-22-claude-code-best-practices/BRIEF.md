# Claude Code best practices — solo-developer agentic engineering reference

**Date:** 2026-05-22
**Source:** Compiled by a backgrounded `claude-code-guide` subagent (Bash + Read + WebFetch + WebSearch against current Anthropic docs, engineering blog, and recent Boris Cherny public materials).
**Scope:** Standing reference for solo-dev agentic engineering. The prior brief at `findings/research/2026-05-22-claude-agents-best-practices/BRIEF.md` was narrower (sub-agents only); this one is broader and is the canonical pointer.

## One honest disagreement, recorded up-front

Section 4 below lists *"Trying to prevent `git add -A` via hook (race condition; use permissions instead)"* as an anti-pattern. The medomni PR #404 hook (`.claude/hooks/pretool-git-add-guard.py`) does exactly this — and its 7-case smoke test passes cleanly without observable races. The user-global `~/.claude/hooks/block-bash-unsafe.sh` has been running the same pattern for weeks without issue. The brief's recommendation to use `permissions.deny` is a valid alternative, but the "race condition" characterization is not supported by either of our working hooks. Documented here for future-self: if a race manifests in practice, switch to permissions; until then, the hook stands.

This disagreement is left visible rather than papered over — that's the discipline floor.

---

## 1. Worktrees: isolation and parallelism

**Pattern:** each worktree is a separate git checkout on its own branch, sharing the `.git` directory with the parent repo.

**When to use:**
- Running 3–5 concurrent Claude sessions (one per worktree) with independent features / fixes.
- Solo devs working on feature A in one session while fixing a bug in another (clean rollback).
- Preventing branch collisions and merge conflicts within a single machine.

**Concurrency model (2026 baseline):**
- 4–8 reliable concurrent worktrees per developer is the documented norm.
- Above that, you're bottlenecked on review (mental), not tooling.
- CI runners cannot rely on shared `.git` object-database tricks that work locally.

**Cleanup / pruning:**
- Add `.claude/worktrees/` to `.gitignore`.
- Automated cleanup on session close if no changes were made.
- Manual: `git worktree list` and `git worktree remove <path>` when orphaned.
- Monitor disk growth; directories can accumulate to multiple GB (we have one at `.claude/worktrees/agent-a0197fd9f2b827515/` consuming ~4GB in medomni).

**Common failure modes:**
- Worktrees diverging from parent if not tracking the same base branch.
- Orphaned worktrees after hard session kills (recover by `git worktree prune`).
- CI scripts assuming fresh checkouts, breaking when tested locally with shared objects.

**Boris Cherny framing:** Anthropic's own engineering teams use worktrees for the Hub-and-Spoke pattern (lead agent coordinates sub-agents across isolated worktrees).

**References:**
- https://www.claudedirectory.org/blog/claude-code-worktrees-guide
- https://code.claude.com/docs/en/common-workflows

---

## 2. Agent teams: multi-agent orchestration

**Pattern:** Experimental (requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`); one lead session spawns teammates in independent contexts with a shared task list, mailbox, and direct agent-to-agent messaging.

**Hub-and-Spoke (lead assigns tasks):**
- Lead creates team, spawns teammates, assigns / approves tasks.
- Teammates work independently; results return via `SendMessage`.
- Best for solo devs: lead stays in control, no coordination overhead.

**Flat collaboration (teammates self-claim):**
- Shared task list with pending / in-progress / completed states.
- Teammates message each other directly without going through lead.
- Requires teammate discipline; less suitable for solo devs without external oversight.

**Optimal for solo developers:**
- **3–5 teammates** is the sweet spot (beyond that, token costs and coordination overhead dominate).
- **Independent research tasks** (e.g., parallel security / performance / test reviews).
- **Feature-per-teammate split** to avoid same-file conflicts.
- **Leading agent supervises**, gives feedback, rejects plans if needed.

**Why NOT for solo devs by default:**
- Much higher token cost than subagents (each teammate = full Claude context).
- Coordination overhead; teammates can block each other on task dependencies.
- Requires explicit `/agent-teams` enable; on by default in CI / enterprise but not personal.

**References:**
- https://code.claude.com/docs/en/agent-teams
- Code w/ Claude 2026 — Simon Willison live-blog: https://simonwillison.net/2026/May/6/code-w-claude-2026/

---

## 3. Sub-agents (`.claude/agents/`) — new in 2026

**Pattern:** Markdown + YAML frontmatter; isolation within a single session via a separate context window. Built-in types: **Explore** (read-only, Haiku, fast), **Plan** (read-only, research), **general-purpose** (full capabilities).

**What's new (vs. prior brief):**
- **Forked subagents** (experimental, `CLAUDE_CODE_FORK_SUBAGENT=1`): inherit full parent context instead of starting fresh; results stay isolated; cheaper than fresh delegation.
- **Persistent memory** (`memory: user|project|local`): subagents build knowledge across sessions, write / maintain `MEMORY.md`.
- **Background execution**: subagents can run concurrently in background; auto-deny prompts (use foreground for interactive approval).
- **Worktree isolation** (`isolation: worktree`): subagent edits land on a temporary git branch, auto-cleanup if no changes.
- **`/agents` UI**: interactive subagent browser; create, edit, delete without restarting.
- **Preloaded skills** (`skills:` frontmatter): inject full skill content at startup so subagent has domain knowledge.

**Common sub-agent roles for solo devs:**
- Code reviewer (read-only, proactive after edits).
- Debugger (edit + bash, root-cause focus).
- Security auditor (read-only, grep / bash, flagged patterns).
- Test writer (edit, isolated from main context).

**Lifecycle:**
- Subagent descriptions are always in context; full content loads only when invoked.
- After auto-compaction, first 5,000 tokens of each invoked skill is re-attached; older ones may drop.
- Subagent transcripts persist independently from main conversation.

**References:**
- https://code.claude.com/docs/en/sub-agents
- https://code.claude.com/docs/llms.txt

---

## 4. Hooks: patterns, security, and lifecycle

**Events:** SessionStart, PreToolUse, PostToolUse, UserPromptSubmit, Stop, SubagentStart, SubagentStop, Notification, PermissionRequest.

**Handler types:** command (shell script), http (remote), mcp_tool (call MCP servers), prompt (single-turn Claude eval), agent (subagent-driven validation).

**Critical security implication:**
- Hooks see tool inputs including secrets in Bash commands, API keys in MCP calls.
- NEVER log hook output in transcript; use `suppressOutput: true` if sensitive.
- Use `if: "Bash(rm *)"` matcher syntax to gate only dangerous invocations.
- Pre-commit hooks that print secret values leaked 3 incidents in 2026 Q1 / Q2 (one of which is documented in our own MEMORY.md `feedback_no_secret_value_dumps.md`).

**Recommended patterns for solo devs:**
1. **PreToolUse guard** (`if: "Bash(rm -rf *|chmod 777 *)"`) to block destructive commands.
2. **PostToolUse lint** (after Edit / Write, run eslint / ruff if fails, signal to Claude).
3. **SessionStart context injection** (load environment, branch name, open PRs count).
4. **SubagentStart setup** (provision temp DB connection; SubagentStop cleanup).

**Anti-patterns (per Anthropic team):**
- Trying to prevent `git add -A` via hook (race condition; use permissions instead) — **see the disagreement note at the top of this brief; our hook works without races.**
- Hooks that timeout and leave dangling processes.
- Expecting hooks to "guide" Claude (use CLAUDE.md or skills for soft guidance; hooks are hard gates).

**New in 2026 (late May):**
- `SubagentStart` / `SubagentStop` events for lifecycle control.
- `"once": true` field to run a hook exactly once per session.
- `terminalSequence` output for OSC 777 desktop notifications.

**References:**
- https://code.claude.com/docs/en/hooks

---

## 5. Skills vs. hooks vs. sub-agents vs. slash commands — decision matrix

| **Goal** | **Choose** | **Why** |
| --- | --- | --- |
| Reusable domain knowledge (API patterns, code style) | Skill (auto-invoked) | Loads on-demand; always available; low context cost |
| Deterministic action (block `rm`, run lint) | Hook | Runs unconditionally; no human bypass; hard gate |
| One-off workflow (`/fix-issue 123`) | Skill with `disable-model-invocation: true` | Manual invocation; side effects; repeatable |
| Long exploration (research, investigation) | Subagent (Explore type) | Separate context; read-only; results summarized |
| Multi-file edits requiring strategic focus | Subagent (general-purpose or custom) | Isolated context; tool restrictions; memory |
| Quick question without history bloat | `/btw` or `/ask` | Dismissed; never enters transcript |
| Context injection per-session | Hook (SessionStart) + `additionalContext` | Runs once; feeds git status, env vars, etc. |

**Cost hierarchy (context tokens):**
1. Slash commands (built-in, tiny).
2. Skills (description always loaded; body on-invoke).
3. Hooks (small; run before main action).
4. Subagents (separate full window; expensive).

**Boris Cherny's emphasis (from "Building Claude Code" interviews):**

> "The harness matters as much as the model." Seven components: CLAUDE.md, hooks, skills, plugins, LSP integrations, MCP servers, subagents. Solo devs should invest in **CLAUDE.md clarity** (keeps context lean), **hooks for hard boundaries** (pre-commit guards), **skills for procedures** (reusable multi-step workflows).

---

## 6. Operating discipline for solo developers

**Pre-registration / lock-in pattern:**
- Write CLAUDE.md FIRST, commit it, make it the contract.
- Use `/init` to auto-generate starter, then manually prune (keep under 500 lines).
- Every CLAUDE.md line should answer: *"Would removing this cause Claude to make mistakes?"* If not, cut.

**Verify-then-claim:**
- Before a multi-hour eval or batch operation, preflight key state (env vars, API keys, database connectivity).
- Run a single-file test first; if it passes, proceed to full batch.
- Example (from our MEMORY.md): *"ALWAYS pre-flight judge API keys before multi-hour evals"* — silent 401 poisons trajectories.

**Refuse-at-hook:**
- Block `git add -A` / `git add .` in repos with artifacts via PreToolUse hook (we ship this; see the disagreement note above).
- Block `rm -rf /` via hook (circuit breaker, even in auto mode).
- Block writes to `.env`, `.git`, `.claude` via hook (protects config drift).

**Narrative consolidation:**
- One canonical CLAUDE.md per project (not scattered README, docs/INSTRUCTIONS, etc.).
- One `MEMORY.md` per long-running agent (append-only; auto-trim to 25KB).
- Never edit evaluation artifacts; treat like clinical data logs (immutable, append-only).

**Time-boxing and CI-blocking:**
- Set `maxTurns: N` in subagent frontmatter to stop runaway loops.
- Use `--stop-on` flag in batch runs: if any step fails, halt.
- Set `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` for unattended automation (prevents spawn explosions).

**Pre-mortem before pre-reg:**
- Before spawning a 5-agent team or batch of 100 parallel runs, write out failure modes:
  - *"What if one agent claims the same task as another?"* → file locking prevents this.
  - *"What if API keys leak in Bash output?"* → hook with `suppressOutput: true`.
  - *"What if network fails mid-request?"* → retry logic in wrapper script.
- Then lock criteria against each: *"I will add a PreToolUse hook to block secret dumps. I will test on 3 files first."*

**Anthropic-recommended posture for solo devs:**
- Start with one session; measure context usage (use `/statusline`).
- Parallelize only when single session is consistently > 60% full.
- Use subagents > agent teams (lower cost, simpler coordination).
- Treat hooks as safety valves, not guidance (permissions + CLAUDE.md do soft guidance).
- Review CLAUDE.md every 3–6 months; prune instructions that are now obviated by model improvements.

---

## 7. Anti-patterns (named failures, 2026)

Documented by Anthropic team in public discussions:

1. **Kitchen-sink session:** start task A, drift to task B, return to A. Context is polluted.
   - Fix: `/clear` between unrelated tasks.

2. **Correct-correct-correct loop:** same issue, corrected twice, still wrong.
   - Fix: after 2 failed corrections, `/clear` and write a better initial prompt.

3. **Over-specified CLAUDE.md:** file too long; rules get lost in noise.
   - Fix: ruthless pruning. If Claude already does it without the instruction, delete. (We just did this in PR #404 — `§0` trim from ~22 lines to 5.)

4. **Trust-then-verify gap:** plausible-looking implementation that fails edge cases.
   - Fix: always provide verification (tests, screenshots, scripts). Never ship unverified.

5. **Infinite exploration:** *"Investigate X"* without scope; Claude reads hundreds of files.
   - Fix: scope investigations narrowly or delegate to subagent in separate context.

6. **The 48-hour session:** long-running unresume-able conversation; context hoarding; outdated code state.
   - Fix: `/rename` sessions; use `--resume` or `--continue` to pick up later; treat like branches.

7. **Env-var dumps in logs:** Bash commands with `export SECRET=value` echoed to stdout.
   - Fix: use `awk -F= '/^KEY=/ {print $1, "len:", length($2)}'` to redact (this is in our MEMORY.md hard-rules block).

8. **RunPod PTY echo (legacy 2026 Q2):** base64-encoded secrets leaked via SSH proxy PTY echo.
   - Fix: use RunPod console UI or direct SSH (now fixed; recommend Brev for direct-SSH pods — this is exactly the constraint that drove `.claude/skills/runpod-ssh-safety/`).

---

## 8. Boris Cherny: public statements and framing

**Vision statement (Pragmatic Engineer interview):**
> "I imagine a world where everyone is able to program. Anyone can just build software anytime." Draws parallel to the printing press: transferred specialized capability to general population, preceded explosion of creative output.

**On agentic architecture:**
> "Agentic coding takes center stage. Best practices have emerged from enterprise teams using the tool for over a year." (Code w/ Claude 2026, Anthropic Dev Conference.)

**On the harness:**
> "The harness matters as much as the model." CLAUDE.md, hooks, skills, plugins, MCP, subagents are the load-bearing patterns.

**Published talks / blog:**
- "Building Claude Code with Boris Cherny" — Pragmatic Engineer newsletter, 2026.
- "Code w/ Claude 2026" — Anthropic Dev Conference, multiple cities, May 2026.
- Official Anthropic engineering blog on Claude Code best practices.

**Note:** Boris Cherny does not have an extensive public X / Twitter thread presence as of 2026-05-22; most guidance is in conference talks and interview format.

---

## 9. Recommended starting stack (solo dev, 2026-05-22)

```bash
# Phase 1: Foundations (Week 1)
claude /init                    # Generate CLAUDE.md
# Prune to <300 lines; keep: bash commands, style rules, testing instructions, gotchas

# Phase 2: Automation (Week 2–3)
# Add PreToolUse hook to block `git add -A` and `rm -rf /`
# Add PostToolUse hook to run linter after edits
# Enable auto mode (`/config` → Permission Mode)

# Phase 3: Parallelism (Week 4+)
# Create subagent: code-reviewer (read-only, Haiku, post-edit)
# Create subagent: debugger (edit + bash, root-cause)
# Test on small task; measure token usage (`/statusline`)

# Phase 4: Scale (when single session > 70% context)
# Spawn agent team for research, review, or feature split
# Use `--worktree feature-name` for parallel CLI sessions
# Measure; adjust team size (3–5 teammates optimal for solo)
```

---

## 10. Relationship to the prior `.claude/agents/` brief

**Overlaps:** sub-agents section (narrower prior; broader here with memory, forking, background execution).

**This brief supersedes** the prior `findings/research/2026-05-22-claude-agents-best-practices/BRIEF.md` as the canonical Claude Code standing reference. The prior brief remains valid for its narrow scope (the `.claude/agents/` (a) / (b) / (c) decision for medomni) and the (b) verdict there still stands: medomni doesn't need `.claude/agents/` until a specific subagent earns its slot.

---

## Sources

- https://code.claude.com/docs/en/best-practices
- https://code.claude.com/docs/en/sub-agents
- https://code.claude.com/docs/en/agent-teams
- https://code.claude.com/docs/en/hooks
- https://code.claude.com/docs/en/skills
- https://claude.com/blog/how-claude-code-works-in-large-codebases-best-practices-and-where-to-start
- https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny
- https://simonwillison.net/2026/May/6/code-w-claude-2026/
- Local `MEMORY.md` — hard rules from 6+ months of incident logs.

---

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
