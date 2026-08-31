---
name: runpod-ssh-safety
description: Use when the user mentions RunPod, ssh.runpod.io, or any flow that pushes secrets through a RunPod ssh session. Covers the PTY-echo failure mode and the 2026-04-29 HF_TOKEN leak. Not applicable to Brev or direct-ssh hosts — medomni's serving GPU is Brev, so this is medomni-rare and lives here rather than in CLAUDE.md §0.
---

# RunPod ssh safety

## When this applies

Only when the operation actually touches RunPod's ssh proxy
(`ssh.runpod.io`). medomni's serving GPU is the Brev pod
the H100 serving pod, which used direct ssh with no PTY-echo issue (see
memory `brev_ssh_direct_access.md`). So in the typical medomni session
none of this applies — that is exactly why it lives in a load-on-demand
skill rather than `CLAUDE.md §0`.

## The failure mode

The RunPod proxy at `ssh.runpod.io` requires PTY allocation for command
execution. PTYs echo stdin to stdout by default — server-side behavior
that cannot be disabled from the client. Any secret pushed via:

- heredoc (`<< EOF` with a secret inside)
- base64-encoded stdin pipe
- inline-env (`KEY=value ssh ...`)
- stdin redirect (`< secrets-file`)

…through this proxy is mirrored to the conversation transcript and to
Claude Code task-output JSONL files at
`/private/tmp/claude-*/tasks/<task-id>.output`. Those files persist
across sessions.

## 2026-04-29 incident

During a Phase 2.2 build accessed through the RunPod proxy, an `HF_TOKEN`
was pushed in base64-heredoc form. The token surfaced verbatim in the
task-output file. Rotated within minutes per the prism42 2026-04-27
precedent.

## Durable mitigations

- **`scripts/_runpod_ssh.sh`** carries a hard secret-grep guard: regex
  blocklist on `HF_TOKEN`, `hf_*`, `nvapi-*`, `sk-*`, `sk-ant-*`,
  `xai-*`, `AIza*`, `ghp_*`, `ghs_*`, plus generic
  `(API_KEY|SECRET|PASSWORD|TOKEN)=` patterns. The script refuses to
  forward when matched.
- **Provision secrets** via the RunPod console's Pod Environment
  Variables UI, or via the user's own interactive ssh session (separate
  from any Claude-driven shell).
- **Never read `.env` values into a shell variable** that gets
  interpolated into a `_runpod_ssh.sh` command body. Use
  `awk -F= '/^KEY=/ {print $2}' .env` and pipe to a file, not to a
  variable.
- The guard's bypass env var
  (`RUNPOD_SSH_ALLOW_SECRET_GREP_BYPASS=1`) exists for false-positive
  recovery only. Never use it to push a real secret.

## Related memory

- `feedback_runpod_proxy_pty_echo.md` (the general lesson)
- `feedback_no_secret_value_dumps.md` (related `systemctl show` and
  `ps -ef | grep docker` failure modes)
- `brev_ssh_direct_access.md` (why this skill is medomni-rare —
  the production GPU uses Brev, not RunPod)
