#!/usr/bin/env python3
"""PreToolUse hook: refuses `git add -A` and `git add .` in medomni.

Medomni hard rule (see `CLAUDE.md` §0 and the user's persistent memory).
Three documented incidents in this user's history traced to a sweep-stage
step that swept evaluation artifacts or agent-owned files into a commit;
recovery in each case relied on `git reflog`. Staging by name is the only
safe pattern in this repo.

This hook exists because m13v's comment on issue #396 is right: depending
on the model to read a §0 line every turn and then choose to obey it is
strictly weaker than refusing the tool call at the hook layer. A §0
marker that fires only when read is hopeful; a hook that fires
unconditionally is enforcement.

Reads `{"tool_input": {"command": "..."}, ...}` from stdin. Exits:
  0 — allow
  2 — block; stderr is surfaced to the agent
"""

import json
import re
import sys

# `git add` followed by an offending arg:
#   `-A`           → all changes
#   `--all`        → all changes
#   `.`            → working-dir sweep (must be a standalone arg, not the
#                    leading `.` of a path like `./web/foo.ts`)
# The leading-context group catches the command whether it starts the line
# or follows a separator (`;`, `&&`, `|`, `&`). The trailing lookahead
# requires the offending arg itself to end at whitespace, end-of-string,
# or a separator — so `git add ./path` and `git add -Apath` do NOT match.
_OFFENDING = re.compile(r"(?:^|[\s;|&])git\s+add\s+(?:-A|--all|\.)(?=\s|$|[;|&])")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        # Malformed hook input — don't block; let the tool through and
        # let other guardrails handle whatever is going on.
        return 0

    cmd = (payload.get("tool_input") or {}).get("command", "")
    if not isinstance(cmd, str):
        return 0

    if _OFFENDING.search(cmd):
        sys.stderr.write(
            "BLOCKED by .claude/hooks/pretool-git-add-guard.py: "
            "`git add -A` / `git add --all` / `git add .` is forbidden in "
            "medomni (CLAUDE.md §0). Stage files by name. Evaluation and "
            "agent artifacts in this repo are immutable, and a sweep-stage "
            "step has swept them into commits three times in this user's "
            "history.\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
