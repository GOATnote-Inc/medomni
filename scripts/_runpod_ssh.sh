#!/usr/bin/env bash
# RunPod ssh-proxy wrapper.
#
# Why this exists: the RunPod proxy at ssh.runpod.io requires an interactive
# PTY for any command exec; non-interactive `ssh runpod-prism 'cmd'` returns
# "Error: Your SSH client doesn't support PTY". The Claude Code Bash tool
# always redirects stdin from /dev/null, which suppresses PTY allocation
# even with -tt.
#
# Workaround: use `script` to allocate a fake PTY, then feed the remote
# command on stdin. This works because the remote shell reads our stdin
# as if a user were typing into it.
#
# Usage:
#   bash scripts/_runpod_ssh.sh 'echo hi; nvidia-smi -L'
#   bash scripts/_runpod_ssh.sh < commands.sh
#
# Output: clean stdout from the remote (banner + control codes stripped).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp /tmp/runpod_ssh.XXXXXX.log)"
trap 'rm -f "$TMP"' EXIT

if [[ $# -eq 0 ]]; then
    # Read command from stdin
    cmd="$(cat)"
else
    cmd="$1"
fi

# ---------------------------------------------------------------------------
# DURABLE SECURITY GUARD — added 2026-04-29 after the HF_TOKEN PTY-echo leak.
#
# The RunPod proxy at ssh.runpod.io ECHOES STDIN TO STDOUT (PTY default
# behavior). Any secret pushed via this wrapper — base64 or otherwise —
# will be mirrored to the conversation transcript and to Claude Code task
# output JSONL files. PTY echo is a server-side property; we cannot turn
# it off from the client.
#
# Hard rule: refuse to forward any command whose body contains a secret-
# shaped string. Force the user to provision secrets via:
#   (a) RunPod console "Pod Environment Variables" UI, OR
#   (b) the user's own interactive ssh session (separate from this wrapper).
#
# False positives are MUCH cheaper than a leak. If a legitimate command
# happens to match, refactor to read the value from a file the user wrote
# out-of-band (cat /workspace/.secrets/<name>), not to interpolate it.
# ---------------------------------------------------------------------------
secret_patterns='HF_TOKEN=|HUGGINGFACE_HUB_TOKEN=|HUGGINGFACEHUB_TOKEN=|API_KEY=|APIKEY=|SECRET=|PASSWORD=|TOKEN=|hf_[a-zA-Z0-9]{20,}|nvapi-[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|sk-ant-[a-zA-Z0-9]{20,}|xai-[a-zA-Z0-9]{20,}|AIza[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{20,}|ghs_[a-zA-Z0-9]{20,}'
if printf '%s' "$cmd" | grep -qE "$secret_patterns"; then
    echo "REFUSED: _runpod_ssh.sh detected secret-shaped string in command body." >&2
    echo "" >&2
    echo "RunPod proxy ssh ECHOES STDIN TO STDOUT (PTY default). Pushing a" >&2
    echo "secret through this wrapper leaks it to the transcript + log files." >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  1. Set the secret as a Pod Environment Variable in the RunPod console" >&2
    echo "     (https://console.runpod.io -> pod -> Edit -> Environment Variables)." >&2
    echo "     The proxy does NOT echo console-set env vars." >&2
    echo "  2. Provision the secret from your own interactive ssh session, e.g.:" >&2
    echo "     ssh -F configs/ssh_runpod.conf runpod-prism" >&2
    echo "     # then in that session:" >&2
    echo "     printf '%%s' '<secret>' > /workspace/.secrets/<name>" >&2
    echo "     chmod 600 /workspace/.secrets/<name>" >&2
    echo "  3. If your command needs to READ a secret, refactor to dereference" >&2
    echo "     the value on-pod, never interpolate it locally:" >&2
    echo "     cmd='val=\$(cat /workspace/.secrets/<name>) && use \$val'" >&2
    echo "" >&2
    echo "If this is a false positive (e.g. a literal string 'TOKEN=' inside" >&2
    echo "documentation), set RUNPOD_SSH_ALLOW_SECRET_GREP_BYPASS=1 to override." >&2
    if [[ "${RUNPOD_SSH_ALLOW_SECRET_GREP_BYPASS:-0}" != "1" ]]; then
        exit 2
    fi
    echo "WARNING: bypass enabled, proceeding despite secret-pattern match." >&2
fi

# Wrap with markers so we can strip the banner reliably.
wrapped="echo '<<<RUNPOD_BEGIN>>>'; { $cmd
}; echo '<<<RUNPOD_END>>>'; exit 0"

/usr/bin/script -q "$TMP" /usr/bin/ssh \
    -F "$REPO/configs/ssh_runpod.conf" \
    -tt \
    -o ConnectTimeout=20 \
    -o ServerAliveInterval=15 \
    runpod-prism <<<"$wrapped" >/dev/null 2>&1

# Strip control codes, extract content between markers.
sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g; s/\r//g' "$TMP" \
  | awk '/<<<RUNPOD_BEGIN>>>/{f=1; next} /<<<RUNPOD_END>>>/{f=0} f'
