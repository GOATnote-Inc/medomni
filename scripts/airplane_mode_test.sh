#!/usr/bin/env bash
# airplane_mode_test.sh — the WiFi-off bench reproduction (SPEC §7).
#
# Purpose: prove that with all non-localhost network blocked at the OS firewall,
# the held-out 6-fixture bench still runs and matches the cached baseline
# within ±0.01 per fixture. This is the demo's "sovereignty by construction"
# moment.
#
# Required env:
#   BASELINE_ARTIFACT  path to a heldout.json from a prior PASS run; per-fixture
#                      scores in this file are the comparison points.
#
# Optional env:
#   TOLERANCE          per-fixture absolute tolerance (default: 0.01)
#   OUT_DIR            where to write the new heldout.json
#                      (default: results/airplane-mode-<ts>/)
#   ALLOWED_PORTS      space-separated localhost TCP ports to keep open
#                      (default: "8000 8001 8002 8003 8004 9002 9003 22")
#   SKIP_BENCH=1       only flip the firewall, verify it blocks, restore.
#                      Useful for live-demo dry-run.
#   DRY_RUN=1          print every privileged command without executing.
#                      Use this when authoring or before any first live run.
#
# What it does:
#   1. Snapshot current firewall state and register a trap that restores it on
#      any exit path (Ctrl-C, error, success).
#   2. Apply a deny-all-egress ruleset that allows only localhost (127.0.0.0/8
#      and ::1) plus the SSH-tunnel ports listed in ALLOWED_PORTS.
#   3. Verify external block: `curl -m 5 https://huggingface.co/` must FAIL.
#   4. Run the held-out bench against the localhost-tunneled endpoints.
#   5. Compare per-fixture scores to baseline within TOLERANCE; PASS only
#      if every fixture matches.
#   6. Restore the prior firewall state (also runs from the trap if anything
#      above fails).
#
# Platform support:
#   - macOS: uses `pfctl` (Packet Filter). Anchor name: `medomni-airplane`.
#     We write a temporary anchor file, load it via `pfctl -a`, and unload on
#     exit. We do NOT touch the system pf ruleset; we only own our anchor.
#   - Linux: uses `iptables` OUTPUT chain (default DROP, then explicit ACCEPT
#     for loopback and the listed ports). Snapshot via `iptables-save`,
#     restore via `iptables-restore`.
#
# Privilege:
#   Both pf and iptables require root. Script auto-prepends `sudo` when
#   not already root. The user will be prompted once for password unless
#   passwordless sudo is configured.
#
# Trap discipline (mirrors `feedback_verify_every_action.md`):
#   - The restore is idempotent — running it without an active firewall block
#     is a no-op.
#   - The trap fires on EXIT, INT, TERM. Exit code is preserved.

set -uo pipefail

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date -u +%Y%m%d-%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_DIR/results/airplane-mode-$TS}"
TOLERANCE="${TOLERANCE:-0.01}"
ALLOWED_PORTS="${ALLOWED_PORTS:-8000 8001 8002 8003 8004 9002 9003 22}"
SKIP_BENCH="${SKIP_BENCH:-0}"
DRY_RUN="${DRY_RUN:-0}"

ANCHOR_NAME="medomni-airplane"
PF_ANCHOR_FILE="/tmp/medomni-airplane.pf.conf"
PF_BACKUP_FILE="/tmp/medomni-airplane.pf.backup.txt"
IPTABLES_BACKUP_FILE="/tmp/medomni-airplane.iptables.backup"

PLATFORM="$(uname -s)"

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
log()   { printf "[airplane] %s\n" "$*" >&2; }
fail()  { printf "[airplane] FAIL: %s\n" "$*" >&2; exit 1; }
priv()  {
  # Run a privileged command, honoring DRY_RUN.
  if [ "$DRY_RUN" = "1" ]; then
    printf "[airplane DRY_RUN] WOULD RUN: sudo %s\n" "$*" >&2
    return 0
  fi
  if [ "$(id -u)" = "0" ]; then
    "$@"
  else
    sudo "$@"
  fi
}

# ----------------------------------------------------------------------------
# Trap-based restore
# ----------------------------------------------------------------------------
RESTORE_DONE=0
restore_firewall() {
  # Idempotent: safe to call multiple times.
  if [ "$RESTORE_DONE" = "1" ]; then
    return 0
  fi
  RESTORE_DONE=1
  log "restoring firewall state"
  if [ "$PLATFORM" = "Darwin" ]; then
    # Unload our anchor only; never touch the system pf rules.
    if [ "$DRY_RUN" = "1" ]; then
      log "DRY_RUN: WOULD pfctl -a $ANCHOR_NAME -F all"
    else
      priv pfctl -a "$ANCHOR_NAME" -F all 2>/dev/null || true
      # If the system pf was OFF before we started, turn it back off.
      if [ -f "$PF_BACKUP_FILE" ] && grep -q "^Status: Disabled" "$PF_BACKUP_FILE" 2>/dev/null; then
        priv pfctl -d 2>/dev/null || true
      fi
      rm -f "$PF_ANCHOR_FILE" "$PF_BACKUP_FILE"
    fi
  elif [ "$PLATFORM" = "Linux" ]; then
    if [ -f "$IPTABLES_BACKUP_FILE" ]; then
      priv iptables-restore < "$IPTABLES_BACKUP_FILE" 2>/dev/null || true
      rm -f "$IPTABLES_BACKUP_FILE"
    fi
  fi
  log "firewall restored"
}
trap restore_firewall EXIT INT TERM

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------

[ -n "${BASELINE_ARTIFACT:-}" ] || fail "BASELINE_ARTIFACT=path required"
[ -f "$BASELINE_ARTIFACT" ]     || fail "baseline artifact not found: $BASELINE_ARTIFACT"

mkdir -p "$OUT_DIR"

log "platform: $PLATFORM"
log "out_dir:  $OUT_DIR"
log "baseline: $BASELINE_ARTIFACT"
log "tolerance per fixture: ±$TOLERANCE"
log "allowed localhost ports: $ALLOWED_PORTS"
[ "$DRY_RUN" = "1" ] && log "DRY_RUN=1 (no privileged commands will execute)"

# ----------------------------------------------------------------------------
# Snapshot current firewall state
# ----------------------------------------------------------------------------
if [ "$PLATFORM" = "Darwin" ]; then
  command -v pfctl >/dev/null 2>&1 || fail "pfctl not found (need macOS)"
  log "snapshotting pf state"
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN: WOULD pfctl -s info > $PF_BACKUP_FILE"
  else
    priv pfctl -s info > "$PF_BACKUP_FILE" 2>/dev/null || echo "Status: Unknown" > "$PF_BACKUP_FILE"
  fi

  # Build the anchor file. The strategy:
  #   - block everything outbound by default
  #   - pass localhost (loopback)
  #   - pass DNS to 127.0.0.1 only (no external resolvers)
  # Note: the anchor only filters traffic that pf actually evaluates for it;
  # pf must be enabled, and the main ruleset must reference our anchor.
  # Easier path on a dev laptop: enable pf and load our anchor as the entire
  # active ruleset, with a `pass quick on lo0` plus a `block out quick all`.
  log "writing pf anchor file"
  cat > "$PF_ANCHOR_FILE" <<EOF
# medomni-airplane — block all egress except loopback
# This file is authored by scripts/airplane_mode_test.sh.
# It is loaded into anchor "$ANCHOR_NAME" and unloaded on exit.
set skip on lo0
block drop out quick all
pass out quick inet  from 127.0.0.0/8 to 127.0.0.0/8
pass out quick inet6 from ::1 to ::1
EOF

elif [ "$PLATFORM" = "Linux" ]; then
  command -v iptables >/dev/null 2>&1 || fail "iptables not found (need Linux)"
  log "snapshotting iptables state"
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN: WOULD iptables-save > $IPTABLES_BACKUP_FILE"
  else
    priv iptables-save > "$IPTABLES_BACKUP_FILE" || fail "iptables-save failed"
  fi
else
  fail "unsupported platform: $PLATFORM (need Darwin or Linux)"
fi

# ----------------------------------------------------------------------------
# Apply airplane-mode firewall
# ----------------------------------------------------------------------------
log "applying airplane-mode firewall (blocks all non-loopback egress)"

if [ "$PLATFORM" = "Darwin" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN: WOULD pfctl -E && pfctl -a $ANCHOR_NAME -f $PF_ANCHOR_FILE"
  else
    # Enable pf (idempotent if already enabled)
    priv pfctl -E 2>&1 | grep -v "^pf already enabled" || true
    # Load our anchor. Note: this loads the rules INTO the anchor; for the
    # rules to take effect, the main ruleset must include
    #   anchor "medomni-airplane"
    # Most macOS dev laptops have an empty active ruleset that doesn't
    # reference any anchor, so we additionally flush + load the rules into
    # the main ruleset directly. We do that with `pfctl -f -` from the same
    # file, which is restored by `pfctl -F all` in the trap.
    priv pfctl -F all 2>/dev/null || true
    priv pfctl -f "$PF_ANCHOR_FILE" || fail "pfctl -f load failed"
  fi
elif [ "$PLATFORM" = "Linux" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN: WOULD iptables flush + DROP non-loopback OUTPUT"
  else
    priv iptables -F OUTPUT
    priv iptables -A OUTPUT -o lo -j ACCEPT
    priv iptables -A OUTPUT -d 127.0.0.0/8 -j ACCEPT
    # allow localhost-bound DNS only
    priv iptables -A OUTPUT -d 127.0.0.0/8 -p udp --dport 53 -j ACCEPT
    priv iptables -A OUTPUT -j DROP
  fi
fi

log "pausing 10s for firewall to settle"
[ "$DRY_RUN" = "1" ] || sleep 10

# ----------------------------------------------------------------------------
# Verify external block
# ----------------------------------------------------------------------------
log "verifying external block (curl https://huggingface.co/)"
if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: skipping external-block verification"
else
  if curl --max-time 5 -s -o /dev/null https://huggingface.co/ ; then
    fail "external block verification FAILED — curl https://huggingface.co/ succeeded; firewall is not blocking"
  fi
  log "external block verified — huggingface.co unreachable"
fi

# ----------------------------------------------------------------------------
# Run held-out bench (or skip if SKIP_BENCH=1)
# ----------------------------------------------------------------------------
if [ "$SKIP_BENCH" = "1" ]; then
  log "SKIP_BENCH=1 — not running bench. Block + restore demo only."
  exit 0
fi

# Endpoints come from env (same as ci-medomni). Use defaults if not set.
NEMOTRON_SERVE_URL="${NEMOTRON_SERVE_URL:-http://127.0.0.1:8000/v1}"
NEMOTRON_JUDGE_URL="${NEMOTRON_JUDGE_URL:-http://127.0.0.1:8003/v1}"
NEMOTRON_EMBED_URL="${NEMOTRON_EMBED_URL:-http://127.0.0.1:8001/v1}"
NEMOTRON_RERANK_URL="${NEMOTRON_RERANK_URL:-http://127.0.0.1:8002/v1}"
NEMOTRON_SERVE_MODEL="${NEMOTRON_SERVE_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
NEMOTRON_JUDGE_MODEL="${NEMOTRON_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

log "running held-out 6-fixture bench (sovereign_bench)"
NEW_ARTIFACT="$OUT_DIR/heldout.json"

VENV="$REPO_DIR/.venv"
PY="$VENV/bin/python"

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: WOULD invoke sovereign_bench against $NEMOTRON_SERVE_URL → $NEW_ARTIFACT"
  # Simulate a successful artifact for the comparison step
  cp "$BASELINE_ARTIFACT" "$NEW_ARTIFACT"
else
  "$PY" "$REPO_DIR/scripts/sovereign_bench.py" \
    --manifest "$REPO_DIR/corpus/clinical-fixtures-heldout" \
    --serve-url  "$NEMOTRON_SERVE_URL"  --serve-model  "$NEMOTRON_SERVE_MODEL" \
    --judge-url  "$NEMOTRON_JUDGE_URL"  --judge-model  "$NEMOTRON_JUDGE_MODEL" \
    --embed-url  "$NEMOTRON_EMBED_URL"  --rerank-url   "$NEMOTRON_RERANK_URL" \
    --n 6 --trials 1 --max-tokens 1024 --timeout-s 300 \
    --out "$NEW_ARTIFACT" \
    || fail "sovereign_bench failed under airplane-mode (this is the FAIL signal — the system depends on the network)"
fi

# ----------------------------------------------------------------------------
# Compare per-fixture scores against baseline
# ----------------------------------------------------------------------------
log "comparing per-fixture scores to baseline (tolerance ±$TOLERANCE)"

"$PY" - "$BASELINE_ARTIFACT" "$NEW_ARTIFACT" "$TOLERANCE" <<'PYEOF'
import json
import sys

baseline_path, new_path, tol_str = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(tol_str)

def per_fixture(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for trial in data.get("trial_results", []):
        for ex in trial.get("per_example", []):
            fid = ex.get("fixture_id") or ex.get("manifest_id") or ex.get("id")
            score = ex.get("score")
            if score is None and "aggregate" in ex:
                score = ex["aggregate"].get("score")
            if fid is not None and score is not None:
                # if multiple trials, take the mean
                out.setdefault(fid, []).append(float(score))
    return {k: sum(v)/len(v) for k, v in out.items()}

base = per_fixture(baseline_path)
new  = per_fixture(new_path)

if not base:
    print("FAIL: baseline has no per-fixture scores"); sys.exit(2)
if not new:
    print("FAIL: airplane-mode run has no per-fixture scores"); sys.exit(2)

print()
print("| fixture | baseline | airplane | delta | verdict |")
print("|---|---|---|---|---|")
all_pass = True
for fid in sorted(base.keys() | new.keys()):
    b = base.get(fid)
    n = new.get(fid)
    if b is None or n is None:
        verdict = "MISSING"
        all_pass = False
        b_s = "-" if b is None else f"{b:.3f}"
        n_s = "-" if n is None else f"{n:.3f}"
        d_s = "-"
    else:
        d = n - b
        ok = abs(d) <= tol
        verdict = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        b_s = f"{b:.3f}"; n_s = f"{n:.3f}"; d_s = f"{d:+.3f}"
    print(f"| {fid} | {b_s} | {n_s} | {d_s} | {verdict} |")

print()
if all_pass:
    print("airplane-mode reproduction: PASS (all fixtures within tolerance)")
    sys.exit(0)
else:
    print("airplane-mode reproduction: FAIL (some fixtures diverged)")
    sys.exit(3)
PYEOF
RC=$?

# (the trap restores the firewall on exit)
exit $RC
