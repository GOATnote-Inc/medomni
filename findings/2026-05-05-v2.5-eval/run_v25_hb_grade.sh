#!/bin/bash
# Wait for V2.5 HB seed file to reach 200 records, then grade.
set -e
SEED=$1
[ -z "$SEED" ] && { echo "usage: $0 <seed>"; exit 2; }
EVAL=/Users/kiteboard/medomni/.claude/worktrees/ship-rule-eval/findings/2026-05-05-v2.5-eval
PY=/Users/kiteboard/medomni/.venv/bin/python3
SCRIPTS=/Users/kiteboard/medomni/.claude/worktrees/ship-rule-eval/scripts

GEN_LOCAL="$EVAL/gen-laptop/healthbench-hard__v25__seed${SEED}.jsonl"
OUT="$EVAL/graded/healthbench-hard__v25__seed${SEED}.jsonl"

# Source key
set -a; source /Users/kiteboard/lostbench/.env; set +a

# Poll until pod has 200 records
while true; do
  N=$(ssh evil-cyan-lobster "wc -l < /workspace/v2.5-eval/gen/healthbench-hard__v25__seed${SEED}.jsonl" 2>/dev/null || echo 0)
  if [ "$N" -ge 200 ]; then
    break
  fi
  echo "[$(date +%H:%M:%S)] waiting for hb v25 seed=$SEED gen ($N/200)"
  sleep 60
done
# Pull
rsync -az "evil-cyan-lobster:/workspace/v2.5-eval/gen/healthbench-hard__v25__seed${SEED}.jsonl" "$GEN_LOCAL"
echo "[$(date +%H:%M:%S)] pulled $GEN_LOCAL"
echo "[$(date +%H:%M:%S)] grading hb v25 seed=$SEED"
"$PY" "$SCRIPTS/grade_hb_parallel.py" "$GEN_LOCAL" "$OUT" 2>&1
