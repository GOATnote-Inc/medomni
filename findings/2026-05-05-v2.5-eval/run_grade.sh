#!/bin/bash
# Laptop-side: pull pod gen output, source canonical .env, run gpt-4.1 grading.
# Per task: never `cat` the .env; use `set -a && source && set +a`.
set -e
REPO=/Users/kiteboard/medomni/.claude/worktrees/ship-rule-eval
EVAL_DIR=$REPO/findings/2026-05-05-v2.5-eval
PY=$REPO/../../../.venv/bin/python3
[ -x "$PY" ] || PY=/Users/kiteboard/medomni/.venv/bin/python3

mkdir -p "$EVAL_DIR/gen-laptop" "$EVAL_DIR/graded"

echo "[$(date +%H:%M:%S)] pulling pod gen output to laptop"
rsync -avz --exclude '_*' evil-cyan-lobster:/workspace/v2.5-eval/gen/ "$EVAL_DIR/gen-laptop/"

echo "[$(date +%H:%M:%S)] sourcing canonical .env (key not echoed)"
set -a; source /Users/kiteboard/lostbench/.env; set +a
[ -n "${OPENAI_API_KEY:-}" ] || { echo "FAIL: OPENAI_API_KEY missing"; exit 1; }

echo "[$(date +%H:%M:%S)] grader pre-flight"
"$PY" -c "
import os
from openai import OpenAI
c = OpenAI()
r = c.chat.completions.create(model='gpt-4.1',
    messages=[{'role':'user','content':'Reply: ok'}],
    max_completion_tokens=8, temperature=0.0)
text = (r.choices[0].message.content or '').strip().lower()
assert 'ok' in text, f'preflight unexpected: {text!r}'
print('preflight passed')
" || { echo "FAIL: gpt-4.1 preflight failed"; exit 1; }

# Grade each (benchmark, arm, seed) JSONL
for SEED in 42 123 7919; do
  for BENCH in medqa pubmedqa medxpertqa-text healthbench-hard; do
    for ARM in v0 v25; do
      IN="$EVAL_DIR/gen-laptop/${BENCH}__${ARM}__seed${SEED}.jsonl"
      OUT="$EVAL_DIR/graded/${BENCH}__${ARM}__seed${SEED}.jsonl"
      if [ ! -s "$IN" ]; then
        echo "[$(date +%H:%M:%S)] SKIP missing $IN"
        continue
      fi
      if [ -s "$OUT" ]; then
        # Verify line counts match
        IN_N=$(wc -l <"$IN")
        OUT_N=$(wc -l <"$OUT")
        if [ "$IN_N" -eq "$OUT_N" ]; then
          echo "[$(date +%H:%M:%S)] SKIP existing $OUT (n=$OUT_N)"
          continue
        fi
      fi
      echo "[$(date +%H:%M:%S)] grading $BENCH $ARM seed=$SEED"
      "$PY" "$REPO/scripts/ship_rule_eval.py" grade \
        --benchmark "$BENCH" \
        --in "$IN" --out "$OUT"
    done
  done
done

echo "[$(date +%H:%M:%S)] all grading done"
ls -la "$EVAL_DIR/graded/" | tail -25
