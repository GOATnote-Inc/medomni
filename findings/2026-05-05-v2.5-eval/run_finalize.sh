#!/bin/bash
# Phase D: stats + leakage + manifest + report.
set -e
REPO=/Users/kiteboard/medomni/.claude/worktrees/ship-rule-eval
EVAL_DIR=$REPO/findings/2026-05-05-v2.5-eval
PY=/Users/kiteboard/medomni/.venv/bin/python3

cd "$REPO"

echo "[$(date +%H:%M:%S)] stats"
"$PY" scripts/ship_rule_eval.py stats --eval-dir "$EVAL_DIR"

echo "[$(date +%H:%M:%S)] leakage"
# Use V1 train corpus as the train-corpus reference for leakage scan.
# Pull from lobster: /workspace/data/v1_train.jsonl is the canonical training file.
mkdir -p "$EVAL_DIR/leakage-input"
rsync -avz evil-cyan-lobster:/workspace/data/v1_train.jsonl "$EVAL_DIR/leakage-input/v1_train.jsonl" 2>&1 | tail -2
"$PY" scripts/ship_rule_eval.py leakage \
  --eval-dir "$EVAL_DIR" \
  --train-jsonl "$EVAL_DIR/leakage-input/v1_train.jsonl"

echo "[$(date +%H:%M:%S)] manifest"
"$PY" scripts/ship_rule_eval.py manifest \
  --eval-dir "$EVAL_DIR" \
  --adapter-path /tmp/.no_adapter_local

echo "[$(date +%H:%M:%S)] report"
"$PY" scripts/ship_rule_eval.py report --eval-dir "$EVAL_DIR"

echo "[$(date +%H:%M:%S)] DONE — files in $EVAL_DIR:"
ls -la "$EVAL_DIR/"
