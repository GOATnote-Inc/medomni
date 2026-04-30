#!/usr/bin/env bash
# health_check_all_pods.sh — multi-pod health probe for the MedOmni v1.0 stack.
#
# Probes every inference endpoint and every authorized pod's basic state.
# Read-only by design: does NOT touch the voice-gateway containers on H200 or
# Brev H100 (per CLAUDE.md §1). Voice pods are only checked for SSH reachability
# and `docker ps` output; their containers are never modified.
#
# Output: a single Markdown table per probe round, optionally logged to
# /var/log/medomni-health.log when run as a cron / systemd timer.
#
# Usage:
#   bash scripts/health_check_all_pods.sh                # one-shot table
#   bash scripts/health_check_all_pods.sh --watch        # repeat every 30s
#   bash scripts/health_check_all_pods.sh --json         # JSON output for piping

set -uo pipefail

# Config — endpoint URLs through laptop port-forwards
B300_OMNI_URL="http://127.0.0.1:8000/v1/models"
B300_EMBED_URL="http://127.0.0.1:8001/v1/models"
B300_RERANK_URL="http://127.0.0.1:8002/v1/models"
B300_JUDGE_URL="http://127.0.0.1:8003/v1/models"
B300_GUARD_URL="http://127.0.0.1:8004/v1/models"   # added in Phase 2.1
PRISM_RERANK_URL="http://127.0.0.1:9002/v1/models" # added in Phase 2.2 (TRT-LLM)
PRISM_JUDGE_URL="http://127.0.0.1:9003/v1/models"  # added in Phase 2.2 (TRT-LLM)

# Pod aliases (Brev CLI hostname or SSH config alias)
B300_HOST="unnecessary-peach-catfish"
H200_HOST="warm-lavender-narwhal"
BREV_H100_HOST="prism-mla-h100"
RUNPOD_PRISM_HOST="runpod-prism"
RUNPOD_PRISM_SSH_OPTS="-F configs/ssh_runpod.conf -tt"

WATCH_MODE=0
JSON_MODE=0
for arg in "$@"; do
  case "$arg" in
    --watch) WATCH_MODE=1 ;;
    --json)  JSON_MODE=1 ;;
  esac
done

probe_http() {
  local url="$1"
  local code
  code=$(curl -sf -m 4 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
  if [ "$code" = "200" ]; then
    printf "UP"
  else
    printf "DOWN(%s)" "$code"
  fi
}

probe_ssh() {
  local host="$1"
  local extra_opts="${2:-}"
  if ssh $extra_opts -o ConnectTimeout=6 -o BatchMode=yes "$host" 'echo ok' </dev/null >/dev/null 2>&1; then
    printf "UP"
  else
    printf "DOWN"
  fi
}

probe_pod_gpu_mem() {
  local host="$1"
  local extra_opts="${2:-}"
  local out
  out=$(ssh $extra_opts -o ConnectTimeout=6 -o BatchMode=yes "$host" \
    'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader' </dev/null 2>/dev/null \
    | tr -d '\r' | tail -1 || echo "?,?")
  printf "%s" "$out"
}

probe_pod_containers() {
  local host="$1"
  local extra_opts="${2:-}"
  ssh $extra_opts -o ConnectTimeout=6 -o BatchMode=yes "$host" \
    'docker ps --format "{{.Names}}" 2>/dev/null | tr "\n" "," | sed "s/,$//"' </dev/null 2>/dev/null \
    | tr -d '\r' | tr -d '\n' \
    || printf "?"
}

run_once() {
  local ts
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  if [ "$JSON_MODE" = "1" ]; then
    printf '{"ts":"%s","endpoints":{' "$ts"
    printf '"b300_omni":"%s",'    "$(probe_http "$B300_OMNI_URL")"
    printf '"b300_embed":"%s",'   "$(probe_http "$B300_EMBED_URL")"
    printf '"b300_rerank":"%s",'  "$(probe_http "$B300_RERANK_URL")"
    printf '"b300_judge":"%s",'   "$(probe_http "$B300_JUDGE_URL")"
    printf '"b300_guard":"%s",'   "$(probe_http "$B300_GUARD_URL")"
    printf '"prism_rerank":"%s",' "$(probe_http "$PRISM_RERANK_URL")"
    printf '"prism_judge":"%s"'   "$(probe_http "$PRISM_JUDGE_URL")"
    printf '},"pods":{'
    printf '"b300":"%s",'       "$(probe_ssh "$B300_HOST")"
    printf '"h200":"%s",'       "$(probe_ssh "$H200_HOST")"
    printf '"brev_h100":"%s",'  "$(probe_ssh "$BREV_H100_HOST")"
    printf '"runpod_prism":"%s"' "$(probe_ssh "$RUNPOD_PRISM_HOST")"
    printf '}}\n'
    return
  fi

  echo "## MedOmni multi-pod health — $ts"
  echo
  echo "### Inference endpoints (laptop port-forwards)"
  echo
  echo "| Endpoint | Service | Pod | State |"
  echo "|---|---|---|---|"
  echo "| 127.0.0.1:8000 | Omni serve              | B300            | $(probe_http "$B300_OMNI_URL") |"
  echo "| 127.0.0.1:8001 | embed (NV-Embed-1B-v2)  | B300            | $(probe_http "$B300_EMBED_URL") |"
  echo "| 127.0.0.1:8002 | rerank (NV-Rerank-1B)   | B300            | $(probe_http "$B300_RERANK_URL") |"
  echo "| 127.0.0.1:8003 | judge (Qwen2.5-7B)      | B300            | $(probe_http "$B300_JUDGE_URL") |"
  echo "| 127.0.0.1:8004 | NemoGuard (Phase 2.1)   | B300            | $(probe_http "$B300_GUARD_URL") |"
  echo "| 127.0.0.1:9002 | rerank TRT-LLM (Ph 2.2) | RunPod prism    | $(probe_http "$PRISM_RERANK_URL") |"
  echo "| 127.0.0.1:9003 | judge TRT-LLM (Ph 2.2)  | RunPod prism    | $(probe_http "$PRISM_JUDGE_URL") |"
  echo
  echo "### Pod-level state"
  echo
  echo "| Pod | SSH | GPU mem (used/total) | Containers |"
  echo "|---|---|---|---|"
  for entry in \
    "B300|$B300_HOST|" \
    "H200(voice)|$H200_HOST|" \
    "Brev H100(voice)|$BREV_H100_HOST|" \
    "RunPod prism|$RUNPOD_PRISM_HOST|$RUNPOD_PRISM_SSH_OPTS"; do
    IFS='|' read -r label host opts <<< "$entry"
    ssh_state=$(probe_ssh "$host" "$opts")
    if [ "$ssh_state" = "UP" ]; then
      mem=$(probe_pod_gpu_mem "$host" "$opts")
      ctr=$(probe_pod_containers "$host" "$opts")
    else
      mem="-"
      ctr="-"
    fi
    echo "| $label | $ssh_state | $mem | $ctr |"
  done
}

if [ "$WATCH_MODE" = "1" ]; then
  while true; do
    clear
    run_once
    sleep 30
  done
else
  run_once
fi
