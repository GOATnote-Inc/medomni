#!/usr/bin/env bash
# snapshot_h200_factory_state.sh — daily backup of the H200 factory data-queue.
#
# Purpose: defense in depth against a Brev pod delete-and-recreate. The
# factory_loop on warm-lavender-narwhal writes generated reasoning data
# to /home/ubuntu/data-queue/ (5 GB cap). Losing that mid-flight loses
# the V2.7+ corpus seed. This script tarballs the queue daily and
# uploads to an object store, with 7-day retention.
#
# Run on the pod itself, scheduled via cron.
# Recommended cron:
#   # daily at 03:17 UTC, log to ~/snapshot.log
#   17 3 * * * /home/ubuntu/medomni/scripts/snapshot_h200_factory_state.sh \
#       >>/home/ubuntu/snapshot.log 2>&1
#
# Idempotent. Each run produces a separate timestamped tarball; pruning
# is by ls + date arithmetic so partially-uploaded tarballs do not
# disrupt the prune cycle.
#
# OBJECT STORE TARGET: not configured in repo state. Set the env var
# OBJECT_STORE_TARGET in the cron environment OR uncomment + edit the
# DEFAULT_TARGET line below to point at your bucket. The placeholder
# `<TODO_OBJECT_STORE>` is used so a misconfigured run fails loudly
# rather than silently skipping uploads.
#
# Supported targets (auto-detected from URL scheme):
#   s3://<bucket>/<prefix>/   — needs aws CLI + creds in env
#   gs://<bucket>/<prefix>/   — needs gsutil + creds
#   r2://<bucket>/<prefix>/   — Cloudflare R2 via aws CLI w/ R2_* env
#                               (export AWS_ENDPOINT_URL accordingly)
#
# Env:
#   OBJECT_STORE_TARGET — e.g. s3://goatnote-medomni-backups/narwhal/
#   DATA_QUEUE_DIR      — default /home/ubuntu/data-queue
#   RETENTION_DAYS      — default 7
#   SNAPSHOT_TMP        — default /tmp
#
# Exit codes:
#   0  — tarball created and uploaded; prune complete
#   2  — missing dep (tar, gzip, aws/gsutil depending on target)
#   3  — DATA_QUEUE_DIR missing or empty
#   4  — OBJECT_STORE_TARGET unset or still <TODO_...>
#   5  — upload failed
#
# Restore (single command, run from any pod with the same target creds):
#   aws s3 cp "$OBJECT_STORE_TARGET/data-queue-LATEST.tar.gz" /tmp/restore.tgz \
#     && tar -xzf /tmp/restore.tgz -C /home/ubuntu/

set -uo pipefail

DATA_QUEUE_DIR="${DATA_QUEUE_DIR:-/home/ubuntu/data-queue}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
SNAPSHOT_TMP="${SNAPSHOT_TMP:-/tmp}"

# DEFAULT_TARGET="s3://goatnote-medomni-backups/narwhal/"
DEFAULT_TARGET="<TODO_OBJECT_STORE>"
TARGET="${OBJECT_STORE_TARGET:-$DEFAULT_TARGET}"

log()  { printf "[snapshot_h200] %s\n" "$*"; }
fail() { log "FAIL: $*"; exit "${2:-1}"; }

# --- Pre-flight ------------------------------------------------------------

for bin in tar gzip; do
  command -v "$bin" >/dev/null 2>&1 || fail "missing dep: $bin" 2
done

if [[ "$TARGET" == "<TODO_OBJECT_STORE>" || -z "$TARGET" ]]; then
  fail "OBJECT_STORE_TARGET not set (and DEFAULT_TARGET still placeholder).
  USER: edit DEFAULT_TARGET in this script OR export OBJECT_STORE_TARGET in cron env.
  Example: export OBJECT_STORE_TARGET=s3://goatnote-medomni-backups/narwhal/" 4
fi

if [[ ! -d "$DATA_QUEUE_DIR" ]]; then
  fail "DATA_QUEUE_DIR=$DATA_QUEUE_DIR does not exist." 3
fi

if [[ -z "$(ls -A "$DATA_QUEUE_DIR" 2>/dev/null)" ]]; then
  fail "DATA_QUEUE_DIR=$DATA_QUEUE_DIR is empty; nothing to snapshot." 3
fi

case "$TARGET" in
  s3://*|r2://*)
    command -v aws >/dev/null 2>&1 || fail "aws CLI required for $TARGET" 2
    UPLOAD_CMD=(aws s3 cp)
    LIST_CMD=(aws s3 ls)
    DELETE_CMD=(aws s3 rm)
    ;;
  gs://*)
    command -v gsutil >/dev/null 2>&1 || fail "gsutil required for $TARGET" 2
    UPLOAD_CMD=(gsutil cp)
    LIST_CMD=(gsutil ls -l)
    DELETE_CMD=(gsutil rm)
    ;;
  *)
    fail "Unsupported scheme in OBJECT_STORE_TARGET=$TARGET" 4
    ;;
esac

# --- Build tarball ---------------------------------------------------------

TS="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname)"
TARBALL="${SNAPSHOT_TMP}/data-queue-${HOST}-${TS}.tar.gz"
LATEST_NAME="data-queue-LATEST.tar.gz"

log "creating tarball $TARBALL from $DATA_QUEUE_DIR"
tar -czf "$TARBALL" -C "$(dirname "$DATA_QUEUE_DIR")" "$(basename "$DATA_QUEUE_DIR")" \
  || fail "tar failed"

SIZE_MB=$(du -m "$TARBALL" | awk '{print $1}')
log "tarball size: ${SIZE_MB} MB"

# --- Upload (timestamped + LATEST alias) -----------------------------------

# normalize trailing slash
TARGET="${TARGET%/}/"

log "uploading to ${TARGET}"
"${UPLOAD_CMD[@]}" "$TARBALL" "${TARGET}$(basename "$TARBALL")" \
  || fail "upload failed (timestamped)" 5
"${UPLOAD_CMD[@]}" "$TARBALL" "${TARGET}${LATEST_NAME}" \
  || fail "upload failed (LATEST alias)" 5

log "upload OK"

# --- Local cleanup ---------------------------------------------------------

rm -f "$TARBALL"
log "removed local tarball"

# --- Remote retention prune (best-effort) ----------------------------------

# List remote tarballs older than RETENTION_DAYS days and remove them.
# Each scheme's listing format differs; parse leniently.
log "pruning remote tarballs older than ${RETENTION_DAYS} days"

CUTOFF_EPOCH=$(date -u -d "${RETENTION_DAYS} days ago" +%s 2>/dev/null \
  || date -u -v-"${RETENTION_DAYS}"d +%s 2>/dev/null \
  || echo 0)

if [[ "$CUTOFF_EPOCH" -eq 0 ]]; then
  log "WARN: cannot compute cutoff date on this host; skipping prune."
  exit 0
fi

# Listing logic per scheme
case "$TARGET" in
  s3://*|r2://*)
    "${LIST_CMD[@]}" "$TARGET" 2>/dev/null | awk '{print $1, $2, $4}' \
      | grep -E "data-queue-.*\.tar\.gz$" \
      | grep -v "${LATEST_NAME}" \
      | while read -r d t fname; do
          [[ -z "$fname" ]] && continue
          obj_epoch=$(date -u -d "$d $t" +%s 2>/dev/null \
                    || date -u -j -f "%Y-%m-%d %H:%M:%S" "$d $t" +%s 2>/dev/null \
                    || echo "$CUTOFF_EPOCH")
          if [[ "$obj_epoch" -lt "$CUTOFF_EPOCH" ]]; then
            log "  prune: $fname (older than ${RETENTION_DAYS}d)"
            "${DELETE_CMD[@]}" "${TARGET}${fname}" >/dev/null 2>&1 || true
          fi
        done
    ;;
  gs://*)
    "${LIST_CMD[@]}" "${TARGET}data-queue-*.tar.gz" 2>/dev/null \
      | awk '$NF ~ /\.tar\.gz$/ && $NF !~ /LATEST/ {print $2, $NF}' \
      | while read -r ts url; do
          obj_epoch=$(date -u -d "$ts" +%s 2>/dev/null || echo "$CUTOFF_EPOCH")
          if [[ "$obj_epoch" -lt "$CUTOFF_EPOCH" ]]; then
            log "  prune: $url"
            "${DELETE_CMD[@]}" "$url" >/dev/null 2>&1 || true
          fi
        done
    ;;
esac

log "verified: snapshot uploaded to ${TARGET}, LATEST alias updated, prune complete"
exit 0
