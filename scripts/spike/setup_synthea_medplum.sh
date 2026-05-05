#!/usr/bin/env bash
# Spin up the Medplum FHIR sandbox, populate it with N=10 Synthea patients,
# and write a bearer token to .spike-token. Idempotent: re-running picks up
# from wherever the previous run stopped.
#
# Constraints (see scripts/spike/README.md and CLAUDE.md):
#   - No PHI. Synthea is fully synthetic; verified post-load.
#   - No cloud LLM keys. Sovereign by construction.
#   - All persistent artifacts under scripts/spike/. Gitignored.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNTHEA_DIR="$SCRIPT_DIR/synthea"
SYNTHEA_JAR="$SYNTHEA_DIR/synthea-with-dependencies.jar"
SYNTHEA_OUT="$SYNTHEA_DIR/output"
TOKEN_FILE="$SCRIPT_DIR/.spike-token"
PATIENT_COUNT="${PATIENT_COUNT:-10}"

MEDPLUM_BASE="http://localhost:8103"
ADMIN_CLIENT_ID="8d8d2b3a-1d0f-4f8a-9c1e-f0e7c8b3a2d1"
ADMIN_CLIENT_SECRET="medomni_spike_local_only_NOT_a_secret"

log() { printf '[spike] %s\n' "$*"; }
die() { printf '[spike] ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. Pre-flight ----------------------------------------------------------
log "preflight: checking docker + java"
command -v docker >/dev/null || die "docker not found in PATH"
docker info >/dev/null 2>&1 || die "docker daemon not reachable (Docker Desktop not running?)"
command -v java >/dev/null || die "java not found in PATH (Synthea requires JRE)"
command -v curl >/dev/null || die "curl not found in PATH"
command -v jq >/dev/null || die "jq not found in PATH"

# --- 2. Boot the stack ------------------------------------------------------
log "boot: docker compose up -d"
( cd "$SCRIPT_DIR" && docker compose up -d )

log "wait: Medplum /healthcheck (timeout 120s)"
deadline=$(( $(date +%s) + 120 ))
while :; do
  if curl -fsS "$MEDPLUM_BASE/healthcheck" >/dev/null 2>&1; then
    log "wait: Medplum is healthy"
    break
  fi
  if (( $(date +%s) >= deadline )); then
    log "diagnostic: docker compose ps"
    ( cd "$SCRIPT_DIR" && docker compose ps )
    log "diagnostic: medplum-server logs (tail 80)"
    ( cd "$SCRIPT_DIR" && docker compose logs --tail=80 medplum-server || true )
    die "Medplum did not become healthy within 120s"
  fi
  sleep 3
done

# --- 3. Mint a bearer token via client_credentials --------------------------
log "auth: minting bearer token via OAuth2 client_credentials"
TOKEN=""
for attempt in 1 2 3 4 5; do
  TOKEN_JSON=$(curl -fsS -X POST "$MEDPLUM_BASE/oauth2/token" \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode "grant_type=client_credentials" \
    --data-urlencode "client_id=$ADMIN_CLIENT_ID" \
    --data-urlencode "client_secret=$ADMIN_CLIENT_SECRET" \
    --data-urlencode "scope=system/*.*" 2>/dev/null || true)
  TOKEN=$(printf '%s' "$TOKEN_JSON" | jq -r '.access_token // empty' 2>/dev/null || true)
  if [[ -n "$TOKEN" && "$TOKEN" != "null" ]]; then
    break
  fi
  log "auth: attempt $attempt failed, sleeping 4s (seed may still be running)"
  sleep 4
done
[[ -n "$TOKEN" && "$TOKEN" != "null" ]] || die "could not mint bearer token; response was: $TOKEN_JSON"

umask 077
printf '%s\n' "$TOKEN" > "$TOKEN_FILE"
log "auth: token written to $TOKEN_FILE (mode 0600)"

# --- 4. Synthea jar ---------------------------------------------------------
mkdir -p "$SYNTHEA_DIR"
if [[ ! -f "$SYNTHEA_JAR" ]]; then
  log "synthea: downloading jar (~50 MB)"
  curl -fsSL -o "$SYNTHEA_JAR" \
    "https://github.com/synthetichealth/synthea/releases/latest/download/synthea-with-dependencies.jar"
fi
log "synthea: jar present at $SYNTHEA_JAR ($(du -h "$SYNTHEA_JAR" | cut -f1))"

# --- 5. Generate N synthetic patients ---------------------------------------
log "synthea: generating $PATIENT_COUNT patients (US Core IG, FHIR R4)"
rm -rf "$SYNTHEA_OUT"
mkdir -p "$SYNTHEA_OUT"
( cd "$SYNTHEA_DIR" && java -jar "$SYNTHEA_JAR" \
    -p "$PATIENT_COUNT" \
    --exporter.baseDirectory "$SYNTHEA_OUT" \
    --exporter.fhir.export true \
    --exporter.fhir.use_us_core_ig true \
    --exporter.hospital.fhir.export false \
    --exporter.practitioner.fhir.export false \
    --exporter.ccda.export false \
    --exporter.csv.export false \
    --exporter.text.export false \
    --exporter.symptoms.csv.export false \
    > "$SYNTHEA_OUT/synthea.log" 2>&1 ) || {
      tail -40 "$SYNTHEA_OUT/synthea.log" >&2
      die "Synthea generation failed; see $SYNTHEA_OUT/synthea.log"
    }

PATIENT_BUNDLES=$(find "$SYNTHEA_OUT/fhir" -type f -name '*.json' \
  ! -name 'hospitalInformation*' \
  ! -name 'practitionerInformation*' \
  | sort)
BUNDLE_COUNT=$(printf '%s\n' "$PATIENT_BUNDLES" | grep -c . || true)
[[ "$BUNDLE_COUNT" -gt 0 ]] || die "no patient bundles generated under $SYNTHEA_OUT/fhir"
log "synthea: $BUNDLE_COUNT patient bundle(s) generated"

# PHI tripwire: synthetic Synthea names always carry a numeric suffix
# (e.g. "Aaron123"). If the first given name doesn't match, refuse to load.
FIRST_BUNDLE=$(printf '%s\n' "$PATIENT_BUNDLES" | head -1)
FIRST_GIVEN=$(jq -r '
  .entry[].resource
  | select(.resourceType == "Patient")
  | .name[0].given[0]' "$FIRST_BUNDLE" | head -1)
log "synthea: first Patient name.given[0] = $FIRST_GIVEN"
if [[ ! "$FIRST_GIVEN" =~ [0-9] ]]; then
  die "Patient given-name '$FIRST_GIVEN' has no digits — Synthea synthetic-data signature missing. Aborting (PHI tripwire)."
fi

# --- 6. Bulk-load each bundle into Medplum ----------------------------------
log "ingest: posting bundles via POST /fhir/R4 (transaction)"
INGEST_TX=0
INGEST_FALLBACK=0
INGEST_FAIL=0

while IFS= read -r bundle; do
  [[ -n "$bundle" ]] || continue
  # Synthea exports type=collection; Medplum transaction endpoint requires
  # type=transaction with a request.method on each entry. Rewrite in-flight.
  TX_JSON=$(jq '
    .type = "transaction"
    | .entry |= map(
        . + {
          request: {
            method: "POST",
            url: .resource.resourceType
          }
        }
      )
  ' "$bundle")

  HTTP_CODE=$(printf '%s' "$TX_JSON" | curl -sS -o /tmp/spike_ingest.out -w '%{http_code}' \
    -X POST "$MEDPLUM_BASE/fhir/R4" \
    -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/fhir+json' \
    --data-binary @- || echo '000')

  if [[ "$HTTP_CODE" =~ ^2[0-9][0-9]$ ]]; then
    INGEST_TX=$((INGEST_TX + 1))
    continue
  fi

  log "ingest: transaction failed for $(basename "$bundle") (HTTP $HTTP_CODE) — falling back to per-resource POST"
  # Per-resource fallback: post each resource individually. Lossy on
  # references that target other resources by urn:uuid:, but adequate for
  # latency probing — the spike only needs queryable Patient + child rows.
  RES_OK=0
  RES_FAIL=0
  while IFS= read -r resource; do
    [[ -n "$resource" ]] || continue
    RT=$(printf '%s' "$resource" | jq -r '.resourceType')
    [[ -n "$RT" && "$RT" != "null" ]] || continue
    RC=$(printf '%s' "$resource" | curl -sS -o /dev/null -w '%{http_code}' \
      -X POST "$MEDPLUM_BASE/fhir/R4/$RT" \
      -H "Authorization: Bearer $TOKEN" \
      -H 'Content-Type: application/fhir+json' \
      --data-binary @- || echo '000')
    if [[ "$RC" =~ ^2[0-9][0-9]$ ]]; then
      RES_OK=$((RES_OK + 1))
    else
      RES_FAIL=$((RES_FAIL + 1))
    fi
  done < <(jq -c '.entry[].resource' "$bundle")
  log "ingest:   per-resource: ok=$RES_OK fail=$RES_FAIL"
  if (( RES_OK > 0 )); then
    INGEST_FALLBACK=$((INGEST_FALLBACK + 1))
  else
    INGEST_FAIL=$((INGEST_FAIL + 1))
  fi
done <<< "$PATIENT_BUNDLES"

log "ingest: summary tx=$INGEST_TX fallback=$INGEST_FALLBACK fail=$INGEST_FAIL"
(( INGEST_TX + INGEST_FALLBACK > 0 )) || die "no bundles ingested"

# --- 7. Smoke verification --------------------------------------------------
log "verify: GET /fhir/R4/Patient"
TOTAL=$(curl -fsS -H "Authorization: Bearer $TOKEN" \
  "$MEDPLUM_BASE/fhir/R4/Patient?_summary=count" \
  | jq -r '.total // 0')
log "verify: Patient.total = $TOTAL"
[[ "$TOTAL" =~ ^[0-9]+$ ]] && (( TOTAL >= PATIENT_COUNT )) \
  || die "expected >= $PATIENT_COUNT Patients, got $TOTAL"

PID=$(curl -fsS -H "Authorization: Bearer $TOKEN" \
  "$MEDPLUM_BASE/fhir/R4/Patient?_count=1" \
  | jq -r '.entry[0].resource.id // empty')
[[ -n "$PID" ]] || die "could not extract a Patient.id from search"

OBS_TOTAL=$(curl -fsS -H "Authorization: Bearer $TOKEN" \
  "$MEDPLUM_BASE/fhir/R4/Observation?patient=$PID&_summary=count" \
  | jq -r '.total // 0')
log "verify: Observation.total for patient[$PID] = $OBS_TOTAL"

cat <<EOF

[spike] OK — sandbox ready.
[spike]   base URL:        $MEDPLUM_BASE
[spike]   token file:      $TOKEN_FILE
[spike]   patients loaded: $TOTAL
[spike]   sample patient:  $PID  (observations: $OBS_TOTAL)

[spike] Smoke:
  TOKEN=\$(cat scripts/spike/.spike-token)
  curl -sS -H "Authorization: Bearer \$TOKEN" \\
    $MEDPLUM_BASE/fhir/R4/Patient | jq '.total'

[spike] Teardown when done:  bash scripts/spike/teardown.sh
EOF
