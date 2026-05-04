# spike — Synthea + Medplum FHIR sandbox

Local Medplum 5.1.x FHIR server fed by Synthea synthetic patients. Exists
to measure real FHIR-fetch latency against the dual-lookup (Pattern B)
augmentation path of the medomni clinical agent. No PHI; no cloud LLM keys.

## Prereqs

- Docker Desktop (running)
- Java 17+ JRE (Synthea requirement)
- `curl`, `jq`

## Run

```
bash scripts/spike/setup_synthea_medplum.sh

TOKEN=$(cat scripts/spike/.spike-token)
# Medplum returns lazy totals; pass _total=accurate (or _summary=count)
# when you want the .total field populated.
curl -sS -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8103/fhir/R4/Patient?_total=accurate" | jq '.total'

bash scripts/spike/teardown.sh
```

`PATIENT_COUNT=N bash setup_synthea_medplum.sh` to generate N != 10.

## Token

`scripts/spike/.spike-token` (mode 0600, gitignored). It's an OAuth2
client_credentials access token minted from the auto-seeded super-admin
ClientApplication. Token lifetime is the server default (~1 h). To refresh:
re-run `setup_synthea_medplum.sh` (idempotent — re-mints token, skips
re-ingest if data is still in the postgres volume) or replay the
client_credentials POST against `/oauth2/token` using the credentials in
`docker-compose.yml`.

## Limitations

- No SMART-on-FHIR OAuth dance. The admin-client bearer token simulates an
  authorized session for the next-track latency probe. Do not treat it as a
  realistic auth surface.
- Synthea bundles are POSTed as transactions; on failure we fall back to
  per-resource POST (lossy on cross-resource references).
- Sandbox is bound to `127.0.0.1:8103` — not reachable from other hosts.
