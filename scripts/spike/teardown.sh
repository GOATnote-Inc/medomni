#!/usr/bin/env bash
# Tear down the spike sandbox. Removes containers, volumes, and local data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[spike] docker compose down -v"
( cd "$SCRIPT_DIR" && docker compose down -v --remove-orphans ) || true

echo "[spike] removing local artifacts"
rm -rf "$SCRIPT_DIR/.medplum-data"
rm -rf "$SCRIPT_DIR/synthea/output"
rm -f  "$SCRIPT_DIR/.spike-token"

echo "[spike] teardown complete."
echo "[spike] (Synthea jar at $SCRIPT_DIR/synthea/synthea-with-dependencies.jar kept; remove manually if unwanted.)"
