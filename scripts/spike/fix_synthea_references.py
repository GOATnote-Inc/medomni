#!/usr/bin/env python3
"""
One-shot fixup for Synthea-loaded Medplum data.

Synthea bundles use `urn:uuid:<uuid>` cross-references. When ingested via
per-resource POST (the fallback path in setup_synthea_medplum.sh when the
transaction-bundle reference rewriter isn't run), references between
resources break: an Observation has `subject.reference: urn:uuid:c37ec...`
but the Patient now lives at Medplum id `38454946-...`. Standard FHIR
search by `?patient=<id>` then returns 0.

Synthea preserves the original urn-uuid in the Patient's
`identifier[].value` under system `https://github.com/synthetichealth/synthea`.
This script:

  1. Walks all Patients, builds a synthea-urn -> medplum-id map.
  2. For each clinical-resource type (Condition, Observation,
     MedicationRequest, AllergyIntolerance, DiagnosticReport),
     paginates through all instances and PATCHes any whose
     subject.reference (or patient.reference for AllergyIntolerance) is
     a urn:uuid: pointing at a known Patient.

Idempotent — already-patched references are skipped.

Usage:
  python3 scripts/spike/fix_synthea_references.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

FHIR_BASE = "http://127.0.0.1:8103/fhir/R4"
TOKEN_PATH = Path(__file__).parent / ".spike-token"
SYNTHEA_SYSTEM = "https://github.com/synthetichealth/synthea"

CLINICAL_TYPES = [
    ("Condition", "subject"),
    ("Observation", "subject"),
    ("MedicationRequest", "subject"),
    ("AllergyIntolerance", "patient"),
    ("DiagnosticReport", "subject"),
]


def fhir_get(token: str, path: str) -> dict:
    req = urllib.request.Request(
        f"{FHIR_BASE}/{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def fhir_patch(token: str, rtype: str, rid: str, ops: list) -> int:
    body = json.dumps(ops).encode()
    req = urllib.request.Request(
        f"{FHIR_BASE}/{rtype}/{rid}",
        data=body,
        method="PATCH",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json-patch+json",
            "Accept": "application/fhir+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def iter_pages(token: str, path_with_count: str):
    """Yield each resource entry, following Bundle.link[next]."""
    url = f"{FHIR_BASE}/{path_with_count}"
    while url:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            bundle = json.loads(r.read())
        for entry in bundle.get("entry", []) or []:
            yield entry["resource"]
        url = None
        for link in bundle.get("link", []) or []:
            if link.get("relation") == "next":
                url = link.get("url")
                break


def main():
    token = TOKEN_PATH.read_text().strip()

    print("[fixup] loading patient urn -> medplum-id map…", file=sys.stderr)
    urn_to_id: dict[str, str] = {}
    for p in iter_pages(token, "Patient?_count=50"):
        synthea_urns = [
            i.get("value")
            for i in p.get("identifier", []) or []
            if i.get("system") == SYNTHEA_SYSTEM and i.get("value")
        ]
        for u in synthea_urns:
            urn_to_id[f"urn:uuid:{u}"] = p["id"]
    print(f"[fixup] {len(urn_to_id)} patient urn references mapped", file=sys.stderr)

    total_patched = 0
    total_skipped = 0
    total_errors = 0
    for rtype, ref_field in CLINICAL_TYPES:
        type_patched = 0
        type_skipped = 0
        type_errors = 0
        for r in iter_pages(token, f"{rtype}?_count=200"):
            ref = (r.get(ref_field) or {}).get("reference", "")
            if not ref.startswith("urn:uuid:"):
                type_skipped += 1
                continue
            mapped = urn_to_id.get(ref)
            if not mapped:
                type_skipped += 1
                continue
            new_ref = f"Patient/{mapped}"
            patch = [{"op": "replace", "path": f"/{ref_field}/reference", "value": new_ref}]
            status = fhir_patch(token, rtype, r["id"], patch)
            if 200 <= status < 300:
                type_patched += 1
            else:
                type_errors += 1
        print(f"[fixup] {rtype:22s} patched={type_patched} skipped={type_skipped} errors={type_errors}", file=sys.stderr)
        total_patched += type_patched
        total_skipped += type_skipped
        total_errors += type_errors

    print(f"[fixup] DONE  total patched={total_patched} skipped={total_skipped} errors={total_errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
