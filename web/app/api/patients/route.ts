// /api/patients — list patients for the picker dropdown.
//
// Server-side only. Same env contract as lib/tools/patient-context.ts:
//   MEDOMNI_FHIR_BASE_URL   e.g. http://localhost:8103/fhir/R4
//   MEDOMNI_FHIR_TOKEN      bearer token for the Medplum sandbox
//
// If MEDOMNI_FHIR_BASE_URL is unset (the public demo / design mode), we
// respond with a one-entry "design fallback" array — the synthetic Maya
// Okafor patient from /assets/app/data.jsx — so the page is never
// patientless. Same fallback fires on any error so the UI stays usable
// even when the FHIR sandbox is briefly down.
//
// Response shape: JSON `Patient[]` with { id, displayName, dob }. The
// FHIR token is never exposed to the browser; this route is the trust
// boundary.

import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TIMEOUT_MS = 8_000;

export interface PatientSummary {
  id: string;
  displayName: string;
  dob: string;
}

// Hard-coded port from /Users/kiteboard/medomni/assets/app/data.jsx const PATIENT.
// Keep in lockstep with that file. The id is a stable sentinel ("design-sample-patient")
// so the picker can auto-select it on first load when no FHIR sandbox is wired.
export const DESIGN_FALLBACK_PATIENT: PatientSummary = {
  id: "design-sample-patient",
  displayName: "Maya Okafor",
  dob: "1991-04-12",
};

const DESIGN_FALLBACK: PatientSummary[] = [DESIGN_FALLBACK_PATIENT];

interface FhirHumanName {
  given?: string[];
  family?: string;
  text?: string;
}

interface FhirPatient {
  resourceType?: "Patient";
  id?: string;
  name?: FhirHumanName[];
  birthDate?: string;
}

interface FhirBundleEntry {
  resource?: FhirPatient;
}

interface FhirBundle {
  resourceType?: "Bundle";
  entry?: FhirBundleEntry[];
}

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(
      () => reject(new Error(`${label} timed out after ${ms}ms`)),
      ms,
    );
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

function patientDisplayName(p: FhirPatient): string {
  const n = p.name?.[0];
  if (!n) return p.id ?? "(unknown)";
  if (n.text) return n.text;
  const given = (n.given ?? []).join(" ").trim();
  const full = [given, n.family].filter(Boolean).join(" ").trim();
  return full || p.id || "(unknown)";
}

function jsonResponse(body: unknown, status = 200): NextResponse {
  const res = NextResponse.json(body, { status });
  res.headers.set("Cache-Control", "no-store");
  return res;
}

export async function GET(): Promise<NextResponse> {
  const baseUrl = process.env.MEDOMNI_FHIR_BASE_URL;
  const token = process.env.MEDOMNI_FHIR_TOKEN;

  // Design / public-demo mode: no FHIR sandbox configured. Hand back the
  // synthetic Maya Okafor entry so the picker has something to render.
  if (!baseUrl || !token) {
    return jsonResponse(DESIGN_FALLBACK);
  }

  const url = `${baseUrl.replace(/\/$/, "")}/Patient?_count=20&_total=accurate`;
  try {
    const res = await withTimeout(
      fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/fhir+json",
        },
      }),
      TIMEOUT_MS,
      "fhir GET Patient list",
    );
    if (!res.ok) {
      const txt = await res.text().catch(() => "(no body)");
      console.error(
        `[api/patients] FHIR ${res.status} ${res.statusText}: ${txt.slice(0, 200)}`,
      );
      return jsonResponse(DESIGN_FALLBACK);
    }
    const bundle = (await res.json()) as FhirBundle;
    const summaries: PatientSummary[] = (bundle.entry ?? [])
      .map((e) => e.resource)
      .filter((r): r is FhirPatient => !!r && r.resourceType === "Patient")
      .map((p) => ({
        id: p.id ?? "",
        displayName: patientDisplayName(p),
        dob: p.birthDate ?? "",
      }))
      .filter((s) => !!s.id);

    if (summaries.length === 0) {
      // Empty FHIR sandbox: keep the page usable with the synthetic entry.
      return jsonResponse(DESIGN_FALLBACK);
    }
    return jsonResponse(summaries);
  } catch (e) {
    console.error(`[api/patients] error: ${(e as Error).message}`);
    return jsonResponse(DESIGN_FALLBACK);
  }
}
