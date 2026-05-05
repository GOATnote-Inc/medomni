// Patient context lookup. Spike-only tool ("Pattern B": dual lookup, no merge
// into PrimeKG). Pulls active conditions, recent vitals/labs, active meds,
// allergies, and recent diagnostic reports for ONE patient from a local
// Medplum FHIR R4 server, then renders a ~200-600 token Markdown block the
// agent loop can consume in parallel with knowledge-base tools.
//
// Server-side only. Reads:
//   MEDOMNI_FHIR_BASE_URL   e.g. http://localhost:8103/fhir/R4
//   MEDOMNI_FHIR_TOKEN      bearer token for the Medplum sandbox
//
// During the spike Medplum is fed Synthea synthetic data ONLY. The
// DemoBanner stays on the UI per medomni/CLAUDE.md until v1 architecture
// review. No PHI ever flows through this code path.
//
// Failure model: per-resource fetches fail soft. If the Patient read 404s
// (or the env vars are missing) we throw, matching primekg.ts — the
// dispatcher in app/api/agent/route.ts converts the throw into an
// {error: string} tool result. Per-resource subordinate fetches degrade
// in place: the section gets a "*(unavailable: ...)*" note rather than
// crashing the agent loop.

const TIMEOUT_MS = 8_000;

const DEFAULT_SCOPES = [
  "Patient",
  "Condition",
  "Observation",
  "MedicationRequest",
  "AllergyIntolerance",
  "DiagnosticReport",
] as const;

export type FhirScope = (typeof DEFAULT_SCOPES)[number];

const CAPS = {
  Condition: 8,
  Observation: 12,
  MedicationRequest: 8,
  AllergyIntolerance: 6,
  DiagnosticReport: 4,
} as const;

const OBS_LOOKBACK_MS = 365 * 24 * 60 * 60 * 1000; // 12 months

export interface PatientContextResult {
  block: string;
  elapsedMs: number;
  resourceCounts: {
    Patient: number;
    Condition: number;
    Observation: number;
    MedicationRequest: number;
    AllergyIntolerance: number;
    DiagnosticReport: number;
  };
  truncated: boolean;
}

// --- minimal FHIR shape stubs (just what we read) -------------------------

interface Coding {
  system?: string;
  code?: string;
  display?: string;
}
interface CodeableConcept {
  coding?: Coding[];
  text?: string;
}
interface FhirPatient {
  resourceType: "Patient";
  id?: string;
  name?: Array<{ given?: string[]; family?: string; text?: string }>;
  gender?: string;
  birthDate?: string;
}
interface FhirCondition {
  resourceType: "Condition";
  id?: string;
  code?: CodeableConcept;
  clinicalStatus?: CodeableConcept;
  onsetDateTime?: string;
  recordedDate?: string;
}
interface FhirQuantity {
  value?: number;
  unit?: string;
  code?: string;
}
interface FhirObservation {
  resourceType: "Observation";
  id?: string;
  code?: CodeableConcept;
  category?: CodeableConcept[];
  effectiveDateTime?: string;
  issued?: string;
  valueQuantity?: FhirQuantity;
  valueString?: string;
  valueCodeableConcept?: CodeableConcept;
  component?: Array<{ code?: CodeableConcept; valueQuantity?: FhirQuantity }>;
  status?: string;
}
interface FhirMedicationRequest {
  resourceType: "MedicationRequest";
  id?: string;
  status?: string;
  medicationCodeableConcept?: CodeableConcept;
  medicationReference?: { display?: string };
  dosageInstruction?: Array<{ text?: string }>;
  authoredOn?: string;
}
interface FhirAllergyIntolerance {
  resourceType: "AllergyIntolerance";
  id?: string;
  code?: CodeableConcept;
  reaction?: Array<{ manifestation?: CodeableConcept[]; severity?: string }>;
  criticality?: string;
  clinicalStatus?: CodeableConcept;
}
interface FhirDiagnosticReport {
  resourceType: "DiagnosticReport";
  id?: string;
  code?: CodeableConcept;
  effectiveDateTime?: string;
  issued?: string;
  conclusion?: string;
  status?: string;
}

interface BundleEntry<T> {
  resource?: T;
}
interface Bundle<T> {
  resourceType: "Bundle";
  entry?: BundleEntry<T>[];
  total?: number;
}

// --- helpers --------------------------------------------------------------

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

function ccLabel(cc?: CodeableConcept): string {
  if (!cc) return "";
  if (cc.text) return cc.text;
  const c = cc.coding?.[0];
  if (!c) return "";
  return c.display || c.code || "";
}

function fmtDate(s?: string): string {
  if (!s) return "";
  // Already ISO; just take YYYY-MM-DD prefix.
  const m = s.match(/^(\d{4}-\d{2}-\d{2})/);
  return m ? m[1] : s;
}

function patientName(p: FhirPatient): string {
  const n = p.name?.[0];
  if (!n) return p.id ?? "(unknown)";
  if (n.text) return n.text;
  const given = (n.given ?? []).join(" ").trim();
  return [given, n.family].filter(Boolean).join(" ").trim() || (p.id ?? "(unknown)");
}

function ageYears(birthDate?: string, now: Date = new Date()): number | null {
  if (!birthDate) return null;
  const m = birthDate.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return null;
  const dob = new Date(`${birthDate}T00:00:00Z`);
  if (Number.isNaN(dob.getTime())) return null;
  let age = now.getUTCFullYear() - dob.getUTCFullYear();
  const md = now.getUTCMonth() - dob.getUTCMonth();
  if (md < 0 || (md === 0 && now.getUTCDate() < dob.getUTCDate())) age -= 1;
  return age;
}

function obsCategory(obs: FhirObservation): string {
  for (const cat of obs.category ?? []) {
    for (const c of cat.coding ?? []) {
      if (c.code) return c.code;
    }
  }
  return "";
}

function isVitalOrLab(obs: FhirObservation): boolean {
  const cat = obsCategory(obs);
  return cat === "vital-signs" || cat === "laboratory";
}

function obsTimestamp(obs: FhirObservation): number {
  const s = obs.effectiveDateTime || obs.issued;
  if (!s) return 0;
  const t = Date.parse(s);
  return Number.isNaN(t) ? 0 : t;
}

function fmtQuantity(q?: FhirQuantity): string {
  if (!q || q.value === undefined) return "";
  const unit = q.unit || q.code || "";
  return unit ? `${q.value} ${unit}` : `${q.value}`;
}

function obsValue(obs: FhirObservation): string {
  if (obs.valueQuantity) return fmtQuantity(obs.valueQuantity);
  if (obs.valueString) return obs.valueString;
  if (obs.valueCodeableConcept) return ccLabel(obs.valueCodeableConcept);
  if (obs.component && obs.component.length > 0) {
    return obs.component
      .map((c) => {
        const lbl = ccLabel(c.code);
        const v = fmtQuantity(c.valueQuantity);
        return lbl && v ? `${lbl} ${v}` : v || lbl;
      })
      .filter(Boolean)
      .join(", ");
  }
  return "";
}

function medLabel(mr: FhirMedicationRequest): string {
  return (
    ccLabel(mr.medicationCodeableConcept) ||
    mr.medicationReference?.display ||
    "(unnamed medication)"
  );
}

function dosage(mr: FhirMedicationRequest): string {
  return (mr.dosageInstruction ?? [])
    .map((d) => d.text)
    .filter((s): s is string => !!s && s.trim() !== "")
    .join("; ");
}

function allergyLabel(a: FhirAllergyIntolerance): string {
  const lbl = ccLabel(a.code) || "(unknown allergen)";
  const reactions = (a.reaction ?? [])
    .flatMap((r) => (r.manifestation ?? []).map(ccLabel))
    .filter(Boolean);
  const r = reactions.length > 0 ? ` — reaction: ${reactions.join(", ")}` : "";
  const crit = a.criticality ? ` [${a.criticality}]` : "";
  return `${lbl}${crit}${r}`;
}

// --- HTTP layer -----------------------------------------------------------

async function fhirGet<T>(
  baseUrl: string,
  token: string,
  path: string,
): Promise<T> {
  const url = `${baseUrl.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
  const res = await withTimeout(
    fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/fhir+json",
      },
    }),
    TIMEOUT_MS,
    `fhir GET ${path.split("?")[0]}`,
  );
  if (!res.ok) {
    const txt = await res.text().catch(() => "(no body)");
    throw new Error(`${res.status} ${res.statusText}: ${txt.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

function bundleResources<T>(b: Bundle<T> | null | undefined): T[] {
  return (b?.entry ?? [])
    .map((e) => e.resource)
    .filter((r): r is T => r !== undefined && r !== null);
}

// --- main -----------------------------------------------------------------

export async function getPatientContext(args: {
  patientId: string;
  queryHint?: string;
  scopes?: string[];
}): Promise<PatientContextResult> {
  const start = Date.now();

  const baseUrl = process.env.MEDOMNI_FHIR_BASE_URL;
  const token = process.env.MEDOMNI_FHIR_TOKEN;
  if (!baseUrl || !token) {
    throw new Error(
      "MEDOMNI_FHIR_BASE_URL and MEDOMNI_FHIR_TOKEN are not configured on " +
        "this deployment. Patient-context lookup is unavailable; answer " +
        "from the question as posed and the knowledge tools.",
    );
  }
  const patientId = (args.patientId ?? "").trim();
  if (!patientId) {
    throw new Error("patientId is required");
  }
  // Defensive: don't let the LLM smuggle in a path traversal.
  if (!/^[A-Za-z0-9._-]{1,64}$/.test(patientId)) {
    throw new Error(`patientId failed validation: ${patientId.slice(0, 32)}`);
  }

  const scopeSet = new Set<string>(
    args.scopes && args.scopes.length > 0 ? args.scopes : (DEFAULT_SCOPES as readonly string[]),
  );

  // Patient is the gate; if this fails we bubble. The other scopes degrade
  // in place.
  const patient = await fhirGet<FhirPatient>(
    baseUrl,
    token,
    `Patient/${encodeURIComponent(patientId)}`,
  );
  if (!patient || patient.resourceType !== "Patient") {
    throw new Error(`Patient/${patientId}: unexpected response shape`);
  }

  const enc = (s: string) => encodeURIComponent(s);

  const conditionsP = scopeSet.has("Condition")
    ? fhirGet<Bundle<FhirCondition>>(
        baseUrl,
        token,
        `Condition?patient=${enc(patientId)}&clinical-status=active&_count=20`,
      ).catch((e: Error) => ({ _error: e.message }) as { _error: string })
    : Promise.resolve(null);

  const observationsP = scopeSet.has("Observation")
    ? fhirGet<Bundle<FhirObservation>>(
        baseUrl,
        token,
        `Observation?patient=${enc(patientId)}&_sort=-date&_count=30`,
      ).catch((e: Error) => ({ _error: e.message }) as { _error: string })
    : Promise.resolve(null);

  const medsP = scopeSet.has("MedicationRequest")
    ? fhirGet<Bundle<FhirMedicationRequest>>(
        baseUrl,
        token,
        `MedicationRequest?patient=${enc(patientId)}&status=active&_count=20`,
      ).catch((e: Error) => ({ _error: e.message }) as { _error: string })
    : Promise.resolve(null);

  const allergiesP = scopeSet.has("AllergyIntolerance")
    ? fhirGet<Bundle<FhirAllergyIntolerance>>(
        baseUrl,
        token,
        `AllergyIntolerance?patient=${enc(patientId)}&_count=10`,
      ).catch((e: Error) => ({ _error: e.message }) as { _error: string })
    : Promise.resolve(null);

  const reportsP = scopeSet.has("DiagnosticReport")
    ? fhirGet<Bundle<FhirDiagnosticReport>>(
        baseUrl,
        token,
        `DiagnosticReport?patient=${enc(patientId)}&_sort=-date&_count=10`,
      ).catch((e: Error) => ({ _error: e.message }) as { _error: string })
    : Promise.resolve(null);

  const [condRaw, obsRaw, medsRaw, allergiesRaw, reportsRaw] = await Promise.all([
    conditionsP,
    observationsP,
    medsP,
    allergiesP,
    reportsP,
  ]);

  const sectionErr = (
    raw: Bundle<unknown> | null | { _error: string },
  ): string | null =>
    raw && typeof raw === "object" && "_error" in raw ? raw._error : null;

  // ---- normalize each resource set -----------------------------------------

  let truncated = false;

  const conditions: FhirCondition[] = (() => {
    if (!condRaw || sectionErr(condRaw)) return [];
    const all = bundleResources<FhirCondition>(condRaw as Bundle<FhirCondition>);
    if (all.length > CAPS.Condition) truncated = true;
    return all.slice(0, CAPS.Condition);
  })();

  const cutoffMs = Date.now() - OBS_LOOKBACK_MS;
  const observations: FhirObservation[] = (() => {
    if (!obsRaw || sectionErr(obsRaw)) return [];
    const all = bundleResources<FhirObservation>(obsRaw as Bundle<FhirObservation>);
    const filtered = all
      .filter(isVitalOrLab)
      .filter((o) => obsTimestamp(o) >= cutoffMs)
      .sort((a, b) => obsTimestamp(b) - obsTimestamp(a));
    if (filtered.length > CAPS.Observation) truncated = true;
    return filtered.slice(0, CAPS.Observation);
  })();

  const meds: FhirMedicationRequest[] = (() => {
    if (!medsRaw || sectionErr(medsRaw)) return [];
    const all = bundleResources<FhirMedicationRequest>(
      medsRaw as Bundle<FhirMedicationRequest>,
    );
    if (all.length > CAPS.MedicationRequest) truncated = true;
    return all.slice(0, CAPS.MedicationRequest);
  })();

  const allergies: FhirAllergyIntolerance[] = (() => {
    if (!allergiesRaw || sectionErr(allergiesRaw)) return [];
    const all = bundleResources<FhirAllergyIntolerance>(
      allergiesRaw as Bundle<FhirAllergyIntolerance>,
    );
    if (all.length > CAPS.AllergyIntolerance) truncated = true;
    return all.slice(0, CAPS.AllergyIntolerance);
  })();

  const reports: FhirDiagnosticReport[] = (() => {
    if (!reportsRaw || sectionErr(reportsRaw)) return [];
    const all = bundleResources<FhirDiagnosticReport>(
      reportsRaw as Bundle<FhirDiagnosticReport>,
    );
    if (all.length > CAPS.DiagnosticReport) truncated = true;
    return all.slice(0, CAPS.DiagnosticReport);
  })();

  // ---- render -------------------------------------------------------------

  const out: string[] = [];

  out.push("## Patient");
  const age = ageYears(patient.birthDate);
  const demo = [
    `Name: ${patientName(patient)}`,
    age !== null ? `${age}y` : null,
    patient.gender ?? null,
    patient.birthDate ? `DOB ${fmtDate(patient.birthDate)}` : null,
    `id: ${patient.id ?? patientId}`,
  ]
    .filter(Boolean)
    .join(" · ");
  out.push(`- ${demo}`);
  if (args.queryHint) {
    out.push(`- Context hint: ${args.queryHint.slice(0, 200)}`);
  }

  out.push("");
  out.push("## Active Conditions");
  const condErr = sectionErr(condRaw);
  if (condErr) {
    out.push(`*(Conditions unavailable: ${condErr.slice(0, 120)})*`);
  } else if (conditions.length === 0) {
    out.push("- (none recorded)");
  } else {
    for (const c of conditions) {
      const lbl = ccLabel(c.code) || "(unspecified)";
      const onset = fmtDate(c.onsetDateTime || c.recordedDate);
      out.push(`- ${lbl}${onset ? ` (onset ${onset})` : ""}`);
    }
  }

  out.push("");
  out.push("## Recent Vitals/Labs");
  const obsErr = sectionErr(obsRaw);
  if (obsErr) {
    out.push(`*(Observations unavailable: ${obsErr.slice(0, 120)})*`);
  } else if (observations.length === 0) {
    out.push("- (no vitals or labs in the last 12 months)");
  } else {
    for (const o of observations) {
      const lbl = ccLabel(o.code) || "(unspecified)";
      const v = obsValue(o);
      const d = fmtDate(o.effectiveDateTime || o.issued);
      out.push(`- ${lbl}: ${v || "(no value)"}${d ? ` [${d}]` : ""}`);
    }
  }

  out.push("");
  out.push("## Active Medications");
  const medsErr = sectionErr(medsRaw);
  if (medsErr) {
    out.push(`*(Medications unavailable: ${medsErr.slice(0, 120)})*`);
  } else if (meds.length === 0) {
    out.push("- (none active)");
  } else {
    for (const m of meds) {
      const lbl = medLabel(m);
      const dose = dosage(m);
      out.push(`- ${lbl}${dose ? ` — ${dose}` : ""}`);
    }
  }

  out.push("");
  out.push("## Allergies");
  const allergiesErr = sectionErr(allergiesRaw);
  if (allergiesErr) {
    out.push(`*(Allergies unavailable: ${allergiesErr.slice(0, 120)})*`);
  } else if (allergies.length === 0) {
    out.push("- (none recorded — confirm with patient)");
  } else {
    for (const a of allergies) {
      out.push(`- ${allergyLabel(a)}`);
    }
  }

  out.push("");
  out.push("## Recent Diagnostic Reports");
  const reportsErr = sectionErr(reportsRaw);
  if (reportsErr) {
    out.push(`*(Diagnostic reports unavailable: ${reportsErr.slice(0, 120)})*`);
  } else if (reports.length === 0) {
    out.push("- (none in record)");
  } else {
    for (const r of reports) {
      const lbl = ccLabel(r.code) || "(unspecified)";
      const d = fmtDate(r.effectiveDateTime || r.issued);
      const concl = r.conclusion ? ` — ${r.conclusion.slice(0, 200)}` : "";
      out.push(`- ${lbl}${d ? ` [${d}]` : ""}${concl}`);
    }
  }

  out.push("");
  out.push(
    "_Source: local Medplum FHIR R4 sandbox · Synthea synthetic data · spike Pattern B (no PrimeKG merge)._",
  );

  const block = out.join("\n");

  return {
    block,
    elapsedMs: Date.now() - start,
    resourceCounts: {
      Patient: 1,
      Condition: conditions.length,
      Observation: observations.length,
      MedicationRequest: meds.length,
      AllergyIntolerance: allergies.length,
      DiagnosticReport: reports.length,
    },
    truncated,
  };
}

// --- design-mode renderer -------------------------------------------------
//
// When MEDOMNI_FHIR_BASE_URL is unset, the live tool throws (no FHIR server
// to read). For the public demo at www.thegoatnote.com/4UWHAt the request
// body still carries `patientId: "design-sample-patient"` from the patient
// picker auto-select, and the page already shows Maya Okafor's record from
// `lib/4uwhat/sample-data.ts`. The agent should be able to reason about the
// SAME data the user is looking at — so we render a parallel Markdown block
// from sample-data.ts and inject it into the system prompt at request time.
//
// Same shape, same renderer style as the live block, so the model behaves
// identically in design-mode and live-mode. When env vars get set later
// for a Medplum-backed deployment, this fallback simply stops firing.
//
// IMPORT STRATEGY: dynamic import inside the function. sample-data.ts is
// imported by the React UI layer (lib/4uwhat/...); keeping the static
// import out of the agent route avoids dragging UI-layer types into the
// server bundle's static graph.

import {
  SAMPLE_PATIENT,
  SAMPLE_VITALS,
  SAMPLE_CONDITIONS,
  SAMPLE_MEDS,
  SAMPLE_LABS,
  SAMPLE_TIMELINE,
  SAMPLE_IMAGING,
  SAMPLE_SURGERIES,
  SAMPLE_CARE_TEAM,
} from "@/lib/4uwhat/sample-data";

const DESIGN_PATIENT_ID = "design-sample-patient";

export function isDesignPatientId(id: string | undefined | null): boolean {
  return id === DESIGN_PATIENT_ID;
}

export function buildDesignPatientContextBlock(): string {
  const out: string[] = [];

  out.push("## Patient");
  const demo = [
    `Name: ${SAMPLE_PATIENT.name}`,
    `${SAMPLE_PATIENT.age}y`,
    SAMPLE_PATIENT.pronouns,
    `DOB ${SAMPLE_PATIENT.dob}`,
    `MRN: ${SAMPLE_PATIENT.mrn}`,
    `Blood: ${SAMPLE_PATIENT.bloodType}`,
    `Ht: ${SAMPLE_PATIENT.height}`,
    `Wt: ${SAMPLE_PATIENT.weight}`,
    `Primary care: ${SAMPLE_PATIENT.primaryCare}`,
  ]
    .filter(Boolean)
    .join(" · ");
  out.push(`- ${demo}`);

  out.push("");
  out.push("## Active Conditions");
  const active = SAMPLE_CONDITIONS.filter((c) => c.status === "active");
  if (active.length === 0) {
    out.push("- (none recorded)");
  } else {
    for (const c of active) {
      out.push(`- ${c.name} (onset ${c.onset}; ICD-10 ${c.icd})`);
    }
  }

  out.push("");
  out.push("## Recent Vitals");
  for (const [key, v] of Object.entries(SAMPLE_VITALS)) {
    const recent = v.spark.length > 0 ? ` (recent trend: ${v.spark.join(", ")})` : "";
    const range = v.range ? ` [${v.range}]` : "";
    const delta = v.delta ? ` ${v.delta}` : "";
    out.push(`- ${v.label} (${key}): ${v.value} ${v.unit}${range}${delta}${recent}`);
  }

  out.push("");
  out.push("## Recent Labs");
  for (const lab of SAMPLE_LABS) {
    const flag = lab.flag !== "normal" ? ` [${lab.flag.toUpperCase()}]` : "";
    const trend = lab.trend.length > 0 ? ` (last ${lab.trend.length}: ${lab.trend.join(", ")})` : "";
    out.push(`- ${lab.name}: ${lab.value} ${lab.unit}${flag} (range ${lab.range}; ${lab.date})${trend}`);
  }

  out.push("");
  out.push("## Active Medications");
  if (SAMPLE_MEDS.length === 0) {
    out.push("- (none active)");
  } else {
    for (const m of SAMPLE_MEDS) {
      const adh = m.adherence !== null ? ` · adherence ${m.adherence}%` : "";
      const refills = m.refills !== null ? ` · ${m.refills} refills` : "";
      out.push(`- ${m.name} ${m.dose} ${m.freq} (since ${m.since}; prescriber ${m.prescriber}${adh}${refills})`);
    }
  }

  // Upcoming appointments — derived from care-team `nextAppointment` fields
  // so the agent can answer "when is my next visit with Dr. Adebayo?".
  out.push("");
  out.push("## Upcoming Appointments");
  const upcoming = SAMPLE_CARE_TEAM.filter((c) => c.nextAppointment).map((c) => ({
    when: c.nextAppointment!.date,
    type: c.nextAppointment!.type,
    with: `${c.name} · ${c.role} · ${c.org}`,
  }));
  if (upcoming.length === 0) {
    out.push("- (none scheduled)");
  } else {
    for (const a of upcoming) {
      out.push(`- ${a.when} — ${a.type} with ${a.with}`);
    }
  }

  // Care-team items-to-watch — clinically relevant follow-up topics each
  // provider has flagged for this patient. Lets the agent answer "what
  // is Dr. Patel watching?" without fabricating.
  out.push("");
  out.push("## Care-Team Watch List");
  const watching = SAMPLE_CARE_TEAM.filter(
    (c) => c.itemsToWatch && c.itemsToWatch.length > 0,
  );
  if (watching.length === 0) {
    out.push("- (no active flags)");
  } else {
    for (const c of watching) {
      out.push(`- **${c.name} (${c.role}):** ${(c.itemsToWatch ?? []).join("; ")}`);
    }
  }

  // Surgical history — the agent should be able to answer "when was my
  // appendectomy?" without speculation. Synthetic data only.
  out.push("");
  out.push("## Surgical History");
  if (SAMPLE_SURGERIES.length === 0) {
    out.push("- (none on record)");
  } else {
    for (const s of SAMPLE_SURGERIES) {
      out.push(
        `- ${s.date} — ${s.procedure}; ${s.surgeon} at ${s.facility}; ${s.outcome}`,
      );
    }
  }

  // Recent encounters — from the unified timeline. Visits + urgent care
  // + vaccinations + portal messages so questions like "when did I last
  // see pulmonology?" or "did I get my flu shot this year?" resolve.
  out.push("");
  out.push("## Recent Encounters (last 12 months)");
  const recent = SAMPLE_TIMELINE.slice(0, 8); // already date-sorted desc in source
  if (recent.length === 0) {
    out.push("- (no recent encounters on record)");
  } else {
    for (const e of recent) {
      out.push(
        `- ${e.date} — ${e.kind.toUpperCase()}: ${e.title} (${e.who} · ${e.loc})`,
      );
    }
  }

  // Imaging summary with reads — lets "what did my chest X-ray show?"
  // resolve from the actual radiologist read field rather than hallucinated.
  out.push("");
  out.push("## Imaging Studies");
  if (SAMPLE_IMAGING.length === 0) {
    out.push("- (none on record)");
  } else {
    for (const im of SAMPLE_IMAGING) {
      out.push(
        `- ${im.date} — ${im.kind} of ${im.region} (read by ${im.radiologist}): ${im.read}`,
      );
    }
  }

  out.push("");
  out.push(
    "_Source: design-sample synthetic patient (Maya Okafor) shipped with the public demo. " +
      "When MEDOMNI_FHIR_BASE_URL is configured, this block is replaced by live FHIR reads " +
      "from the configured Medplum server via Pattern B's get_patient_context tool._",
  );

  return out.join("\n");
}

/**
 * Build a patient-context block suitable for prefixing the system prompt.
 * Returns null if no usable context can be loaded (caller should proceed
 * without patient context).
 *
 * - Design mode (patientId === "design-sample-patient"): synchronous render
 *   from sample-data.ts. Always succeeds.
 * - Live mode (env vars set, any other patientId): calls getPatientContext.
 *   Returns null on failure rather than throwing — the agent should still
 *   respond, just without patient context.
 */
export async function buildPatientContextForSystemPrompt(
  patientId: string | undefined | null,
): Promise<string | null> {
  if (!patientId) return null;

  if (isDesignPatientId(patientId)) {
    return buildDesignPatientContextBlock();
  }

  if (!process.env.MEDOMNI_FHIR_BASE_URL || !process.env.MEDOMNI_FHIR_TOKEN) {
    // Live mode requested but not configured. Skip silently — the model
    // will see no patient block and the user-facing answer will fall back
    // to general knowledge. (Better than crashing the request.)
    return null;
  }

  try {
    const result = await getPatientContext({ patientId });
    return result.block;
  } catch (err) {
    console.error(
      "[patient-context] live FHIR fetch failed for system-prompt injection:",
      (err as Error).message,
    );
    return null;
  }
}
