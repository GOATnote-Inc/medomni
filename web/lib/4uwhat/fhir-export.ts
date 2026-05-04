// 4UWHAt — FHIR R4 Bundle export.
//
// Pure, dependency-free assembler that maps the synthetic Maya Okafor
// patient slice (lib/4uwhat/sample-data.ts) into a FHIR R4 Bundle of
// type "collection". Used by ShareDrawer to give the demo's "patient
// owns their record and can take it anywhere FHIR is spoken" story
// teeth: download, copy-to-clipboard, or stub-share to a receiving
// system.
//
// FHIR R4 references:
//   Bundle:       https://www.hl7.org/fhir/bundle.html
//   Patient:      https://www.hl7.org/fhir/patient.html
//   Condition:    https://www.hl7.org/fhir/condition.html
//   Observation:  https://www.hl7.org/fhir/observation.html
//   MedReq:       https://www.hl7.org/fhir/medicationrequest.html
//   AllergyInt:   https://www.hl7.org/fhir/allergyintolerance.html
//
// LOINC mapping policy: best-effort. If we can pin a code with high
// confidence we emit `code.coding[0]` with the LOINC system. Otherwise
// we emit `code.text` only — no invented codes (per CLAUDE.md /
// memory's "evaluation artifacts are immutable" discipline applied
// to clinical codes).
//
// PHI note: the source patient is synthetic (Maya Okafor). Safe to
// download/copy/share to any environment.

import {
  SAMPLE_PATIENT,
  SAMPLE_CONDITIONS,
  SAMPLE_VITALS,
  SAMPLE_LABS,
  SAMPLE_MEDS,
  type SampleVital,
  type SampleLab,
  type SampleCondition,
  type SampleMed,
  type SamplePatient,
} from "./sample-data";

// ── FHIR R4 minimal type stubs ────────────────────────────────────────

export interface FhirCoding {
  system?: string;
  code?: string;
  display?: string;
}

export interface FhirCodeableConcept {
  coding?: FhirCoding[];
  text?: string;
}

export interface FhirIdentifier {
  system: string;
  value: string;
}

export interface FhirReference {
  reference: string;
  display?: string;
}

export interface FhirQuantity {
  value: number;
  unit: string;
  system?: string;
  code?: string;
}

export interface FhirPatientResource {
  resourceType: "Patient";
  id: string;
  identifier?: FhirIdentifier[];
  name?: Array<{ use?: string; text?: string; family?: string; given?: string[] }>;
  gender?: string;
  birthDate?: string;
}

export interface FhirConditionResource {
  resourceType: "Condition";
  id: string;
  clinicalStatus?: FhirCodeableConcept;
  code: FhirCodeableConcept;
  subject: FhirReference;
  onsetDateTime?: string;
}

export interface FhirObservationResource {
  resourceType: "Observation";
  id: string;
  status: "final";
  category?: FhirCodeableConcept[];
  code: FhirCodeableConcept;
  subject: FhirReference;
  effectiveDateTime?: string;
  valueQuantity?: FhirQuantity;
  valueString?: string;
  component?: Array<{
    code: FhirCodeableConcept;
    valueQuantity?: FhirQuantity;
  }>;
}

export interface FhirMedicationRequestResource {
  resourceType: "MedicationRequest";
  id: string;
  status: "active";
  intent: "order";
  medicationCodeableConcept: FhirCodeableConcept;
  subject: FhirReference;
  authoredOn?: string;
  dosageInstruction?: Array<{ text: string }>;
  requester?: { display: string };
}

export interface FhirAllergyIntoleranceResource {
  resourceType: "AllergyIntolerance";
  id: string;
  clinicalStatus?: FhirCodeableConcept;
  verificationStatus?: FhirCodeableConcept;
  code: FhirCodeableConcept;
  patient: FhirReference;
}

export type FhirResource =
  | FhirPatientResource
  | FhirConditionResource
  | FhirObservationResource
  | FhirMedicationRequestResource
  | FhirAllergyIntoleranceResource;

export interface FhirBundleEntry {
  fullUrl: string;
  resource: FhirResource;
}

export interface FhirBundle {
  resourceType: "Bundle";
  type: "collection";
  timestamp: string;
  identifier: FhirIdentifier;
  entry: FhirBundleEntry[];
}

// ── helpers ──────────────────────────────────────────────────────────

const LOINC = "http://loinc.org";
const ICD10_CM = "http://hl7.org/fhir/sid/icd-10-cm";
const SNOMED = "http://snomed.info/sct";
const CLINICAL_STATUS_SYS =
  "http://terminology.hl7.org/CodeSystem/condition-clinical";
const ALLERGY_CLINICAL_SYS =
  "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical";
const ALLERGY_VERIF_SYS =
  "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification";
const OBS_CATEGORY_SYS =
  "http://terminology.hl7.org/CodeSystem/observation-category";
const IDENTIFIER_SYS = "https://medomni.example/identifier/mrn";
const BUNDLE_ID_SYS = "https://medomni.example/identifier/export";

// Browser-portable UUID v4. Falls back to a Math.random based generator
// so this module works in jsdom-less test runners (npx tsx) and older
// browsers without crypto.randomUUID.
function uuid(): string {
  // Prefer crypto.randomUUID where available.
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  // RFC4122 v4 fallback.
  const b = new Array(16);
  for (let i = 0; i < 16; i++) b[i] = Math.floor(Math.random() * 256);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = b.map((x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(
    16,
    20,
  )}-${h.slice(20, 32)}`;
}

function urn(id: string): string {
  return `urn:uuid:${id}`;
}

function isoTimestamp(d: Date = new Date()): string {
  // FHIR instant: ISO 8601 with Z (UTC) — millisecond precision is fine.
  return d.toISOString();
}

// ── LOINC maps ───────────────────────────────────────────────────────
//
// Vitals: best-effort. Where we can pin a LOINC, we do; otherwise the
// renderer falls back to code.text and omits coding.

interface LoincPin {
  code: string;
  display: string;
}

const VITAL_LOINC: Record<string, LoincPin> = {
  hr: { code: "8867-4", display: "Heart rate" },
  bp: { code: "85354-9", display: "Blood pressure panel" },
  spo2: { code: "2708-6", display: "Oxygen saturation in Arterial blood" },
  hrv: { code: "80404-7", display: "R-R interval.standard deviation" },
  weight: { code: "29463-7", display: "Body weight" },
  // Mean sleep duration: 93832-4 is "Sleep duration"; best-effort pin.
  sleep: { code: "93832-4", display: "Sleep duration" },
};

// BP component LOINCs (when we split 118/72 into two components).
const BP_SYSTOLIC: LoincPin = { code: "8480-6", display: "Systolic blood pressure" };
const BP_DIASTOLIC: LoincPin = { code: "8462-4", display: "Diastolic blood pressure" };

const LAB_LOINC: Record<string, LoincPin> = {
  ldl: { code: "2089-1", display: "LDL cholesterol [Mass/volume] in Serum or Plasma" },
  hdl: { code: "2085-9", display: "HDL cholesterol [Mass/volume] in Serum or Plasma" },
  tg: { code: "2571-8", display: "Triglyceride [Mass/volume] in Serum or Plasma" },
  a1c: { code: "4548-4", display: "Hemoglobin A1c/Hemoglobin.total in Blood" },
  vitd: { code: "1989-3", display: "25-hydroxyvitamin D2+D3 [Mass/volume] in Serum or Plasma" },
  fer: { code: "2276-4", display: "Ferritin [Mass/volume] in Serum or Plasma" },
  tsh: { code: "3016-3", display: "Thyrotropin [Units/volume] in Serum or Plasma" },
  // hs-CRP: code-text-only fall-through (no high-confidence pin used here).
};

const VITALS_CATEGORY: FhirCodeableConcept = {
  coding: [
    {
      system: OBS_CATEGORY_SYS,
      code: "vital-signs",
      display: "Vital Signs",
    },
  ],
  text: "Vital Signs",
};

const LAB_CATEGORY: FhirCodeableConcept = {
  coding: [
    {
      system: OBS_CATEGORY_SYS,
      code: "laboratory",
      display: "Laboratory",
    },
  ],
  text: "Laboratory",
};

// ── builders ─────────────────────────────────────────────────────────

function buildPatient(p: SamplePatient, patientId: string): FhirPatientResource {
  // Split "Maya Okafor" into given/family. Robust to extra middle names.
  const parts = p.name.trim().split(/\s+/);
  const family = parts.length > 1 ? parts[parts.length - 1] : p.name;
  const given = parts.length > 1 ? parts.slice(0, -1) : [];

  return {
    resourceType: "Patient",
    id: patientId,
    identifier: [{ system: IDENTIFIER_SYS, value: p.mrn }],
    name: [
      {
        use: "official",
        text: p.name,
        family,
        given,
      },
    ],
    birthDate: p.dob,
  };
}

// Map a sample-data condition status to the FHIR clinicalStatus
// terminology binding. "active" / "resolved" map cleanly.
function clinicalStatusCC(status: SampleCondition["status"]): FhirCodeableConcept {
  return {
    coding: [
      {
        system: CLINICAL_STATUS_SYS,
        code: status,
        display: status === "active" ? "Active" : "Resolved",
      },
    ],
  };
}

// Best-effort onset → onsetDateTime. Sample data uses "2009" or "Mar 2024"
// or "OTC". If we can pin a year we emit YYYY (FHIR partial-date allowed
// in onsetDateTime as XML schema dateTime, which permits YYYY|YYYY-MM
// |YYYY-MM-DD|YYYY-MM-DDThh:mm:ss(±zzz)). Otherwise omit and put the
// raw string into a code.text on the encompassing condition (we leave
// onsetDateTime undefined in that case).
function parseOnset(onset: string): string | undefined {
  const yearOnly = onset.match(/^(\d{4})$/);
  if (yearOnly) return yearOnly[1];
  // "Mar 2024" → 2024-03
  const monMap: Record<string, string> = {
    Jan: "01", Feb: "02", Mar: "03", Apr: "04", May: "05", Jun: "06",
    Jul: "07", Aug: "08", Sep: "09", Oct: "10", Nov: "11", Dec: "12",
  };
  const mY = onset.match(/^([A-Za-z]{3})\s+(\d{4})$/);
  if (mY && monMap[mY[1]]) return `${mY[2]}-${monMap[mY[1]]}`;
  return undefined;
}

function buildCondition(
  c: SampleCondition,
  patientFullUrl: string,
): FhirConditionResource {
  const onset = parseOnset(c.onset);
  const out: FhirConditionResource = {
    resourceType: "Condition",
    id: `cond-${c.id}`,
    clinicalStatus: clinicalStatusCC(c.status),
    code: {
      coding: [
        {
          system: ICD10_CM,
          code: c.icd,
          display: c.name,
        },
      ],
      text: c.name,
    },
    subject: { reference: patientFullUrl, display: SAMPLE_PATIENT.name },
  };
  if (onset) out.onsetDateTime = onset;
  return out;
}

// Derive a unit-of-measure code from a unit string. We do not invent
// UCUM codes — when we don't have a confident map, we omit the `code`
// field on FhirQuantity (UCUM `system` is then meaningless and also
// omitted). The `unit` text always survives so the receiver sees the
// original units.
function ucum(unit: string): { system?: string; code?: string } {
  // Map only the units we control in sample-data.
  const map: Record<string, string> = {
    bpm: "/min",
    "%": "%",
    mmHg: "mm[Hg]",
    ms: "ms",
    lb: "[lb_av]",
    "mg/dL": "mg/dL",
    "ng/mL": "ng/mL",
    "mIU/L": "m[iU]/L",
    "mg/L": "mg/L",
  };
  const code = map[unit];
  return code ? { system: "http://unitsofmeasure.org", code } : {};
}

function vitalToObservation(
  key: string,
  v: SampleVital,
  patientFullUrl: string,
): FhirObservationResource {
  const pin = VITAL_LOINC[key];
  // Use the most-recent date we have on hand: sample-data doesn't carry
  // a real timestamp on vitals, but the labs share a draw date of
  // 2026-04-22 in the design — we use the same date so a downstream
  // receiver sees a coherent encounter date.
  const effectiveDateTime = "2026-04-22";

  const code: FhirCodeableConcept = pin
    ? {
        coding: [{ system: LOINC, code: pin.code, display: pin.display }],
        text: v.label,
      }
    : { text: v.label };

  // Special case: BP is "118/72". Render as a panel with sys/dia
  // components.
  if (key === "bp" && typeof v.value === "string" && v.value.includes("/")) {
    const [sysStr, diaStr] = v.value.split("/").map((s) => s.trim());
    const sys = Number(sysStr);
    const dia = Number(diaStr);
    if (!Number.isNaN(sys) && !Number.isNaN(dia)) {
      const u = ucum(v.unit);
      return {
        resourceType: "Observation",
        id: `vital-${key}`,
        status: "final",
        category: [VITALS_CATEGORY],
        code,
        subject: { reference: patientFullUrl, display: SAMPLE_PATIENT.name },
        effectiveDateTime,
        component: [
          {
            code: { coding: [{ system: LOINC, code: BP_SYSTOLIC.code, display: BP_SYSTOLIC.display }] },
            valueQuantity: { value: sys, unit: v.unit, ...u },
          },
          {
            code: { coding: [{ system: LOINC, code: BP_DIASTOLIC.code, display: BP_DIASTOLIC.display }] },
            valueQuantity: { value: dia, unit: v.unit, ...u },
          },
        ],
      };
    }
  }

  // Numeric values render as valueQuantity; non-numeric ("7h 14m" sleep)
  // fall back to valueString so we never lose information.
  const numeric =
    typeof v.value === "number"
      ? v.value
      : Number(typeof v.value === "string" ? v.value : NaN);

  const out: FhirObservationResource = {
    resourceType: "Observation",
    id: `vital-${key}`,
    status: "final",
    category: [VITALS_CATEGORY],
    code,
    subject: { reference: patientFullUrl, display: SAMPLE_PATIENT.name },
    effectiveDateTime,
  };
  if (Number.isFinite(numeric)) {
    const u = ucum(v.unit);
    out.valueQuantity = { value: numeric, unit: v.unit, ...u };
  } else {
    out.valueString = `${v.value} ${v.unit}`.trim();
  }
  return out;
}

function labToObservation(
  l: SampleLab,
  patientFullUrl: string,
): FhirObservationResource {
  const pin = LAB_LOINC[l.id];
  const code: FhirCodeableConcept = pin
    ? {
        coding: [{ system: LOINC, code: pin.code, display: pin.display }],
        text: l.name,
      }
    : { text: l.name };
  const u = ucum(l.unit);
  return {
    resourceType: "Observation",
    id: `lab-${l.id}`,
    status: "final",
    category: [LAB_CATEGORY],
    code,
    subject: { reference: patientFullUrl, display: SAMPLE_PATIENT.name },
    effectiveDateTime: l.date,
    valueQuantity: { value: l.value, unit: l.unit, ...u },
  };
}

function medToRequest(
  m: SampleMed,
  patientFullUrl: string,
): FhirMedicationRequestResource {
  return {
    resourceType: "MedicationRequest",
    id: `med-${m.id}`,
    status: "active",
    intent: "order",
    medicationCodeableConcept: { text: m.name },
    subject: { reference: patientFullUrl, display: SAMPLE_PATIENT.name },
    dosageInstruction: [{ text: `${m.dose} ${m.freq}` }],
    requester: { display: m.prescriber },
  };
}

// "NKA — no known drug allergies" is a real, codeable status.
// SNOMED 409137002 = "No known drug allergy". We attach the pin
// because it's stable and widely-used by FHIR servers.
function buildNkaAllergy(patientFullUrl: string): FhirAllergyIntoleranceResource {
  return {
    resourceType: "AllergyIntolerance",
    id: "allergy-nka",
    clinicalStatus: {
      coding: [{ system: ALLERGY_CLINICAL_SYS, code: "active", display: "Active" }],
    },
    verificationStatus: {
      coding: [{ system: ALLERGY_VERIF_SYS, code: "confirmed", display: "Confirmed" }],
    },
    code: {
      coding: [{ system: SNOMED, code: "409137002", display: "No known drug allergy" }],
      text: "No known drug allergies",
    },
    patient: { reference: patientFullUrl, display: SAMPLE_PATIENT.name },
  };
}

// ── public entry ─────────────────────────────────────────────────────

/**
 * Assemble a FHIR R4 collection Bundle from the synthetic Maya Okafor
 * patient slice. Pure: no I/O, no env reads, deterministic in the
 * resource ordering (Patient first, then Conditions in source order,
 * vitals, labs, medications, allergies).
 *
 * Each entry's `fullUrl` is a fresh `urn:uuid:...` so cross-references
 * (subject / patient) resolve correctly inside the Bundle if a receiver
 * processes it as a transaction.
 */
export function buildFhirBundle(now: Date = new Date()): FhirBundle {
  const patientId = uuid();
  const patientFullUrl = urn(patientId);

  const entries: FhirBundleEntry[] = [];

  // 1. Patient
  entries.push({
    fullUrl: patientFullUrl,
    resource: buildPatient(SAMPLE_PATIENT, patientId),
  });

  // 2. Conditions
  for (const c of SAMPLE_CONDITIONS) {
    entries.push({
      fullUrl: urn(uuid()),
      resource: buildCondition(c, patientFullUrl),
    });
  }

  // 3. Vital-signs observations
  for (const [key, v] of Object.entries(SAMPLE_VITALS)) {
    entries.push({
      fullUrl: urn(uuid()),
      resource: vitalToObservation(key, v, patientFullUrl),
    });
  }

  // 4. Lab observations
  for (const l of SAMPLE_LABS) {
    entries.push({
      fullUrl: urn(uuid()),
      resource: labToObservation(l, patientFullUrl),
    });
  }

  // 5. Medication requests
  for (const m of SAMPLE_MEDS) {
    entries.push({
      fullUrl: urn(uuid()),
      resource: medToRequest(m, patientFullUrl),
    });
  }

  // 6. Single NKA AllergyIntolerance (Maya is NKA per the design).
  entries.push({
    fullUrl: urn(uuid()),
    resource: buildNkaAllergy(patientFullUrl),
  });

  return {
    resourceType: "Bundle",
    type: "collection",
    timestamp: isoTimestamp(now),
    identifier: { system: BUNDLE_ID_SYS, value: `medomni-export-${uuid()}` },
    entry: entries,
  };
}

/**
 * Quick one-line summary used by the Share UI.
 * "1 Patient · 4 Conditions · 14 Observations · 4 Medications · 1 AllergyIntolerance · ~38 KB"
 */
export interface FhirBundleStats {
  resourceCounts: Record<string, number>;
  totalEntries: number;
  byteSize: number;
}

export function bundleStats(b: FhirBundle): FhirBundleStats {
  const counts: Record<string, number> = {};
  for (const e of b.entry) {
    const t = e.resource.resourceType;
    counts[t] = (counts[t] ?? 0) + 1;
  }
  // Pretty JSON byte-size (matches what we offer for download).
  const json = JSON.stringify(b, null, 2);
  // TextEncoder is available in Node ≥18 and modern browsers.
  const enc = (globalThis as { TextEncoder?: typeof TextEncoder }).TextEncoder;
  const byteSize = enc ? new enc().encode(json).length : json.length;
  return { resourceCounts: counts, totalEntries: b.entry.length, byteSize };
}

export function formatBundleSummary(b: FhirBundle): string {
  const s = bundleStats(b);
  const order = [
    "Patient",
    "Condition",
    "Observation",
    "MedicationRequest",
    "AllergyIntolerance",
  ];
  const segs = order
    .filter((t) => s.resourceCounts[t])
    .map((t) => `${s.resourceCounts[t]} ${t}${s.resourceCounts[t] === 1 ? "" : "s"}`);
  const kb = (s.byteSize / 1024).toFixed(1);
  return `${segs.join(" · ")} · ~${kb} KB`;
}
