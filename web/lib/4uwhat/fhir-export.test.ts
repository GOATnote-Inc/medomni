// Inline tests for fhir-export.ts. No test runner is configured in
// web/package.json, so this file is a runnable assertion script in the
// same shape as patient-context.test.ts. Exits non-zero on first failure.
//
// Run:
//   cd web && npx tsx lib/4uwhat/fhir-export.test.ts
//
// Covers (per B2 spec):
//   1. resourceType: "Bundle" + type: "collection"
//   2. exactly one Patient resource
//   3. one Condition per SAMPLE_CONDITIONS
//   4. one Observation per (SAMPLE_VITALS + SAMPLE_LABS)
//   5. JSON-serializable round-trip with no loss
//   6. formatBundleSummary produces a non-empty summary string

import {
  buildFhirBundle,
  bundleStats,
  formatBundleSummary,
} from "./fhir-export";
import {
  SAMPLE_CONDITIONS,
  SAMPLE_LABS,
  SAMPLE_VITALS,
  SAMPLE_MEDS,
  SAMPLE_IMAGING,
} from "./sample-data";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function test(name: string, fn: () => void): void {
  try {
    fn();
    passed += 1;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failed += 1;
    const msg = `${name}: ${(e as Error).message}`;
    failures.push(msg);
    console.error(`  FAIL ${msg}`);
  }
}

function assert(cond: unknown, msg: string): void {
  if (!cond) throw new Error(msg);
}

function assertEq<T>(actual: T, expected: T, msg: string): void {
  if (actual !== expected) {
    throw new Error(`${msg} (expected ${String(expected)}, got ${String(actual)})`);
  }
}

console.log("fhir-export.test.ts");

test("Bundle has resourceType: Bundle and type: collection", () => {
  const b = buildFhirBundle();
  assertEq(b.resourceType, "Bundle", "resourceType");
  assertEq(b.type, "collection", "type");
  assert(typeof b.timestamp === "string", "timestamp present");
  // ISO 8601 sanity.
  assert(/^\d{4}-\d{2}-\d{2}T/.test(b.timestamp), "timestamp ISO-ish");
  assert(b.identifier && b.identifier.value.startsWith("medomni-export-"),
    "identifier value tagged medomni-export-*");
});

test("Bundle.entry includes exactly one Patient resource", () => {
  const b = buildFhirBundle();
  const patients = b.entry.filter((e) => e.resource.resourceType === "Patient");
  assertEq(patients.length, 1, "patient count");
  const p = patients[0].resource;
  if (p.resourceType !== "Patient") throw new Error("type narrow failed");
  assert(p.id !== "", "patient has id");
  assert(p.identifier && p.identifier[0]?.value === "P42-0096-MAYA", "MRN identifier");
  assert(p.birthDate === "1991-04-12", "birthDate matches sample-data");
});

test("Bundle.entry includes one Condition per SAMPLE_CONDITIONS", () => {
  const b = buildFhirBundle();
  const conds = b.entry.filter((e) => e.resource.resourceType === "Condition");
  assertEq(conds.length, SAMPLE_CONDITIONS.length, "condition count");
  // ICD codes round-tripped.
  const icds = new Set(SAMPLE_CONDITIONS.map((c) => c.icd));
  for (const c of conds) {
    if (c.resource.resourceType !== "Condition") throw new Error("type narrow");
    const code = c.resource.code.coding?.[0]?.code;
    assert(code !== undefined && icds.has(code),
      `condition coding code present and from sample-data (${code})`);
  }
});

test("Bundle.entry includes one Observation per (SAMPLE_VITALS + SAMPLE_LABS)", () => {
  const b = buildFhirBundle();
  const obs = b.entry.filter((e) => e.resource.resourceType === "Observation");
  const expected = Object.keys(SAMPLE_VITALS).length + SAMPLE_LABS.length;
  assertEq(obs.length, expected, "observation count");
  // Categories present.
  const cats = obs
    .map((e) => {
      if (e.resource.resourceType !== "Observation") return "";
      return e.resource.category?.[0]?.coding?.[0]?.code ?? "";
    })
    .filter(Boolean);
  assert(cats.includes("vital-signs"), "vital-signs category present");
  assert(cats.includes("laboratory"), "laboratory category present");
});

test("Bundle.entry includes one MedicationRequest per SAMPLE_MEDS", () => {
  const b = buildFhirBundle();
  const meds = b.entry.filter((e) => e.resource.resourceType === "MedicationRequest");
  assertEq(meds.length, SAMPLE_MEDS.length, "medication count");
});

test("Bundle includes exactly one AllergyIntolerance (NKA stub)", () => {
  const b = buildFhirBundle();
  const al = b.entry.filter((e) => e.resource.resourceType === "AllergyIntolerance");
  assertEq(al.length, 1, "allergy count");
});

test("Bundle.entry includes one ImagingStudy per SAMPLE_IMAGING", () => {
  const b = buildFhirBundle();
  const studies = b.entry.filter((e) => e.resource.resourceType === "ImagingStudy");
  assertEq(studies.length, SAMPLE_IMAGING.length, "imaging count");
  for (const s of studies) {
    if (s.resource.resourceType !== "ImagingStudy") throw new Error("type narrow");
    assertEq(s.resource.status, "available", `${s.resource.id} status`);
    assert(typeof s.resource.subject?.reference === "string"
      && s.resource.subject.reference.startsWith("urn:uuid:"),
      `${s.resource.id} has urn:uuid subject reference`);
    assert(Array.isArray(s.resource.modality) && s.resource.modality.length === 1,
      `${s.resource.id} has exactly one modality coding`);
    assert(typeof s.resource.started === "string" && s.resource.started.length > 0,
      `${s.resource.id} has started date`);
    // Description carries region + read narrative.
    assert(typeof s.resource.description === "string"
      && s.resource.description.includes("—"),
      `${s.resource.id} description has region + read narrative`);
    // Interpreter present (display-only Practitioner reference).
    assert(Array.isArray(s.resource.interpreter)
      && s.resource.interpreter.length === 1
      && typeof s.resource.interpreter[0]?.display === "string",
      `${s.resource.id} has display-only interpreter`);
    // numberOfSeries / numberOfInstances populated (>= 1).
    assert(typeof s.resource.numberOfSeries === "number" && s.resource.numberOfSeries >= 1,
      `${s.resource.id} numberOfSeries >= 1`);
    assert(typeof s.resource.numberOfInstances === "number" && s.resource.numberOfInstances >= 1,
      `${s.resource.id} numberOfInstances >= 1`);
  }
});

test("ImagingStudy modality codings honor DCM-vs-text policy", () => {
  const b = buildFhirBundle();
  const byKind = new Map<string, string>();
  for (const im of SAMPLE_IMAGING) byKind.set(`imaging-${im.id}`, im.kind);
  const studies = b.entry.filter((e) => e.resource.resourceType === "ImagingStudy");
  for (const s of studies) {
    if (s.resource.resourceType !== "ImagingStudy") throw new Error("type narrow");
    const kind = byKind.get(s.resource.id) ?? "";
    const m = s.resource.modality[0];
    if (kind === "X-ray") {
      assertEq(m.system, "http://dicom.nema.org/resources/ontology/DCM",
        "X-ray modality system DCM");
      assertEq(m.code, "CR", "X-ray modality code CR");
    } else if (kind === "MRI") {
      assertEq(m.system, "http://dicom.nema.org/resources/ontology/DCM",
        "MRI modality system DCM");
      assertEq(m.code, "MR", "MRI modality code MR");
    } else {
      // Panoramic / other — no high-confidence DCM pin; display-only.
      assert(m.system === undefined && m.code === undefined,
        `${kind} modality should be display-only (no system/code)`);
      assert(typeof m.display === "string" && m.display.length > 0,
        `${kind} modality has display string`);
    }
  }
});

test("ImagingStudy fullUrls are deterministic across rebuilds", () => {
  const a = buildFhirBundle(new Date("2026-05-04T00:00:00.000Z"));
  const b = buildFhirBundle(new Date("2026-05-04T12:00:00.000Z"));
  const studyUrlsA = a.entry
    .filter((e) => e.resource.resourceType === "ImagingStudy")
    .map((e) => e.fullUrl);
  const studyUrlsB = b.entry
    .filter((e) => e.resource.resourceType === "ImagingStudy")
    .map((e) => e.fullUrl);
  assertEq(studyUrlsA.length, studyUrlsB.length, "study count stable");
  for (let i = 0; i < studyUrlsA.length; i++) {
    assertEq(studyUrlsA[i], studyUrlsB[i],
      `ImagingStudy fullUrl deterministic across rebuilds at index ${i}`);
  }
});

test("Each entry has urn:uuid: fullUrl", () => {
  const b = buildFhirBundle();
  for (const e of b.entry) {
    assert(/^urn:uuid:[0-9a-f-]{36}$/.test(e.fullUrl),
      `fullUrl is urn:uuid (${e.fullUrl})`);
  }
});

test("All Conditions/Observations/Meds/Allergies/ImagingStudy subject reference the Patient", () => {
  const b = buildFhirBundle();
  const patient = b.entry.find((e) => e.resource.resourceType === "Patient");
  assert(patient !== undefined, "patient entry exists");
  const pUrl = patient!.fullUrl;
  for (const e of b.entry) {
    const r = e.resource;
    if (r.resourceType === "Patient") continue;
    let ref = "";
    if (
      r.resourceType === "Condition" ||
      r.resourceType === "Observation" ||
      r.resourceType === "MedicationRequest" ||
      r.resourceType === "ImagingStudy"
    ) {
      ref = r.subject.reference;
    } else if (r.resourceType === "AllergyIntolerance") {
      ref = r.patient.reference;
    }
    assertEq(ref, pUrl, `${r.resourceType} ${r.id} subject reference -> patient`);
  }
});

test("Bundle round-trips through JSON.parse(JSON.stringify(...)) identically", () => {
  const b = buildFhirBundle(new Date("2026-05-04T00:00:00.000Z"));
  const json = JSON.stringify(b);
  const round = JSON.parse(json);
  // Deep equality via re-stringify; any non-JSON-safe fields (Date,
  // Function, undefined) would differ here.
  assertEq(JSON.stringify(round), json, "round-trip stable");
  assertEq(round.resourceType, "Bundle", "round-trip resourceType");
  assertEq(round.entry.length, b.entry.length, "round-trip entry count");
});

test("formatBundleSummary produces a non-empty human summary including ImagingStudy", () => {
  const b = buildFhirBundle();
  const summary = formatBundleSummary(b);
  assert(summary.length > 0, "non-empty");
  assert(summary.includes("Patient"), "mentions Patient");
  assert(summary.includes("ImagingStudy"), "mentions ImagingStudy");
  assert(summary.includes("KB"), "mentions size");
  const stats = bundleStats(b);
  assert(stats.totalEntries === b.entry.length, "stats totalEntries matches");
  assert(stats.byteSize > 0, "byteSize > 0");
  assertEq(stats.resourceCounts.ImagingStudy ?? 0, SAMPLE_IMAGING.length,
    "stats include ImagingStudy count");
});

console.log(`\n  ${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.error("FAILURES:");
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
