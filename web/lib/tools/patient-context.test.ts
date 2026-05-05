// Inline tests for patient-context.ts. No test runner is configured in
// web/package.json, so this file is a runnable assertion script intended
// for `tsx` or `node --import tsx`. Exits non-zero on first failure.
//
// Run:
//   cd web && npx tsx lib/tools/patient-context.test.ts
//
// Covers (per S2 spec):
//   1. Happy path — Synthea-shaped Bundles
//   2. Missing env vars → throw
//   3. Patient-not-found (404) → throw
//   4. Partial failure (Patient OK, Observation 503) → degraded section
//   5. Truncation flag set when over caps

import { getPatientContext } from "./patient-context";

// --- minimal test harness -------------------------------------------------

let passed = 0;
let failed = 0;
const failures: string[] = [];

async function test(name: string, fn: () => Promise<void> | void): Promise<void> {
  try {
    await fn();
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

// --- env / fetch mock plumbing --------------------------------------------

const origFetch = globalThis.fetch;
const origBaseUrl = process.env.MEDOMNI_FHIR_BASE_URL;
const origToken = process.env.MEDOMNI_FHIR_TOKEN;

function setEnv(base: string | undefined, token: string | undefined): void {
  if (base === undefined) delete process.env.MEDOMNI_FHIR_BASE_URL;
  else process.env.MEDOMNI_FHIR_BASE_URL = base;
  if (token === undefined) delete process.env.MEDOMNI_FHIR_TOKEN;
  else process.env.MEDOMNI_FHIR_TOKEN = token;
}

function restoreEnv(): void {
  setEnv(origBaseUrl, origToken);
}

function jsonResponse(body: unknown, init: { status?: number } = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "Content-Type": "application/fhir+json" },
  });
}

type FetchHandler = (urlStr: string) => Response | Promise<Response>;

function installFetch(handler: FetchHandler): void {
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const urlStr = typeof input === "string" ? input : input.toString();
    return handler(urlStr);
  }) as typeof fetch;
}

function restoreFetch(): void {
  globalThis.fetch = origFetch;
}

// --- Synthea-shaped fixtures ---------------------------------------------

function syntheaPatient() {
  return {
    resourceType: "Patient",
    id: "synthea-pt-1",
    name: [{ given: ["Jane"], family: "Doe" }],
    gender: "female",
    birthDate: "1962-04-12",
  };
}

function bundle<T>(resources: T[]): { resourceType: "Bundle"; entry: Array<{ resource: T }> } {
  return {
    resourceType: "Bundle",
    entry: resources.map((r) => ({ resource: r })),
  };
}

function condition(display: string): unknown {
  return {
    resourceType: "Condition",
    id: `cond-${display.toLowerCase().replace(/\s+/g, "-")}`,
    code: { coding: [{ system: "http://snomed.info/sct", code: "X", display }] },
    clinicalStatus: { coding: [{ code: "active" }] },
    onsetDateTime: "2020-06-15",
  };
}

function vitalObs(display: string, value: number, unit: string, dateIso: string): unknown {
  return {
    resourceType: "Observation",
    id: `obs-${display.toLowerCase().replace(/\s+/g, "-")}`,
    code: { coding: [{ display }] },
    category: [
      { coding: [{ system: "http://terminology.hl7.org/CodeSystem/observation-category", code: "vital-signs" }] },
    ],
    effectiveDateTime: dateIso,
    valueQuantity: { value, unit },
    status: "final",
  };
}

function labObs(display: string, value: number, unit: string, dateIso: string): unknown {
  return {
    resourceType: "Observation",
    id: `lab-${display.toLowerCase().replace(/\s+/g, "-")}`,
    code: { coding: [{ display }] },
    category: [{ coding: [{ code: "laboratory" }] }],
    effectiveDateTime: dateIso,
    valueQuantity: { value, unit },
  };
}

function staleObs(): unknown {
  return labObs("Glucose", 92, "mg/dL", "2010-01-01"); // way outside 12mo window
}

function socialHistoryObs(): unknown {
  // Should be filtered out (not vital-signs / not laboratory).
  return {
    resourceType: "Observation",
    id: "obs-social",
    code: { coding: [{ display: "Tobacco smoking status" }] },
    category: [{ coding: [{ code: "social-history" }] }],
    effectiveDateTime: new Date().toISOString(),
    valueCodeableConcept: { text: "Never smoker" },
  };
}

function medRequest(name: string, dose: string): unknown {
  return {
    resourceType: "MedicationRequest",
    id: `med-${name.toLowerCase().replace(/\s+/g, "-")}`,
    status: "active",
    medicationCodeableConcept: { coding: [{ display: name }] },
    dosageInstruction: [{ text: dose }],
  };
}

function allergy(name: string, severity: string): unknown {
  return {
    resourceType: "AllergyIntolerance",
    id: `allergy-${name.toLowerCase()}`,
    code: { coding: [{ display: name }] },
    criticality: severity,
    reaction: [{ manifestation: [{ coding: [{ display: "rash" }] }] }],
  };
}

function diagnosticReport(name: string, dateIso: string, conclusion: string): unknown {
  return {
    resourceType: "DiagnosticReport",
    id: `dr-${name.toLowerCase().replace(/\s+/g, "-")}`,
    code: { coding: [{ display: name }] },
    effectiveDateTime: dateIso,
    conclusion,
  };
}

function imagingStudy(opts: {
  id: string;
  startedIso: string;
  modalityCode: string;
  modalityDisplay: string;
  description: string;
  radiologist: string;
  series?: number;
  instances?: number;
}): unknown {
  return {
    resourceType: "ImagingStudy",
    id: opts.id,
    status: "available",
    modality: [
      {
        system: "http://dicom.nema.org/resources/ontology/DCM",
        code: opts.modalityCode,
        display: opts.modalityDisplay,
      },
    ],
    started: opts.startedIso,
    description: opts.description,
    numberOfSeries: opts.series ?? 1,
    numberOfInstances: opts.instances ?? 1,
    interpreter: [{ display: opts.radiologist }],
    note: [{ text: opts.description }],
  };
}

// --- now() in spec uses real wall clock; for fixtures we need real "recent" timestamps
function recentIso(daysAgo: number): string {
  const d = new Date(Date.now() - daysAgo * 24 * 60 * 60 * 1000);
  return d.toISOString();
}

// --- the tests ------------------------------------------------------------

async function runAll(): Promise<void> {
  await test("missing env vars throws", async () => {
    setEnv(undefined, undefined);
    let threw = false;
    try {
      await getPatientContext({ patientId: "synthea-pt-1" });
    } catch (e) {
      threw = true;
      assert(
        /MEDOMNI_FHIR_BASE_URL/.test((e as Error).message),
        "error message should mention env vars",
      );
    }
    assert(threw, "expected throw on missing env");
  });

  await test("patient-not-found (404) throws", async () => {
    setEnv("http://fhir.local/fhir/R4", "tok");
    installFetch((url) => {
      if (url.includes("/Patient/")) {
        return new Response(
          JSON.stringify({ resourceType: "OperationOutcome", issue: [{ diagnostics: "not found" }] }),
          { status: 404 },
        );
      }
      return jsonResponse(bundle([]));
    });
    let threw = false;
    try {
      await getPatientContext({ patientId: "missing-pt" });
    } catch (e) {
      threw = true;
      assert(/404/.test((e as Error).message), `expected 404 in error, got: ${(e as Error).message}`);
    }
    assert(threw, "expected throw on 404");
    restoreFetch();
  });

  await test("happy path renders all sections", async () => {
    setEnv("http://fhir.local/fhir/R4", "tok");
    installFetch((url) => {
      if (url.includes("/Patient/")) return jsonResponse(syntheaPatient());
      if (url.includes("Condition?")) {
        return jsonResponse(bundle([condition("Type 2 diabetes mellitus"), condition("Hypertension")]));
      }
      if (url.includes("Observation?")) {
        return jsonResponse(
          bundle([
            vitalObs("Systolic BP", 142, "mm[Hg]", recentIso(3)),
            labObs("HbA1c", 7.6, "%", recentIso(30)),
            staleObs(),
            socialHistoryObs(),
          ]),
        );
      }
      if (url.includes("MedicationRequest?")) {
        return jsonResponse(bundle([medRequest("Metformin", "500 mg PO BID")]));
      }
      if (url.includes("AllergyIntolerance?")) {
        return jsonResponse(bundle([allergy("Penicillin", "high")]));
      }
      if (url.includes("DiagnosticReport?")) {
        return jsonResponse(
          bundle([diagnosticReport("Lipid panel", recentIso(60), "LDL elevated.")]),
        );
      }
      if (url.includes("ImagingStudy?")) {
        return jsonResponse(
          bundle([
            imagingStudy({
              id: "study-cxr-1",
              startedIso: recentIso(45),
              modalityCode: "CR",
              modalityDisplay: "Computed Radiography",
              description: "Chest, 2-view — No acute findings.",
              radiologist: "Dr. M. Hsu",
              series: 2,
              instances: 2,
            }),
          ]),
        );
      }
      return jsonResponse(bundle([]));
    });

    const out = await getPatientContext({ patientId: "synthea-pt-1", queryHint: "diabetes mgmt" });
    assert(out.block.startsWith("## Patient"), "block must start with Patient section");
    assert(out.block.includes("Jane Doe"), "patient name missing");
    assert(out.block.includes("## Active Conditions"), "missing Conditions header");
    assert(out.block.includes("Type 2 diabetes mellitus"), "missing condition label");
    assert(out.block.includes("## Recent Vitals/Labs"), "missing Vitals/Labs header");
    assert(out.block.includes("Systolic BP"), "missing vital");
    assert(out.block.includes("HbA1c"), "missing lab");
    assert(!out.block.includes("Tobacco"), "social-history must be filtered out");
    assert(!out.block.includes("Glucose"), "stale (>12mo) observation must be filtered out");
    assert(out.block.includes("Metformin"), "missing medication");
    assert(out.block.includes("Penicillin"), "missing allergy");
    assert(out.block.includes("Lipid panel"), "missing diagnostic report");
    assert(out.block.includes("## Imaging Studies"), "missing Imaging Studies header");
    assert(
      out.block.includes("Computed Radiography"),
      "missing imaging modality display",
    );
    assert(
      out.block.includes("Chest, 2-view"),
      "missing imaging description text",
    );
    assert(
      out.block.includes("Dr. M. Hsu"),
      "missing imaging interpreter display",
    );
    assert(out.block.includes("diabetes mgmt"), "queryHint must surface in block");
    assert(out.truncated === false, "truncated should be false");
    assert(out.resourceCounts.Patient === 1, "patient count should be 1");
    assert(out.resourceCounts.Condition === 2, "should have 2 conditions");
    assert(out.resourceCounts.Observation === 2, "should have 2 observations after filtering");
    assert(out.resourceCounts.MedicationRequest === 1, "should have 1 med");
    assert(out.resourceCounts.AllergyIntolerance === 1, "should have 1 allergy");
    assert(out.resourceCounts.DiagnosticReport === 1, "should have 1 report");
    assert(out.resourceCounts.ImagingStudy === 1, "should have 1 imaging study");
    restoreFetch();
  });

  await test("imaging studies are sorted by `started` desc and capped at 5", async () => {
    setEnv("http://fhir.local/fhir/R4", "tok");
    // Build 7 studies with mixed dates — newer ones must surface first, and
    // the cap (5) must trip the truncated flag.
    const studies = [
      imagingStudy({
        id: "old1", startedIso: recentIso(400),
        modalityCode: "MR", modalityDisplay: "Magnetic Resonance Imaging",
        description: "Old MRI", radiologist: "Dr. A",
      }),
      imagingStudy({
        id: "new1", startedIso: recentIso(5),
        modalityCode: "CR", modalityDisplay: "Computed Radiography",
        description: "Recent CXR", radiologist: "Dr. B",
      }),
      imagingStudy({
        id: "mid1", startedIso: recentIso(120),
        modalityCode: "CT", modalityDisplay: "Computed Tomography",
        description: "Mid CT", radiologist: "Dr. C",
      }),
      imagingStudy({
        id: "mid2", startedIso: recentIso(60),
        modalityCode: "US", modalityDisplay: "Ultrasound",
        description: "Mid US", radiologist: "Dr. D",
      }),
      imagingStudy({
        id: "old2", startedIso: recentIso(800),
        modalityCode: "MR", modalityDisplay: "Magnetic Resonance Imaging",
        description: "Older MRI", radiologist: "Dr. E",
      }),
      imagingStudy({
        id: "new2", startedIso: recentIso(2),
        modalityCode: "CR", modalityDisplay: "Computed Radiography",
        description: "Newest CXR", radiologist: "Dr. F",
      }),
      imagingStudy({
        id: "mid3", startedIso: recentIso(30),
        modalityCode: "MG", modalityDisplay: "Mammography",
        description: "Mid MG", radiologist: "Dr. G",
      }),
    ];
    installFetch((url) => {
      if (url.includes("/Patient/")) return jsonResponse(syntheaPatient());
      if (url.includes("ImagingStudy?")) return jsonResponse(bundle(studies));
      return jsonResponse(bundle([]));
    });
    const out = await getPatientContext({ patientId: "synthea-pt-1" });
    assert(out.truncated === true, "truncated should fire when over the cap of 5");
    assert(
      out.resourceCounts.ImagingStudy === 5,
      `cap is 5, got ${out.resourceCounts.ImagingStudy}`,
    );
    // Newest descriptions must be rendered; oldest ("Older MRI") must be cut.
    assert(out.block.includes("Newest CXR"), "newest study must render");
    assert(out.block.includes("Recent CXR"), "second-newest study must render");
    assert(!out.block.includes("Older MRI"), "oldest (>800d) study must be dropped by the cap");
    restoreFetch();
  });

  await test("imaging studies render `(none on record)` when bundle is empty", async () => {
    setEnv("http://fhir.local/fhir/R4", "tok");
    installFetch((url) => {
      if (url.includes("/Patient/")) return jsonResponse(syntheaPatient());
      return jsonResponse(bundle([]));
    });
    const out = await getPatientContext({ patientId: "synthea-pt-1" });
    assert(
      /## Imaging Studies\n- \(none on record\)/.test(out.block),
      "imaging empty-state must render literal `- (none on record)`",
    );
    assert(out.resourceCounts.ImagingStudy === 0, "imaging count should be 0");
    restoreFetch();
  });

  await test("partial failure (Observation 503) → degraded section", async () => {
    setEnv("http://fhir.local/fhir/R4", "tok");
    installFetch((url) => {
      if (url.includes("/Patient/")) return jsonResponse(syntheaPatient());
      if (url.includes("Observation?")) {
        return new Response("upstream gone", { status: 503 });
      }
      if (url.includes("Condition?")) return jsonResponse(bundle([condition("Asthma")]));
      return jsonResponse(bundle([]));
    });
    const out = await getPatientContext({ patientId: "synthea-pt-1" });
    assert(
      /Observations unavailable: 503/.test(out.block),
      `expected degraded note, got block:\n${out.block}`,
    );
    assert(out.block.includes("Asthma"), "other sections must still render");
    assert(out.resourceCounts.Observation === 0, "observation count should be 0");
    restoreFetch();
  });

  await test("truncation flag set when over caps", async () => {
    setEnv("http://fhir.local/fhir/R4", "tok");
    const manyConditions = Array.from({ length: 12 }, (_, i) => condition(`Condition ${i}`));
    installFetch((url) => {
      if (url.includes("/Patient/")) return jsonResponse(syntheaPatient());
      if (url.includes("Condition?")) return jsonResponse(bundle(manyConditions));
      return jsonResponse(bundle([]));
    });
    const out = await getPatientContext({ patientId: "synthea-pt-1" });
    assert(out.truncated === true, "truncated should be true with 12 conditions");
    assert(out.resourceCounts.Condition === 8, `cap is 8, got ${out.resourceCounts.Condition}`);
    restoreFetch();
  });
}

void (async () => {
  try {
    await runAll();
  } finally {
    restoreEnv();
    restoreFetch();
  }
  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) {
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(1);
  }
})();
