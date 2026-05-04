// Direct latency probe for the new get_patient_context tool — bypasses
// the full agent loop and measures the FHIR-fetch leg in isolation.
//
// Why: Agent E's projected 25-95ms for graph traversal extrapolated from
// cuGraph benchmarks. The unmeasured cost is the FHIR fetch + bundle
// assembly. This probe isolates that leg by calling getPatientContext
// directly against the local Medplum sandbox, with N runs per patient,
// reporting p50/p95/p99.
//
// Prereqs:
//   1. scripts/spike/setup_synthea_medplum.sh has run; Medplum is up at
//      http://127.0.0.1:8103 with >= 10 Synthea patients loaded.
//   2. scripts/spike/.spike-token exists with a valid bearer token.
//
// Run:
//   cd /Users/kiteboard/medomni/web && npx tsx ../scripts/spike/bench/probe_patient_context.ts
//
// Output: Markdown table to stdout with per-patient + aggregate p50/p95/p99
// of the get_patient_context tool's wall-clock latency.

import { readFileSync } from "fs";
import { resolve } from "path";
import { getPatientContext } from "../../../web/lib/tools/patient-context";

const FHIR_BASE = "http://127.0.0.1:8103/fhir/R4";
const TOKEN_PATH = resolve(__dirname, "../.spike-token");
const RUNS_PER_PATIENT = 5;
const WARMUP_RUNS = 1;

function pct(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

function fmt(n: number): string {
  return n.toFixed(0);
}

async function listPatientIds(token: string): Promise<string[]> {
  const r = await fetch(`${FHIR_BASE}/Patient?_count=20&_total=accurate`, {
    headers: { Authorization: `Bearer ${token}`, Accept: "application/fhir+json" },
  });
  if (!r.ok) throw new Error(`Patient list failed: HTTP ${r.status} ${await r.text()}`);
  const bundle = (await r.json()) as { entry?: Array<{ resource: { id: string } }> };
  return (bundle.entry ?? []).map((e) => e.resource.id);
}

async function main() {
  const token = readFileSync(TOKEN_PATH, "utf8").trim();
  if (!token) throw new Error(`empty token at ${TOKEN_PATH}`);

  process.env.MEDOMNI_FHIR_BASE_URL = FHIR_BASE;
  process.env.MEDOMNI_FHIR_TOKEN = token;

  console.log("# Pattern B FHIR-fetch latency probe");
  console.log(`- Medplum: ${FHIR_BASE}`);
  console.log(`- Runs per patient: ${RUNS_PER_PATIENT} (after ${WARMUP_RUNS} warmup)`);
  console.log("");

  const patientIds = await listPatientIds(token);
  if (patientIds.length === 0) throw new Error("no patients in Medplum");
  console.log(`Discovered ${patientIds.length} patients\n`);

  const allMs: number[] = [];
  const allResourceCounts: number[] = [];
  let truncatedCount = 0;
  let errCount = 0;

  console.log("| patient | n | p50 ms | p95 ms | resources (P/C/O/M/A/D) | truncated |");
  console.log("|---|---|---|---|---|---|");

  for (const pid of patientIds) {
    // warmup
    for (let i = 0; i < WARMUP_RUNS; i++) {
      try {
        await getPatientContext({ patientId: pid });
      } catch {}
    }
    const ms: number[] = [];
    let lastResult: Awaited<ReturnType<typeof getPatientContext>> | null = null;
    for (let i = 0; i < RUNS_PER_PATIENT; i++) {
      const t0 = Date.now();
      try {
        lastResult = await getPatientContext({ patientId: pid });
        ms.push(Date.now() - t0);
      } catch (e) {
        errCount++;
      }
    }
    if (lastResult && ms.length > 0) {
      const rc = lastResult.resourceCounts;
      const total = rc.Patient + rc.Condition + rc.Observation + rc.MedicationRequest + rc.AllergyIntolerance + rc.DiagnosticReport;
      allMs.push(...ms);
      allResourceCounts.push(total);
      if (lastResult.truncated) truncatedCount++;
      console.log(
        `| ${pid.slice(0, 8)}… | ${ms.length} | ${fmt(pct(ms, 50))} | ${fmt(pct(ms, 95))} | ${rc.Patient}/${rc.Condition}/${rc.Observation}/${rc.MedicationRequest}/${rc.AllergyIntolerance}/${rc.DiagnosticReport} | ${lastResult.truncated ? "yes" : "no"} |`,
      );
    }
  }

  console.log("\n## Aggregate (all patients, all runs)");
  console.log(`- N samples: ${allMs.length}`);
  console.log(`- Errors: ${errCount}`);
  console.log(`- Truncated: ${truncatedCount}/${patientIds.length} patients`);
  console.log(`- Latency p50: ${fmt(pct(allMs, 50))} ms`);
  console.log(`- Latency p95: ${fmt(pct(allMs, 95))} ms`);
  console.log(`- Latency p99: ${fmt(pct(allMs, 99))} ms`);
  console.log(`- Mean resources per patient: ${fmt(allResourceCounts.reduce((a, b) => a + b, 0) / allResourceCounts.length)}`);

  console.log("\n## Decision criteria");
  const p95 = pct(allMs, 95);
  if (p95 < 500) {
    console.log(`Pattern B FHIR-fetch p95 = ${fmt(p95)} ms — well within budget. SHIP.`);
  } else if (p95 < 1500) {
    console.log(`Pattern B FHIR-fetch p95 = ${fmt(p95)} ms — fits if existing PrimeKG p95 < 500ms. Verify in full bench.`);
  } else {
    console.log(`Pattern B FHIR-fetch p95 = ${fmt(p95)} ms — too slow for parallel dispatch. Consider precomputed cache (Pattern C).`);
  }
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
