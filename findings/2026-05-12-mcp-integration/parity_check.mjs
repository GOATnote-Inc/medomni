#!/usr/bin/env node
// Parity check: medomni's hand-rolled clinical_calculate vs HealthCraft MCP's
// applyDecisionRule on the 5 overlapping rules, plus MCP-only canonical-table
// checks on HEART + Wells PE (the unlock demonstrations medomni cannot do).
//
// 50 cases total: 8 per overlapping rule (40) + 5 HEART + 5 Wells PE.
//
// Runs on Node 18+ (built-in fetch). No npm install.
//
// Output:
//   <this-dir>/parity_check_results.json
//   stdout summary: per-rule pass rate + overall pass rate.
//
// The 5 calculator functions below are LIFTED VERBATIM (translated to JS)
// from /Users/kiteboard/medomni/web/lib/tools/clinical-calculator.ts. If that
// file changes, update these to keep parity-test integrity.

import { writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const RESULTS_PATH = join(__dirname, "parity_check_results.json");
const MCP_URL = process.env.MCP_ED_RULES_URL || "https://mcp.thegoatnote.com/mcp";

// ---------------------------------------------------------------------------
// LIFTED medomni calculators (TS -> JS, logic preserved)
// ---------------------------------------------------------------------------

function chadsvasc(args) {
  const age = typeof args.age === "number" ? args.age : 0;
  const sex = String(args.sex ?? "").toLowerCase();
  let s = 0;
  if (args.chf) s += 1;
  if (args.hypertension) s += 1;
  if (age >= 75) s += 2;
  else if (age >= 65) s += 1;
  if (args.diabetes) s += 1;
  if (args.stroke_tia_thromboembolism) s += 2;
  if (args.vascular_disease) s += 1;
  if (sex === "female" || sex === "f") s += 1;
  return { score_name: "CHA2DS2-VASc", score: s };
}

function hasbled(args) {
  const KEYS = [
    "hypertension_uncontrolled", "abnormal_renal", "abnormal_liver",
    "stroke_history", "bleeding_history", "labile_inr",
    "age_over_65", "drugs_predisposing_bleeding", "alcohol_use",
  ];
  let s = 0;
  for (const k of KEYS) if (args[k]) s += 1;
  return { score_name: "HAS-BLED", score: s };
}

function meld(args) {
  const tbili = Math.max(args.bilirubin_mgdl ?? 1.0, 1.0);
  const inr = Math.max(args.inr ?? 1.0, 1.0);
  let creat = Math.max(args.creatinine_mgdl ?? 1.0, 1.0);
  if (args.on_dialysis || creat > 4.0) creat = 4.0;
  const meld_i = Math.round(
    9.57 * Math.log(creat) +
    3.78 * Math.log(tbili) +
    11.2 * Math.log(inr) +
    6.43,
  );
  const meld = Math.max(6, Math.min(40, meld_i));
  let final = meld;
  if (typeof args.sodium_meql === "number" && meld > 11) {
    const na = Math.max(125, Math.min(137, args.sodium_meql));
    final = Math.round(meld + 1.32 * (137 - na) - (0.033 * meld * (137 - na)));
    final = Math.max(6, Math.min(40, final));
  }
  return { score_name: "MELD-Na", score: final };
}

function wellsDvt(args) {
  const FLAGS = [
    "active_cancer",
    "paralysis_paresis_recent_immobilization",
    "bedridden_3d_or_major_surgery_12wk",
    "localized_tenderness_along_deep_veins",
    "entire_leg_swollen",
    "calf_swelling_3cm_vs_other",
    "pitting_edema_symptomatic_leg",
    "collateral_superficial_veins",
    "previously_documented_dvt",
  ];
  let s = 0;
  for (const k of FLAGS) if (args[k]) s += 1;
  if (args.alternative_diagnosis_at_least_as_likely) s -= 2;
  return { score_name: "Wells DVT", score: s };
}

function perc(args) {
  const KEYS = [
    "age_under_50", "hr_under_100", "spo2_at_least_95",
    "no_unilateral_leg_swelling", "no_hemoptysis",
    "no_recent_surgery_trauma_4wk", "no_prior_pe_dvt", "no_estrogen_use",
  ];
  const failed = KEYS.filter((k) => !args[k]);
  return {
    score_name: "PERC",
    score: 8 - failed.length,
    perc_negative: failed.length === 0,
  };
}

// ---------------------------------------------------------------------------
// MCP wire (JSON-RPC over HTTP)
// ---------------------------------------------------------------------------

let nextId = 1;

async function rpc(method, params) {
  const id = nextId++;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 10_000);
  try {
    const r = await fetch(MCP_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
      },
      body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
      signal: ctrl.signal,
    });
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}` };
    const body = await r.json();
    if (body.error) return { ok: false, error: body.error.message };
    return { ok: true, data: body.result };
  } catch (e) {
    return { ok: false, error: String(e?.message ?? e) };
  } finally {
    clearTimeout(timer);
  }
}

async function applyRule(ruleName, variables) {
  const r = await rpc("tools/call", {
    name: "applyDecisionRule",
    arguments: { ruleName, variables },
  });
  if (!r.ok) return r;
  const inner = r.data?.structuredContent ?? r.data;
  const result = inner?.data?.result;
  if (!result || typeof result.score !== "number") {
    return {
      ok: false,
      error: "MCP returned no numeric score: " + JSON.stringify(inner).slice(0, 300),
    };
  }
  return {
    ok: true,
    score: result.score,
    risk_level: result.risk_level,
    recommendation: result.recommendation,
    ruleVersion: inner?.data?.ruleVersionShort,
  };
}

// ---------------------------------------------------------------------------
// Translation: medomni args -> MCP variable dict (exact display names from
// getRuleSchema). PERC has inverted semantics (medomni asks "criterion met",
// MCP asks "criterion present-as-failure").
// ---------------------------------------------------------------------------

function cha2dsToMcp(a) {
  const age = a.age ?? 0;
  const sexFemale = String(a.sex ?? "").toLowerCase().startsWith("f") ? 1 : 0;
  return {
    "Congestive heart failure": a.chf ? 1 : 0,
    "Hypertension": a.hypertension ? 1 : 0,
    "Age >= 75": age >= 75 ? 2 : 0,
    "Age 65-74": age >= 65 && age < 75 ? 1 : 0,
    "Diabetes": a.diabetes ? 1 : 0,
    "Stroke/TIA history": a.stroke_tia_thromboembolism ? 2 : 0,
    "Vascular disease": a.vascular_disease ? 1 : 0,
    "Sex (female)": sexFemale,
  };
}

function hasbledToMcp(a) {
  return {
    "Hypertension uncontrolled": a.hypertension_uncontrolled ? 1 : 0,
    "Abnormal renal function": a.abnormal_renal ? 1 : 0,
    "Abnormal liver function": a.abnormal_liver ? 1 : 0,
    "Stroke history": a.stroke_history ? 1 : 0,
    "Bleeding history": a.bleeding_history ? 1 : 0,
    "Labile INR": a.labile_inr ? 1 : 0,
    "Elderly (>65)": a.age_over_65 ? 1 : 0,
    "Drugs (antiplatelet/NSAID)": a.drugs_predisposing_bleeding ? 1 : 0,
    "Alcohol >= 8 drinks/week": a.alcohol_use ? 1 : 0,
  };
}

function meldToMcp(a) {
  let creat = Math.max(a.creatinine_mgdl ?? 1.0, 1.0);
  if (a.on_dialysis || creat > 4.0) creat = 4.0;
  return {
    "Creatinine (mg/dL)": creat,
    "Bilirubin (mg/dL)": Math.max(a.bilirubin_mgdl ?? 1.0, 1.0),
    "INR": Math.max(a.inr ?? 1.0, 1.0),
    "Sodium (mmol/L)": a.sodium_meql ?? 137,
  };
}

function wellsDvtToMcp(a) {
  return {
    "Active cancer": a.active_cancer ? 1 : 0,
    "Paralysis, paresis, or recent cast": a.paralysis_paresis_recent_immobilization ? 1 : 0,
    "Bedridden > 3 days or major surgery within 12 weeks": a.bedridden_3d_or_major_surgery_12wk ? 1 : 0,
    "Localized tenderness along deep venous system": a.localized_tenderness_along_deep_veins ? 1 : 0,
    "Entire leg swollen": a.entire_leg_swollen ? 1 : 0,
    "Calf swelling > 3 cm compared to other leg": a.calf_swelling_3cm_vs_other ? 1 : 0,
    "Pitting edema": a.pitting_edema_symptomatic_leg ? 1 : 0,
    "Collateral superficial veins": a.collateral_superficial_veins ? 1 : 0,
    "Previously documented DVT": a.previously_documented_dvt ? 1 : 0,
    "Alternative diagnosis at least as likely": a.alternative_diagnosis_at_least_as_likely ? -2 : 0,
  };
}

function percToMcp(a) {
  // INVERTED: medomni's `age_under_50=true` -> MCP `Age >= 50 = 0` (criterion absent).
  return {
    "Age >= 50": a.age_under_50 ? 0 : 1,
    "Heart rate >= 100": a.hr_under_100 ? 0 : 1,
    "SpO2 < 95% on room air": a.spo2_at_least_95 ? 0 : 1,
    "Unilateral leg swelling": a.no_unilateral_leg_swelling ? 0 : 1,
    "Hemoptysis": a.no_hemoptysis ? 0 : 1,
    "Recent surgery or trauma": a.no_recent_surgery_trauma_4wk ? 0 : 1,
    "Prior DVT or PE": a.no_prior_pe_dvt ? 0 : 1,
    "Hormone use (estrogen)": a.no_estrogen_use ? 0 : 1,
  };
}

// ---------------------------------------------------------------------------
// 50 test cases
// ---------------------------------------------------------------------------

const CASES = [
  // CHA2DS2-VASc x 8 (parity vs medomni)
  { id: "cha-01", rule: "CHA2DS2-VASc", inputs: { age: 60, sex: "male" } },
  { id: "cha-02", rule: "CHA2DS2-VASc", inputs: { age: 65, sex: "male", hypertension: true } },
  { id: "cha-03", rule: "CHA2DS2-VASc", inputs: { age: 75, sex: "female", hypertension: true, diabetes: true } },
  { id: "cha-04", rule: "CHA2DS2-VASc", inputs: { age: 80, sex: "male", hypertension: true, chf: true, stroke_tia_thromboembolism: true } },
  { id: "cha-05", rule: "CHA2DS2-VASc", inputs: { age: 50, sex: "female" } },
  { id: "cha-06", rule: "CHA2DS2-VASc", inputs: { age: 70, sex: "male", diabetes: true, vascular_disease: true } },
  { id: "cha-07", rule: "CHA2DS2-VASc", inputs: { age: 85, sex: "female", chf: true, hypertension: true, diabetes: true, stroke_tia_thromboembolism: true, vascular_disease: true } },
  { id: "cha-08", rule: "CHA2DS2-VASc", inputs: { age: 65, sex: "female", vascular_disease: true } },

  // HAS-BLED x 8
  { id: "hb-01", rule: "HAS-BLED", inputs: {} },
  { id: "hb-02", rule: "HAS-BLED", inputs: { hypertension_uncontrolled: true } },
  { id: "hb-03", rule: "HAS-BLED", inputs: { hypertension_uncontrolled: true, abnormal_renal: true } },
  { id: "hb-04", rule: "HAS-BLED", inputs: { stroke_history: true, bleeding_history: true } },
  { id: "hb-05", rule: "HAS-BLED", inputs: { labile_inr: true, age_over_65: true, drugs_predisposing_bleeding: true } },
  { id: "hb-06", rule: "HAS-BLED", inputs: { hypertension_uncontrolled: true, abnormal_renal: true, abnormal_liver: true, stroke_history: true, bleeding_history: true, labile_inr: true, age_over_65: true, drugs_predisposing_bleeding: true, alcohol_use: true } },
  { id: "hb-07", rule: "HAS-BLED", inputs: { hypertension_uncontrolled: true, age_over_65: true, alcohol_use: true } },
  { id: "hb-08", rule: "HAS-BLED", inputs: { abnormal_liver: true, bleeding_history: true, drugs_predisposing_bleeding: true } },

  // MELD-Na x 8 (allow +/- 1 rounding tolerance)
  { id: "meld-01", rule: "MELD-Na", inputs: { bilirubin_mgdl: 1.0, inr: 1.0, creatinine_mgdl: 1.0, sodium_meql: 137 } },
  { id: "meld-02", rule: "MELD-Na", inputs: { bilirubin_mgdl: 2.0, inr: 1.5, creatinine_mgdl: 1.5, sodium_meql: 132 } },
  { id: "meld-03", rule: "MELD-Na", inputs: { bilirubin_mgdl: 5.0, inr: 2.0, creatinine_mgdl: 2.0, sodium_meql: 128 } },
  { id: "meld-04", rule: "MELD-Na", inputs: { bilirubin_mgdl: 0.8, inr: 0.9, creatinine_mgdl: 0.7, sodium_meql: 138 } },
  { id: "meld-05", rule: "MELD-Na", inputs: { bilirubin_mgdl: 10.0, inr: 3.0, creatinine_mgdl: 3.5, sodium_meql: 130 } },
  { id: "meld-06", rule: "MELD-Na", inputs: { bilirubin_mgdl: 1.5, inr: 1.2, creatinine_mgdl: 1.2, sodium_meql: 135 } },
  { id: "meld-07", rule: "MELD-Na", inputs: { bilirubin_mgdl: 3.0, inr: 1.8, creatinine_mgdl: 2.5, sodium_meql: 128 } },
  { id: "meld-08", rule: "MELD-Na", inputs: { bilirubin_mgdl: 1.0, inr: 1.0, creatinine_mgdl: 4.0, on_dialysis: true, sodium_meql: 125 } },

  // Wells DVT x 8
  { id: "wd-01", rule: "Wells DVT", inputs: {} },
  { id: "wd-02", rule: "Wells DVT", inputs: { active_cancer: true, localized_tenderness_along_deep_veins: true } },
  { id: "wd-03", rule: "Wells DVT", inputs: { calf_swelling_3cm_vs_other: true, entire_leg_swollen: true, localized_tenderness_along_deep_veins: true, pitting_edema_symptomatic_leg: true } },
  { id: "wd-04", rule: "Wells DVT", inputs: { active_cancer: true, paralysis_paresis_recent_immobilization: true, bedridden_3d_or_major_surgery_12wk: true, localized_tenderness_along_deep_veins: true, entire_leg_swollen: true, calf_swelling_3cm_vs_other: true, pitting_edema_symptomatic_leg: true, collateral_superficial_veins: true, previously_documented_dvt: true, alternative_diagnosis_at_least_as_likely: true } },
  { id: "wd-05", rule: "Wells DVT", inputs: { bedridden_3d_or_major_surgery_12wk: true, paralysis_paresis_recent_immobilization: true, entire_leg_swollen: true } },
  { id: "wd-06", rule: "Wells DVT", inputs: { active_cancer: true, alternative_diagnosis_at_least_as_likely: true } },
  { id: "wd-07", rule: "Wells DVT", inputs: { localized_tenderness_along_deep_veins: true, calf_swelling_3cm_vs_other: true } },
  { id: "wd-08", rule: "Wells DVT", inputs: { pitting_edema_symptomatic_leg: true, entire_leg_swollen: true, previously_documented_dvt: true } },

  // PERC x 8 (inverted-verdict comparison; raw scores are inverses)
  { id: "perc-01", rule: "PERC", inputs: { age_under_50: true, hr_under_100: true, spo2_at_least_95: true, no_unilateral_leg_swelling: true, no_hemoptysis: true, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: true, no_estrogen_use: true } },
  { id: "perc-02", rule: "PERC", inputs: { age_under_50: false, hr_under_100: true, spo2_at_least_95: true, no_unilateral_leg_swelling: true, no_hemoptysis: true, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: true, no_estrogen_use: true } },
  { id: "perc-03", rule: "PERC", inputs: { age_under_50: true, hr_under_100: false, spo2_at_least_95: true, no_unilateral_leg_swelling: true, no_hemoptysis: true, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: true, no_estrogen_use: true } },
  { id: "perc-04", rule: "PERC", inputs: { age_under_50: true, hr_under_100: true, spo2_at_least_95: false, no_unilateral_leg_swelling: true, no_hemoptysis: true, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: true, no_estrogen_use: true } },
  { id: "perc-05", rule: "PERC", inputs: { age_under_50: true, hr_under_100: true, spo2_at_least_95: true, no_unilateral_leg_swelling: false, no_hemoptysis: true, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: true, no_estrogen_use: true } },
  { id: "perc-06", rule: "PERC", inputs: { age_under_50: true, hr_under_100: true, spo2_at_least_95: true, no_unilateral_leg_swelling: true, no_hemoptysis: false, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: true, no_estrogen_use: true } },
  { id: "perc-07", rule: "PERC", inputs: { age_under_50: true, hr_under_100: true, spo2_at_least_95: true, no_unilateral_leg_swelling: true, no_hemoptysis: true, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: false, no_estrogen_use: false } },
  { id: "perc-08", rule: "PERC", inputs: { age_under_50: false, hr_under_100: false, spo2_at_least_95: true, no_unilateral_leg_swelling: true, no_hemoptysis: true, no_recent_surgery_trauma_4wk: true, no_prior_pe_dvt: true, no_estrogen_use: true } },

  // HEART x 5 (MCP-only, expected = published table sum)
  { id: "heart-01", rule: "HEART", mcpVariables: { History: 0, ECG: 0, Age: 0, "Risk factors": 0, Troponin: 0 }, expected: 0 },
  { id: "heart-02", rule: "HEART", mcpVariables: { History: 1, ECG: 1, Age: 1, "Risk factors": 1, Troponin: 1 }, expected: 5 },
  { id: "heart-03", rule: "HEART", mcpVariables: { History: 2, ECG: 2, Age: 2, "Risk factors": 2, Troponin: 2 }, expected: 10 },
  { id: "heart-04", rule: "HEART", mcpVariables: { History: 2, ECG: 0, Age: 1, "Risk factors": 1, Troponin: 0 }, expected: 4 },
  { id: "heart-05", rule: "HEART", mcpVariables: { History: 1, ECG: 2, Age: 2, "Risk factors": 0, Troponin: 1 }, expected: 6 },

  // Wells PE x 5 (MCP-only)
  { id: "wpe-01", rule: "Wells PE", mcpVariables: { "Clinical signs/symptoms of DVT": 0, "PE is #1 diagnosis or equally likely": 0, "Heart rate > 100": 0, "Immobilization or surgery in past 4 weeks": 0, "Previous PE or DVT": 0, "Hemoptysis": 0, "Malignancy": 0 }, expected: 0 },
  { id: "wpe-02", rule: "Wells PE", mcpVariables: { "Clinical signs/symptoms of DVT": 3, "PE is #1 diagnosis or equally likely": 3, "Heart rate > 100": 1.5, "Immobilization or surgery in past 4 weeks": 1.5, "Previous PE or DVT": 1.5, "Hemoptysis": 1, "Malignancy": 1 }, expected: 12.5 },
  { id: "wpe-03", rule: "Wells PE", mcpVariables: { "Clinical signs/symptoms of DVT": 3, "PE is #1 diagnosis or equally likely": 3, "Heart rate > 100": 0, "Immobilization or surgery in past 4 weeks": 0, "Previous PE or DVT": 0, "Hemoptysis": 0, "Malignancy": 0 }, expected: 6 },
  { id: "wpe-04", rule: "Wells PE", mcpVariables: { "Clinical signs/symptoms of DVT": 0, "PE is #1 diagnosis or equally likely": 3, "Heart rate > 100": 1.5, "Immobilization or surgery in past 4 weeks": 1.5, "Previous PE or DVT": 0, "Hemoptysis": 1, "Malignancy": 0 }, expected: 7 },
  { id: "wpe-05", rule: "Wells PE", mcpVariables: { "Clinical signs/symptoms of DVT": 3, "PE is #1 diagnosis or equally likely": 0, "Heart rate > 100": 1.5, "Immobilization or surgery in past 4 weeks": 0, "Previous PE or DVT": 1.5, "Hemoptysis": 0, "Malignancy": 1 }, expected: 7 },
];

const RULE_CFG = {
  "CHA2DS2-VASc": { mcpName: "CHA2DS2-VASc", translate: cha2dsToMcp, calc: chadsvasc, tol: 0 },
  "HAS-BLED": { mcpName: "HAS-BLED Score", translate: hasbledToMcp, calc: hasbled, tol: 0 },
  "MELD-Na": { mcpName: "MELD-Na", translate: meldToMcp, calc: meld, tol: 1 },
  "Wells DVT": { mcpName: "Wells Criteria for DVT", translate: wellsDvtToMcp, calc: wellsDvt, tol: 0 },
  "PERC": { mcpName: "PERC Rule", translate: percToMcp, calc: perc, percVerdict: true },
  "HEART": { mcpName: "HEART Score", tol: 0 },
  "Wells PE": { mcpName: "Wells Criteria for PE", tol: 1e-6 },
};

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

async function runCase(c) {
  const cfg = RULE_CFG[c.rule];
  if (!cfg) return { id: c.id, rule: c.rule, ok: false, error: "unknown rule in test config" };

  if (c.mcpVariables !== undefined) {
    const mcp = await applyRule(cfg.mcpName, c.mcpVariables);
    if (!mcp.ok) return { id: c.id, rule: c.rule, kind: "mcp_only", ok: false, error: mcp.error };
    const tol = cfg.tol ?? 0;
    const match = Math.abs(mcp.score - c.expected) <= tol;
    return {
      id: c.id, rule: c.rule, kind: "mcp_only_canonical",
      mcp_score: mcp.score, expected: c.expected, match,
      mcp_risk_level: mcp.risk_level, ruleVersion: mcp.ruleVersion, ok: true,
      detail: `mcp=${mcp.score} expected=${c.expected} tol=${tol}`,
    };
  }

  const m = cfg.calc(c.inputs);
  const mcpVars = cfg.translate(c.inputs);
  const mcp = await applyRule(cfg.mcpName, mcpVars);
  if (!mcp.ok) return { id: c.id, rule: c.rule, kind: "parity", ok: false, error: mcp.error, inputs: c.inputs, mcp_variables: mcpVars };

  let match, detail;
  if (cfg.percVerdict) {
    const medomniNeg = m.perc_negative;
    const mcpNeg = mcp.score === 0;
    match = medomniNeg === mcpNeg;
    detail = `medomni_perc_negative=${medomniNeg} mcp_score=${mcp.score} mcp_perc_negative=${mcpNeg}`;
  } else {
    const tol = cfg.tol ?? 0;
    match = Math.abs(m.score - mcp.score) <= tol;
    detail = `medomni=${m.score} mcp=${mcp.score} tol=${tol}`;
  }
  return {
    id: c.id, rule: c.rule, kind: "parity", ok: true,
    inputs: c.inputs, mcp_variables: mcpVars,
    medomni_score: m.score, mcp_score: mcp.score, mcp_risk_level: mcp.risk_level,
    match, detail, ruleVersion: mcp.ruleVersion,
  };
}

async function main() {
  console.log(`MCP_URL=${MCP_URL}`);
  console.log(`Total cases: ${CASES.length}\n`);
  const out = [];
  for (const c of CASES) {
    const r = await runCase(c);
    out.push(r);
    const flag = !r.ok ? "ERROR" : r.match ? "PASS" : "MISMATCH";
    const msg = r.detail ?? r.error ?? "";
    console.log(`[${flag.padEnd(8)}] ${r.rule.padEnd(13)} ${r.id.padEnd(9)} ${msg}`);
  }

  const byRule = {};
  for (const r of out) {
    byRule[r.rule] ??= { total: 0, pass: 0, mismatch: 0, error: 0 };
    byRule[r.rule].total++;
    if (!r.ok) byRule[r.rule].error++;
    else if (r.match) byRule[r.rule].pass++;
    else byRule[r.rule].mismatch++;
  }
  console.log("\nSummary:");
  for (const [rule, st] of Object.entries(byRule)) {
    const rate = ((st.pass / st.total) * 100).toFixed(1);
    console.log(
      `  ${rule.padEnd(13)} ${String(st.pass).padStart(2)}/${st.total} pass (${rate}%)  ` +
      `mismatch=${st.mismatch}  error=${st.error}`,
    );
  }
  const pass = out.filter((r) => r.ok && r.match).length;
  const overallRate = ((pass / out.length) * 100).toFixed(1);
  console.log(`  ${"OVERALL".padEnd(13)} ${pass}/${out.length} pass (${overallRate}%)`);

  writeFileSync(
    RESULTS_PATH,
    JSON.stringify({
      mcp_url: MCP_URL,
      ts: new Date().toISOString(),
      summary: { total: out.length, pass, byRule },
      cases: out,
    }, null, 2),
  );
  console.log(`\nWrote ${RESULTS_PATH}`);
  process.exit(pass === out.length ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
