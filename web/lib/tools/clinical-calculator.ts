// Clinical scoring calculators. Pure-JS, no backend, no network.
//
// Scoring systems implemented:
// - CHA2DS2-VASc: stroke risk in non-valvular AFib (drives anticoag decision)
// - HAS-BLED: bleeding risk on anticoagulation (paired with CHA2DS2-VASc)
// - MELD: end-stage liver disease severity / transplant prioritization
// - Wells DVT: pre-test probability of DVT
// - PERC: rule out pulmonary embolism in low-risk patients
//
// Each calc takes a structured input object and returns score + risk band
// + recommendation summary. The model dispatches by `score` arg.

export interface CalculatorResult {
  score_name: string;
  inputs: Record<string, unknown>;
  score: number;
  risk_band: string;
  interpretation: string;
  citation: string;
}

// CHA2DS2-VASc: each yes = points
function chadsvasc(args: {
  age?: number;
  sex?: "male" | "female" | "M" | "F";
  chf?: boolean;
  hypertension?: boolean;
  diabetes?: boolean;
  stroke_tia_thromboembolism?: boolean;
  vascular_disease?: boolean;
}): CalculatorResult {
  const age = typeof args.age === "number" ? args.age : 0;
  const sex = String(args.sex ?? "").toLowerCase();
  let s = 0;
  const breakdown: string[] = [];
  if (args.chf) { s += 1; breakdown.push("CHF +1"); }
  if (args.hypertension) { s += 1; breakdown.push("HTN +1"); }
  if (age >= 75) { s += 2; breakdown.push("age≥75 +2"); }
  else if (age >= 65) { s += 1; breakdown.push("age 65-74 +1"); }
  if (args.diabetes) { s += 1; breakdown.push("DM +1"); }
  if (args.stroke_tia_thromboembolism) { s += 2; breakdown.push("prior stroke/TIA +2"); }
  if (args.vascular_disease) { s += 1; breakdown.push("vascular disease +1"); }
  if (sex === "female" || sex === "f") { s += 1; breakdown.push("female +1"); }

  const annual_stroke_risk_pct: Record<number, string> = {
    0: "0.2%", 1: "0.6%", 2: "2.2%", 3: "3.2%", 4: "4.8%",
    5: "7.2%", 6: "9.7%", 7: "11.2%", 8: "10.8%", 9: "12.2%",
  };
  const risk = annual_stroke_risk_pct[s] ?? ">12%";

  let interp: string;
  if (s === 0) interp = "Score 0 (men) or 1 (women): no antithrombotic recommended.";
  else if (s === 1 && (sex === "male" || sex === "m"))
    interp = "Score 1 (male): consider OAC; weigh against bleed risk (HAS-BLED).";
  else
    interp = `Score ${s}: oral anticoagulation recommended (apixaban first-line per AHA/ACC/HRS 2023; warfarin only for mechanical valves or severe CKD). Annual stroke risk ~${risk}.`;

  return {
    score_name: "CHA2DS2-VASc",
    inputs: { ...args, _breakdown: breakdown.join(", ") },
    score: s,
    risk_band: `annual stroke risk ${risk}`,
    interpretation: interp,
    citation: "AHA/ACC/HRS 2023 Guideline for AFib (and 2024 focused update)",
  };
}

function hasbled(args: {
  hypertension_uncontrolled?: boolean;
  abnormal_renal?: boolean;
  abnormal_liver?: boolean;
  stroke_history?: boolean;
  bleeding_history?: boolean;
  labile_inr?: boolean;
  age_over_65?: boolean;
  drugs_predisposing_bleeding?: boolean;
  alcohol_use?: boolean;
}): CalculatorResult {
  let s = 0;
  const b: string[] = [];
  for (const [k, label] of [
    ["hypertension_uncontrolled", "HTN uncontrolled"],
    ["abnormal_renal", "abnormal renal"],
    ["abnormal_liver", "abnormal liver"],
    ["stroke_history", "stroke history"],
    ["bleeding_history", "bleeding history/predisposition"],
    ["labile_inr", "labile INR"],
    ["age_over_65", "age >65"],
    ["drugs_predisposing_bleeding", "drugs predisposing bleeding"],
    ["alcohol_use", "alcohol use"],
  ] as const) {
    if ((args as Record<string, unknown>)[k]) { s += 1; b.push(`${label} +1`); }
  }
  let band: string;
  let interp: string;
  if (s <= 2) {
    band = "low";
    interp = `Score ${s}: low bleeding risk (~1.0-3.4 bleeds per 100 patient-years). Anticoagulation usually appropriate if CHA2DS2-VASc warrants.`;
  } else if (s === 3) {
    band = "moderate";
    interp = "Score 3: moderate bleed risk. Anticoagulate if indicated, but monitor closely + address modifiable risk factors.";
  } else {
    band = "high";
    interp = `Score ${s}: high bleed risk. Address modifiable factors (HTN, alcohol, labile INR, NSAIDs). Anticoagulation may still be net-beneficial — do not withhold solely based on HAS-BLED.`;
  }
  return {
    score_name: "HAS-BLED",
    inputs: { ...args, _breakdown: b.join(", ") },
    score: s,
    risk_band: `${band} bleeding risk`,
    interpretation: interp,
    citation: "Pisters et al. 2010, Chest; AHA/ACC/HRS 2023 endorses for shared decision-making",
  };
}

function meld(args: {
  bilirubin_mgdl?: number;
  inr?: number;
  creatinine_mgdl?: number;
  on_dialysis?: boolean;
  sodium_meql?: number;
}): CalculatorResult {
  // MELD-Na (UNOS post-2016)
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

  let band: string;
  if (final < 10) band = "compensated";
  else if (final < 15) band = "moderate (transplant evaluation)";
  else if (final < 20) band = "high mortality risk";
  else if (final < 30) band = "very high — transplant priority";
  else band = "critical — highest priority";

  return {
    score_name: "MELD-Na",
    inputs: args,
    score: final,
    risk_band: band,
    interpretation: `MELD-Na ${final}: ${band}. 3-month mortality scales ~2% (MELD<10) → ~52% (MELD 30-39). Above 15 generally triggers transplant referral.`,
    citation: "UNOS MELD-Na (since 2016); Kim et al. NEJM 2008",
  };
}

function wellsDvt(args: {
  active_cancer?: boolean;
  paralysis_paresis_recent_immobilization?: boolean;
  bedridden_3d_or_major_surgery_12wk?: boolean;
  localized_tenderness_along_deep_veins?: boolean;
  entire_leg_swollen?: boolean;
  calf_swelling_3cm_vs_other?: boolean;
  pitting_edema_symptomatic_leg?: boolean;
  collateral_superficial_veins?: boolean;
  previously_documented_dvt?: boolean;
  alternative_diagnosis_at_least_as_likely?: boolean;
}): CalculatorResult {
  let s = 0;
  const b: string[] = [];
  const yes = (k: keyof typeof args, label: string, v = 1) => {
    if (args[k]) { s += v; b.push(`${label} +${v}`); }
  };
  yes("active_cancer", "active cancer");
  yes("paralysis_paresis_recent_immobilization", "paralysis/immobilization");
  yes("bedridden_3d_or_major_surgery_12wk", "bedridden ≥3d / major surgery <12wk");
  yes("localized_tenderness_along_deep_veins", "tenderness along deep veins");
  yes("entire_leg_swollen", "entire leg swollen");
  yes("calf_swelling_3cm_vs_other", "calf swelling >3cm vs contralateral");
  yes("pitting_edema_symptomatic_leg", "pitting edema");
  yes("collateral_superficial_veins", "collateral superficial veins");
  yes("previously_documented_dvt", "prior DVT");
  if (args.alternative_diagnosis_at_least_as_likely) { s -= 2; b.push("alternative dx ≥ likely -2"); }

  let band: string;
  let interp: string;
  if (s >= 3) {
    band = "high";
    interp = "Wells ≥3: DVT likely (~75% prevalence). Proceed to compression US; D-dimer not useful at this pre-test probability.";
  } else if (s >= 1) {
    band = "moderate";
    interp = "Wells 1-2: moderate. Get high-sensitivity D-dimer; if negative, DVT excluded; if positive, US.";
  } else {
    band = "low";
    interp = "Wells ≤0: low (~5% prevalence). Negative high-sensitivity D-dimer rules out DVT.";
  }
  return {
    score_name: "Wells DVT",
    inputs: { ...args, _breakdown: b.join(", ") },
    score: s,
    risk_band: `${band} pre-test probability`,
    interpretation: interp,
    citation: "Wells et al. NEJM 2003; ACR Appropriateness Criteria 2023",
  };
}

function perc(args: {
  age_under_50?: boolean;
  hr_under_100?: boolean;
  spo2_at_least_95?: boolean;
  no_unilateral_leg_swelling?: boolean;
  no_hemoptysis?: boolean;
  no_recent_surgery_trauma_4wk?: boolean;
  no_prior_pe_dvt?: boolean;
  no_estrogen_use?: boolean;
}): CalculatorResult {
  // PERC: ALL 8 must be true to "rule out" PE in low pre-test probability
  const items: Array<[keyof typeof args, string]> = [
    ["age_under_50", "age<50"],
    ["hr_under_100", "HR<100"],
    ["spo2_at_least_95", "SpO2≥95% on room air"],
    ["no_unilateral_leg_swelling", "no unilateral leg swelling"],
    ["no_hemoptysis", "no hemoptysis"],
    ["no_recent_surgery_trauma_4wk", "no recent surgery/trauma in 4wk"],
    ["no_prior_pe_dvt", "no prior PE/DVT"],
    ["no_estrogen_use", "no estrogen use"],
  ];
  const failed = items.filter(([k]) => !args[k]).map(([_, label]) => label);
  const all_true = failed.length === 0;
  return {
    score_name: "PERC",
    inputs: args,
    score: 8 - failed.length,
    risk_band: all_true ? "PERC-negative" : "PERC-positive (cannot rule out)",
    interpretation: all_true
      ? "All 8 criteria met → in patients with low gestalt pre-test probability of PE (<15%), PE is essentially excluded; no D-dimer, no CT-PA needed."
      : `Failed: ${failed.join(", ")}. PERC cannot rule out PE; obtain D-dimer (high-sensitivity) and proceed per Wells / Geneva pathway.`,
    citation: "Kline et al. JTH 2008; ACEP Clinical Policy 2018",
  };
}

const CALCULATORS = {
  "cha2ds2-vasc": chadsvasc,
  chadsvasc: chadsvasc,
  "has-bled": hasbled,
  hasbled: hasbled,
  meld: meld,
  "meld-na": meld,
  "wells-dvt": wellsDvt,
  wells: wellsDvt,
  perc: perc,
} as const;

export const SUPPORTED_SCORES = Array.from(
  new Set(Object.values({
    "CHA2DS2-VASc": "cha2ds2-vasc",
    "HAS-BLED": "has-bled",
    "MELD-Na": "meld",
    "Wells DVT": "wells-dvt",
    PERC: "perc",
  })),
);

export async function clinicalCalculate(args: {
  score: string;
  inputs?: Record<string, unknown>;
}): Promise<CalculatorResult> {
  const key = (args.score ?? "").toLowerCase().trim();
  const fn = (CALCULATORS as Record<string, (i: Record<string, unknown>) => CalculatorResult>)[key];
  if (!fn) {
    throw new Error(
      `Unknown score "${args.score}". Supported: ${SUPPORTED_SCORES.join(", ")}.`,
    );
  }
  return fn((args.inputs ?? {}) as never);
}
