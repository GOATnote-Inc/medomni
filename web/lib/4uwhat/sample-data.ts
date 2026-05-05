// 4UWHAt — design fallback patient slice.
// TypeScript port of /Users/kiteboard/medomni/assets/app/data.jsx, used by
// the Records OS dashboard when no live FHIR patient is selected (or the
// `get_patient_context` tool fails). Persona: synthetic — Maya Okafor —
// matches the design prototype 1:1. NOT PHI; safe to ship publicly.

export interface SamplePatient {
  name: string;
  dob: string;
  age: number;
  mrn: string;
  pronouns: string;
  bloodType: string;
  height: string;
  weight: string;
  primaryCare: string;
  careTeamSize: number;
}

export interface SampleVital {
  value: string | number;
  unit: string;
  label: string;
  delta: string;
  good: boolean;
  range: string;
  hint: string;
  source: string;
  spark: number[];
}

export interface SampleCondition {
  id: string;
  name: string;
  onset: string;
  status: "active" | "resolved";
  icd: string;
}

export interface SampleMed {
  id: string;
  name: string;
  dose: string;
  freq: string;
  since: string;
  adherence: number | null;
  refills: number | null;
  prescriber: string;
}

export interface SampleLab {
  id: string;
  name: string;
  value: number;
  unit: string;
  range: string;
  flag: "normal" | "low" | "high";
  date: string;
  hint: string;
  source: string;
  trend: number[];
  dates?: string[];
}

export interface SampleTimelineEntry {
  date: string;
  t: string;
  kind: "visit" | "lab" | "imaging" | "vacc" | "msg";
  title: string;
  who: string;
  loc: string;
  tag: string;
}

export interface SampleImaging {
  id: string;
  date: string;
  kind: string;
  region: string;
  read: string;
  radiologist: string;
}

export interface SampleCareTeamAppointment {
  /** ISO date (YYYY-MM-DD). Time-of-day omitted by design — demo data. */
  date: string;
  /** Short description rendered next to the date in the drawer card. */
  type: string;
}

export interface SampleCareTeamMember {
  id: string;
  name: string;
  role: string;
  org: string;
  avatar: string;
  online: boolean;
  /** Next scheduled appointment with this provider. Optional — drawer
   *  renders "(none scheduled)" when absent. */
  nextAppointment?: SampleCareTeamAppointment;
  /** Open clinical threads worth tracking between visits. Each entry is
   *  a short, plain-language sentence — anchored to Maya's actual labs/
   *  conditions/meds elsewhere in this file so the chips stay coherent
   *  with the rest of the synthetic record. */
  itemsToWatch?: string[];
}

export interface SampleShare {
  who: string;
  scope: string;
  since: string;
  exp: string;
  kind: "clinician" | "app" | "research";
}

export const SAMPLE_PATIENT: SamplePatient = {
  name: "Maya Okafor",
  dob: "1991-04-12",
  age: 34,
  mrn: "P42-0096-MAYA",
  pronouns: "she/her",
  bloodType: "O+",
  height: "5'7\"",
  weight: "141 lb",
  primaryCare: "Dr. R. Adebayo",
  careTeamSize: 4,
};

export const SAMPLE_VITALS: Record<string, SampleVital> = {
  hr: {
    value: 58,
    unit: "bpm",
    label: "Resting HR",
    delta: "-3 vs 30d",
    good: true,
    range: "60–100 typical · athlete 40–60",
    hint:
      "You're in the athletic range. Lower = stronger heart, but persistent <40 should be checked.",
    source: "Apple Watch · last 12 nights",
    spark: [62, 61, 60, 61, 59, 60, 58, 59, 57, 58, 59, 58],
  },
  bp: {
    value: "118/72",
    unit: "mmHg",
    label: "Blood pressure",
    delta: "in range",
    good: true,
    range: "<120/80 optimal · <130/80 normal",
    hint:
      "Stage 1 hypertension starts at 130/80. You're comfortably below.",
    source: "Clinic + Withings · 30 days",
    spark: [122, 120, 121, 119, 120, 118, 119, 118, 119, 118, 117, 118],
  },
  spo2: {
    value: 98,
    unit: "%",
    label: "SpO₂",
    delta: "normal",
    good: true,
    range: "95–100% normal · <92% concerning",
    hint:
      "Blood oxygen saturation. Persistent dips below 92% warrant a pulmonology check.",
    source: "Apple Watch · spot checks",
    spark: [97, 98, 98, 97, 98, 99, 98, 98, 97, 98, 98, 98],
  },
  hrv: {
    value: 62,
    unit: "ms",
    label: "HRV (7d avg)",
    delta: "+8 vs 30d",
    good: true,
    range: "Personal: 45–70 ms · age 30–40 median ~50",
    hint:
      "Heart rate variability — recovery proxy. Higher is generally better.",
    source: "Oura ring · 7 day rolling",
    spark: [50, 52, 55, 54, 58, 60, 59, 61, 60, 62, 61, 62],
  },
  weight: {
    value: 141,
    unit: "lb",
    label: "Weight",
    delta: "-1.2 vs 90d",
    good: true,
    range: "BMI 22.1 · normal 18.5–24.9",
    hint:
      "Weight at 5'7\" puts BMI in the normal range. Trend: -1.2 lb over 90 days.",
    source: "Withings scale · weekly",
    spark: [
      143, 143, 142.5, 142, 142, 141.8, 141.5, 141.5, 141.2, 141, 141, 141,
    ],
  },
  sleep: {
    value: "7h 14m",
    unit: "avg",
    label: "Sleep (7d)",
    delta: "+22m vs 30d",
    good: true,
    range: "7–9 hr/night recommended (adults)",
    hint:
      "Best 7-day average in 2 years. Bedtime drift down 38 min vs last quarter.",
    source: "Oura ring · 7 nights",
    spark: [6.5, 7, 6.8, 7.2, 7.4, 7.0, 7.1, 7.3, 7.2, 7.4, 7.5, 7.2],
  },
};

export const SAMPLE_CONDITIONS: SampleCondition[] = [
  { id: "asthma", name: "Asthma, mild persistent", onset: "2009", status: "active", icd: "J45.30" },
  { id: "dyslip", name: "Dyslipidemia", onset: "2023", status: "active", icd: "E78.5" },
  { id: "iron", name: "Iron deficiency anemia", onset: "2022", status: "resolved", icd: "D50.9" },
  { id: "wisdom", name: "Impacted wisdom teeth", onset: "2018", status: "resolved", icd: "K01.1" },
];

export const SAMPLE_MEDS: SampleMed[] = [
  { id: "rosu", name: "Rosuvastatin", dose: "5 mg", freq: "nightly", since: "Mar 2024", adherence: 0.94, refills: 2, prescriber: "Dr. Adebayo" },
  { id: "albu", name: "Albuterol HFA", dose: "90 mcg", freq: "as needed", since: "2009", adherence: null, refills: 4, prescriber: "Dr. Patel" },
  { id: "fluti", name: "Fluticasone", dose: "110 mcg", freq: "2× daily", since: "Jan 2025", adherence: 0.88, refills: 1, prescriber: "Dr. Patel" },
  { id: "d3", name: "Vitamin D₃", dose: "2000 IU", freq: "daily", since: "OTC", adherence: null, refills: null, prescriber: "Self" },
];

export const SAMPLE_LABS: SampleLab[] = [
  {
    id: "ldl", name: "LDL cholesterol", value: 92, unit: "mg/dL", range: "<100", flag: "normal", date: "2026-04-22",
    hint: '"Bad" cholesterol. On statin therapy, target may be <70.', source: "Quest · drawn at clinic",
    trend: [142, 138, 121, 108, 99, 92], dates: ["2024-01", "2024-06", "2024-11", "2025-04", "2025-10", "2026-04"],
  },
  { id: "hdl", name: "HDL cholesterol", value: 64, unit: "mg/dL", range: ">40", flag: "normal", date: "2026-04-22",
    hint: '"Good" cholesterol. Higher is protective; >60 considered ideal.', source: "Quest", trend: [58, 60, 61, 63, 64, 64] },
  { id: "tg", name: "Triglycerides", value: 88, unit: "mg/dL", range: "<150", flag: "normal", date: "2026-04-22",
    hint: "Blood fats. Normal <150. Down 38% in 24 months.", source: "Quest", trend: [142, 130, 112, 98, 92, 88] },
  { id: "a1c", name: "Hemoglobin A1c", value: 5.2, unit: "%", range: "<5.7", flag: "normal", date: "2026-04-22",
    hint: "3-month average blood sugar. Pre-diabetic 5.7–6.4 · diabetic ≥6.5.", source: "Quest", trend: [5.4, 5.3, 5.3, 5.2, 5.2, 5.2] },
  { id: "vitd", name: "Vitamin D, 25-OH", value: 28, unit: "ng/mL", range: "30–100", flag: "low", date: "2026-04-22",
    hint: "Below 30 = insufficient. Trending up — 22 → 28 over 24 months.", source: "Quest", trend: [22, 24, 26, 27, 27, 28] },
  { id: "fer", name: "Ferritin", value: 64, unit: "ng/mL", range: "15–150", flag: "normal", date: "2026-04-22",
    hint: "Iron stores. Was 11 in 2022 (deficient); fully replenished.", source: "Quest", trend: [11, 22, 38, 52, 60, 64] },
  { id: "tsh", name: "TSH", value: 1.8, unit: "mIU/L", range: "0.4–4.0", flag: "normal", date: "2026-04-22",
    hint: "Thyroid stimulating hormone. Mid-range = healthy thyroid.", source: "Quest", trend: [1.6, 1.7, 1.9, 1.8, 1.8, 1.8] },
  { id: "crp", name: "hs-CRP", value: 0.6, unit: "mg/L", range: "<1.0", flag: "normal", date: "2026-04-22",
    hint: "Inflammation marker. <1 low cardiovascular risk; 1–3 average; >3 elevated.", source: "Quest", trend: [1.4, 1.1, 0.9, 0.7, 0.7, 0.6] },
];

export const SAMPLE_TIMELINE: SampleTimelineEntry[] = [
  { date: "2026-04-22", t: "09:14", kind: "visit", title: "Annual physical", who: "Dr. R. Adebayo", loc: "Riverside Primary Care", tag: "PRIMARY" },
  { date: "2026-04-22", t: "10:02", kind: "lab", title: "Lipid panel + CBC + CMP", who: "Quest Diagnostics", loc: "Drawn at clinic", tag: "LAB" },
  { date: "2026-03-08", t: "14:30", kind: "imaging", title: "Chest X-ray, 2-view", who: "Dr. M. Hsu (Radiology)", loc: "Riverside Imaging", tag: "IMG" },
  { date: "2026-02-19", t: "08:45", kind: "visit", title: "Pulmonology follow-up", who: "Dr. Priya Patel", loc: "Riverside Pulmonary", tag: "SPEC" },
  { date: "2026-01-30", t: "11:00", kind: "vacc", title: "Influenza vaccine", who: "CVS Pharmacy", loc: "Walk-in", tag: "VAX" },
  { date: "2025-11-04", t: "16:20", kind: "visit", title: "Urgent care — sprained ankle", who: "Dr. K. Nguyen", loc: "Westside Urgent", tag: "URG" },
  { date: "2025-10-15", t: "09:50", kind: "lab", title: "Lipid panel", who: "Quest Diagnostics", loc: "Mail-in kit", tag: "LAB" },
  { date: "2025-09-01", t: "10:00", kind: "msg", title: "Renewed Fluticasone Rx", who: "Dr. Patel", loc: "via portal", tag: "MSG" },
];

export interface SampleSurgery {
  date: string;       // ISO YYYY-MM-DD
  procedure: string;
  surgeon: string;
  facility: string;
  outcome: string;
}

export const SAMPLE_SURGERIES: SampleSurgery[] = [
  { date: "2018-09-12", procedure: "Laparoscopic appendectomy", surgeon: "Dr. J. Chen", facility: "Renown Regional Medical Center", outcome: "Uncomplicated; same-day discharge." },
  { date: "2010-06-04", procedure: "Wisdom teeth extraction (4)", surgeon: "Dr. T. Cohen, DDS", facility: "Cohen Oral Surgery", outcome: "Routine; full healing in 2 weeks." },
];

export const SAMPLE_IMAGING: SampleImaging[] = [
  { id: "cxr", date: "2026-03-08", kind: "X-ray", region: "Chest, 2-view", read: "No acute findings.", radiologist: "Dr. M. Hsu" },
  { id: "mri", date: "2025-11-04", kind: "MRI", region: "Right ankle", read: "Grade II ATFL sprain. No fracture.", radiologist: "Dr. L. Sato" },
  { id: "pan", date: "2024-06-12", kind: "Panoramic", region: "Dental", read: "Restorations stable.", radiologist: "Dr. T. Cohen" },
];

export const SAMPLE_CARE_TEAM: SampleCareTeamMember[] = [
  {
    id: "ade",
    name: "Dr. R. Adebayo",
    role: "Primary care",
    org: "Riverside Primary Care",
    avatar: "RA",
    online: true,
    nextAppointment: {
      date: "2026-10-21",
      type: "Annual physical (in-person)",
    },
    itemsToWatch: [
      "LDL 92 on rosuvastatin 5 mg — recheck lipids in 8 weeks",
      "Vitamin D 28 ng/mL still insufficient — confirm D₃ 2000 IU adherence",
      "Resting HR trending into athletic range (58 bpm) — note baseline at next visit",
    ],
  },
  {
    id: "pat",
    name: "Dr. P. Patel",
    role: "Pulmonology",
    org: "Riverside Pulmonary",
    avatar: "PP",
    online: false,
    nextAppointment: {
      date: "2026-08-19",
      type: "Asthma follow-up + spirometry",
    },
    itemsToWatch: [
      "Fluticasone 110 mcg adherence 88% — reinforce twice-daily use",
      "Albuterol refills (4 remaining) — flag if usage climbs above 2× per week",
      "Chest X-ray (Mar 8) clear — re-image only if symptom escalation",
    ],
  },
  {
    id: "won",
    name: "A. Wong, RD",
    role: "Dietitian",
    org: "In-network",
    avatar: "AW",
    online: true,
    nextAppointment: {
      date: "2026-06-04",
      type: "Nutrition follow-up (telehealth)",
    },
    itemsToWatch: [
      "Triglycerides 88 mg/dL — keep current Mediterranean pattern",
      "Vitamin D dietary sources (fatty fish, fortified dairy) given low 25-OH",
      "Weight trending −1.2 lb / 90d at BMI 22.1 — maintain, don't restrict",
    ],
  },
  {
    id: "rui",
    name: "M. Ruiz, PharmD",
    role: "Pharmacist",
    org: "CVS · Mission St.",
    avatar: "MR",
    online: false,
    // Pharmacist consults are typically open-walk-in; leave appointment
    // blank to exercise the "(none scheduled)" empty-state in the drawer.
    itemsToWatch: [
      "Rosuvastatin 5 mg refills (2 remaining) — auto-refill enrollment available",
      "Fluticasone inhaler technique check at next pickup",
      "OTC vitamin D₃ 2000 IU — confirm brand + lot tracked in med list",
    ],
  },
];

export const SAMPLE_SHARES: SampleShare[] = [
  { who: "Dr. R. Adebayo", scope: "Full record", since: "2022-08-01", exp: "—", kind: "clinician" },
  { who: "Dr. P. Patel", scope: "Pulmonary + meds", since: "2025-01-15", exp: "—", kind: "clinician" },
  { who: "Apple Health", scope: "Vitals + activity (read)", since: "2024-03-02", exp: "always", kind: "app" },
  { who: "Wellness Study #284", scope: "De-identified labs", since: "2026-01-10", exp: "2026-12-31", kind: "research" },
];

export const SAMPLE_AI_SUGGESTIONS: string[] = [
  "Why is my LDL trending down?",
  "Summarize my last visit in plain language",
  "Compare my HRV before and after I started running",
  "Anything I should ask Dr. Adebayo at the next visit?",
];
