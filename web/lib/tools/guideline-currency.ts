// Guideline currency check. Local lookup against a curated registry of
// stale → current guideline shifts (e.g. "clarithromycin triple as
// H. pylori first-line" → "bismuth quadruple is first-line where local
// resistance > 15 %, ACG 2024"). Bundled with the route — no network
// call, no service dependency.
//
// Behavior contract (P0 fix, A2):
//   The registry is small and curated. When a query has no genuinely
//   relevant entry, we MUST return `notInRegistry: true` with an empty
//   `matches` array and a short `registryTopics` summary, NOT a list of
//   string-distance nearest neighbors. The agent reads `notInRegistry`
//   and falls back to `pubmed_search`. Returning unrelated entries was
//   the user-trust bug from the demo screenshot.
//
// Source of truth: prism42-nemotron-med data/guideline_currency_registry.json
// (Phase 2.6 deliverable). Mirror this file when the upstream changes.

import registry from "@/data/guideline_currency_registry.json";

export interface GuidelineEntry {
  id: string;
  concept: string;
  stale_default: string;
  current_default: string;
  citation: string;
  trigger_patterns?: string[];
}

interface Registry {
  schema_version: string;
  entries: GuidelineEntry[];
}

const REG = registry as Registry;

export interface GuidelineCurrencyMatch {
  id: string;
  concept: string;
  stale_default: string;
  current_default: string;
  citation: string;
  score: number;
}

export interface GuidelineCurrencyResult {
  query: string;
  match_count: number;
  matches: GuidelineCurrencyMatch[];
  registry_size: number;
  // P0: explicit out-of-registry signal. True when the query did not
  // produce any match clearing the relevance threshold.
  notInRegistry: boolean;
  // Compact list of concepts the registry *does* track, surfaced only
  // when notInRegistry === true. Helps the agent decide whether to
  // refine the query or fall back to pubmed_search.
  registryTopics?: string[];
}

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s\-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length >= 3);
}

const STOPWORDS = new Set([
  "and", "the", "for", "with", "without", "about", "this", "that",
  "are", "was", "were", "have", "has", "had", "from", "but", "not",
  "you", "your", "its", "their", "there", "first", "line", "default",
  "currently", "still", "given", "use", "used", "using",
]);

// Split haystack into the same word shape we tokenize the query into,
// then test set membership. This is the key behavior change vs the
// previous `haystack.includes(t)` substring match: "therapy" no longer
// matches inside "monotherapy", so off-topic queries stop scoring on
// generic medical filler words.
function haystackTokens(entry: GuidelineEntry): Set<string> {
  return new Set(tokenize(`${entry.concept} ${entry.stale_default}`));
}

// Generic clinical filler tokens that should NEVER pull a registry hit
// on their own — they appear in many entries and historically caused
// the demo bug ("asthma controller therapy" matching PPSV23/BPH/H.pylori
// because all three contained "therapy"). These tokens count toward the
// hit count only when accompanied by at least one discriminative hit.
const GENERIC_TOKENS = new Set([
  "therapy", "treatment", "drug", "drugs", "medication", "medications",
  "dose", "dosing", "dosage", "schedule", "regimen", "agent", "agents",
  "patient", "patients", "adult", "adults", "elderly", "older", "young",
  "clinical", "medical", "guideline", "guidelines", "recommend",
  "recommendation", "recommendations", "current", "standard", "preferred",
  "monotherapy", "combination", "prevention", "screen", "screening",
  "target", "goal", "level", "levels", "value", "values", "range",
  "initiation", "discontinue", "discontinuation", "stop", "start",
]);

function isDiscriminative(token: string): boolean {
  return token.length >= 4 && !GENERIC_TOKENS.has(token);
}

interface Score {
  hits: number;
  ratio: number;
  discriminativeHits: number;
}

function score(qTokens: Set<string>, hayTokens: Set<string>): Score {
  if (qTokens.size === 0) return { hits: 0, ratio: 0, discriminativeHits: 0 };
  let hits = 0;
  let discriminativeHits = 0;
  for (const t of qTokens) {
    if (hayTokens.has(t)) {
      hits += 1;
      if (isDiscriminative(t)) discriminativeHits += 1;
    }
  }
  return { hits, ratio: hits / qTokens.size, discriminativeHits };
}

// Relevance threshold. An entry must clear ALL of:
//   - ratio >= 0.5 (at least half the meaningful query tokens hit)
//   - discriminativeHits >= 1 (at least one non-generic token must match —
//     this is what kills the "asthma controller therapy" → PPSV23 bug)
//   - For qSize === 1: ratio === 1 AND the token is discriminative
const MIN_RATIO = 0.5;

function clears(qSize: number, s: Score): boolean {
  if (s.discriminativeHits < 1) return false;
  if (qSize === 1) return s.ratio === 1;
  return s.ratio >= MIN_RATIO;
}

// Curated topic summary surfaced when notInRegistry === true. We keep
// this short and human-readable rather than dumping all 44 concepts; the
// agent uses it to decide whether to refine the query, not to enumerate
// the registry.
const REGISTRY_TOPIC_SUMMARY: string[] = [
  "H. pylori first-line therapy (ACG 2024)",
  "DOAC vs warfarin in non-valvular AFib",
  "Adult pneumococcal vaccination (PCV13/PCV15/PCV20/PPSV23)",
  "BPH medical therapy (combination vs monotherapy)",
  "VTE prophylaxis after total knee arthroplasty",
  "Opioid taper thresholds (CDC 2022)",
  "GLP-1 RA contraindications",
  "Aspirin for primary prevention (USPSTF 2022 / ASPREE)",
  "Adjuvant bisphosphonates / denosumab in breast cancer",
  "HPV vaccination + cervical cancer screening",
  "RZV (Shingrix) for herpes zoster",
  "Tdap during pregnancy",
  "HIV PEP / PrEP timing and initiation",
  "SGLT2 inhibitors in HFrEF (diabetic + non-diabetic)",
  "PCSK9 inhibitor + statin combination",
  "Statin therapy for ASCVD primary prevention (ACC/AHA 2018, AHA 2025 update)",
  "Asthma controller therapy (GINA 2024)",
  "Hypertension BP target (ACC/AHA 2017, 2023 update)",
  "Vitamin D supplementation (Endocrine Society 2024)",
  "Hypothyroidism first-line therapy (levothyroxine vs desiccated)",
  "Antibiotic stewardship for viral URI",
  "PPI step-down strategy for GERD",
  "CKD nephrology referral threshold",
  "A1c target in older adults",
  "SSRI adolescent boxed warning",
  "Tamoxifen / SERM / AI for breast-cancer risk reduction",
  "Naloxone post-administration / 911 call",
  "Omega-3 OTC vs icosapent ethyl",
];

export async function guidelineCurrencyCheck(args: {
  query: string;
  maxResults?: number;
}): Promise<GuidelineCurrencyResult> {
  const query = args.query.trim();
  if (!query) {
    return {
      query,
      match_count: 0,
      matches: [],
      registry_size: REG.entries.length,
      notInRegistry: true,
      registryTopics: REGISTRY_TOPIC_SUMMARY,
    };
  }
  const maxResults = Math.min(Math.max(args.maxResults ?? 3, 1), 8);

  const qTokens = new Set(tokenize(query).filter((t) => !STOPWORDS.has(t)));
  const qSize = qTokens.size;

  const ranked = REG.entries
    .map((e) => ({ entry: e, s: score(qTokens, haystackTokens(e)) }))
    .filter((r) => clears(qSize, r.s))
    .sort((a, b) => b.s.ratio - a.s.ratio || b.s.discriminativeHits - a.s.discriminativeHits || b.s.hits - a.s.hits)
    .slice(0, maxResults);

  if (ranked.length === 0) {
    return {
      query,
      match_count: 0,
      matches: [],
      registry_size: REG.entries.length,
      notInRegistry: true,
      registryTopics: REGISTRY_TOPIC_SUMMARY,
    };
  }

  return {
    query,
    match_count: ranked.length,
    matches: ranked.map((r) => ({
      id: r.entry.id,
      concept: r.entry.concept,
      stale_default: r.entry.stale_default,
      current_default: r.entry.current_default,
      citation: r.entry.citation,
      score: Math.round(r.s.ratio * 100) / 100,
    })),
    registry_size: REG.entries.length,
    notInRegistry: false,
  };
}
