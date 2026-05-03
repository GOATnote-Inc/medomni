// Guideline currency check. Local lookup against a curated registry of
// stale → current guideline shifts (e.g. "clarithromycin triple as
// H. pylori first-line" → "bismuth quadruple is first-line where local
// resistance > 15 %, ACG 2024"). 40 entries, ~30 KB JSON, bundled with
// the route — no network call, no service dependency.
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

export interface GuidelineCurrencyResult {
  query: string;
  match_count: number;
  matches: Array<{
    id: string;
    concept: string;
    stale_default: string;
    current_default: string;
    citation: string;
    score: number;
  }>;
  registry_size: number;
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

function score(query: string, entry: GuidelineEntry): number {
  const qTokens = new Set(tokenize(query).filter((t) => !STOPWORDS.has(t)));
  if (qTokens.size === 0) return 0;
  const haystack = `${entry.concept} ${entry.stale_default}`.toLowerCase();
  let hits = 0;
  for (const t of qTokens) {
    if (haystack.includes(t)) hits += 1;
  }
  // Normalize so a 3-token query that fully hits scores ~1.0
  return hits / qTokens.size;
}

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
    };
  }
  const maxResults = Math.min(Math.max(args.maxResults ?? 3, 1), 8);

  const ranked = REG.entries
    .map((e) => ({ entry: e, s: score(query, e) }))
    .filter((r) => r.s > 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, maxResults);

  return {
    query,
    match_count: ranked.length,
    matches: ranked.map((r) => ({
      id: r.entry.id,
      concept: r.entry.concept,
      stale_default: r.entry.stale_default,
      current_default: r.entry.current_default,
      citation: r.entry.citation,
      score: Math.round(r.s * 100) / 100,
    })),
    registry_size: REG.entries.length,
  };
}
