// Skill registry — server-side loader for the public /skills page.
//
// Reads the same markdown files that `web/lib/agent/skills.ts` consumes at
// runtime, parses YAML-ish frontmatter, and surfaces a structured manifest
// for the registry UI. Trust-through-transparency: every skill the agent
// can fire is enumerated here, with name + description + trigger keywords +
// the full markdown body (rendered client-side via react-markdown).
//
// The file enumeration is intentionally hardcoded (not a glob) so the
// public surface only ships skills we've explicitly authored. New skills
// must be added to SKILL_FILES below before they appear on /skills.

import fs from "node:fs";
import path from "node:path";
import type { SkillIntent } from "./skills";

const SKILLS_DIR = path.join(process.cwd(), "lib", "agent", "skills");

// Triggers come from skills.ts — single source of truth. We import the
// constants by name when possible; for the public page we duplicate the
// arrays here as a literal so we don't widen the export surface of
// skills.ts (the runtime classifier uses lowercase keyword arrays; the
// registry UI shows them as-is for clinician inspection).
const TRIGGER_FALLBACK: Record<SkillIntent, string[]> = {
  differential: [
    "differential",
    "ddx",
    "rule out",
    "what could",
    "what else",
    "what are the possible",
    "broaden",
    "narrow",
  ],
  calc: [
    "score",
    "scoring",
    "calculate",
    "calculation",
    "compute",
    "cha2ds2",
    "has-bled",
    "hasbled",
    "wells",
    "perc",
    "meld",
    "gfr",
    "creatinine clearance",
  ],
  handoff: [
    "order",
    "place",
    "write",
    "draft",
    "schedule",
    "refer",
    "consult",
    "transfer",
    "admit",
    "discharge",
    "handoff",
    "sign-out",
  ],
  default: [],
};

export type SkillId = "system_prompt_v1" | "differential" | "calc" | "handoff";

export interface SkillManifest {
  id: SkillId;
  // Slug used in the URL (?active=<slug>). Matches the SkillIntent union when
  // the skill is router-dispatchable; system_prompt_v1 uses its file name.
  slug: string;
  // Human-readable title — derived from frontmatter `name` or the first H1.
  title: string;
  // One-line description from frontmatter or the first non-frontmatter line.
  description: string;
  // Keyword list that fires this skill in the runtime classifier. Empty for
  // the system prompt (always loaded, not router-gated).
  triggers: string[];
  // Whether the runtime classifier can route to this skill. False for
  // system_prompt_v1 (which is always-on context, not a routed skill).
  routable: boolean;
  // Raw markdown body, frontmatter stripped. Rendered client-side.
  markdown: string;
  // Original on-disk filename (relative to web/lib/agent/skills/).
  fileName: string;
}

const SKILL_FILES: Array<{ id: SkillId; routable: boolean }> = [
  { id: "system_prompt_v1", routable: false },
  { id: "differential", routable: true },
  { id: "calc", routable: true },
  { id: "handoff", routable: true },
];

interface ParsedFrontmatter {
  name?: string;
  description?: string;
  triggers: string[];
  body: string;
}

/**
 * Tiny YAML-ish frontmatter parser. Handles the three keys we actually use
 * across `web/lib/agent/skills/*.md`:
 *   name: <string>
 *   description: <string>
 *   trigger:
 *     - "<keyword>"
 *     - "<keyword>"
 *
 * Not a general YAML parser — quoted values get their surrounding quotes
 * stripped, that's it. Adding a YAML dependency for ~12 lines of
 * frontmatter would be over-engineering.
 */
function parseFrontmatter(raw: string): ParsedFrontmatter {
  const result: ParsedFrontmatter = { triggers: [], body: raw };
  if (!raw.startsWith("---")) return result;
  const end = raw.indexOf("\n---", 3);
  if (end === -1) return result;
  const fmText = raw.slice(3, end).trim();
  const body = raw.slice(end + 4).replace(/^\n/, "");
  result.body = body;

  const lines = fmText.split("\n");
  let inTriggerList = false;
  for (const rawLine of lines) {
    const line = rawLine.replace(/\r$/, "");
    if (!line.trim()) continue;
    // List item under `trigger:` or `triggers:`
    const listMatch = line.match(/^\s+-\s+(.+)$/);
    if (listMatch && inTriggerList) {
      result.triggers.push(stripQuotes(listMatch[1].trim()));
      continue;
    }
    inTriggerList = false;
    const kv = line.match(/^([a-zA-Z_][a-zA-Z0-9_-]*):\s*(.*)$/);
    if (!kv) continue;
    const key = kv[1].toLowerCase();
    const value = kv[2].trim();
    if (key === "name") result.name = stripQuotes(value);
    else if (key === "description") result.description = stripQuotes(value);
    else if (key === "trigger" || key === "triggers") {
      // The value can be inline or open a list. If empty string after the
      // colon, expect indented `- item` lines below.
      inTriggerList = value.length === 0;
    }
  }
  return result;
}

function stripQuotes(s: string): string {
  if (s.length >= 2) {
    const first = s[0];
    const last = s[s.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return s.slice(1, -1);
    }
  }
  return s;
}

function deriveTitle(id: SkillId, fm: ParsedFrontmatter): string {
  if (fm.name) return fm.name;
  // Fall back to first H1 in the body.
  const h1 = fm.body.match(/^#\s+(.+)$/m);
  if (h1) return h1[1].trim();
  return id;
}

function deriveDescription(fm: ParsedFrontmatter): string {
  if (fm.description) return fm.description;
  // Fall back to first non-empty, non-heading line in body.
  for (const line of fm.body.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("#")) continue;
    if (trimmed.startsWith("---")) continue;
    return trimmed;
  }
  return "";
}

function readSkillFile(id: SkillId): string {
  const file = path.join(SKILLS_DIR, `${id}.md`);
  return fs.readFileSync(file, "utf-8");
}

export function loadSkillRegistry(): SkillManifest[] {
  return SKILL_FILES.map(({ id, routable }) => {
    const raw = readSkillFile(id);
    const fm = parseFrontmatter(raw);
    // For routable skills, prefer triggers from skills.ts (the actual
    // runtime classifier source) over frontmatter. The frontmatter is
    // illustrative; the lowercase array in skills.ts is what fires.
    const intentKey = (id as string) as SkillIntent;
    const triggers =
      routable && TRIGGER_FALLBACK[intentKey]?.length
        ? TRIGGER_FALLBACK[intentKey]
        : fm.triggers;
    return {
      id,
      slug: id === "system_prompt_v1" ? "system" : id,
      title: deriveTitle(id, fm),
      description: deriveDescription(fm),
      triggers,
      routable,
      markdown: fm.body,
      fileName: `${id}.md`,
    };
  });
}
