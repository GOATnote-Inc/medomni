// Skills router for the V_final inference profile.
//
// Pattern (per Boris Cherny / Claude Code skills): markdown-driven progressive
// disclosure. The /api/agent route reads a `?profile=v_final` query param;
// when set, this module classifies the most recent user message and returns
// a skill markdown block to splice into the system prompt before dispatching
// to the B300 vllm endpoint.
//
// Canonical authoring location: mvp/medomni-inference/skills/*.md.
// Runtime build copy: web/lib/agent/skills/*.md (kept in sync via
// `make sync-skills` from repo root). Edit ONLY the canonical authoring
// copies; the sync target overwrites the runtime copies.
//
// Intent classification is deliberately a tiny keyword heuristic in this PR.
// Subagent #2 (skills router with proper LLM-based intent classification)
// supersedes this when it lands. The point of v1 is to wire the
// /api/agent → skill markdown → system prompt path end-to-end so each new
// skill ships in a markdown PR rather than a 17-hr training cycle.

import fs from "node:fs";
import path from "node:path";

const SKILLS_DIR = path.join(process.cwd(), "lib", "agent", "skills");

function readSkill(name: string): string {
  try {
    return fs.readFileSync(path.join(SKILLS_DIR, `${name}.md`), "utf-8");
  } catch {
    return "";
  }
}

// Loaded once per cold start. Vercel functions are short-lived so this
// matches the prompt-cache TTL pattern documented in
// mvp/medomni-inference/system_prompt_v1.md.
const SYSTEM_PROMPT_V_FINAL = readSkill("system_prompt_v1");
const SKILL_DIFFERENTIAL = readSkill("differential");
const SKILL_CALC = readSkill("calc");
const SKILL_HANDOFF = readSkill("handoff");

export type SkillIntent = "differential" | "calc" | "handoff" | "default";

const DIFFERENTIAL_KEYWORDS = [
  "differential",
  "ddx",
  "rule out",
  "what could",
  "what else",
  "what are the possible",
  "broaden",
  "narrow",
];

const CALC_KEYWORDS = [
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
];

const HANDOFF_KEYWORDS = [
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
];

export function classifyIntent(userText: string): SkillIntent {
  const lower = userText.toLowerCase();
  // Calc beats differential when a score keyword fires alongside DDx phrasing
  // ("calculate the CHA2DS2-VASc"). Handoff is a write-side intent and
  // shouldn't fire if the user's just thinking aloud.
  if (CALC_KEYWORDS.some((k) => lower.includes(k))) return "calc";
  if (HANDOFF_KEYWORDS.some((k) => lower.includes(k))) return "handoff";
  if (DIFFERENTIAL_KEYWORDS.some((k) => lower.includes(k))) return "differential";
  return "default";
}

/**
 * Build the V_final-profile system content by splicing the V_final base
 * prompt + the appropriate skill on top of the existing system prompt.
 *
 * Order: existing SYSTEM_PROMPT (sovereignty + tool guidance) → V_final
 * plan-then-act header → matched skill block. The existing prompt stays
 * authoritative on safety/PHI rules; the V_final additions sit underneath
 * as expansion guidance.
 */
export function buildVFinalSystemContent(
  baseSystemPrompt: string,
  userText: string,
): { content: string; intent: SkillIntent } {
  const intent = classifyIntent(userText);
  const skillBlock =
    intent === "differential"
      ? SKILL_DIFFERENTIAL
      : intent === "calc"
        ? SKILL_CALC
        : intent === "handoff"
          ? SKILL_HANDOFF
          : "";

  const sections: string[] = [baseSystemPrompt];
  if (SYSTEM_PROMPT_V_FINAL) {
    sections.push("---", "# V_final inference profile (plan-then-act)", SYSTEM_PROMPT_V_FINAL);
  }
  if (skillBlock) {
    sections.push(
      "---",
      `# Active skill: ${intent}`,
      skillBlock,
    );
  }
  return { content: sections.join("\n\n"), intent };
}

export function isVFinalProfile(searchParams: URLSearchParams): boolean {
  return searchParams.get("profile") === "v_final";
}
