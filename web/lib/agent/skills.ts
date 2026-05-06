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
// v2 (iter-182): the keyword heuristic in `classifyIntent` is now the
// fallback path. The default classifier is `classifyIntentLLM`, which fires a
// 30-token chat-completion at the catfish vllm endpoint with a tight system
// prompt asking the model to pick exactly one of {differential, calc,
// handoff, default} and respond as `{"intent":"<label>"}`. Wrapped in a
// 2-second AbortController timeout — on timeout, network error, parse
// failure, or unexpected label we transparently degrade to the keyword
// heuristic. Net effect: same default-off ?profile=v_final opt-in, same
// skill markdown payloads, slightly more robust router for the messy clinical
// phrasings the keyword list misses ("how worried should I be about clots
// in this AFib patient?" → differential, even though no DDx keyword fires).
//
// Why not `response_format={"type":"json_object"}`? vllm 0.6.x with
// the qwen3_coder tool parser + nemotron_v3 reasoning parser respects the
// grammar flag, but we keep the surface area minimal — we ask the model to
// emit a JSON object and parse the first {…} we find. If a future vllm
// upgrade tightens this, the response_format flag is a one-liner add.

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

const VALID_INTENTS: ReadonlySet<SkillIntent> = new Set<SkillIntent>([
  "differential",
  "calc",
  "handoff",
  "default",
]);

const CLASSIFIER_SYSTEM_PROMPT = `You are an intent classifier for a medical reasoning assistant. The user is a clinician (RN/NP/PA/MD). You must read the most recent user message and pick ONE label that best describes what they want next.

Labels (pick exactly one):
- "differential" — the user is asking for a list of possibilities, what could be going on, what to rule out, broaden/narrow the workup, or general DDx-shaped reasoning.
- "calc" — the user wants a clinical score or calculation (CHA2DS2-VASc, HAS-BLED, MELD-Na, Wells DVT, PERC, GFR, creatinine clearance, etc.).
- "handoff" — the user wants to write something downstream: place an order, consult, refer, transfer, admit, discharge, sign-out, draft a note.
- "default" — none of the above; a general clinical question, definition lookup, or chitchat.

Output JSON only, exactly: {"intent":"<label>"}
No prose, no reasoning, no extra keys.`;

const CLASSIFIER_TIMEOUT_MS = 2_000;
const CLASSIFIER_MAX_TOKENS = 30;
const CLASSIFIER_MODEL_ID = "nemotron";

/**
 * LLM-based intent classifier. Calls the catfish vllm chat-completions
 * endpoint with a tight system prompt and a 2-second timeout. Returns one of
 * {differential, calc, handoff, default} on success; throws on timeout,
 * network error, non-2xx status, or parse/label failure. Callers should
 * fall back to `classifyIntent` (keyword heuristic) on any throw.
 */
export async function classifyIntentLLM(text: string): Promise<SkillIntent> {
  const tunnelUrl = process.env.MEDOMNI_TUNNEL_URL;
  if (!tunnelUrl) {
    throw new Error("MEDOMNI_TUNNEL_URL not set");
  }
  const trimmed = text.trim();
  if (!trimmed) return "default";

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), CLASSIFIER_TIMEOUT_MS);
  try {
    const upstream = await fetch(`${tunnelUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: ctrl.signal,
      body: JSON.stringify({
        model: CLASSIFIER_MODEL_ID,
        messages: [
          { role: "system", content: CLASSIFIER_SYSTEM_PROMPT },
          { role: "user", content: trimmed },
        ],
        stream: false,
        temperature: 0.0,
        max_tokens: CLASSIFIER_MAX_TOKENS,
        response_format: { type: "json_object" },
      }),
    });
    if (!upstream.ok) {
      throw new Error(`classifier vllm ${upstream.status}`);
    }
    const json = (await upstream.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const raw = json.choices?.[0]?.message?.content ?? "";
    return parseIntentJson(raw);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Extract `intent` from the model's response. The model is instructed to
 * emit `{"intent":"<label>"}` exactly, but we tolerate (a) leading/trailing
 * whitespace, (b) bare label tokens (`calc`), and (c) wrapped JSON inside
 * other text by grabbing the first {...} block. Throws on anything we can't
 * coerce into a valid SkillIntent so the caller can fall back.
 */
export function parseIntentJson(raw: string): SkillIntent {
  if (!raw) throw new Error("empty classifier response");
  const trimmed = raw.trim();

  // Tolerate bare label tokens.
  const bare = trimmed.toLowerCase().replace(/^["']|["']$/g, "") as SkillIntent;
  if (VALID_INTENTS.has(bare)) return bare;

  // Find the first {...} block.
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error(`no json object in classifier response: ${trimmed.slice(0, 80)}`);
  }
  const slice = trimmed.slice(start, end + 1);
  let parsed: unknown;
  try {
    parsed = JSON.parse(slice);
  } catch (e) {
    throw new Error(`classifier json parse failed: ${(e as Error).message}`);
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("classifier response is not an object");
  }
  const intent = (parsed as { intent?: unknown }).intent;
  if (typeof intent !== "string") {
    throw new Error("classifier response missing string `intent` field");
  }
  const lower = intent.toLowerCase().trim() as SkillIntent;
  if (!VALID_INTENTS.has(lower)) {
    throw new Error(`classifier returned unknown label: ${intent.slice(0, 40)}`);
  }
  return lower;
}

/**
 * LLM classifier with keyword-heuristic fallback. Always resolves to a valid
 * `SkillIntent` — never throws. Use this from production paths.
 */
export async function classifyIntentWithFallback(
  userText: string,
): Promise<SkillIntent> {
  try {
    return await classifyIntentLLM(userText);
  } catch (e) {
    // Server log so we can spot when the LLM router is degrading.
    console.warn(
      `[skills] LLM classifier fell back to keyword heuristic: ${(e as Error).message}`,
    );
    return classifyIntent(userText);
  }
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
export async function buildVFinalSystemContent(
  baseSystemPrompt: string,
  userText: string,
): Promise<{ content: string; intent: SkillIntent }> {
  const intent = await classifyIntentWithFallback(userText);
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
