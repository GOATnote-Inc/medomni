// /api/agent — server-side agent loop. Stays alongside /api/ask.
//
// Catfish vllm-omni-b300 is now launched with
//   --enable-auto-tool-choice --tool-call-parser qwen3_coder
//   --reasoning-parser nemotron_v3
// so vllm streams structured `delta.tool_calls[]` + `delta.reasoning` (note:
// vllm renames the field from `reasoning_content` to `reasoning` once the
// reasoning parser is configured). The 4-format envelope parser from PR #11
// is gone; we read the structured fields directly.
//
// We still drive the loop manually via `createUIMessageStream` rather than
// `streamText({tools})` because the user side mixes text and audio: each
// user message is forwarded to vllm as an OpenAI-compat `content` array of
// text + image_url + audio_url blocks. Going through AI SDK's
// `convertToModelMessages` would force us to negotiate FileUIPart ↔
// audio_url through whichever provider package handles it, which is
// undersupported for vllm-style audio. Manual passthrough is one screen of
// code with no provider wrangling.

import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  type UIMessage,
  type UIMessageStreamWriter,
} from "ai";
import type { NextRequest } from "next/server";
import { pubmedSearch, type PubMedSearchResult } from "@/lib/tools/pubmed";
import { primekgLookup, type PrimeKGSubgraphResult } from "@/lib/tools/primekg";
import {
  guidelineCurrencyCheck,
  type GuidelineCurrencyResult,
} from "@/lib/tools/guideline-currency";
import {
  clinicalCalculate,
  SUPPORTED_SCORES,
  type CalculatorResult,
} from "@/lib/tools/clinical-calculator";

export const runtime = "nodejs";
export const maxDuration = 300;
export const dynamic = "force-dynamic";

const MODEL_ID = "nemotron";
const MAX_STEPS = 4;
// Audio is the largest input type; a 60s WAV at 16 kHz mono PCM is ~2.6 MB
// base64. /api/ask uses 4 MB; we mirror that cap.
const MAX_BODY_BYTES = 4_000_000;

const TOOL_SPEC = [
  {
    type: "function" as const,
    function: {
      name: "pubmed_search",
      description:
        "Search PubMed for biomedical literature. Returns up to 10 records with PMID, title, journal, year, authors, and abstract.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "PubMed query. Field tags follow the term, not precede it: '2020:2024[dp]' for a date range, '2023[dp]' for one year, 'shock, septic'[mh] for a MeSH heading. Bare year ranges match nothing. Booleans (AND, OR, NOT) must be uppercase. Call once per turn unless the result is clearly wrong.",
          },
          maxResults: {
            type: "integer",
            description: "Maximum records to return (1-10, default 5).",
            minimum: 1,
            maximum: 10,
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "primekg_lookup",
      description:
        "Look up an entity in PrimeKG (a curated medical knowledge graph: drugs, diseases, genes, phenotypes, anatomy) and return its k-hop neighborhood. Returns connected drugs (e.g. drug-drug interactions), associated diseases/phenotypes, and pathway annotations as a structured text block. Use when the user asks about drug interactions, contraindications, disease-gene associations, or relationships between named medical entities.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "Name of the entity to look up (e.g. 'warfarin', 'ibrutinib', 'atrial fibrillation', 'BRCA1'). Free-text; the service runs case-insensitive substring matching against PrimeKG node names.",
          },
          maxHops: {
            type: "integer",
            description: "Graph traversal depth (1-3, default 2). Higher = broader neighborhood.",
            minimum: 1,
            maximum: 3,
          },
          maxNodes: {
            type: "integer",
            description: "Maximum nodes in returned subgraph (10-150, default 60).",
            minimum: 10,
            maximum: 150,
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "guideline_currency_check",
      description:
        "Check whether a clinical default the user mentions (or that you are about to recommend) is the CURRENT standard or a stale recommendation that has been superseded since 2023. Returns matched entries from a curated registry with the stale default, the current default, and the citing guideline body+year. Use early in your reasoning whenever the question touches a topic where guidance has shifted recently (DOAC vs warfarin, H. pylori first-line, adult pneumococcal vaccination, BPH therapy, opioid taper, GLP-1 contraindications, denosumab discontinuation, SGLT2 in non-diabetic HFrEF, etc.) — call once with the clinical concept, then incorporate the current default into your final answer.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "Clinical concept or phrase to check (e.g. 'H. pylori first-line therapy', 'PCV13 adult vaccination', 'warfarin AFib first-line', 'denosumab stop'). Free-text; the registry runs token overlap.",
          },
          maxResults: {
            type: "integer",
            description: "Maximum entries to return (1-8, default 3).",
            minimum: 1,
            maximum: 8,
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "clinical_calculate",
      description:
        "Compute a standard clinical risk/severity score with a structured input. Supported: cha2ds2-vasc (stroke risk in non-valvular AFib), has-bled (bleed risk on anticoag), meld-na (end-stage liver severity), wells-dvt (DVT pre-test probability), perc (rule out PE in low-risk patients). Use whenever the user gives enough patient detail to compute one of these scores — DON'T guess the math; call the tool. Returns score + risk band + interpretation + citation.",
      parameters: {
        type: "object",
        properties: {
          score: {
            type: "string",
            description: "Score name. One of: cha2ds2-vasc, has-bled, meld-na, wells-dvt, perc.",
            enum: ["cha2ds2-vasc", "has-bled", "meld-na", "wells-dvt", "perc"],
          },
          inputs: {
            type: "object",
            description: "Score-specific inputs. Examples:\n- cha2ds2-vasc: {age:72, sex:'female', chf:false, hypertension:true, diabetes:true, stroke_tia_thromboembolism:false, vascular_disease:false}\n- has-bled: {hypertension_uncontrolled:true, abnormal_renal:false, stroke_history:false, bleeding_history:false, labile_inr:false, age_over_65:true, drugs_predisposing_bleeding:true, alcohol_use:false}\n- meld-na: {bilirubin_mgdl:2.1, inr:1.6, creatinine_mgdl:1.4, sodium_meql:131, on_dialysis:false}\n- wells-dvt: {active_cancer:false, localized_tenderness_along_deep_veins:true, entire_leg_swollen:true, calf_swelling_3cm_vs_other:true, alternative_diagnosis_at_least_as_likely:false}\n- perc: {age_under_50:true, hr_under_100:true, spo2_at_least_95:true, no_unilateral_leg_swelling:true, no_hemoptysis:true, no_recent_surgery_trauma_4wk:true, no_prior_pe_dvt:true, no_estrogen_use:true}",
          },
        },
        required: ["score"],
      },
    },
  },
];

const SYSTEM_PROMPT = `You are MedOmni, a medical reasoning assistant served sovereign on NVIDIA Blackwell B300 hardware. You help clinicians (RNs, NPs, PAs, MDs) and trained healthcare workers think through clinical scenarios.

Discipline:
- State your reasoning briefly, then your recommendation.
- Cite guidelines (AHA/ACC, USPSTF, FDA, IDSA, ASCO, EBCTCG, etc.) with year and the specific recommendation.
- If uncertain or the evidence is contested, say so.
- Never fabricate guideline versions, study names, or numerical thresholds.
- Never replace patient-specific clinical judgment.
- Never request, accept, or echo PHI. If the user pastes identifiable patient information, ask them to redact and re-ask.

When the user message contains audio:
- Begin your reasoning with "Transcript: <verbatim transcript>" on the first line so the user can see what you heard.
- Then continue normally — reasoning, optional tool call, final answer.

Tool use:
- Call guideline_currency_check FIRST when your draft answer relies on a clinical default that may have shifted since 2023 (DOAC vs warfarin, H. pylori first-line, pneumococcal vaccination, opioid taper, GLP-1 contraindications, denosumab discontinuation, SGLT2 in HFrEF, etc.). One quick call up front prevents you from confidently citing a stale default.
- Call clinical_calculate whenever the user gives enough patient detail to compute a standard score (CHA2DS2-VASc, HAS-BLED, MELD-Na, Wells DVT, PERC). Do NOT estimate these scores in your head — the tool is exact.
- Call pubmed_search when recent literature would meaningfully change your answer.
- Call primekg_lookup when the question turns on relationships between named medical entities — drug-drug interactions, drug-disease contraindications, gene-disease associations, pathway membership. Especially valuable for polypharmacy questions.
- Three tool calls per turn is the hard cap. After that, answer from prior knowledge and acknowledge the limit.
- If a tool returns an "error" field or zero matches, do not retry — answer from prior knowledge and note what was unavailable.

This is a public demo. Be tight; every word counts.`;

// --- Vllm message shape ----------------------------------------------------
//
// The user can attach audio + image to a single turn. We forward each
// UIMessage to vllm as an OpenAI-compat content array (string for text-only,
// array of typed blocks otherwise).

type ContentBlock =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "audio_url"; audio_url: { url: string } };

interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentBlock[] | null;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

interface UIMessagePartLike {
  type: string;
  text?: string;
  mediaType?: string;
  url?: string;
}

function uiMessagesToChat(messages: UIMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "assistant" && m.role !== "system") continue;
    const blocks: ContentBlock[] = [];
    let textAccum = "";
    for (const part of (m.parts ?? []) as UIMessagePartLike[]) {
      if (part.type === "text" && typeof part.text === "string") {
        textAccum += part.text;
      } else if (part.type === "file" && part.url) {
        if (part.mediaType?.startsWith("audio/")) {
          blocks.push({ type: "audio_url", audio_url: { url: part.url } });
        } else if (part.mediaType?.startsWith("image/")) {
          blocks.push({ type: "image_url", image_url: { url: part.url } });
        }
      }
    }
    if (textAccum.trim()) {
      blocks.push({ type: "text", text: textAccum });
    }
    if (blocks.length === 0) continue;
    if (blocks.length === 1 && blocks[0].type === "text") {
      out.push({ role: m.role, content: blocks[0].text });
    } else {
      out.push({ role: m.role, content: blocks });
    }
  }
  return out;
}

// --- SSE parsing -----------------------------------------------------------

async function* iterSseLines(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, nl).replace(/\r$/, "");
      buffer = buffer.slice(nl + 1);
      if (line.startsWith("data: ")) yield line.slice(6);
    }
  }
  if (buffer.startsWith("data: ")) yield buffer.slice(6);
}

interface DeltaToolCall {
  id?: string;
  type?: "function";
  index?: number;
  function?: { name?: string; arguments?: string };
}

interface VllmDelta {
  role?: string;
  content?: string;
  reasoning?: string;
  reasoning_content?: string;
  tool_calls?: DeltaToolCall[];
}

interface VllmChoice {
  delta?: VllmDelta;
  finish_reason?: string | null;
}

interface ToolCallAcc {
  id: string;
  name: string;
  argsRaw: string;
  index: number;
}

interface StepResult {
  finishReason: string | null;
  toolCalls: ToolCallAcc[];
  contentEmitted: string;
}

async function streamOneStep(
  tunnelUrl: string,
  history: ChatMessage[],
  writer: UIMessageStreamWriter,
  step: number,
): Promise<StepResult> {
  const payload = {
    model: MODEL_ID,
    messages: history,
    stream: true,
    temperature: 0.6,
    top_p: 0.95,
    max_tokens: 16384,
    chat_template_kwargs: {
      enable_thinking: true,
    },
    tools: TOOL_SPEC,
    tool_choice: "auto",
  };

  const upstream = await fetch(`${tunnelUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!upstream.ok || !upstream.body) {
    const text = upstream.body ? await upstream.text() : "(no body)";
    throw new Error(`vllm ${upstream.status}: ${text.slice(0, 300)}`);
  }

  const reasoningId = `reasoning_${step}`;
  const textId = `text_${step}`;
  let reasoningOpen = false;
  let textOpen = false;
  let contentEmitted = "";
  let finishReason: string | null = null;

  // Map index → accumulator. vllm emits the first tool_call chunk with
  // {id, type, name, index, arguments:""}, then subsequent chunks with the
  // same index and only `function.arguments` deltas.
  const toolAcc = new Map<number, ToolCallAcc>();
  const toolStartedFor = new Set<number>();

  for await (const data of iterSseLines(upstream.body)) {
    if (data === "[DONE]") break;
    let parsed: { choices?: VllmChoice[] };
    try {
      parsed = JSON.parse(data) as { choices?: VllmChoice[] };
    } catch {
      continue;
    }
    const choice = parsed.choices?.[0];
    if (!choice) continue;
    const delta = choice.delta;
    if (delta) {
      const reasoningDelta = delta.reasoning ?? delta.reasoning_content;
      if (reasoningDelta) {
        if (!reasoningOpen) {
          writer.write({ type: "reasoning-start", id: reasoningId });
          reasoningOpen = true;
        }
        writer.write({ type: "reasoning-delta", id: reasoningId, delta: reasoningDelta });
      }
      if (delta.content) {
        if (!textOpen) {
          writer.write({ type: "text-start", id: textId });
          textOpen = true;
        }
        writer.write({ type: "text-delta", id: textId, delta: delta.content });
        contentEmitted += delta.content;
      }
      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          let acc = toolAcc.get(idx);
          if (!acc) {
            acc = {
              id: tc.id ?? `call_${step}_${idx}_${Date.now()}`,
              name: tc.function?.name ?? "",
              argsRaw: "",
              index: idx,
            };
            toolAcc.set(idx, acc);
          }
          if (tc.id && !acc.id) acc.id = tc.id;
          if (tc.function?.name && !acc.name) acc.name = tc.function.name;
          if (acc.name && !toolStartedFor.has(idx)) {
            writer.write({
              type: "tool-input-start",
              toolCallId: acc.id,
              toolName: acc.name,
            });
            toolStartedFor.add(idx);
          }
          if (tc.function?.arguments) {
            acc.argsRaw += tc.function.arguments;
            if (acc.name) {
              writer.write({
                type: "tool-input-delta",
                toolCallId: acc.id,
                inputTextDelta: tc.function.arguments,
              });
            }
          }
        }
      }
    }
    if (choice.finish_reason) finishReason = choice.finish_reason;
  }

  if (reasoningOpen) writer.write({ type: "reasoning-end", id: reasoningId });
  if (textOpen) writer.write({ type: "text-end", id: textId });

  const toolCalls = [...toolAcc.values()].sort((a, b) => a.index - b.index);
  return { finishReason, toolCalls, contentEmitted };
}

// --- Tool execution --------------------------------------------------------

type ToolResult =
  | PubMedSearchResult
  | PrimeKGSubgraphResult
  | GuidelineCurrencyResult
  | CalculatorResult
  | { error: string };

async function runTool(name: string, argsRaw: string): Promise<ToolResult> {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(argsRaw || "{}") as Record<string, unknown>;
  } catch (e) {
    return { error: `invalid tool arguments JSON: ${(e as Error).message}` };
  }

  if (name === "pubmed_search") {
    const query = typeof parsed.query === "string" ? parsed.query : "";
    const maxResults = typeof parsed.maxResults === "number" ? parsed.maxResults : 5;
    if (!query.trim()) return { error: "query is required" };
    try {
      return await pubmedSearch({ query, maxResults });
    } catch (e) {
      return { error: (e as Error).message };
    }
  }

  if (name === "primekg_lookup") {
    const query = typeof parsed.query === "string" ? parsed.query : "";
    const maxHops = typeof parsed.maxHops === "number" ? parsed.maxHops : 2;
    const maxNodes = typeof parsed.maxNodes === "number" ? parsed.maxNodes : 60;
    if (!query.trim()) return { error: "query is required" };
    try {
      return await primekgLookup({ query, maxHops, maxNodes });
    } catch (e) {
      return { error: (e as Error).message };
    }
  }

  if (name === "guideline_currency_check") {
    const query = typeof parsed.query === "string" ? parsed.query : "";
    const maxResults = typeof parsed.maxResults === "number" ? parsed.maxResults : 3;
    if (!query.trim()) return { error: "query is required" };
    try {
      return await guidelineCurrencyCheck({ query, maxResults });
    } catch (e) {
      return { error: (e as Error).message };
    }
  }

  if (name === "clinical_calculate") {
    const score = typeof parsed.score === "string" ? parsed.score : "";
    const inputs = typeof parsed.inputs === "object" && parsed.inputs !== null
      ? parsed.inputs as Record<string, unknown>
      : {};
    if (!score.trim()) {
      return { error: `score is required. Supported: ${SUPPORTED_SCORES.join(", ")}` };
    }
    try {
      return await clinicalCalculate({ score, inputs });
    } catch (e) {
      return { error: (e as Error).message };
    }
  }

  return { error: `unknown tool: ${name}` };
}

// --- Route handler ---------------------------------------------------------

export async function POST(req: NextRequest) {
  const tunnelUrl = process.env.MEDOMNI_TUNNEL_URL;
  if (!tunnelUrl) {
    return new Response(JSON.stringify({ error: "MEDOMNI_TUNNEL_URL not set on server." }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }

  const cl = req.headers.get("content-length");
  if (cl && parseInt(cl, 10) > MAX_BODY_BYTES) {
    return new Response(
      JSON.stringify({
        error: `Body too large (${cl} bytes; max ${MAX_BODY_BYTES}). Audio recordings are capped at 60 s.`,
      }),
      { status: 413, headers: { "Content-Type": "application/json" } },
    );
  }

  let body: { messages?: UIMessage[] };
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const incoming = body.messages ?? [];
  if (incoming.length === 0) {
    return new Response(JSON.stringify({ error: "messages array is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const stream = createUIMessageStream({
    async execute({ writer }) {
      const history: ChatMessage[] = [
        { role: "system", content: SYSTEM_PROMPT },
        ...uiMessagesToChat(incoming),
      ];

      let totalToolCalls = 0;

      for (let step = 0; step < MAX_STEPS; step++) {
        let result: StepResult;
        try {
          result = await streamOneStep(tunnelUrl, history, writer, step);
        } catch (e) {
          writer.write({ type: "error", errorText: (e as Error).message });
          return;
        }

        if (result.finishReason !== "tool_calls" || result.toolCalls.length === 0) {
          // Final answer (or refusal). Stream is closed by writer's
          // text-end/reasoning-end events emitted in streamOneStep.
          return;
        }

        // Append the assistant turn with the structured tool calls to history,
        // exactly as OpenAI tool-calling expects.
        history.push({
          role: "assistant",
          content: result.contentEmitted || null,
          tool_calls: result.toolCalls.map((tc) => ({
            id: tc.id,
            type: "function",
            function: { name: tc.name, arguments: tc.argsRaw || "{}" },
          })),
        });

        for (const tc of result.toolCalls) {
          totalToolCalls += 1;

          // Always emit input-available first so the UI part transitions
          // input-streaming → input-available → output-* cleanly.
          let inputObj: unknown;
          try {
            inputObj = JSON.parse(tc.argsRaw || "{}");
          } catch {
            inputObj = { _raw: tc.argsRaw };
          }
          writer.write({
            type: "tool-input-available",
            toolCallId: tc.id,
            toolName: tc.name,
            input: inputObj,
          });

          if (totalToolCalls > 2) {
            const quotaMsg = "Tool quota for this turn is exceeded. Do not call any more tools. Answer the user from prior knowledge and clearly note which information could not be retrieved.";
            writer.write({
              type: "tool-output-available",
              toolCallId: tc.id,
              output: { error: quotaMsg },
            });
            history.push({
              role: "tool",
              tool_call_id: tc.id,
              content: JSON.stringify({ error: quotaMsg }),
            });
            continue;
          }

          const toolOutput = await runTool(tc.name, tc.argsRaw);
          writer.write({
            type: "tool-output-available",
            toolCallId: tc.id,
            output: toolOutput,
          });
          history.push({
            role: "tool",
            tool_call_id: tc.id,
            content: JSON.stringify(toolOutput),
          });
        }
      }

      writer.write({
        type: "error",
        errorText: `Agent loop exceeded ${MAX_STEPS} steps without a final answer.`,
      });
    },
    onError: (e) => `Agent error: ${(e as Error).message}`,
  });

  return createUIMessageStreamResponse({ stream });
}
