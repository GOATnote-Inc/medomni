// /api/agent — server-side agent loop. Stays alongside /api/ask (which is the
// non-agent one-shot proxy); does not modify it.
//
// Why a manual loop instead of `streamText({tools: ...})`: the live vllm-omni
// container on catfish is launched WITHOUT `--enable-auto-tool-choice` /
// `--tool-call-parser`, so the OpenAI-style tool-calls API isn't wired up.
// Verified 2026-05-02 via `docker inspect`. Until catfish is reconfigured we
// run a prompted-tool loop: model emits a `<tool_call>{json}</tool_call>`
// marker in its content stream, we parse, execute, and feed the result back
// as a synthetic user turn. The wire format and React side stay AI-SDK
// native via `createUIMessageStream` so when vllm gets the tool-call flag,
// swapping to `streamText({tools})` is a contained change.

import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  type UIMessage,
  type UIMessageStreamWriter,
} from "ai";
import type { NextRequest } from "next/server";
import { pubmedSearch, type PubMedSearchResult } from "@/lib/tools/pubmed";

export const runtime = "nodejs";
export const maxDuration = 300;
export const dynamic = "force-dynamic";

const MODEL_ID = "nemotron";
const MAX_STEPS = 4;
const MAX_BODY_BYTES = 1_000_000;

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
            description: "PubMed query string. Supports MeSH terms, field tags, and date ranges, e.g. 'norepinephrine vasopressin septic shock 2020:2024'.",
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
];

const SYSTEM_PROMPT = `You are MedOmni, a medical reasoning assistant served sovereign on NVIDIA Blackwell B300 hardware. You help clinicians (RNs, NPs, PAs, MDs) and trained healthcare workers think through clinical scenarios.

Discipline:
- State your reasoning briefly, then your recommendation.
- Cite guidelines (AHA/ACC, USPSTF, FDA, IDSA, ASCO, EBCTCG, etc.) with year and the specific recommendation.
- If uncertain or evidence is contested, say so.
- Never fabricate guideline versions, study names, or numerical thresholds.
- Never replace patient-specific clinical judgment.
- Never request, accept, or echo PHI. If the user pastes identifiable patient information, ask them to redact and re-ask.

Tool: pubmed_search
You may search PubMed when current literature would meaningfully change your answer — drug-drug interactions, recent trial results, less-common diagnoses, or when you'd otherwise hedge with "I'm not sure of the latest evidence."

Parameters:
  query (string, required) — a valid PubMed search query.
  maxResults (integer, optional, default 5, max 10)

PubMed query syntax notes (these matter — getting them wrong returns zero hits):
  - Date ranges need a field tag, e.g. "vasopressin AND septic shock AND 2020:2024[dp]". Bare "2020:2024" is treated as a literal token and matches nothing.
  - Field tags: [dp] = date of publication, [mh] = MeSH, [tiab] = title/abstract, [au] = author.
  - Boolean operators are uppercase: AND, OR, NOT.
  - Prefer specific MeSH terms when you know them, e.g. "shock, septic"[mh].

When you call the tool, the response is held until the search returns. The result will arrive as a <tool_result name="pubmed_search">{...}</tool_result> block in the next user turn. After the result, continue your answer and cite specific PMIDs you used. You may call the tool at most twice per turn. If you do not need PubMed, answer directly.

This is a public demo. Be tight; every word counts.`;

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

function uiMessagesToChat(messages: UIMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "assistant" && m.role !== "system") continue;
    let text = "";
    for (const part of m.parts ?? []) {
      if (part.type === "text" && typeof part.text === "string") {
        text += part.text;
      }
    }
    if (text) out.push({ role: m.role, content: text });
  }
  return out;
}

// Nemotron-3-Nano-Omni emits at least three tool-call formats non-
// deterministically depending on the prompt and chat template state:
//   A. Anthropic-style: <invoke name="X">...<parameter K>V</parameter>...</invoke>
//   B. Hermes/qwen-style: <|tool_call|>...<|X|>...<parameter=K>V</parameter>
//      ...</tool_call>   (closing tags often inconsistent)
//   C. The format we instruct: <tool_call name="X">{...json...}</tool_call>
// The parser scans for any envelope, extracts the tool name, and harvests
// parameter key/value pairs with permissive separators.
const ENVELOPE_RES: Array<{ open: RegExp; close: RegExp }> = [
  { open: /<invoke\b[^>]*>/, close: /<\/invoke>/ },
  { open: /<\|tool_call\|>/, close: /<\/(?:\|tool_call\||tool_call)>/ },
  { open: /<tool_call\b[^>]*>/, close: /<\/tool_call>/ },
];
const NAME_PATTERNS = [
  /<invoke\s+name="([a-zA-Z_][a-zA-Z0-9_]*)"/,
  /<tool_call\s+name="([a-zA-Z_][a-zA-Z0-9_]*)"/,
  /<\|([a-zA-Z_][a-zA-Z0-9_]*)\|>/,
];
// Matches both <parameter=key> (Hermes, equals) and <parameter key> (Anthropic, space).
const PARAM_RE = /<parameter[\s=]+([a-zA-Z_][a-zA-Z0-9_]*)\s*>([\s\S]*?)<\/parameter>/g;

interface ParsedToolCall {
  name: string;
  input: Record<string, unknown>;
  matchStart: number;
  matchEnd: number;
}

function coerceParamValue(raw: string): unknown {
  const t = raw.trim();
  if (/^-?\d+$/.test(t)) return parseInt(t, 10);
  if (/^-?\d+\.\d+$/.test(t)) return parseFloat(t);
  if (t === "true") return true;
  if (t === "false") return false;
  return t;
}

function parseToolCall(text: string): ParsedToolCall | null {
  // Format C (JSON body) — preferred when present, exact match.
  const xmlJson = text.match(/<tool_call\s+name="([a-zA-Z_][a-zA-Z0-9_]*)">([\s\S]*?)<\/tool_call>/);
  if (xmlJson) {
    try {
      const input = JSON.parse(xmlJson[2].trim()) as Record<string, unknown>;
      return {
        name: xmlJson[1],
        input,
        matchStart: xmlJson.index ?? 0,
        matchEnd: (xmlJson.index ?? 0) + xmlJson[0].length,
      };
    } catch {
      // fall through
    }
  }

  for (const env of ENVELOPE_RES) {
    const om = text.match(env.open);
    if (!om) continue;
    const openIdx = om.index ?? 0;
    const after = text.slice(openIdx + om[0].length);
    const cm = after.match(env.close);
    if (!cm) continue;
    const innerStart = openIdx + om[0].length;
    const matchEnd = innerStart + (cm.index ?? 0) + cm[0].length;
    const envelopeText = text.slice(openIdx, matchEnd);

    let name: string | null = null;
    for (const np of NAME_PATTERNS) {
      const nm = envelopeText.match(np);
      if (nm) {
        name = nm[1];
        break;
      }
    }
    // Single-tool short-circuit: when the envelope is present but the model
    // omitted the tool name (observed with Nemotron when the tool spec is
    // present but the chat template's tool-call schema is incomplete), we
    // assume the only tool we have. Revisit if a second tool is added.
    if (!name && (text.includes("<tool_call") || text.includes("<invoke"))) {
      name = "pubmed_search";
    }
    if (!name) continue;

    const input: Record<string, unknown> = {};
    for (const pm of envelopeText.matchAll(PARAM_RE)) {
      input[pm[1].trim()] = coerceParamValue(pm[2]);
    }
    return { name, input, matchStart: openIdx, matchEnd };
  }
  return null;
}

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

interface StepResult {
  reasoning: string;
  content: string;
  finishReason: string | null;
}

async function streamOneStep(
  tunnelUrl: string,
  history: ChatMessage[],
  writer: UIMessageStreamWriter,
  reasoningId: string,
): Promise<StepResult> {
  const payload = {
    model: MODEL_ID,
    messages: history,
    stream: true,
    temperature: 0.6,
    top_p: 0.95,
    max_tokens: 16384,
    thinking_token_budget: 12288,
    chat_template_kwargs: {
      enable_thinking: true,
      reasoning_budget: 11264,
    },
    // Send the OpenAI tool spec so the chat template injects the model's
    // trained tool-call schema. The container is launched WITHOUT
    // --enable-auto-tool-choice, so vllm refuses tool_choice="auto" and
    // refuses bare `tools` (defaults to "auto"). Workaround: pass
    // tool_choice="none". vllm still forwards the schema through the chat
    // template; the model decides on its own whether to emit a tool call;
    // we parse the resulting plaintext envelope ourselves.
    tools: TOOL_SPEC,
    tool_choice: "none",
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

  let reasoning = "";
  let content = "";
  let finishReason: string | null = null;
  let reasoningStarted = false;

  for await (const data of iterSseLines(upstream.body)) {
    if (data === "[DONE]") break;
    let parsed: {
      choices?: Array<{
        delta?: { content?: string; reasoning_content?: string };
        finish_reason?: string | null;
      }>;
    };
    try {
      parsed = JSON.parse(data);
    } catch {
      continue;
    }
    const choice = parsed.choices?.[0];
    if (!choice) continue;
    const reasoningDelta = choice.delta?.reasoning_content;
    const contentDelta = choice.delta?.content;
    if (reasoningDelta) {
      if (!reasoningStarted) {
        writer.write({ type: "reasoning-start", id: reasoningId });
        reasoningStarted = true;
      }
      reasoning += reasoningDelta;
      writer.write({ type: "reasoning-delta", id: reasoningId, delta: reasoningDelta });
    }
    if (contentDelta) {
      content += contentDelta;
    }
    if (choice.finish_reason) {
      finishReason = choice.finish_reason;
    }
  }

  if (reasoningStarted) {
    writer.write({ type: "reasoning-end", id: reasoningId });
  }
  return { reasoning, content, finishReason };
}

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
    return new Response(JSON.stringify({ error: `Body too large (${cl} bytes; max ${MAX_BODY_BYTES}).` }), {
      status: 413,
      headers: { "Content-Type": "application/json" },
    });
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

      let toolCallCount = 0;

      for (let step = 0; step < MAX_STEPS; step++) {
        const reasoningId = `reasoning_${step}`;
        let result: StepResult;
        try {
          result = await streamOneStep(tunnelUrl, history, writer, reasoningId);
        } catch (e) {
          writer.write({ type: "error", errorText: (e as Error).message });
          return;
        }

        const tc = parseToolCall(result.content);
        if (process.env.MEDOMNI_AGENT_DEBUG === "1") {
          // Diagnostic surface so we can see why a turn went empty.
          writer.write({
            type: "data-debug",
            data: {
              step,
              contentPreview: result.content.slice(0, 600),
              parsedTool: tc,
              finishReason: result.finishReason,
            },
          });
        }

        if (!tc || tc.name !== "pubmed_search" || toolCallCount >= 2) {
          // Final answer. Strip any malformed/over-budget tool-call envelopes
          // so the user-visible text is clean prose.
          const finalText = result.content
            .replace(/<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/g, "")
            .replace(/<\|tool_call\|>[\s\S]*?<\/(?:\|tool_call\||tool_call)>/g, "")
            .replace(/<invoke\b[^>]*>[\s\S]*?<\/invoke>/g, "")
            .trim();
          if (finalText) {
            const textId = `text_${step}`;
            writer.write({ type: "text-start", id: textId });
            writer.write({ type: "text-delta", id: textId, delta: finalText });
            writer.write({ type: "text-end", id: textId });
          }
          return;
        }

        toolCallCount += 1;
        const toolCallId = `pubmed_${step}_${Date.now()}`;
        const query = String(tc.input.query ?? "").trim();
        const maxResults = typeof tc.input.maxResults === "number" ? tc.input.maxResults : 5;

        writer.write({ type: "tool-input-start", toolCallId, toolName: "pubmed_search" });
        writer.write({
          type: "tool-input-available",
          toolCallId,
          toolName: "pubmed_search",
          input: { query, maxResults },
        });

        let toolOutput: PubMedSearchResult | { error: string };
        try {
          toolOutput = await pubmedSearch({ query, maxResults });
        } catch (e) {
          toolOutput = { error: (e as Error).message };
        }
        writer.write({ type: "tool-output-available", toolCallId, output: toolOutput });

        history.push({ role: "assistant", content: result.content });
        history.push({
          role: "user",
          content: `<tool_result name="pubmed_search">${JSON.stringify(toolOutput)}</tool_result>\n\nUsing this result, continue your answer for the original question. Cite specific PMIDs you used. Do not call the tool again unless strictly necessary.`,
        });
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
