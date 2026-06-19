// /api/ask — SSE proxy from browser to vllm Nemotron-3-Nano-Omni on the H100.
// Sampling-mode aware: detects audio_url / image_url / text-only and applies
// the right Nemotron-Omni params (per model card + REDTEAM-SAMPLING.md).
//
// Body cap: reject > 4 MB to clean-413 instead of leaking past Vercel's 4.5 MB limit.

import type { NextRequest } from "next/server";
import { getAnthropic, llmProvider, OPUS_MODEL_ID } from "@/lib/llm/anthropic";
import { toAnthropicMessages, type ChatMsg } from "@/lib/llm/translate";

export const runtime = "nodejs";
export const maxDuration = 120;
export const dynamic = "force-dynamic";

// vllm container is launched with --served-model-name nemotron, so it only
// accepts that alias (NOT the full HF path). vLLM path = rollback only.
const MODEL_ID = "nemotron";

const SYSTEM_PROMPT = `You are MedOmni, a medical reasoning assistant served sovereign on dedicated hardware. Your job is to help clinicians (RNs, NPs, PAs, MDs) and trained healthcare workers think through clinical scenarios.

Discipline:
- State your reasoning briefly, then your recommendation.
- When you cite a guideline (e.g., AHA/ACC, USPSTF, FDA, EBCTCG, ASCO, IDSA), name the year and the specific recommendation.
- If you are uncertain or the evidence is contested, say so explicitly.
- Do NOT fabricate guideline versions, study names, or numerical thresholds.
- Do NOT replace patient-specific clinical judgment. The user is responsible for verifying every recommendation against the patient in front of them.
- Never request, accept, or echo PHI. If the user pastes identifiable patient information, ask them to redact and re-ask.

This is a public demo. Be tight; every word counts.`;

const MAX_BODY_BYTES = 4_000_000;

// Audio (audio_url) dropped in the 2026-06 Claude migration. Image kept.
type ContentBlock =
  | { type: "text"; text?: string }
  | { type: "image_url"; image_url?: { url?: string } }
  | { type: string; [k: string]: unknown };

interface Message {
  role: string;
  content: string | ContentBlock[];
}

function detectMode(messages: Message[]): "image" | "text" {
  for (const m of messages) {
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block.type === "image_url") return "image";
      }
    }
  }
  return "text";
}

// --- Anthropic (Claude) ask path -------------------------------------------
//
// Streams Claude but emits the same OpenAI-shaped SSE the legacy vLLM
// pass-through produced (`data: {choices:[{delta:{content}}]}` … `[DONE]`),
// so any /api/ask consumer keeps working unchanged. Only text deltas are
// forwarded; thinking happens (quality) but is not surfaced on this path.

function extractSystem(messages: Message[]): string {
  const sys = messages
    .filter((m) => m.role === "system")
    .map((m) => (typeof m.content === "string" ? m.content : ""))
    .filter(Boolean);
  return sys.length > 0 ? sys.join("\n\n") : SYSTEM_PROMPT;
}

function askSseChunk(content: string): string {
  return `data: ${JSON.stringify({
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  })}\n\n`;
}

function streamAnthropicAsk(messages: Message[], mode: "image" | "text"): Response {
  const system = extractSystem(messages);
  const aMessages = toAnthropicMessages(messages as unknown as ChatMsg[]);
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        const s = getAnthropic().messages.stream({
          model: OPUS_MODEL_ID,
          max_tokens: 8192,
          system,
          messages: aMessages,
          thinking: { type: "adaptive", display: "omitted" },
          output_config: { effort: "medium" },
        });
        for await (const event of s) {
          if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
            controller.enqueue(encoder.encode(askSseChunk(event.delta.text)));
          }
        }
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      } catch (e) {
        controller.enqueue(encoder.encode(askSseChunk(`\n[error: ${(e as Error).message}]`)));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      } finally {
        controller.close();
      }
    },
  });
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
      "X-Medomni-Mode": mode,
    },
  });
}

export async function POST(req: NextRequest) {
  const provider = llmProvider();
  const tunnelUrl = process.env.MEDOMNI_TUNNEL_URL;
  if (provider === "vllm" && !tunnelUrl) {
    return new Response(JSON.stringify({ error: "MEDOMNI_TUNNEL_URL not set on server." }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Body-size guard (REDTEAM-AUDIO §10)
  const cl = req.headers.get("content-length");
  if (cl && parseInt(cl, 10) > MAX_BODY_BYTES) {
    return new Response(
      JSON.stringify({ error: `Body too large (${cl} bytes; max ${MAX_BODY_BYTES}). Recordings are capped at 60 s.` }),
      { status: 413, headers: { "Content-Type": "application/json" } },
    );
  }

  let body: { messages?: Message[] };
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const userMessages = body.messages ?? [];
  if (userMessages.length === 0) {
    return new Response(JSON.stringify({ error: "messages array is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const hasSystem = userMessages.some((m) => m.role === "system");
  const messages = hasSystem
    ? userMessages
    : [{ role: "system", content: SYSTEM_PROMPT } as Message, ...userMessages];

  const mode = detectMode(userMessages);

  if (provider === "anthropic") {
    return streamAnthropicAsk(messages, mode);
  }

  // --- vLLM path (rollback only) ------------------------------------------
  // Empirical finding (2026-04-30 live test): with `enable_thinking=false` +
  // audio_url + nemotron_v3 reasoning-parser, ALL output streams to
  // delta.reasoning and delta.content stays empty. Confirmed via tunnel curl:
  // thinking=true → 150 reasoning + 27 content chunks; thinking=false → 0
  // content. So the model-card's ASR settings break our composer's render
  // path.
  //
  // Fix: use thinking-mode params for ALL modalities. Audio gets a smaller
  // reasoning_budget so transcription completes faster, but enable_thinking
  // stays TRUE so delta.content actually emits. Mode detection stays for
  // observability via X-Medomni-Mode header.
  const upstreamPayload: Record<string, unknown> = {
    model: MODEL_ID,
    messages,
    stream: true,
    temperature: 0.6,
    top_p: 0.95,
    max_tokens: 20480,
    thinking_token_budget: 17408,
    chat_template_kwargs: {
      enable_thinking: true,
      reasoning_budget: 16384,
    },
  };

  let upstream: Response;
  try {
    upstream = await fetch(`${tunnelUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(upstreamPayload),
    });
  } catch (e) {
    return new Response(
      JSON.stringify({ error: `Upstream connection failed: ${(e as Error).message}` }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(
      JSON.stringify({ error: `Upstream returned ${upstream.status}`, detail: text.slice(0, 500), mode }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  if (!upstream.body) {
    return new Response(JSON.stringify({ error: "Upstream returned empty body" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
      "X-Medomni-Mode": mode,
    },
  });
}
