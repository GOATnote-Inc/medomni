// /api/ask — SSE proxy from browser to vllm Nemotron-3-Nano-Omni on B300.
// Sampling-mode aware: detects audio_url / image_url / text-only and applies
// the right Nemotron-Omni params (per model card + REDTEAM-SAMPLING.md).
//
// Body cap: reject > 4 MB to clean-413 instead of leaking past Vercel's 4.5 MB limit.

import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 120;
export const dynamic = "force-dynamic";

const MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4";

const SYSTEM_PROMPT = `You are MedOmni, a medical reasoning assistant served sovereign on NVIDIA Blackwell B300 hardware. Your job is to help clinicians (RNs, NPs, PAs, MDs) and trained healthcare workers think through clinical scenarios.

Discipline:
- State your reasoning briefly, then your recommendation.
- When you cite a guideline (e.g., AHA/ACC, USPSTF, FDA, EBCTCG, ASCO, IDSA), name the year and the specific recommendation.
- If you are uncertain or the evidence is contested, say so explicitly.
- Do NOT fabricate guideline versions, study names, or numerical thresholds.
- Do NOT replace patient-specific clinical judgment. The user is responsible for verifying every recommendation against the patient in front of them.
- Never request, accept, or echo PHI. If the user pastes identifiable patient information, ask them to redact and re-ask.

This is a public demo. Be tight; every word counts.`;

const MAX_BODY_BYTES = 4_000_000;

type ContentBlock =
  | { type: "text"; text?: string }
  | { type: "image_url"; image_url?: { url?: string } }
  | { type: "audio_url"; audio_url?: { url?: string } }
  | { type: string; [k: string]: unknown };

interface Message {
  role: string;
  content: string | ContentBlock[];
}

function detectMode(messages: Message[]): "asr" | "image" | "text" {
  for (const m of messages) {
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block.type === "audio_url") return "asr";
      }
    }
  }
  for (const m of messages) {
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block.type === "image_url") return "image";
      }
    }
  }
  return "text";
}

export async function POST(req: NextRequest) {
  const tunnelUrl = process.env.MEDOMNI_TUNNEL_URL;
  if (!tunnelUrl) {
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

  // Sampling matrix per REDTEAM-SAMPLING.md (server-side guard)
  let upstreamPayload: Record<string, unknown>;
  if (mode === "asr") {
    upstreamPayload = {
      model: MODEL_ID,
      messages,
      stream: true,
      temperature: 0.0,
      top_k: 1,
      max_tokens: 2048,
      chat_template_kwargs: { enable_thinking: false },
    };
  } else {
    upstreamPayload = {
      model: MODEL_ID,
      messages,
      stream: true,
      temperature: 0.6,
      top_p: 0.95,
      max_tokens: 20480,
      thinking_token_budget: 17408,
      chat_template_kwargs: { enable_thinking: true, reasoning_budget: 16384 },
    };
  }

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
