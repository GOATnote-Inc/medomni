// /api/ask — SSE proxy from the browser to vllm on the Brev B300 pod
// via a Cloudflare quick tunnel. The upstream model is
// nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 (Blackwell-only NVFP4).
//
// MEDOMNI_TUNNEL_URL is set in Vercel project env. If unset, we return a
// non-streaming JSON error so the UI can render the "wiring up" notice
// instead of crashing.
//
// nodejs runtime — edge runtime has historically had streaming-fetch quirks
// for SSE proxies that don't terminate cleanly. The Plan agent's stack-rank
// prefers nodejs for this use case.

import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;
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

export async function POST(req: NextRequest) {
  const tunnelUrl = process.env.MEDOMNI_TUNNEL_URL;

  if (!tunnelUrl) {
    return new Response(
      JSON.stringify({
        error: "MEDOMNI_TUNNEL_URL not set on the server. The model endpoint is being plugged in.",
      }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }

  let body: { messages?: Array<{ role: string; content: string }> };
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

  // Prepend system prompt unless the caller already supplied one
  const hasSystem = userMessages.some((m) => m.role === "system");
  const messages = hasSystem
    ? userMessages
    : [{ role: "system", content: SYSTEM_PROMPT }, ...userMessages];

  // Call vllm with streaming
  let upstream: Response;
  try {
    upstream = await fetch(`${tunnelUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL_ID,
        messages,
        stream: true,
        max_tokens: 2048,
        temperature: 0.3,
      }),
    });
  } catch (e) {
    return new Response(
      JSON.stringify({
        error: `Upstream connection failed: ${(e as Error).message}`,
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(
      JSON.stringify({
        error: `Upstream returned ${upstream.status}`,
        detail: text.slice(0, 500),
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  if (!upstream.body) {
    return new Response(JSON.stringify({ error: "Upstream returned empty body" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Pass the SSE stream through unmodified — the client parses delta.content
  // and delta.reasoning fields.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
