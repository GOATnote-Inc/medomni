// /api/tts — server-side proxy to the sovereign Kokoro-FastAPI server on
// lobster (H200). Tier 2 of the layered TTS design:
//
//   Tier 0 — browser speechSynthesis (last-resort fallback)
//   Tier 1 — Kokoro WebGPU in-browser (~160 MB cache, then ~100ms TTFB)
//   Tier 2 — server-side Kokoro on lobster via Cloudflare quick-tunnel
//            (this route — no browser download, ~400ms TTFB)
//
// The lobster URL + token are server-side env vars only; the client never
// sees the tunnel hostname or the bearer token. Only NEXT_PUBLIC_MEDOMNI_USE_SERVER_TTS
// flips client-visible behavior.
//
// Request:  POST /4UWHAt/api/tts  body { text: string; voice?: string; format?: "mp3"|"wav"|"opus"|"pcm" }
// Response: audio/mpeg (or matching content-type for the requested format)
//
// Sovereignty note: the audio bytes transit Cloudflare's edge but no model
// weights, no LLM keys, no PHI cross any external boundary. Synthetic-data
// demo only. Cloudflared can be torn down with `pkill cloudflared` on lobster.

import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 30;
export const dynamic = "force-dynamic";

const DEFAULT_VOICE = "af_heart";
const DEFAULT_FORMAT = "mp3";
const ALLOWED_FORMATS = new Set(["mp3", "wav", "opus", "pcm", "flac", "aac"]);

// Cap input length so an over-eager client can't ask Kokoro to synthesize
// a 20K-char paragraph in one shot. Sentence-buffered consumers send
// <500 chars per call.
const MAX_INPUT_CHARS = 2000;

interface TtsRequestBody {
  text?: string;
  voice?: string;
  format?: string;
}

export async function POST(req: NextRequest) {
  const ttsUrl = process.env.MEDOMNI_TTS_URL;
  const ttsToken = process.env.MEDOMNI_TTS_TOKEN;
  if (!ttsUrl) {
    return new Response(
      JSON.stringify({ error: "MEDOMNI_TTS_URL not set on this deployment." }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }

  let body: TtsRequestBody;
  try {
    body = (await req.json()) as TtsRequestBody;
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const text = (body.text ?? "").trim();
  if (!text) {
    return new Response(JSON.stringify({ error: "text is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (text.length > MAX_INPUT_CHARS) {
    return new Response(
      JSON.stringify({
        error: `text too long (${text.length} chars; max ${MAX_INPUT_CHARS}). Split client-side at sentence boundaries.`,
      }),
      { status: 413, headers: { "Content-Type": "application/json" } },
    );
  }

  const voice = (body.voice ?? DEFAULT_VOICE).trim();
  const format = (body.format ?? DEFAULT_FORMAT).toLowerCase();
  if (!ALLOWED_FORMATS.has(format)) {
    return new Response(
      JSON.stringify({
        error: `unsupported format: ${format}. allowed: ${Array.from(ALLOWED_FORMATS).join(", ")}`,
      }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 25_000);

  let upstream: Response;
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "audio/*",
    };
    if (ttsToken) {
      // Forward bearer token. Today's Kokoro-FastAPI doesn't validate it,
      // but a future auth-proxy sidecar on lobster will. Forward-compatible.
      headers["Authorization"] = `Bearer ${ttsToken}`;
    }
    upstream = await fetch(`${ttsUrl.replace(/\/$/, "")}/v1/audio/speech`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "kokoro",
        input: text,
        voice,
        response_format: format,
      }),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    return new Response(
      JSON.stringify({
        error: `tts upstream unreachable: ${(err as Error).message}`,
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }
  clearTimeout(timer);

  if (!upstream.ok) {
    const errText = await upstream.text().catch(() => "(no body)");
    return new Response(
      JSON.stringify({
        error: `tts upstream ${upstream.status}: ${errText.slice(0, 200)}`,
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  const upstreamCT = upstream.headers.get("content-type") ?? "audio/mpeg";
  const buf = await upstream.arrayBuffer();

  return new Response(buf, {
    status: 200,
    headers: {
      "Content-Type": upstreamCT,
      "Content-Length": String(buf.byteLength),
      "Cache-Control": "no-store",
    },
  });
}
