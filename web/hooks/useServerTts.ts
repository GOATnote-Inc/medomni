"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { BASE_PATH } from "@/lib/basePath";

// Tier 2 TTS: server-side Kokoro via /api/tts proxy → Cloudflare tunnel
// → lobster H200. Same Kokoro-82M model as the in-browser WebGPU hook
// (useKokoroTts.ts) but inference happens on the GPU pod, eliminating
// the ~160 MB browser download. Network adds ~400 ms TTFB; total ~500 ms
// for a short sentence — well inside the conversational budget.
//
// Mirrors the public surface of useSpeechSynthesis / useKokoroTts so the
// unified `useTts` façade can route here without consumer changes.

export type ServerTtsState = "idle" | "speaking";

export interface UseServerTtsOptions {
  enabled: boolean;
  voiceId?: string;
}

export interface UseServerTtsReturn {
  speak: (text: string) => void;
  cancel: () => void;
  state: ServerTtsState;
  supported: boolean;
}

const DEFAULT_VOICE = "af_heart";

export function useServerTts(opts: UseServerTtsOptions): UseServerTtsReturn {
  const { enabled, voiceId = DEFAULT_VOICE } = opts;

  const [state, setState] = useState<ServerTtsState>("idle");
  const [supported, setSupported] = useState(false);

  // Queue of pending utterance text. Drained serially so phrases don't
  // overlap. Each `speak` call enqueues; the drain loop pulls and POSTs
  // /api/tts then plays the resulting audio.
  const queueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const enabledRef = useRef(enabled);
  const voiceRef = useRef(voiceId);
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);
  useEffect(() => {
    voiceRef.current = voiceId;
  }, [voiceId]);

  // Detect support: requires window + AudioContext + fetch.
  useEffect(() => {
    if (typeof window === "undefined") {
      setSupported(false);
      return;
    }
    const hasAudio = !!(window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext);
    setSupported(hasAudio && typeof fetch === "function");
  }, []);

  const ensureCtx = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    if (ctxRef.current && ctxRef.current.state !== "closed") {
      // Resume on user-gesture; iOS/Safari leave contexts suspended otherwise.
      if (ctxRef.current.state === "suspended") {
        ctxRef.current.resume().catch(() => {});
      }
      return ctxRef.current;
    }
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    ctxRef.current = new Ctor();
    return ctxRef.current;
  }, []);

  const drain = useCallback(async () => {
    if (typeof window === "undefined") return;
    if (speakingRef.current) return;
    if (!enabledRef.current) {
      queueRef.current = [];
      return;
    }
    const next = queueRef.current.shift();
    if (!next) {
      setState("idle");
      return;
    }

    speakingRef.current = true;
    setState("speaking");

    const ctx = ensureCtx();
    if (!ctx) {
      // No AudioContext — drop and continue.
      speakingRef.current = false;
      setState("idle");
      return;
    }

    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const res = await fetch(`${BASE_PATH}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: next,
          voice: voiceRef.current,
          format: "mp3",
        }),
        signal: ac.signal,
      });
      if (!res.ok) {
        // Swallow + continue — TTS is non-critical.
        speakingRef.current = false;
        // eslint-disable-next-line no-console
        console.warn(`[useServerTts] /api/tts ${res.status}`);
        await drain();
        return;
      }
      const buf = await res.arrayBuffer();
      const audio = await ctx.decodeAudioData(buf.slice(0));
      const src = ctx.createBufferSource();
      src.buffer = audio;
      src.connect(ctx.destination);
      sourceRef.current = src;
      src.onended = () => {
        speakingRef.current = false;
        sourceRef.current = null;
        // Recurse to drain remaining queue.
        drain();
      };
      src.start();
    } catch (err) {
      const e = err as { name?: string };
      if (e.name === "AbortError") {
        speakingRef.current = false;
        return;
      }
      // eslint-disable-next-line no-console
      console.warn(`[useServerTts] error: ${(err as Error).message}`);
      speakingRef.current = false;
      drain();
    } finally {
      abortRef.current = null;
    }
  }, [ensureCtx]);

  const speak = useCallback(
    (text: string) => {
      if (typeof window === "undefined") return;
      if (!enabledRef.current) return;
      const trimmed = text.trim();
      if (!trimmed) return;
      queueRef.current.push(trimmed);
      drain();
    },
    [drain],
  );

  const cancel = useCallback(() => {
    queueRef.current = [];
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (sourceRef.current) {
      try {
        sourceRef.current.stop();
      } catch {
        // already stopped
      }
      sourceRef.current = null;
    }
    speakingRef.current = false;
    setState("idle");
  }, []);

  // If consumer flips enabled off mid-stream, stop talking immediately.
  useEffect(() => {
    if (!enabled) cancel();
  }, [enabled, cancel]);

  // On unmount, kill any in-flight playback.
  useEffect(() => {
    return () => {
      if (sourceRef.current) {
        try {
          sourceRef.current.stop();
        } catch {
          // ignore
        }
      }
      if (abortRef.current) abortRef.current.abort();
      if (ctxRef.current && ctxRef.current.state !== "closed") {
        ctxRef.current.close().catch(() => {});
      }
    };
  }, []);

  return { speak, cancel, state, supported };
}
