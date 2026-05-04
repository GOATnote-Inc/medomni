"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Tier 1 TTS: Kokoro.js running 100% client-side via WebGPU (or WASM
// fallback). Audio is generated on-device — no audio bytes leave the
// browser. Sovereign by construction.
//
// First-run cost: ~160 MB model download (Transformers.js IndexedDB
// cache) + per-voice ~520 KB style tensor (Cache Storage under
// `kokoro-voices`). Subsequent loads are instant.
//
// This hook deliberately mirrors the public surface of
// `useSpeechSynthesis` so that `useTts` (the unified façade) can swap
// implementations without the consumer noticing.

const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

export type KokoroState = "idle" | "loading" | "speaking";

export interface KokoroVoice {
  id: string; // e.g. "af_heart"
  name: string; // e.g. "Heart"
  language: string; // e.g. "en-us"
  gender: string; // "Female" | "Male"
  overallGrade: string; // e.g. "A", "C+", "D"
  traits?: string;
}

export interface UseKokoroTtsOptions {
  enabled: boolean;
  voiceId?: string;
}

export interface UseKokoroTtsReturn {
  speak: (text: string) => void;
  cancel: () => void;
  state: KokoroState;
  loadProgress: number; // 0..1 while state === "loading"
  voices: KokoroVoice[];
  supported: boolean;
  loaded: boolean;
}

// Static voice catalog — lifted from the kokoro-js source. We keep this
// list in code (not a dynamic import) so VoicePicker can render entries
// before the 30 MB kokoro-js bundle is even fetched. The IDs MUST match
// the ones shipped inside kokoro-js.
const KOKORO_VOICES: KokoroVoice[] = [
  { id: "af_heart", name: "Heart", language: "en-us", gender: "Female", traits: "♥", overallGrade: "A" },
  { id: "af_bella", name: "Bella", language: "en-us", gender: "Female", traits: "fire", overallGrade: "A-" },
  { id: "af_nicole", name: "Nicole", language: "en-us", gender: "Female", traits: "headphones", overallGrade: "B-" },
  { id: "af_aoede", name: "Aoede", language: "en-us", gender: "Female", overallGrade: "C+" },
  { id: "af_kore", name: "Kore", language: "en-us", gender: "Female", overallGrade: "C+" },
  { id: "af_sarah", name: "Sarah", language: "en-us", gender: "Female", overallGrade: "C+" },
  { id: "af_nova", name: "Nova", language: "en-us", gender: "Female", overallGrade: "C" },
  { id: "af_alloy", name: "Alloy", language: "en-us", gender: "Female", overallGrade: "C" },
  { id: "af_sky", name: "Sky", language: "en-us", gender: "Female", overallGrade: "C-" },
  { id: "am_michael", name: "Michael", language: "en-us", gender: "Male", overallGrade: "C+" },
  { id: "am_fenrir", name: "Fenrir", language: "en-us", gender: "Male", overallGrade: "C+" },
  { id: "am_puck", name: "Puck", language: "en-us", gender: "Male", overallGrade: "C+" },
  { id: "am_echo", name: "Echo", language: "en-us", gender: "Male", overallGrade: "D" },
  { id: "am_eric", name: "Eric", language: "en-us", gender: "Male", overallGrade: "D" },
  { id: "am_liam", name: "Liam", language: "en-us", gender: "Male", overallGrade: "D" },
  { id: "am_onyx", name: "Onyx", language: "en-us", gender: "Male", overallGrade: "D" },
  { id: "bf_emma", name: "Emma", language: "en-gb", gender: "Female", traits: "uk", overallGrade: "B-" },
  { id: "bf_isabella", name: "Isabella", language: "en-gb", gender: "Female", overallGrade: "C" },
  { id: "bm_george", name: "George", language: "en-gb", gender: "Male", overallGrade: "C" },
  { id: "bm_fable", name: "Fable", language: "en-gb", gender: "Male", traits: "uk", overallGrade: "C" },
];

export const KOKORO_DEFAULT_VOICE_ID = "af_heart";
export const KOKORO_VOICE_CATALOG = KOKORO_VOICES;

// Module-level cache of the loaded TTS instance + the in-flight
// `from_pretrained` promise. Two consumers (Records OS Ask + the older
// /agent surface) hitting this hook in the same tab share the same 160 MB
// model — we never want to fetch twice.
let kokoroInstance: unknown = null;
let kokoroLoadPromise: Promise<unknown> | null = null;

interface MinimalKokoroAPI {
  generate: (
    text: string,
    opts: { voice: string; speed?: number },
  ) => Promise<{ audio: Float32Array; sampling_rate: number }>;
}

async function loadKokoro(
  onProgress: (pct: number) => void,
): Promise<MinimalKokoroAPI> {
  if (kokoroInstance) return kokoroInstance as MinimalKokoroAPI;
  if (kokoroLoadPromise) return kokoroLoadPromise as Promise<MinimalKokoroAPI>;

  kokoroLoadPromise = (async () => {
    // Dynamic import keeps the 30 MB kokoro-js (+ Transformers.js)
    // bundle out of the initial page load. Only consumers who actually
    // pick a Kokoro voice pay this cost.
    const mod = await import("kokoro-js");
    const KokoroTTS = mod.KokoroTTS;
    const useWebGPU =
      typeof navigator !== "undefined" && "gpu" in navigator;

    const tts = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
      // WebGPU prefers fp32; WASM fallback uses q8 for size/speed.
      dtype: useWebGPU ? "fp32" : "q8",
      device: useWebGPU ? "webgpu" : "wasm",
      progress_callback: (info: unknown) => {
        // Transformers.js emits {status, file, progress, loaded, total}
        // events. We surface the most recent `progress` (0..100) value
        // back to the UI. When `status === "ready"` we're done.
        const e = info as { status?: string; progress?: number };
        if (e && typeof e.progress === "number") {
          onProgress(Math.min(1, Math.max(0, e.progress / 100)));
        }
        if (e && e.status === "ready") {
          onProgress(1);
        }
      },
    });
    kokoroInstance = tts;
    return tts as unknown as MinimalKokoroAPI;
  })();

  try {
    return (await kokoroLoadPromise) as MinimalKokoroAPI;
  } catch (err) {
    // On failure, allow a future call to retry from scratch.
    kokoroLoadPromise = null;
    throw err;
  }
}

export function useKokoroTts(opts: UseKokoroTtsOptions): UseKokoroTtsReturn {
  const { enabled, voiceId = KOKORO_DEFAULT_VOICE_ID } = opts;

  const [state, setState] = useState<KokoroState>("idle");
  const [loadProgress, setLoadProgress] = useState(0);
  const [loaded, setLoaded] = useState(!!kokoroInstance);
  const [supported, setSupported] = useState(false);

  // Queue of pending utterances. We drain serially so phrases don't
  // overlap. Each `speak` call enqueues; the drain loop pulls and feeds
  // the AudioContext.
  const queueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const enabledRef = useRef(enabled);
  const voiceIdRef = useRef(voiceId);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const activeSourceRef = useRef<AudioBufferSourceNode | null>(null);

  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);
  useEffect(() => {
    voiceIdRef.current = voiceId;
  }, [voiceId]);

  // WebGPU detection. Without WebGPU, kokoro-js falls back to WASM —
  // which works but is several × slower. We still report `supported:
  // true` for WASM so the consumer can opt-in deliberately. The Tier 1
  // toggle in VoicePicker hints at this trade-off in copy.
  useEffect(() => {
    if (typeof window === "undefined") {
      setSupported(false);
      return;
    }
    // Both WebGPU (preferred) and WASM (fallback) require a browser env.
    setSupported(true);
  }, []);

  const ensureAudioCtx = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    if (!audioCtxRef.current) {
      const Ctor: typeof AudioContext | undefined =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return null;
      audioCtxRef.current = new Ctor({ sampleRate: 24000 });
    }
    return audioCtxRef.current;
  }, []);

  const playPcm = useCallback(
    async (pcm: Float32Array, sampleRate: number): Promise<void> => {
      const ctx = ensureAudioCtx();
      if (!ctx) return;
      // Browser autoplay policy: AudioContext is created in suspended
      // state on most browsers until a user gesture. The user clicked
      // "voice on" / picked a voice to land here, so resume() is safe.
      if (ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch {
          // ignore
        }
      }
      const buffer = ctx.createBuffer(1, pcm.length, sampleRate);
      // Copy into a fresh Float32Array<ArrayBuffer> so the DOM types
      // are happy. Kokoro returns Float32Array<ArrayBufferLike> (could
      // be a SharedArrayBuffer view); copyToChannel demands the
      // non-shared variant.
      const channel = new Float32Array(pcm.length);
      channel.set(pcm);
      buffer.copyToChannel(channel, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      activeSourceRef.current = source;
      await new Promise<void>((resolve) => {
        source.onended = () => {
          if (activeSourceRef.current === source) {
            activeSourceRef.current = null;
          }
          resolve();
        };
        source.start();
      });
    },
    [ensureAudioCtx],
  );

  const drain = useCallback(async () => {
    if (typeof window === "undefined") return;
    if (speakingRef.current) return;
    if (!enabledRef.current) {
      queueRef.current = [];
      return;
    }

    // Make sure the model is loaded. If we're in the middle of loading
    // already, await that promise — the queue will drain once ready.
    if (!kokoroInstance && !kokoroLoadPromise) {
      // First speak() call ever — kick off the lazy load now.
      try {
        setState("loading");
        await loadKokoro((p) => setLoadProgress(p));
        setLoaded(true);
      } catch (err) {
        console.error("[useKokoroTts] load failed", err);
        setState("idle");
        queueRef.current = [];
        return;
      }
    } else if (!kokoroInstance && kokoroLoadPromise) {
      try {
        setState("loading");
        await kokoroLoadPromise;
        setLoaded(true);
      } catch {
        setState("idle");
        queueRef.current = [];
        return;
      }
    }

    const next = queueRef.current.shift();
    if (!next) {
      setState("idle");
      return;
    }
    speakingRef.current = true;
    setState("speaking");
    try {
      const tts = kokoroInstance as MinimalKokoroAPI;
      const out = await tts.generate(next, { voice: voiceIdRef.current });
      // Re-check enabled — the consumer may have toggled off while we
      // were generating audio.
      if (!enabledRef.current) {
        speakingRef.current = false;
        setState("idle");
        return;
      }
      await playPcm(out.audio, out.sampling_rate);
    } catch (err) {
      console.error("[useKokoroTts] generate/play failed", err);
    } finally {
      speakingRef.current = false;
    }
    // Recurse to drain the rest of the queue.
    drain();
  }, [playPcm]);

  const speak = useCallback(
    (text: string) => {
      if (typeof window === "undefined") return;
      if (!enabledRef.current) return;
      const trimmed = text.trim();
      if (!trimmed) return;
      queueRef.current.push(trimmed);
      // Fire and forget — drain handles its own state transitions.
      void drain();
    },
    [drain],
  );

  const cancel = useCallback(() => {
    queueRef.current = [];
    if (activeSourceRef.current) {
      try {
        activeSourceRef.current.stop();
      } catch {
        // already stopped
      }
      activeSourceRef.current = null;
    }
    speakingRef.current = false;
    setState((prev) => (prev === "loading" ? "loading" : "idle"));
  }, []);

  // If the consumer flips `enabled` off mid-stream, stop talking
  // immediately. Same semantics as useSpeechSynthesis.
  useEffect(() => {
    if (!enabled) {
      cancel();
    }
  }, [enabled, cancel]);

  // Tear down the AudioContext on unmount to release the audio device.
  useEffect(() => {
    return () => {
      if (activeSourceRef.current) {
        try {
          activeSourceRef.current.stop();
        } catch {
          // ignore
        }
        activeSourceRef.current = null;
      }
      if (audioCtxRef.current) {
        try {
          void audioCtxRef.current.close();
        } catch {
          // ignore
        }
        audioCtxRef.current = null;
      }
    };
  }, []);

  return {
    speak,
    cancel,
    state,
    loadProgress,
    voices: KOKORO_VOICES,
    supported,
    loaded,
  };
}
