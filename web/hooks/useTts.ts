"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";

import { useKokoroTts, KOKORO_DEFAULT_VOICE_ID } from "@/hooks/useKokoroTts";
import { useSpeechSynthesis } from "@/hooks/useSpeechSynthesis";

// Unified TTS façade. Consumers (Records OS Ask bar, /agent page) import
// THIS hook only — it routes to either Tier 0 (browser speechSynthesis)
// or Tier 1 (Kokoro.js WebGPU) based on the `selectedVoice` token.
//
// Selection token format:
//   "browser:<voiceURI>"   → Tier 0
//   "kokoro:<voiceId>"     → Tier 1
//   ""                     → Tier 0 with auto-pick (the existing
//                            useSpeechSynthesis preferred-voice logic)

export type TtsVoiceTier = "browser" | "kokoro";

export interface ParsedTtsVoice {
  tier: TtsVoiceTier;
  id: string; // voiceURI (browser) or kokoro voice id
}

export function parseSelectedVoice(token: string | null | undefined): ParsedTtsVoice {
  if (!token) return { tier: "browser", id: "" };
  if (token.startsWith("kokoro:")) return { tier: "kokoro", id: token.slice("kokoro:".length) };
  if (token.startsWith("browser:")) return { tier: "browser", id: token.slice("browser:".length) };
  // Legacy callers that stored a raw voiceURI in localStorage land here.
  return { tier: "browser", id: token };
}

export interface UseTtsOptions {
  enabled: boolean;
  selectedVoice: string;
  onLoadProgress?: (pct: number) => void;
}

export interface UseTtsReturn {
  speak: (text: string) => void;
  cancel: () => void;
  state: "idle" | "loading" | "speaking";
  loadProgress: number;
  loaded: boolean;
  /** Underlying tier in active use — useful for status pills. */
  tier: TtsVoiceTier;
  /** Tier 0 voices the browser exposes. Pass to VoicePicker. */
  browserVoices: SpeechSynthesisVoice[];
  /** Whether the runtime advertises WebGPU. Pass to VoicePicker. */
  webgpuSupported: boolean;
}

export function useTts(opts: UseTtsOptions): UseTtsReturn {
  const { enabled, selectedVoice, onLoadProgress } = opts;
  const parsed = useMemo(() => parseSelectedVoice(selectedVoice), [selectedVoice]);

  // Both hooks always mount; we just gate which one's `speak` we route
  // to. This is the simplest correct pattern — toggling between hooks
  // by tier would violate the rules of hooks. The cost is small: each
  // hook's effects only fire once on mount.
  const browser = useSpeechSynthesis({
    enabled: enabled && parsed.tier === "browser",
    voiceURI: parsed.tier === "browser" ? parsed.id || undefined : undefined,
  });

  const kokoro = useKokoroTts({
    enabled: enabled && parsed.tier === "kokoro",
    voiceId: parsed.tier === "kokoro" ? parsed.id || KOKORO_DEFAULT_VOICE_ID : undefined,
  });

  // When the active tier flips, silence the other side so we don't get
  // double-tracked audio during the swap.
  const lastTierRef = useRef<TtsVoiceTier>(parsed.tier);
  useEffect(() => {
    if (lastTierRef.current !== parsed.tier) {
      if (lastTierRef.current === "browser") browser.cancel();
      else kokoro.cancel();
      lastTierRef.current = parsed.tier;
    }
  }, [parsed.tier, browser, kokoro]);

  // Forward Kokoro load progress back to the consumer (so VoicePicker
  // can render "downloading 45%").
  useEffect(() => {
    if (parsed.tier !== "kokoro") return;
    onLoadProgress?.(kokoro.loadProgress);
  }, [parsed.tier, kokoro.loadProgress, onLoadProgress]);

  const speak = useCallback(
    (text: string) => {
      if (parsed.tier === "kokoro") kokoro.speak(text);
      else browser.speak(text);
    },
    [parsed.tier, browser, kokoro],
  );

  const cancel = useCallback(() => {
    // Cancel BOTH — defense in depth against tier-swap races.
    browser.cancel();
    kokoro.cancel();
  }, [browser, kokoro]);

  // Map the underlying tier-specific state to the unified shape.
  let state: "idle" | "loading" | "speaking";
  let loaded: boolean;
  let loadProgress: number;
  if (parsed.tier === "kokoro") {
    state = kokoro.state;
    loaded = kokoro.loaded;
    loadProgress = kokoro.loadProgress;
  } else {
    state = browser.state === "speaking" ? "speaking" : "idle";
    loaded = browser.supported; // browser TTS is always "loaded" if supported
    loadProgress = 1;
  }

  return {
    speak,
    cancel,
    state,
    loadProgress,
    loaded,
    tier: parsed.tier,
    browserVoices: browser.voices,
    webgpuSupported: kokoro.supported,
  };
}
