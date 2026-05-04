"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Tier 1: browser-native `window.speechSynthesis`. No deps, no network.
// Tier 2 (future PR): when `process.env.NEXT_PUBLIC_MEDOMNI_TTS_URL` is set,
// this hook will WebSocket to that endpoint and stream PCM audio chunks
// back to a WebAudio sink — sovereign Riva Magpie / Parakeet-TTS path.
// Don't implement here yet; the toggle/queue contract is the same so the
// swap is a one-file change.

export type SpeechState = "idle" | "speaking" | "paused";

export interface UseSpeechSynthesisOptions {
  enabled: boolean;
  rate?: number;
  pitch?: number;
  voiceURI?: string;
}

export interface UseSpeechSynthesisReturn {
  speak: (text: string) => void;
  cancel: () => void;
  state: SpeechState;
  voices: SpeechSynthesisVoice[];
  supported: boolean;
}

// Quality scoring for browser-native voices. The browser exposes whatever
// the OS makes available; quality varies wildly. We score by markers in
// the voice name (Premium / Enhanced / Neural / Online — these are the
// modern neural voices Apple/Microsoft/Google ship) plus a few known-good
// names, then break ties on language. Higher score wins.
//
// On macOS Safari + Chrome: "Samantha (Premium)", "Daniel (Premium)" if
// the user has downloaded enhanced voices via System Settings → Spoken
// Content → System Voice → Customize. Without that download, Samantha
// reads as the legacy 1990s formant voice.
//
// On Edge + Chrome (recent): Microsoft "Aria Online (Natural)", "Jenny",
// "Guy", "Davis" — neural cloud voices that ship with the browser.
//
// On Chrome (no Edge): Google US English / UK English — neural.
//
// Older fallback: Karen, Alex (en-AU/en-US legacy voices).
//
// If everything is missing or low-quality, the Tier 1 hook
// (Kokoro.js WebGPU) is the proper upgrade path — see useKokoroTts.ts.

interface ScoredVoice {
  voice: SpeechSynthesisVoice;
  score: number;
}

const QUALITY_MARKERS: Array<{ pattern: RegExp; bonus: number }> = [
  { pattern: /\bPremium\b/i, bonus: 100 },
  { pattern: /\bEnhanced\b/i, bonus: 90 },
  { pattern: /\bNeural\b/i, bonus: 90 },
  { pattern: /\bNatural\b/i, bonus: 85 },
  { pattern: /\bOnline\b/i, bonus: 80 },
  { pattern: /\bPlus\b/i, bonus: 50 },
  // Known good defaults (no marker but high baseline quality on their platforms)
  { pattern: /^Google US English$/i, bonus: 70 },
  { pattern: /^Google UK English/i, bonus: 60 },
  { pattern: /^Microsoft Aria/i, bonus: 70 },
  { pattern: /^Microsoft Jenny/i, bonus: 65 },
  { pattern: /^Microsoft Guy/i, bonus: 60 },
  { pattern: /^Microsoft Davis/i, bonus: 60 },
  // Apple Siri voices are higher quality than the legacy named voices on macOS
  { pattern: /^Siri\b/i, bonus: 80 },
  { pattern: /^Samantha$/i, bonus: 25 }, // legacy without Premium marker
  { pattern: /^Daniel$/i, bonus: 20 },
  { pattern: /^Alex$/i, bonus: 20 },
  { pattern: /^Karen$/i, bonus: 18 },
];

function langBonus(lang: string): number {
  if (!lang) return 0;
  if (lang === "en-US") return 10;
  if (lang === "en-GB" || lang === "en-AU") return 7;
  if (lang.startsWith("en")) return 5;
  return 0;
}

export function scoreVoice(voice: SpeechSynthesisVoice): number {
  let score = langBonus(voice.lang);
  for (const { pattern, bonus } of QUALITY_MARKERS) {
    if (pattern.test(voice.name)) score += bonus;
  }
  // localService: voices that ship with the OS sound more reliable than
  // network voices for short utterances (no startup lag). Tiny tiebreaker.
  if (voice.localService) score += 1;
  return score;
}

/** Sorted descending by quality score. Useful for surfacing in a UI picker. */
export function rankVoices(
  voices: SpeechSynthesisVoice[],
): SpeechSynthesisVoice[] {
  const scored: ScoredVoice[] = voices
    .filter((v) => v.lang.startsWith("en")) // English only for the medical demo
    .map((voice) => ({ voice, score: scoreVoice(voice) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.map((s) => s.voice);
}

function pickVoice(
  voices: SpeechSynthesisVoice[],
  voiceURI?: string,
): SpeechSynthesisVoice | undefined {
  if (voices.length === 0) return undefined;
  if (voiceURI) {
    const match = voices.find((v) => v.voiceURI === voiceURI);
    if (match) return match;
  }
  const ranked = rankVoices(voices);
  if (ranked.length > 0) return ranked[0];
  // No English voices at all — last resort, return the first available.
  return voices[0];
}

export function useSpeechSynthesis(
  opts: UseSpeechSynthesisOptions,
): UseSpeechSynthesisReturn {
  // Default rate is 1.0 (was 1.05 — slight rush made already-robotic voices
  // sound more rushed). Pitch 1.0 is neutral.
  const { enabled, rate = 1.0, pitch = 1.0, voiceURI } = opts;

  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [state, setState] = useState<SpeechState>("idle");
  const [supported, setSupported] = useState(false);

  // Queue of pending utterance text. We drain serially so phrases don't
  // overlap. Each `speak` call enqueues; the drain loop pulls and feeds
  // `speechSynthesis.speak`.
  const queueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const enabledRef = useRef(enabled);

  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  // Detect support + load voices. Safari quirk: voices arrive async via
  // the `voiceschanged` event; the first `getVoices()` call may return [].
  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setSupported(false);
      return;
    }
    setSupported(true);
    const synth = window.speechSynthesis;

    const load = () => {
      const v = synth.getVoices();
      if (v.length > 0) setVoices(v);
    };
    load();
    synth.addEventListener("voiceschanged", load);
    return () => {
      synth.removeEventListener("voiceschanged", load);
    };
  }, []);

  const drain = useCallback(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
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
    const synth = window.speechSynthesis;
    const utter = new SpeechSynthesisUtterance(next);
    utter.rate = rate;
    utter.pitch = pitch;
    const picked = pickVoice(synth.getVoices(), voiceURI);
    if (picked) {
      utter.voice = picked;
      utter.lang = picked.lang;
    } else {
      utter.lang = "en-US";
    }
    speakingRef.current = true;
    setState("speaking");
    utter.onend = () => {
      speakingRef.current = false;
      // Drain the rest of the queue.
      drain();
    };
    utter.onerror = () => {
      speakingRef.current = false;
      drain();
    };
    synth.speak(utter);
  }, [rate, pitch, voiceURI]);

  const speak = useCallback(
    (text: string) => {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
      if (!enabledRef.current) return;
      const trimmed = text.trim();
      if (!trimmed) return;
      queueRef.current.push(trimmed);
      drain();
    },
    [drain],
  );

  const cancel = useCallback(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    queueRef.current = [];
    speakingRef.current = false;
    window.speechSynthesis.cancel();
    setState("idle");
  }, []);

  // If the consumer flips `enabled` off mid-stream, stop talking immediately.
  useEffect(() => {
    if (!enabled) {
      cancel();
    }
  }, [enabled, cancel]);

  // Stop any in-flight speech when the component using this hook unmounts.
  useEffect(() => {
    return () => {
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  return { speak, cancel, state, voices, supported };
}
