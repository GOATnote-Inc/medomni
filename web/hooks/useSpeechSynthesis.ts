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

// Names ranked by clinical credibility (clear, neutral en-US neural voices).
// First match wins.
const PREFERRED_VOICE_NAMES = [
  "Google US English",
  "Microsoft Aria Online (Natural) - English (United States)",
  "Microsoft Aria",
  "Microsoft Jenny",
  "Samantha",
  "Karen",
  "Alex",
];

function pickVoice(
  voices: SpeechSynthesisVoice[],
  voiceURI?: string,
): SpeechSynthesisVoice | undefined {
  if (voices.length === 0) return undefined;
  if (voiceURI) {
    const match = voices.find((v) => v.voiceURI === voiceURI);
    if (match) return match;
  }
  for (const name of PREFERRED_VOICE_NAMES) {
    const match = voices.find((v) => v.name === name || v.name.startsWith(name));
    if (match) return match;
  }
  // Prefer any en-US voice over a non-English default.
  const enUS = voices.find((v) => v.lang === "en-US");
  if (enUS) return enUS;
  return voices.find((v) => v.lang.startsWith("en"));
}

export function useSpeechSynthesis(
  opts: UseSpeechSynthesisOptions,
): UseSpeechSynthesisReturn {
  const { enabled, rate = 1.05, pitch = 1.0, voiceURI } = opts;

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
