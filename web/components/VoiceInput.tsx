"use client";

import { useEffect, useRef, useState } from "react";

// Browser-only push-to-talk via Web Speech API. Transcript drops into
// the parent's textarea via onTranscript(text). No audio leaves the device.
//
// Caveats per current MDN/Chrome status:
// - Chrome / Safari (desktop & iOS): supported via webkitSpeechRecognition
// - Firefox: NOT supported (fallback message shown)
// - On iOS, requires HTTPS (we have it)

declare global {
  interface Window {
    webkitSpeechRecognition?: new () => SpeechRecognition;
    SpeechRecognition?: new () => SpeechRecognition;
  }
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  [index: number]: { transcript: string; confidence: number };
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
  onerror: ((this: SpeechRecognition, ev: Event) => void) | null;
  onend: ((this: SpeechRecognition, ev: Event) => void) | null;
}

interface VoiceInputProps {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

export function VoiceInput({ onTranscript, disabled }: VoiceInputProps) {
  const [recording, setRecording] = useState(false);
  const [supported, setSupported] = useState<boolean | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const transcriptRef = useRef<string>("");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    setSupported(!!Ctor);
  }, []);

  function start() {
    if (recording || disabled) return;
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Ctor) {
      setSupported(false);
      return;
    }
    const recog = new Ctor();
    recog.continuous = true;
    recog.interimResults = true;
    recog.lang = "en-US";
    transcriptRef.current = "";

    recog.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      let final = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          final += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }
      if (final) {
        transcriptRef.current += final;
      }
      onTranscript(transcriptRef.current + interim);
    };

    recog.onerror = (e: Event) => {
      console.warn("[voice] error", e);
      setRecording(false);
    };

    recog.onend = () => {
      setRecording(false);
    };

    try {
      recog.start();
      recognitionRef.current = recog;
      setRecording(true);
    } catch (e) {
      console.warn("[voice] start failed", e);
      setRecording(false);
    }
  }

  function stop() {
    recognitionRef.current?.stop();
    setRecording(false);
  }

  if (supported === false) {
    return (
      <button
        type="button"
        disabled
        title="Voice input unavailable in this browser. Try Chrome or Safari."
        aria-label="Voice unsupported"
        className="p-2 rounded-md text-slate-300 cursor-not-allowed"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
      </button>
    );
  }

  return (
    <button
      type="button"
      onMouseDown={start}
      onMouseUp={stop}
      onMouseLeave={() => recording && stop()}
      onTouchStart={(e) => {
        e.preventDefault();
        start();
      }}
      onTouchEnd={stop}
      disabled={disabled}
      title={recording ? "Recording — release to stop" : "Hold to record"}
      aria-label={recording ? "Stop recording" : "Start recording (push to talk)"}
      className={
        recording
          ? "p-2 rounded-md bg-rose-500 text-white animate-pulse"
          : "p-2 rounded-md text-slate-500 hover:bg-slate-200 hover:text-slate-800 transition-colors"
      }
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
    </button>
  );
}
