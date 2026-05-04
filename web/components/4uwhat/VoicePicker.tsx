"use client";

import { useEffect, useMemo, useState } from "react";

import { KOKORO_DEFAULT_VOICE_ID, KOKORO_VOICE_CATALOG } from "@/hooks/useKokoroTts";

// Voice picker that combines Tier 0 (browser-native) and Tier 1
// (Kokoro WebGPU) voices into a single dropdown. Persists the user's
// pick under `medomni:tts:voice` so the choice rides across both the
// /agent surface and any future Records OS Ask bar.
//
// Selection token format mirrors `useTts.parseSelectedVoice`:
//   "browser:<voiceURI>"
//   "kokoro:<voiceId>"

export const TTS_VOICE_STORAGE_KEY = "medomni:tts:voice";

export interface VoicePickerProps {
  selectedVoice: string;
  onChange: (next: string) => void;
  /** Tier 0 voices from `useSpeechSynthesis().voices`. */
  browserVoices: SpeechSynthesisVoice[];
  /** Whether the Kokoro model has finished loading in this tab. */
  kokoroLoaded: boolean;
  /** Active Kokoro load progress (0..1). */
  kokoroLoadProgress: number;
  /** Tier-currently-in-use. */
  activeTier: "browser" | "kokoro";
  /** True if the runtime supports WebGPU (best Kokoro perf). */
  webgpuSupported: boolean;
  disabled?: boolean;
}

export function VoicePicker({
  selectedVoice,
  onChange,
  browserVoices,
  kokoroLoaded,
  kokoroLoadProgress,
  activeTier,
  webgpuSupported,
  disabled,
}: VoicePickerProps) {
  const [open, setOpen] = useState(false);

  // Sort browser voices: en-US first, then everything else, alphabetical.
  const sortedBrowser = useMemo(() => {
    const arr = [...browserVoices];
    arr.sort((a, b) => {
      const aPref = a.lang.startsWith("en-US") ? 0 : a.lang.startsWith("en") ? 1 : 2;
      const bPref = b.lang.startsWith("en-US") ? 0 : b.lang.startsWith("en") ? 1 : 2;
      if (aPref !== bPref) return aPref - bPref;
      return a.name.localeCompare(b.name);
    });
    return arr;
  }, [browserVoices]);

  // Sort Kokoro voices by overallGrade (best first), then by name.
  const sortedKokoro = useMemo(() => {
    const arr = [...KOKORO_VOICE_CATALOG];
    arr.sort((a, b) => a.overallGrade.localeCompare(b.overallGrade) || a.name.localeCompare(b.name));
    return arr;
  }, []);

  // Close the menu on outside click. Cheap implementation — single
  // document listener, scoped via a ref-less id check.
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      if (target.closest('[data-voice-picker-root="true"]')) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const labelForBrowser = (v: SpeechSynthesisVoice) => {
    const trimmed = v.name.replace(/\s*\([^)]*\)\s*$/, "");
    return `${trimmed} · ${v.lang}`;
  };

  const labelForKokoro = (v: (typeof sortedKokoro)[number]) => {
    const lang = v.language === "en-us" ? "US" : v.language === "en-gb" ? "UK" : v.language;
    return `${v.name} · ${v.gender} · ${lang} · grade ${v.overallGrade}`;
  };

  const currentLabel = (() => {
    if (selectedVoice.startsWith("kokoro:")) {
      const id = selectedVoice.slice("kokoro:".length);
      const v = sortedKokoro.find((k) => k.id === id);
      return v ? `AI · ${v.name}` : "AI · custom";
    }
    if (selectedVoice.startsWith("browser:")) {
      const uri = selectedVoice.slice("browser:".length);
      const v = sortedBrowser.find((b) => b.voiceURI === uri);
      return v ? v.name.replace(/\s*\([^)]*\)\s*$/, "") : "Browser · default";
    }
    return "auto";
  })();

  // Status hint shown next to the picker. Three cases for the AI tier:
  //   - active tier === kokoro && !loaded && progress > 0 → downloading
  //   - active tier === kokoro && loaded → ready
  //   - else → blank
  const status = (() => {
    if (activeTier !== "kokoro") return null;
    if (!kokoroLoaded) {
      const pct = Math.round(kokoroLoadProgress * 100);
      return pct > 0 ? `downloading model… ${pct}%` : "downloading 160MB model…";
    }
    return "ready · on-device";
  })();

  return (
    <div className="relative inline-flex items-center gap-2" data-voice-picker-root="true">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-slate-300 bg-white text-slate-700 text-[11px] font-medium hover:bg-slate-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Choose voice"
      >
        <VoiceGlyph />
        <span>{currentLabel}</span>
        <span aria-hidden className="text-slate-400">▾</span>
      </button>
      {status && (
        <span
          className={`text-[10px] italic ${
            kokoroLoaded ? "text-emerald-700" : "text-amber-700"
          }`}
        >
          {status}
        </span>
      )}
      {open && (
        <div
          role="listbox"
          className="absolute top-full left-0 mt-1 z-20 w-72 max-h-80 overflow-auto rounded-md border border-slate-200 bg-white shadow-lg text-[12px]"
        >
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-slate-500 bg-slate-50 border-b border-slate-200">
            Browser voices · Tier 0
          </div>
          {sortedBrowser.length === 0 && (
            <div className="px-3 py-2 text-slate-400 italic">No browser voices available</div>
          )}
          {sortedBrowser.map((v) => {
            const token = `browser:${v.voiceURI}`;
            const active = selectedVoice === token;
            return (
              <button
                key={`b-${v.voiceURI}`}
                type="button"
                role="option"
                aria-selected={active}
                onClick={() => {
                  onChange(token);
                  setOpen(false);
                }}
                className={`w-full text-left px-3 py-1.5 hover:bg-slate-50 ${
                  active ? "bg-emerald-50 text-emerald-900 font-medium" : "text-slate-700"
                }`}
              >
                {labelForBrowser(v)}
              </button>
            );
          })}
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-slate-500 bg-slate-50 border-y border-slate-200 flex items-center justify-between">
            <span>AI voices · Kokoro {webgpuSupported ? "WebGPU" : "WASM"}</span>
            {!webgpuSupported && (
              <span className="text-[9px] text-amber-700 normal-case" title="WebGPU not detected; Kokoro will fall back to WASM (slower)">
                WASM
              </span>
            )}
          </div>
          {sortedKokoro.map((v) => {
            const token = `kokoro:${v.id}`;
            const active = selectedVoice === token;
            const isDefault = v.id === KOKORO_DEFAULT_VOICE_ID;
            return (
              <button
                key={`k-${v.id}`}
                type="button"
                role="option"
                aria-selected={active}
                onClick={() => {
                  onChange(token);
                  setOpen(false);
                }}
                className={`w-full text-left px-3 py-1.5 hover:bg-slate-50 ${
                  active ? "bg-emerald-50 text-emerald-900 font-medium" : "text-slate-700"
                }`}
              >
                {labelForKokoro(v)}
                {isDefault && <span className="ml-1 text-emerald-700 text-[10px]">· default</span>}
              </button>
            );
          })}
          <div className="px-3 py-2 text-[10px] text-slate-500 bg-slate-50 border-t border-slate-200">
            AI voices download a 160MB model on first use (cached after).
          </div>
        </div>
      )}
    </div>
  );
}

function VoiceGlyph() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}
