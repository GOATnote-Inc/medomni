"use client";

import type { SpeechState } from "@/hooks/useSpeechSynthesis";

interface VoiceOutToggleProps {
  enabled: boolean;
  onChange: (next: boolean) => void;
  state: SpeechState;
  supported: boolean;
  disabled?: boolean;
}

// Three visual states:
// - voice off (enabled=false)
// - voice on, idle (enabled=true, state==="idle")
// - speaking (enabled=true, state==="speaking") — animated dot
//
// Style matches the header tool-pill row on /agent: rounded-full border,
// 11px text, hover-darken. Click flips localStorage['medomni:voiceOut'].
export function VoiceOutToggle({
  enabled,
  onChange,
  state,
  supported,
  disabled,
}: VoiceOutToggleProps) {
  if (!supported) {
    // The Speech Synthesis API is missing (very old browsers). Render a
    // disabled hint so the absence of the toggle doesn't look like a bug.
    return (
      <span
        className="px-2.5 py-1 rounded-full border border-slate-200 bg-slate-50 text-slate-400 text-[11px] cursor-not-allowed"
        title="Browser does not support speech synthesis"
      >
        <SpeakerGlyph muted /> voice n/a
      </span>
    );
  }

  const speaking = enabled && state === "speaking";
  const label = !enabled ? "voice off" : speaking ? "speaking" : "voice on";

  // Visual treatments echo the existing pill colorways: neutral when off,
  // emerald when ready, sky-pulse when actively speaking.
  const cls = !enabled
    ? "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
    : speaking
      ? "border-sky-300 bg-sky-50 text-sky-800"
      : "border-emerald-300 bg-emerald-50 text-emerald-800 hover:bg-emerald-100";

  return (
    <button
      type="button"
      onClick={() => onChange(!enabled)}
      disabled={disabled}
      aria-pressed={enabled}
      aria-label={`Voice output ${enabled ? "enabled" : "disabled"}`}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${cls}`}
    >
      <SpeakerGlyph muted={!enabled} />
      <span>{label}</span>
      {speaking && (
        <span
          className="inline-block w-1.5 h-1.5 rounded-full bg-sky-500 animate-pulse"
          aria-hidden
        />
      )}
    </button>
  );
}

function SpeakerGlyph({ muted }: { muted: boolean }) {
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
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      {muted ? (
        <>
          <line x1="23" y1="9" x2="17" y2="15" />
          <line x1="17" y1="9" x2="23" y2="15" />
        </>
      ) : (
        <>
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        </>
      )}
    </svg>
  );
}
