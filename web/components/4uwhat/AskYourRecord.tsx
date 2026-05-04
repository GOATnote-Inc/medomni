"use client";

// 4UWHAt — AskYourRecord
// Embedded command bar for the Records OS dashboard. Reuses the same
// `useChat` + `DefaultChatTransport` pattern as `web/app/agent/page.tsx`
// — that's the working contract — but renders streaming reasoning, tool
// calls, and the final answer in a tight dashboard-footer layout instead
// of the full-page agent surface.
//
// On every turn the request body carries:
//   - patientId (from `usePatientId()`) — drives `get_patient_context`
//   - persona  (from `usePersona()`)    — informs the system prompt
//
// BASE_PATH is required: useChat's DefaultChatTransport calls raw fetch()
// which does not auto-prefix the Next.js basePath. See lib/basePath.ts.
//
// Voice I/O (B3, regression fix from /agent):
//   - mic button next to the input → AudioRecorder (mirrors /agent path)
//   - VoiceOutToggle in the panel header → useSpeechSynthesis reads the
//     streaming assistant deltas sentence-by-sentence
//   - localStorage key `medomni:voiceOut` matches /agent so the toggle
//     state persists across the two surfaces.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { BASE_PATH } from "@/lib/basePath";
import { Eyebrow } from "./Eyebrow";
import { Mono } from "./Mono";
import { PrismMark } from "./PrismMark";
import { AudioRecorder } from "@/components/AudioRecorder";
import { VoiceOutToggle } from "@/components/VoiceOutToggle";
import { useTts } from "@/hooks/useTts";
import { VoicePicker } from "@/components/4uwhat/VoicePicker";

const VOICE_SELECT_STORAGE_KEY = "medomni:tts:voice";
import { usePatientId } from "@/hooks/usePatientId";
import { usePersona } from "@/hooks/usePersona";

const VOICE_OUT_STORAGE_KEY = "medomni:voiceOut";

const cardSurface: CSSProperties = {
  background: "var(--p42-ink, #0a0a0a)",
  border: "1px solid rgba(255,255,255,0.07)",
};

const inputShellFocus: CSSProperties = {
  background: "var(--p42-ink, #0a0a0a)",
  border: "2px solid var(--accent)",
  padding: "12px 14px",
  display: "flex",
  alignItems: "center",
  gap: 10,
  boxShadow: "var(--glow-pink)",
};

const inputStyle: CSSProperties = {
  flex: 1,
  background: "transparent",
  border: "none",
  outline: "none",
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  color: "rgba(255,255,255,0.95)",
};

const submitStyle: CSSProperties = {
  background: "var(--accent)",
  color: "#fff",
  border: "1px solid var(--accent)",
  padding: "6px 12px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 11,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
  cursor: "pointer",
};

interface FilePart {
  type: "file";
  mediaType: string;
  url: string;
  filename?: string;
}

interface AskYourRecordProps {
  /** Suggestion chips rendered above the input until first turn. */
  suggestions?: string[];
  className?: string;
  style?: CSSProperties;
}

// Strip markdown fences/asterisks/etc. for TTS. Mirrors /agent's helper
// (kept inline rather than imported so the /agent route stays the
// canonical owner — this is the Records-OS version of the same idea).
function stripMarkdownForSpeech(s: string): string {
  return s
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

// Flush at sentence boundaries so utterances sound natural.
function splitForSpeech(buffer: string): { chunks: string[]; rest: string } {
  const re = /[^.!?\n]+[.!?\n]+["')\]]?\s*/g;
  const chunks: string[] = [];
  let lastEnd = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(buffer)) !== null) {
    chunks.push(match[0]);
    lastEnd = re.lastIndex;
  }
  return { chunks, rest: buffer.slice(lastEnd) };
}

export function AskYourRecord({
  suggestions,
  className,
  style,
}: AskYourRecordProps) {
  const [patientId] = usePatientId();
  const [persona] = usePersona();
  const [input, setInput] = useState("");
  const [pendingAudio, setPendingAudio] = useState<{
    url: string;
    durationMs: number;
  } | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);

  // Voice-out toggle state. Default off; persists via localStorage to
  // match /agent so flipping the toggle on either surface carries to
  // the other on the next mount.
  const [voiceOutEnabled, setVoiceOutEnabled] = useState(false);
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(VOICE_OUT_STORAGE_KEY);
      if (raw === "true") setVoiceOutEnabled(true);
    } catch {
      // localStorage can throw in strict privacy modes; fail silent.
    }
  }, []);
  const handleVoiceOutChange = (next: boolean) => {
    setVoiceOutEnabled(next);
    try {
      window.localStorage.setItem(VOICE_OUT_STORAGE_KEY, String(next));
    } catch {
      // ignore
    }
  };

  // Persist + read voice selection across surfaces (the /agent route uses
  // the same key so toggling on either page carries over).
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(VOICE_SELECT_STORAGE_KEY);
      if (raw) setSelectedVoice(raw);
    } catch {
      // ignore
    }
  }, []);
  const handleVoiceChange = useCallback((v: string) => {
    setSelectedVoice(v);
    try {
      window.localStorage.setItem(VOICE_SELECT_STORAGE_KEY, v);
    } catch {
      // ignore
    }
  }, []);

  const {
    speak: ttsSpeak,
    cancel: ttsCancel,
    state: ttsState,
    loaded: ttsLoaded,
    loadProgress: ttsLoadProgress,
    tier: ttsTier,
    browserVoices,
    webgpuSupported,
  } = useTts({ enabled: voiceOutEnabled, selectedVoice });
  const ttsSupported = true; // useTts always supported (browser fallback)

  // Spoken-offset bookkeeping per assistant text-part. Same shape as
  // /agent — keys are `${messageId}#${partIdx}`.
  const spokenLenRef = useRef<Map<string, number>>(new Map());
  const speechBufferRef = useRef<Map<string, string>>(new Map());

  // Refs hold the live patientId/persona so the transport's body callback
  // (created once via useMemo) reads the latest values at request time
  // rather than capturing them on first render.
  const patientIdRef = useRef(patientId);
  const personaRef = useRef(persona);
  useEffect(() => {
    patientIdRef.current = patientId;
  }, [patientId]);
  useEffect(() => {
    personaRef.current = persona;
  }, [persona]);

  // One transport per component lifetime. Recreating it on every render
  // would reset the chat ID and corrupt streaming state.
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: `${BASE_PATH}/api/agent`,
        // The agent route reads `body.patientId` for get_patient_context
        // and `body.persona` (currently advisory) for the system prompt.
        body: () => ({
          patientId: patientIdRef.current,
          persona: personaRef.current,
        }),
      }),
    [],
  );

  const { messages, sendMessage, status, stop } = useChat({ transport });

  const busy = status === "submitted" || status === "streaming";
  const hasAudio = pendingAudio !== null;

  // Walk every assistant text-part each render, queue only the NEW
  // characters past the last spoken offset, flush at sentence boundaries.
  // O(deltas) per render — same shape as /agent's effect.
  useEffect(() => {
    if (!voiceOutEnabled) return;
    for (const m of messages) {
      if (m.role !== "assistant") continue;
      m.parts.forEach((part, i) => {
        if (part.type !== "text") return;
        const key = `${m.id}#${i}`;
        const fullText = (part as { text?: string }).text ?? "";
        const prev = spokenLenRef.current.get(key) ?? 0;
        if (fullText.length <= prev) return;
        const delta = fullText.slice(prev);
        spokenLenRef.current.set(key, fullText.length);
        const cleaned = stripMarkdownForSpeech(delta);
        if (!cleaned) return;
        const buf =
          (speechBufferRef.current.get(key) ?? "") + cleaned + " ";
        const { chunks, rest } = splitForSpeech(buf);
        speechBufferRef.current.set(key, rest);
        for (const c of chunks) {
          const trimmed = c.trim();
          if (trimmed) ttsSpeak(trimmed);
        }
      });
    }
  }, [messages, voiceOutEnabled, ttsSpeak]);

  // Flush leftover buffer when streaming ends (last sentence may not
  // end with terminal punctuation).
  useEffect(() => {
    if (busy) return;
    if (!voiceOutEnabled) return;
    for (const [key, rest] of speechBufferRef.current.entries()) {
      const trimmed = rest.trim();
      if (trimmed) ttsSpeak(trimmed);
      speechBufferRef.current.set(key, "");
    }
  }, [busy, voiceOutEnabled, ttsSpeak]);

  // While the user is recording, silence any in-flight TTS so the mic
  // doesn't pick up the speaker output.
  useEffect(() => {
    if (hasAudio) ttsCancel();
  }, [hasAudio, ttsCancel]);

  function handleAudio(url: string, durationMs: number) {
    setPendingAudio({ url, durationMs });
    setAudioError(null);
  }

  function clearAudio() {
    setPendingAudio(null);
  }

  function submit() {
    if (busy) return;
    // Any new turn cancels in-flight TTS — the user has the floor again.
    ttsCancel();
    const trimmed = input.trim();

    // Audio attached → send audio + optional follow-up text. Mirrors
    // /agent's voice-mode submit path 1:1.
    if (pendingAudio) {
      const followup =
        trimmed ||
        "Transcribe the audio and answer the clinical question about this record. Cite guidelines and PMIDs where they meaningfully change the answer.";
      void sendMessage({
        text: followup,
        files: [
          {
            type: "file",
            mediaType: "audio/wav",
            url: pendingAudio.url,
            filename: `recording-${pendingAudio.durationMs}ms.wav`,
          } as FilePart,
        ],
      });
      setInput("");
      setPendingAudio(null);
      return;
    }

    // Text-only path.
    if (!trimmed) return;
    void sendMessage({ text: trimmed });
    setInput("");
  }

  return (
    <section
      className={className}
      style={{
        // Flex column so the message list takes available space and the input
        // bar pins to the bottom. `min-height: 0` is the well-known flex
        // gotcha that lets the messages region actually shrink and scroll
        // instead of pushing the input off-screen. The parent container is
        // expected to constrain height (the right rail aside in RecordsOS).
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        // Hard cap so even a stray un-bounded parent can't make us grow into
        // siblings. The rail's other sections (care team, recent activity)
        // get their own scroll regions; this just keeps the chat sane.
        maxHeight: "min(70vh, 640px)",
        gap: 12,
        ...style,
      }}
      aria-label="Ask your record"
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          flexShrink: 0,
          minWidth: 0,
          flexWrap: "wrap",
        }}
      >
        <Eyebrow>ASK YOUR RECORD</Eyebrow>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            minWidth: 0,
          }}
        >
          <VoiceOutToggle
            enabled={voiceOutEnabled}
            onChange={handleVoiceOutChange}
            state={ttsState === "loading" ? "speaking" : ttsState}
            supported={ttsSupported}
          />
          <VoicePicker
            selectedVoice={selectedVoice}
            onChange={handleVoiceChange}
            browserVoices={browserVoices}
            kokoroLoaded={ttsLoaded}
            kokoroLoadProgress={ttsLoadProgress}
            activeTier={ttsTier}
            webgpuSupported={webgpuSupported}
          />
          <Mono size={9}>
            {patientId ? `PT · ${patientId.slice(0, 16)}` : "NO PATIENT SELECTED"}
            {persona ? ` · ${persona.toUpperCase()}` : ""}
          </Mono>
        </div>
      </div>

      {/* Streamed turn list. Grows to fill available vertical space and
          scrolls internally so the input bar below stays pinned. The user
          previously saw reasoning rendering BELOW the input bar — caused by
          a fixed `maxHeight: 220` parent + un-pinned input that let long
          reasoning blocks visually appear under the form on certain
          viewports. Flex-column with min-height:0 + overflow-y:auto makes
          the scroll authoritative. */}
      {messages.length > 0 && (
        <div
          style={{
            ...cardSurface,
            padding: 14,
            flex: 1,
            minHeight: 0,
            overflowY: "auto",
            overflowX: "hidden",
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {messages.map((m) => (
            <div
              key={m.id}
              style={{ display: "flex", flexDirection: "column", gap: 6 }}
            >
              <Mono size={9} color="rgba(255,255,255,0.45)">
                {m.role === "user" ? "YOU" : "MEDOMNI"}
              </Mono>
              {m.parts.map((part, i) => {
                const key = `${m.id}-${i}`;
                if (part.type === "text") {
                  return (
                    <div
                      key={key}
                      style={{
                        fontFamily: "var(--font-display)",
                        fontSize: 13,
                        lineHeight: 1.45,
                        color: "rgba(255,255,255,0.92)",
                        whiteSpace: "pre-wrap",
                        // Long URLs / tokens / drug-name strings shouldn't
                        // force horizontal scroll inside the chat panel.
                        overflowWrap: "anywhere",
                        wordBreak: "break-word",
                      }}
                    >
                      {(part as { text?: string }).text}
                    </div>
                  );
                }
                if (part.type === "reasoning") {
                  const reasoningPart = part as {
                    text?: string;
                    state?: string;
                  };
                  return (
                    <details
                      key={key}
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: 11,
                        color: "rgba(255,255,255,0.55)",
                      }}
                      open={reasoningPart.state === "streaming"}
                    >
                      <summary
                        style={{
                          cursor: "pointer",
                          color: "rgba(255,255,255,0.4)",
                          textTransform: "uppercase",
                          letterSpacing: "0.12em",
                          fontSize: 9,
                          fontWeight: 700,
                        }}
                      >
                        reasoning
                      </summary>
                      <div
                        style={{
                          marginTop: 6,
                          padding: 8,
                          borderLeft: "2px solid rgba(255,255,255,0.12)",
                          fontStyle: "italic",
                          whiteSpace: "pre-wrap",
                          overflowWrap: "anywhere",
                          wordBreak: "break-word",
                        }}
                      >
                        {reasoningPart.text}
                      </div>
                    </details>
                  );
                }
                if (
                  part.type === "file" &&
                  (part as FilePart).mediaType?.startsWith("audio/")
                ) {
                  return (
                    <div
                      key={key}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "6px 10px",
                        border: "1px solid rgba(255,0,150,0.25)",
                        background: "rgba(255,0,150,0.04)",
                        fontFamily: "var(--font-mono)",
                        fontSize: 10.5,
                        color: "rgba(255,255,255,0.75)",
                        alignSelf: "flex-start",
                      }}
                    >
                      <span style={{ color: "var(--accent)", fontWeight: 700 }}>
                        VOICE
                      </span>
                      <audio
                        src={(part as FilePart).url}
                        controls
                        style={{ height: 24 }}
                      />
                    </div>
                  );
                }
                if (
                  typeof part.type === "string" &&
                  part.type.startsWith("tool-")
                ) {
                  // Tool cards are rendered as compact mono badges. The full
                  // inputs/outputs render in /agent for deep inspection; the
                  // dashboard footer just signals what the agent reached for.
                  const toolName = part.type.replace(/^tool-/, "");
                  const toolPart = part as {
                    type: string;
                    state?: string;
                    input?: { query?: string; score?: string };
                  };
                  const state = toolPart.state ?? "unknown";
                  const q = toolPart.input?.query ?? toolPart.input?.score;
                  return (
                    <div
                      key={key}
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: 10.5,
                        padding: "5px 8px",
                        border: "1px solid rgba(255,0,150,0.25)",
                        background: "rgba(255,0,150,0.04)",
                        color: "rgba(255,255,255,0.75)",
                        display: "inline-flex",
                        alignItems: "center",
                        flexWrap: "wrap",
                        gap: 8,
                        alignSelf: "flex-start",
                        maxWidth: "100%",
                        overflowWrap: "anywhere",
                        wordBreak: "break-word",
                      }}
                    >
                      <span style={{ color: "var(--accent)", fontWeight: 700 }}>
                        tool
                      </span>
                      <span>{toolName}</span>
                      <span style={{ color: "rgba(255,255,255,0.35)" }}>·</span>
                      <span>{state.replace(/-/g, " ")}</span>
                      {q ? (
                        <>
                          <span style={{ color: "rgba(255,255,255,0.35)" }}>
                            ·
                          </span>
                          <span style={{ color: "rgba(255,255,255,0.55)" }}>
                            {String(q)}
                          </span>
                        </>
                      ) : null}
                    </div>
                  );
                }
                return null;
              })}
            </div>
          ))}
        </div>
      )}

      {/* Audio-attached chip + recording-error banner sit just above the
          input shell so the user can clear or retry without losing focus. */}
      {audioError && (
        <div
          style={{
            flexShrink: 0,
            padding: "8px 10px",
            border: "1px solid rgba(244,63,94,0.45)",
            background: "rgba(244,63,94,0.08)",
            color: "rgba(254,205,211,0.95)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            lineHeight: 1.4,
          }}
          role="alert"
        >
          {audioError}
        </div>
      )}
      {pendingAudio && (
        <div
          style={{
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "8px 10px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.04)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "rgba(255,255,255,0.75)",
          }}
        >
          <span style={{ color: "var(--accent)", fontWeight: 700 }}>VOICE</span>
          <span>
            recorded · {(pendingAudio.durationMs / 1000).toFixed(1)}s · 16 kHz mono
          </span>
          <button
            type="button"
            onClick={clearAudio}
            style={{
              marginLeft: "auto",
              background: "transparent",
              color: "rgba(255,255,255,0.55)",
              border: "1px solid rgba(255,255,255,0.18)",
              padding: "3px 8px",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              cursor: "pointer",
            }}
          >
            clear
          </button>
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        style={{
          // Input bar is pinned to the bottom of the chat panel: it never
          // shrinks and never scrolls. Combined with the messages region's
          // `flex: 1; overflow-y: auto` above, this is the canonical
          // chat-panel layout (Slack/iMessage/etc).
          flexShrink: 0,
          minWidth: 0,
        }}
        data-testid="askyourrecord-form"
      >
        <div style={inputShellFocus}>
          <PrismMark size={16} color="var(--accent)" />
          <input
            type="text"
            placeholder={
              hasAudio
                ? "optional follow-up text_"
                : "ask anything about this record_"
            }
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.currentTarget.value)}
            style={inputStyle}
            aria-label="Ask a question about this record"
          />
          {/* Mic — tap to start, tap to stop. AudioRecorder owns the
              recording state itself, so we only feed it onAudio/onError. */}
          <AudioRecorder
            onAudio={handleAudio}
            onError={setAudioError}
            disabled={busy || hasAudio}
          />
          {busy ? (
            <button
              type="button"
              onClick={() => stop()}
              style={{
                ...submitStyle,
                background: "transparent",
                color: "var(--accent)",
              }}
            >
              stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!hasAudio && input.trim().length === 0}
              style={{
                ...submitStyle,
                opacity: !hasAudio && input.trim().length === 0 ? 0.5 : 1,
              }}
            >
              ask
            </button>
          )}
        </div>
      </form>

      {messages.length === 0 && suggestions && suggestions.length > 0 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 6,
            flexShrink: 0,
            minWidth: 0,
          }}
        >
          {suggestions.map((s, i) => (
            <button
              key={i}
              type="button"
              onClick={() => {
                setInput(s);
              }}
              style={{
                display: "block",
                width: "100%",
                textAlign: "left",
                padding: "8px 12px",
                background: "var(--p42-ink, #0a0a0a)",
                border: "1px solid rgba(255,255,255,0.06)",
                color: "rgba(255,255,255,0.85)",
                fontFamily: "var(--font-display)",
                fontSize: 11.5,
                fontWeight: 500,
                cursor: "pointer",
                lineHeight: 1.4,
                overflowWrap: "anywhere",
                wordBreak: "break-word",
              }}
            >
              <span style={{ color: "var(--accent)", marginRight: 8 }}>→</span>
              {s}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
