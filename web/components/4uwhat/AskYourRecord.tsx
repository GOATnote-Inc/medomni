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

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { BASE_PATH } from "@/lib/basePath";
import { Eyebrow } from "./Eyebrow";
import { Mono } from "./Mono";
import { PrismMark } from "./PrismMark";
import { usePatientId } from "@/hooks/usePatientId";
import { usePersona } from "@/hooks/usePersona";

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

interface AskYourRecordProps {
  /** Suggestion chips rendered above the input until first turn. */
  suggestions?: string[];
  className?: string;
  style?: CSSProperties;
}

export function AskYourRecord({
  suggestions,
  className,
  style,
}: AskYourRecordProps) {
  const [patientId] = usePatientId();
  const [persona] = usePersona();
  const [input, setInput] = useState("");

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

  function submit() {
    if (busy) return;
    const trimmed = input.trim();
    if (!trimmed) return;
    void sendMessage({ text: trimmed });
    setInput("");
  }

  return (
    <section
      className={className}
      style={style}
      aria-label="Ask your record"
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <Eyebrow>ASK YOUR RECORD</Eyebrow>
        <Mono size={9}>
          {patientId ? `PT · ${patientId.slice(0, 16)}` : "NO PATIENT SELECTED"}
          {persona ? ` · ${persona.toUpperCase()}` : ""}
        </Mono>
      </div>

      {/* Streamed turn list (lightweight). Renders the latest assistant
          message + reasoning + tool calls; older turns collapse. */}
      {messages.length > 0 && (
        <div
          style={{
            ...cardSurface,
            padding: 14,
            marginBottom: 12,
            maxHeight: 220,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {messages.map((m) => (
            <div key={m.id} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
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
                      }}
                    >
                      {part.text}
                    </div>
                  );
                }
                if (part.type === "reasoning") {
                  return (
                    <details
                      key={key}
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: 11,
                        color: "rgba(255,255,255,0.55)",
                      }}
                      open={part.state === "streaming"}
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
                        }}
                      >
                        {part.text}
                      </div>
                    </details>
                  );
                }
                if (typeof part.type === "string" && part.type.startsWith("tool-")) {
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
                        gap: 8,
                        alignSelf: "flex-start",
                      }}
                    >
                      <span style={{ color: "var(--accent)", fontWeight: 700 }}>tool</span>
                      <span>{toolName}</span>
                      <span style={{ color: "rgba(255,255,255,0.35)" }}>·</span>
                      <span>{state.replace(/-/g, " ")}</span>
                      {q ? (
                        <>
                          <span style={{ color: "rgba(255,255,255,0.35)" }}>·</span>
                          <span style={{ color: "rgba(255,255,255,0.55)" }}>{String(q)}</span>
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

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <div style={inputShellFocus}>
          <PrismMark size={16} color="var(--accent)" />
          <input
            type="text"
            placeholder="ask anything about this record_"
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.currentTarget.value)}
            style={inputStyle}
            aria-label="Ask a question about this record"
          />
          {busy ? (
            <button
              type="button"
              onClick={() => stop()}
              style={{ ...submitStyle, background: "transparent", color: "var(--accent)" }}
            >
              stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={input.trim().length === 0}
              style={{
                ...submitStyle,
                opacity: input.trim().length === 0 ? 0.5 : 1,
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
            marginTop: 14,
            display: "flex",
            flexDirection: "column",
            gap: 6,
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
