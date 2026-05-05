"use client";

// 4UWHAt — ReceiptCard
// Per-turn audit card for the /receipts page. Surfaces the same data the
// in-line verification badge under each assistant turn renders, but in
// long-form: full prompt, full response, every tool call with args,
// verification numbers, latency. Click anywhere on the header to expand
// or collapse the body.

import { useMemo, useState, type CSSProperties } from "react";
import { Eyebrow } from "./Eyebrow";
import { Mono } from "./Mono";
import type { Receipt } from "@/lib/4uwhat/receipts";

interface ReceiptCardProps {
  receipt: Receipt;
  /** Whether body sections start expanded. Defaults to false. */
  initiallyExpanded?: boolean;
}

// Match the dark-canvas card surface used elsewhere in 4UWHAt.
const cardSurface: CSSProperties = {
  background: "var(--p42-bg-card, #0e0e0e)",
  border: "1px solid rgba(255,255,255,0.07)",
};

const sectionStyle: CSSProperties = {
  borderTop: "1px solid rgba(255,255,255,0.06)",
  padding: "14px 18px",
  display: "flex",
  flexDirection: "column",
  gap: 8,
};

const monoBlock: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  lineHeight: 1.5,
  color: "rgba(255,255,255,0.85)",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  overflowWrap: "anywhere",
  margin: 0,
};

const collapseBtn: CSSProperties = {
  background: "transparent",
  color: "rgba(255,255,255,0.55)",
  border: "1px solid rgba(255,255,255,0.18)",
  padding: "3px 8px",
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.06em",
  cursor: "pointer",
  marginTop: 4,
  alignSelf: "flex-start",
};

function formatRelative(ms: number, nowMs: number): string {
  const delta = nowMs - ms;
  if (delta < 0) return "just now";
  const sec = Math.floor(delta / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}d ago`;
  return new Date(ms).toLocaleDateString();
}

function formatLocalTime(ms: number): string {
  try {
    return new Date(ms).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return String(ms);
  }
}

function formatLocalDate(ms: number): string {
  try {
    return new Date(ms).toLocaleDateString([], {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return "";
  }
}

interface ExpandableTextProps {
  text: string;
  /** Approximate line clamp before "expand" affordance kicks in. */
  maxLines?: number;
}

function ExpandableText({ text, maxLines = 6 }: ExpandableTextProps) {
  const [open, setOpen] = useState(false);
  const lineCount = useMemo(() => text.split(/\r?\n/).length, [text]);
  const overflow = lineCount > maxLines || text.length > maxLines * 120;
  const display =
    open || !overflow
      ? text
      : text.split(/\r?\n/).slice(0, maxLines).join("\n");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <pre style={monoBlock}>{display || "(empty)"}</pre>
      {overflow && (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          style={collapseBtn}
          aria-expanded={open}
        >
          {open ? "collapse" : "expand"}
        </button>
      )}
    </div>
  );
}

export function ReceiptCard({
  receipt,
  initiallyExpanded = false,
}: ReceiptCardProps) {
  const [expanded, setExpanded] = useState(initiallyExpanded);
  // Compute "now" once on first render so relative timestamps don't
  // flicker every keystroke. The /receipts page mounts the list fresh
  // on each navigation, which is good enough for an audit log.
  const nowMs = useMemo(() => Date.now(), []);

  const r = receipt;
  const v = r.verification;

  return (
    <article
      style={{
        ...cardSurface,
        display: "flex",
        flexDirection: "column",
      }}
      aria-label={`Receipt ${r.id}`}
    >
      {/* Header — always visible, click to expand/collapse */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        style={{
          all: "unset",
          cursor: "pointer",
          display: "flex",
          flexDirection: "column",
          gap: 8,
          padding: "14px 18px",
          // Subtle hover affordance, mirroring DetailDrawer / Clickable.
          // No background change here — header stays readable on dark.
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: 10,
              flexWrap: "wrap",
              minWidth: 0,
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 16,
                fontWeight: 600,
                color: "rgba(255,255,255,0.95)",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {formatLocalTime(r.timestamp)}
            </span>
            <Mono size={10} color="rgba(255,255,255,0.5)">
              {formatLocalDate(r.timestamp)} · {formatRelative(r.timestamp, nowMs)}
            </Mono>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              flexWrap: "wrap",
            }}
          >
            {r.patientId && (
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  letterSpacing: "0.08em",
                  padding: "2px 6px",
                  border: "1px solid rgba(255,0,150,0.25)",
                  background: "rgba(255,0,150,0.04)",
                  color: "rgba(255,255,255,0.75)",
                }}
              >
                PT · {r.patientId.slice(0, 16)}
              </span>
            )}
            {r.persona && (
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  letterSpacing: "0.08em",
                  padding: "2px 6px",
                  border: "1px solid rgba(255,255,255,0.12)",
                  color: "rgba(255,255,255,0.7)",
                }}
              >
                {r.persona.toUpperCase()}
              </span>
            )}
            <Mono size={10} color="rgba(255,255,255,0.5)">
              {v.model}
              {r.latencyMs != null ? ` · ${r.latencyMs} ms` : ""}
            </Mono>
            <span
              aria-hidden
              style={{
                color: "var(--accent)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                marginLeft: 4,
              }}
            >
              {expanded ? "▾" : "▸"}
            </span>
          </div>
        </div>

        {/* One-line preview when collapsed: prompt's first ~140 chars */}
        {!expanded && (
          <div
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 13,
              color: "rgba(255,255,255,0.7)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              maxWidth: "100%",
            }}
          >
            {r.prompt || "(empty prompt)"}
          </div>
        )}

        {/* Verification line — same numbers as the in-line badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
            marginTop: 2,
          }}
        >
          <Mono size={9} color="var(--accent)">
            VERIFIED · {v.toolsCalled} TOOLS USED · {v.checksPassed}/{v.checksTotal} CHECKS PASSED
          </Mono>
        </div>
      </button>

      {expanded && (
        <>
          <div style={sectionStyle}>
            <Eyebrow>PROMPT</Eyebrow>
            <ExpandableText text={r.prompt} maxLines={8} />
          </div>

          <div style={sectionStyle}>
            <Eyebrow>
              TOOL CALLS · {r.toolCalls.length}
            </Eyebrow>
            {r.toolCalls.length === 0 ? (
              <Mono size={11} color="rgba(255,255,255,0.5)">
                (no tools called on this turn)
              </Mono>
            ) : (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                {r.toolCalls.map((tc, i) => (
                  <ToolCallRow
                    key={`${tc.name}-${i}`}
                    name={tc.name}
                    args={tc.args}
                  />
                ))}
              </div>
            )}
          </div>

          <div style={sectionStyle}>
            <Eyebrow>RESPONSE</Eyebrow>
            <ExpandableText text={r.response} maxLines={6} />
          </div>

          <div style={sectionStyle}>
            <Eyebrow>VERIFICATION</Eyebrow>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(0, auto) minmax(0, 1fr)",
                gap: "4px 14px",
                fontFamily: "var(--font-mono)",
                fontSize: 11.5,
                lineHeight: 1.5,
                color: "rgba(255,255,255,0.85)",
              }}
            >
              <span style={{ color: "rgba(255,255,255,0.5)" }}>tools called</span>
              <span>{v.toolsCalled}</span>
              <span style={{ color: "rgba(255,255,255,0.5)" }}>checks</span>
              <span>
                {v.checksPassed}/{v.checksTotal} passed
              </span>
              <span style={{ color: "rgba(255,255,255,0.5)" }}>model</span>
              <span>{v.model}</span>
              {r.latencyMs != null && (
                <>
                  <span style={{ color: "rgba(255,255,255,0.5)" }}>latency</span>
                  <span>{r.latencyMs} ms</span>
                </>
              )}
              <span style={{ color: "rgba(255,255,255,0.5)" }}>id</span>
              <span style={{ overflowWrap: "anywhere" }}>{r.id}</span>
            </div>
          </div>
        </>
      )}
    </article>
  );
}

interface ToolCallRowProps {
  name: string;
  args: unknown;
}

function ToolCallRow({ name, args }: ToolCallRowProps) {
  const [open, setOpen] = useState(false);
  let argsJson: string;
  try {
    argsJson = JSON.stringify(args, null, 2);
  } catch {
    argsJson = String(args);
  }
  const empty =
    args == null ||
    (typeof args === "object" && Object.keys(args as object).length === 0);
  return (
    <div
      style={{
        border: "1px solid rgba(255,0,150,0.18)",
        background: "rgba(255,0,150,0.03)",
        padding: "8px 10px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            fontWeight: 700,
            color: "var(--accent)",
            letterSpacing: "0.04em",
          }}
        >
          {name}
        </span>
        {!empty && (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            style={collapseBtn}
            aria-expanded={open}
          >
            {open ? "hide args" : "show args"}
          </button>
        )}
        {empty && (
          <Mono size={10} color="rgba(255,255,255,0.5)">
            (no args)
          </Mono>
        )}
      </div>
      {open && !empty && (
        <pre
          style={{
            ...monoBlock,
            marginTop: 8,
            padding: 8,
            background: "rgba(0,0,0,0.4)",
            border: "1px solid rgba(255,255,255,0.06)",
            fontSize: 11.5,
          }}
        >
          {argsJson}
        </pre>
      )}
    </div>
  );
}
