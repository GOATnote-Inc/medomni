"use client";

// 4UWHAt — TraceTimeline
//
// Phase 1 of the flagship-demo plan ("make the stack visible"): renders the
// measured inference trace that the `/api/agent` route streams as custom
// data parts under each assistant answer. Every row is a REAL stage that
// ran — a Nemotron-3-Nano-Omni generation step or a tool call — with its
// measured wall time. The header carries the turn summary: time-to-first-
// token, sustained tokens/sec, total time.
//
// Honesty (SPEC §6 Phase 1 / §8): this component renders only what the
// server measured. It invents nothing. If a turn produced no stages yet
// (stream still warming up) it renders nothing.
//
// Style: matches the dark Records-OS surface — `#0a0a0a` ink, hairline
// borders, the mono micro-type used across 4UWHAt. Kind-colored markers
// come from the shared `traceKindColor()` so a Phase-2 kind ('guardrail',
// 'retrieval') lights up automatically.

import { useState, type CSSProperties } from "react";
import {
  traceKindColor,
  type TraceStage,
  type TurnMetrics,
} from "@/lib/4uwhat/trace";

interface TraceTimelineProps {
  /** Measured stages for this answer, in stream order. */
  stages: TraceStage[];
  /** The turn summary, once the turn has completed. */
  metrics?: TurnMetrics;
  /** Whether the turn is still streaming — drives the live "measuring" hint. */
  streaming?: boolean;
}

// Format a millisecond duration compactly: sub-second in ms, else seconds
// to one decimal. Keeps the timeline scannable.
function fmtMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

const containerStyle: CSSProperties = {
  marginTop: 4,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.015)",
  fontFamily: "var(--font-mono)",
};

const summaryStyle: CSSProperties = {
  cursor: "pointer",
  listStyle: "none",
  userSelect: "none",
  display: "flex",
  alignItems: "center",
  gap: 8,
  flexWrap: "wrap",
  padding: "6px 8px",
  fontSize: 9.5,
  color: "rgba(255,255,255,0.7)",
};

// One metric chip in the summary row (TTFT, tok/s, total).
function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "baseline", gap: 4 }}>
      <span style={{ color: "rgba(255,255,255,0.4)" }}>{label}</span>
      <span style={{ color: "rgba(255,255,255,0.92)", fontWeight: 600 }}>
        {value}
      </span>
    </span>
  );
}

export function TraceTimeline({
  stages,
  metrics,
  streaming = false,
}: TraceTimelineProps) {
  // Default-open while streaming so a visitor watches the stack run live;
  // collapsible afterward to keep the answer list tight.
  const [open, setOpen] = useState(true);

  // Nothing measured yet — render nothing rather than an empty shell.
  if (stages.length === 0 && !metrics) return null;

  // Scale bar widths to the longest stage so relative cost is visible at a
  // glance. Guard the all-zero case (sub-ms stages) to avoid divide-by-zero.
  const maxDuration = Math.max(1, ...stages.map((s) => s.durationMs));

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
      style={containerStyle}
    >
      <summary style={summaryStyle}>
        <span
          style={{
            color: "var(--accent)",
            fontWeight: 700,
            letterSpacing: "0.1em",
          }}
        >
          INFERENCE TRACE
        </span>
        <span style={{ color: "rgba(255,255,255,0.3)" }}>·</span>
        <span style={{ color: "rgba(255,255,255,0.5)" }}>
          {stages.length} STAGE{stages.length === 1 ? "" : "S"}
        </span>
        {metrics ? (
          <>
            <span style={{ color: "rgba(255,255,255,0.3)" }}>·</span>
            <MetricChip label="TTFT" value={fmtMs(metrics.ttftMs)} />
            <span style={{ color: "rgba(255,255,255,0.3)" }}>·</span>
            <MetricChip
              label="est. tok/s"
              value={
                metrics.tokPerSec > 0 ? metrics.tokPerSec.toFixed(1) : "—"
              }
            />
            <span style={{ color: "rgba(255,255,255,0.3)" }}>·</span>
            <MetricChip label="total" value={fmtMs(metrics.totalMs)} />
          </>
        ) : streaming ? (
          <>
            <span style={{ color: "rgba(255,255,255,0.3)" }}>·</span>
            <span style={{ color: "rgba(255,255,255,0.4)" }}>measuring…</span>
          </>
        ) : null}
      </summary>

      <div
        style={{
          borderTop: "1px solid rgba(255,255,255,0.06)",
          padding: "8px 8px 10px",
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        {stages.map((stage) => {
          const color = traceKindColor(stage.kind);
          const widthPct = Math.max(
            3,
            Math.round((stage.durationMs / maxDuration) * 100),
          );
          return (
            <div
              key={stage.id}
              style={{ display: "flex", flexDirection: "column", gap: 3 }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 7,
                  fontSize: 10,
                }}
              >
                {/* Kind-colored marker. */}
                <span
                  aria-hidden
                  style={{
                    width: 7,
                    height: 7,
                    flexShrink: 0,
                    background: color,
                    boxShadow: `0 0 6px ${color}`,
                  }}
                />
                <span
                  style={{
                    color: "rgba(255,255,255,0.9)",
                    fontWeight: 600,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {stage.label}
                </span>
                <span
                  style={{
                    marginLeft: "auto",
                    color: "rgba(255,255,255,0.85)",
                    fontVariantNumeric: "tabular-nums",
                    flexShrink: 0,
                  }}
                >
                  {fmtMs(stage.durationMs)}
                </span>
              </div>
              {/* Proportional duration bar. */}
              <div
                style={{
                  height: 3,
                  background: "rgba(255,255,255,0.06)",
                  marginLeft: 14,
                }}
              >
                <div
                  style={{
                    width: `${widthPct}%`,
                    height: "100%",
                    background: color,
                  }}
                />
              </div>
              {stage.detail ? (
                <div
                  style={{
                    marginLeft: 14,
                    fontSize: 9,
                    color: "rgba(255,255,255,0.42)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {stage.detail}
                </div>
              ) : null}
            </div>
          );
        })}

        {/* Token count footnote — honest "est." label per the trace model;
            the vLLM stream carries no usage object mid-turn. */}
        {metrics ? (
          <div
            style={{
              marginTop: 2,
              paddingTop: 6,
              borderTop: "1px dashed rgba(255,255,255,0.08)",
              fontSize: 9,
              color: "rgba(255,255,255,0.4)",
            }}
          >
            ~{metrics.tokens} output tokens · timings measured server-side ·
            token count estimated
          </div>
        ) : null}
      </div>
    </details>
  );
}
