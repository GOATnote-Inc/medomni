"use client";

// 4UWHAt — SystemPanel
//
// Phase 1 of the flagship-demo plan ("make the stack visible"): a compact,
// persistent panel that shows what is actually serving this demo — the
// model, the serving stack, the sovereignty property, and live vLLM
// serving metrics polled from `/api/telemetry`.
//
// Honesty (SPEC §6 Phase 1 / §8): the model / stack / sovereign lines are
// true by construction of this deployment. The metrics are live reads from
// the vLLM Prometheus endpoint; when that endpoint is unreachable the panel
// shows an explicit "metrics unavailable" state — never a fabricated value.
//
// Sovereignty line ("no third-party AI APIs"): this is the load-bearing
// differentiator — every inference runs on hardware GOATnote operates, with
// no cloud LLM API on the path (CLAUDE.md §2).

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { BASE_PATH } from "@/lib/basePath";
import { Eyebrow } from "./Eyebrow";
import { Mono } from "./Mono";

// Poll cadence for the live metrics. ~5s is responsive without hammering
// the route (which itself does a 4s-timeout read of the vLLM endpoint).
const POLL_INTERVAL_MS = 5000;

// --- Telemetry response shape (mirrors /api/telemetry) --------------------

interface ServingMetricsAvailable {
  available: true;
  scrapedAtMs: number;
  runningRequests: number | null;
  waitingRequests: number | null;
  generationTokensTotal: number | null;
  promptTokensTotal: number | null;
  kvCacheUsage: number | null;
  requestsFinishedTotal: number | null;
}

interface ServingMetricsUnavailable {
  available: false;
  reason: string;
}

type ServingMetrics = ServingMetricsAvailable | ServingMetricsUnavailable;

interface TelemetryResponse {
  model: string;
  servingStack: string;
  sovereign: boolean;
  metrics: ServingMetrics;
}

// --- formatting helpers ---------------------------------------------------

// Compact integer formatting with thousands separators (token counters get
// large). `null` -> em dash so an absent metric reads honestly.
function fmtCount(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

// KV-cache usage is a 0..1 fraction from vLLM; render as a percentage.
function fmtPct(frac: number | null): string {
  if (frac === null || !Number.isFinite(frac)) return "—";
  return `${Math.round(frac * 100)}%`;
}

// --- styles ---------------------------------------------------------------

const panelStyle: CSSProperties = {
  background: "var(--p42-ink, #0a0a0a)",
  border: "1px solid rgba(255,255,255,0.07)",
  padding: "14px 16px",
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const rowStyle: CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  justifyContent: "space-between",
  gap: 12,
};

// One label/value row.
function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={rowStyle}>
      <Mono size={9} color="rgba(255,255,255,0.4)">
        {label}
      </Mono>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          fontWeight: 600,
          color: "rgba(255,255,255,0.92)",
          fontVariantNumeric: "tabular-nums",
          textAlign: "right",
        }}
      >
        {value}
      </span>
    </div>
  );
}

export function SystemPanel({
  className,
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  const [data, setData] = useState<TelemetryResponse | null>(null);
  // Distinguishes "first load not done" from "loaded, metrics unavailable".
  const [loadError, setLoadError] = useState(false);
  // Guards a state update after unmount (the poll is async).
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    async function poll() {
      try {
        // BASE_PATH-prefixed: raw fetch() does not inherit Next's basePath,
        // and the demo is served behind the /4UWHAt reverse proxy.
        const res = await fetch(`${BASE_PATH}/api/telemetry`, {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`telemetry ${res.status}`);
        const json = (await res.json()) as TelemetryResponse;
        if (!mountedRef.current) return;
        setData(json);
        setLoadError(false);
      } catch {
        // Network/route failure — keep the last good facts on screen but
        // flag the metrics as unavailable. Never throw out of the poll.
        if (!mountedRef.current) return;
        setLoadError(true);
      }
    }

    void poll();
    const timer = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(timer);
    };
  }, []);

  // Static facts shown before the first response lands. These are the
  // deployment's true configuration; the route confirms them live.
  const model = data?.model ?? "Nemotron-3-Nano-Omni";
  const servingStack = data?.servingStack ?? "vLLM";
  const sovereign = data?.sovereign ?? true;

  // Metrics block: only "available" once the route says so AND the last
  // fetch did not error.
  const metrics =
    data && data.metrics.available && !loadError ? data.metrics : null;
  const metricsReason =
    data && !data.metrics.available
      ? data.metrics.reason
      : loadError
        ? "telemetry endpoint unreachable"
        : null;

  return (
    <section
      className={className}
      style={{ ...panelStyle, ...style }}
      aria-label="System status"
    >
      <div style={rowStyle}>
        <Eyebrow>SYSTEM</Eyebrow>
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 5,
          }}
        >
          <span
            aria-hidden
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: metrics ? "#22d3ee" : "rgba(255,255,255,0.3)",
              boxShadow: metrics ? "0 0 6px #22d3ee" : "none",
            }}
          />
          <Mono size={8.5} color="rgba(255,255,255,0.45)">
            {metrics ? "LIVE" : "FACTS"}
          </Mono>
        </span>
      </div>

      {/* Static, true-by-construction facts. */}
      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        <Row label="MODEL" value={model} />
        <Row label="SERVING" value={servingStack} />
      </div>

      {/* Sovereignty line — the differentiator. */}
      {sovereign ? (
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 7,
            padding: "7px 9px",
            border: "1px solid rgba(255,0,150,0.2)",
            background: "rgba(255,0,150,0.03)",
          }}
        >
          <span
            aria-hidden
            style={{ color: "var(--accent)", fontWeight: 700, fontSize: 11 }}
          >
            ✓
          </span>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 9.5,
              lineHeight: 1.45,
              color: "rgba(255,255,255,0.7)",
            }}
          >
            Sovereign — runs on hardware we operate · no third-party AI APIs
          </span>
        </div>
      ) : null}

      {/* Live serving metrics, or an explicit unavailable state. */}
      <div
        style={{
          borderTop: "1px solid rgba(255,255,255,0.06)",
          paddingTop: 9,
          display: "flex",
          flexDirection: "column",
          gap: 7,
        }}
      >
        <Mono size={8.5} color="rgba(255,255,255,0.35)">
          LIVE SERVING METRICS
        </Mono>
        {metrics ? (
          <>
            <Row
              label="REQUESTS RUNNING"
              value={fmtCount(metrics.runningRequests)}
            />
            <Row
              label="REQUESTS WAITING"
              value={fmtCount(metrics.waitingRequests)}
            />
            <Row
              label="KV-CACHE USAGE"
              value={fmtPct(metrics.kvCacheUsage)}
            />
            <Row
              label="GENERATION TOKENS"
              value={fmtCount(metrics.generationTokensTotal)}
            />
            <Row
              label="REQUESTS SERVED"
              value={fmtCount(metrics.requestsFinishedTotal)}
            />
          </>
        ) : (
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 9.5,
              lineHeight: 1.45,
              color: "rgba(255,255,255,0.4)",
              fontStyle: "italic",
            }}
          >
            Metrics unavailable
            {metricsReason ? ` — ${metricsReason}.` : "."} The model is still
            serving; only this telemetry read failed.
          </div>
        )}
      </div>
    </section>
  );
}
