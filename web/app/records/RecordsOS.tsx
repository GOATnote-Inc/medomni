"use client";

// 4UWHAt — Records OS dashboard.
// Three-column desktop layout (nav rail · module grid · AI rail) ported
// from /Users/kiteboard/medomni/assets/app/variant-records-os.jsx. Reads
// the active patientId via `usePatientId()`; if null (or until live FHIR
// patient context wires up beyond the get_patient_context tool path), it
// falls back to the design's synthetic "Maya Okafor" sample slice — that
// data is non-PHI, ships in the public bundle, and matches the design 1:1.
//
// Sovereignty: all in-page primitives are static. The only outbound
// network calls happen inside the embedded AskYourRecord command bar,
// which talks to /4UWHAt/api/agent on the same origin.

import { useState, type CSSProperties } from "react";
import {
  Avatar,
  Dot,
  Eyebrow,
  Mono,
  Sparkline,
  Stripe,
  Tag,
  Tooltip,
  Wordmark,
} from "@/components/4uwhat";
import { AskYourRecord } from "@/components/4uwhat/AskYourRecord";
// Direct import: A2's parallel branch lands the PatientPicker file at
// this path. Bypassing the barrel keeps this file's imports isolated to
// what A2 actually exports as a component module.
import { PatientPicker } from "@/components/4uwhat/PatientPicker";
import {
  SAMPLE_AI_SUGGESTIONS,
  SAMPLE_CARE_TEAM,
  SAMPLE_CONDITIONS,
  SAMPLE_LABS,
  SAMPLE_MEDS,
  SAMPLE_PATIENT,
  SAMPLE_SHARES,
  SAMPLE_TIMELINE,
  SAMPLE_VITALS,
} from "@/lib/4uwhat/sample-data";

// ── Shared inline styles (copied from variant-records-os.jsx) ─────────
// `minWidth: 0` on every card lets long medication names / lab analyte
// names wrap inside the card instead of forcing the parent grid column to
// expand and collide with siblings. `overflow: hidden` is the second guard
// for any absolutely-positioned child or oversized SVG that tries to
// escape (sparklines occasionally overshoot by 1-2px due to stroke caps).
const cardBase: CSSProperties = {
  background: "var(--p42-bg-card, #0e0e0e)",
  border: "1px solid rgba(255,255,255,0.07)",
  padding: 18,
  minWidth: 0,
  overflow: "hidden",
};

const tableHead: CSSProperties = {
  display: "grid",
  // minmax(0, …) on every column — long analyte names ("Lipoprotein(a)
  // mass") otherwise force the analyte column wider than its share and
  // shove value/range/trend off-edge.
  gridTemplateColumns:
    "minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 0.9fr) minmax(0, 0.9fr) minmax(0, 0.5fr)",
  gap: 12,
  padding: "8px 0",
  borderBottom: "1px solid rgba(255,255,255,0.08)",
  fontFamily: "var(--font-mono)",
  fontSize: 9,
  fontWeight: 700,
  letterSpacing: "0.12em",
  color: "rgba(255,255,255,0.4)",
  textTransform: "uppercase",
  minWidth: 0,
};

const tableRow: CSSProperties = {
  display: "grid",
  gridTemplateColumns:
    "minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 0.9fr) minmax(0, 0.9fr) minmax(0, 0.5fr)",
  gap: 12,
  padding: "10px 0",
  borderBottom: "1px solid rgba(255,255,255,0.04)",
  fontSize: 12.5,
  alignItems: "center",
  minWidth: 0,
  // Cell content (analyte name, range string, sparkline) wraps inside
  // its column rather than overflowing into the next.
  overflowWrap: "anywhere",
  wordBreak: "break-word",
};

const medRow: CSSProperties = {
  display: "flex",
  gap: 10,
  alignItems: "center",
  padding: "8px 10px",
  border: "1px solid rgba(255,255,255,0.06)",
  minWidth: 0,
};

const timelineRow: CSSProperties = {
  display: "flex",
  gap: 12,
  alignItems: "center",
  padding: "10px 0",
  borderBottom: "1px solid rgba(255,255,255,0.05)",
  minWidth: 0,
};

const btnGhost: CSSProperties = {
  background: "transparent",
  color: "rgba(255,255,255,0.7)",
  border: "1px solid #2a2a2a",
  padding: "7px 12px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 11.5,
  cursor: "pointer",
};

const btnPrimary: CSSProperties = {
  background: "var(--accent)",
  color: "#fff",
  border: "1px solid var(--accent)",
  padding: "7px 14px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 11.5,
  cursor: "pointer",
};

const NAV: Array<{ id: string; label: string; count: number | null; k: string }> = [
  { id: "overview", label: "Overview", count: null, k: "O" },
  { id: "timeline", label: "Timeline", count: 142, k: "T" },
  { id: "labs", label: "Labs", count: 8, k: "L" },
  { id: "meds", label: "Medications", count: 4, k: "M" },
  { id: "cond", label: "Conditions", count: 4, k: "C" },
  { id: "vitals", label: "Vitals", count: null, k: "V" },
  { id: "imaging", label: "Imaging", count: 3, k: "I" },
  { id: "wear", label: "Wearables", count: null, k: "W" },
  { id: "notes", label: "Visit notes", count: 12, k: "N" },
  { id: "team", label: "Care team", count: 4, k: "P" },
  { id: "genome", label: "Genome", count: null, k: "G" },
  { id: "shares", label: "Sharing", count: 4, k: "S" },
];

interface KeyValProps {
  k: string;
  v: string;
  tip?: { label: string; range: string; hint: string; source: string };
}

function KeyVal({ k, v, tip }: KeyValProps) {
  const inner = (
    <span style={{ fontSize: 13, fontWeight: 500 }}>{v}</span>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Mono size={9}>{k}</Mono>
      {tip ? <Tooltip {...tip}>{inner}</Tooltip> : inner}
    </div>
  );
}

interface ActivityRowProps {
  time: string;
  actor: string;
  verb: string;
  obj: string;
  accent?: boolean;
}

function ActivityRow({ time, actor, verb, obj, accent = false }: ActivityRowProps) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
      <Mono size={9} style={{ width: 36, flexShrink: 0, paddingTop: 2 }}>
        {time.toUpperCase()}
      </Mono>
      <Dot
        size={5}
        color={accent ? "var(--accent)" : "rgba(255,255,255,0.35)"}
        style={{ marginTop: 6, flexShrink: 0 }}
      />
      <div style={{ fontSize: 12, lineHeight: 1.4, flex: 1 }}>
        <span
          style={{
            color: accent ? "var(--accent)" : "rgba(255,255,255,0.95)",
            fontWeight: 600,
          }}
        >
          {actor}
        </span>{" "}
        <span style={{ color: "rgba(255,255,255,0.5)" }}>{verb}</span>{" "}
        <span style={{ color: "#fff" }}>{obj}</span>
      </div>
    </div>
  );
}

export function RecordsOS() {
  const [activeModule, setActiveModule] = useState("overview");

  // Design fallback. When patientId is set but no live patient feed has
  // been wired in, render the sample slice so the page is never blank.
  const patient = SAMPLE_PATIENT;
  const vitals = SAMPLE_VITALS;
  const conditions = SAMPLE_CONDITIONS;
  const meds = SAMPLE_MEDS;
  const labs = SAMPLE_LABS;
  const timeline = SAMPLE_TIMELINE;
  const careTeam = SAMPLE_CARE_TEAM;
  const shares = SAMPLE_SHARES;

  return (
    <div
      style={{
        background: "#000",
        color: "#fff",
        display: "grid",
        // Three-column desktop layout. Both rails are explicit fixed widths
        // (220px nav, 360px AI rail) so they cannot squeeze the fluid
        // center column. `minmax(0, 1fr)` on the center column is the CSS
        // Grid gotcha fix: bare `1fr` defaults to `minmax(auto, 1fr)`,
        // which lets oversized children (long lab names, wide tables,
        // sparklines) push the column wider than the viewport. `minmax(0,
        // 1fr)` clamps it to the available space so children honor it.
        gridTemplateColumns: "220px minmax(0, 1fr) 360px",
        fontFamily: "var(--font-display)",
        minHeight: "calc(100vh - 40px)",
        // Bound the dashboard to the viewport height so the rails (nav +
        // AI rail) can scroll internally instead of pushing the page taller
        // than the screen, which is what was making the chat panel input
        // appear "mid-scroll" when reasoning streamed in.
        height: "calc(100vh - 40px)",
        // No horizontal page scroll, ever. The rails are fixed widths so
        // the only way to break out is via a runaway grid child; the
        // overflow-x guard catches that.
        overflowX: "hidden",
        overflowY: "hidden",
        maxWidth: "100vw",
        borderTop: "1px solid #1f1f1f",
      }}
    >
      {/* ── LEFT RAIL ───────────────────────────────────────────────── */}
      <aside
        style={{
          borderRight: "1px solid #1f1f1f",
          display: "flex",
          flexDirection: "column",
          background: "#000",
          // Grid item — without min-height:0 the inner `<nav>` cannot
          // actually scroll (the flex column would expand the rail to fit
          // every nav item).
          minHeight: 0,
          minWidth: 0,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "20px 20px 16px",
            borderBottom: "1px solid #1f1f1f",
            flexShrink: 0,
          }}
        >
          <Wordmark size={13} />
          <Mono size={9} style={{ marginTop: 10, display: "block" }} color="rgba(255,255,255,0.35)">
            HEALTH / v4.2
          </Mono>
        </div>

        {/* Patient picker */}
        <div
          style={{
            padding: "12px 20px",
            borderBottom: "1px solid #1f1f1f",
            flexShrink: 0,
            minWidth: 0,
          }}
        >
          <Mono size={9} color="rgba(255,255,255,0.4)" style={{ marginBottom: 6, display: "block" }}>
            PATIENT
          </Mono>
          <PatientPicker />
        </div>

        {/* Patient identity */}
        <div
          style={{
            padding: "16px 20px",
            borderBottom: "1px solid #1f1f1f",
            flexShrink: 0,
            minWidth: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
            <Avatar
              initials={patient.name
                .split(" ")
                .map((s) => s[0])
                .slice(0, 2)
                .join("")
                .toUpperCase()}
              size={32}
              accent={true}
            />
            <div
              style={{
                minWidth: 0,
                overflowWrap: "anywhere",
                wordBreak: "break-word",
              }}
            >
              <div style={{ fontWeight: 600, fontSize: 13, lineHeight: 1.2 }}>{patient.name}</div>
              <Mono size={9} color="rgba(255,255,255,0.4)">
                {patient.mrn}
              </Mono>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ padding: "8px 8px", flex: 1, overflowY: "auto", overflowX: "hidden", minHeight: 0 }}>
          {NAV.map((n) => {
            const active = n.id === activeModule;
            return (
              <button
                key={n.id}
                type="button"
                onClick={() => setActiveModule(n.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  width: "100%",
                  padding: "8px 12px",
                  border: "none",
                  cursor: "pointer",
                  background: active ? "rgba(255,0,150,0.08)" : "transparent",
                  color: active ? "var(--accent)" : "rgba(255,255,255,0.7)",
                  borderLeft: "2px solid",
                  borderLeftColor: active ? "var(--accent)" : "transparent",
                  fontFamily: "inherit",
                  textAlign: "left",
                }}
              >
                <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <Mono size={9} color={active ? "var(--accent)" : "rgba(255,255,255,0.3)"}>
                    {n.k}
                  </Mono>
                  <span style={{ fontSize: 13, fontWeight: active ? 600 : 500 }}>{n.label}</span>
                </span>
                {n.count != null && (
                  <Mono size={9} color={active ? "var(--accent)" : "rgba(255,255,255,0.35)"}>
                    {String(n.count).padStart(3, "0")}
                  </Mono>
                )}
              </button>
            );
          })}
        </nav>

        {/* Footer status */}
        <div
          style={{
            padding: "12px 20px",
            borderTop: "1px solid #1f1f1f",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Dot color="var(--accent)" glow={true} size={6} />
            <Mono size={9}>SYNCED · 2 MIN AGO</Mono>
          </div>
        </div>
      </aside>

      {/* ── MAIN ────────────────────────────────────────────────────── */}
      <main
        style={{
          display: "flex",
          flexDirection: "column",
          // `min-width: 0` (already present) is the grid gotcha fix; pair
          // with `min-height: 0` so the inner content scroll region works.
          minWidth: 0,
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        {/* Top bar */}
        <div
          style={{
            height: 56,
            borderBottom: "1px solid #1f1f1f",
            padding: "0 28px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <Eyebrow>OVERVIEW</Eyebrow>
            <span style={{ color: "rgba(255,255,255,0.2)" }}>/</span>
            <Mono color="rgba(255,255,255,0.55)">YOUR RECORD · APR 22, 2026</Mono>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button type="button" style={btnGhost}>
              Export
            </button>
            <button type="button" style={btnGhost}>
              Share
            </button>
            <button type="button" style={btnPrimary}>
              + New entry
            </button>
          </div>
        </div>

        {/* Content scroll area */}
        <div
          style={{
            padding: "24px 28px",
            flex: 1,
            minHeight: 0,
            minWidth: 0,
            // The dashboard content (hero row, vitals strip, labs/meds row,
            // timeline) is taller than the viewport. Without a scroll
            // container here, the whole page would grow and the rails
            // would lose their internal-scroll behavior.
            overflowY: "auto",
            overflowX: "hidden",
          }}
        >
          {/* Hero row */}
          <div
            style={{
              display: "grid",
              // `minmax(0, …fr)` everywhere a grid child renders content
              // that could be wider than the column (long names, big
              // numerics, sparklines). Without this, `1.2fr 1fr` defaults
              // to `minmax(auto, …)` and any oversized inline element
              // (like the 36px display name "Maya Okafor") stretches the
              // column past the available width.
              gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 1fr)",
              gap: 16,
              marginBottom: 16,
              minWidth: 0,
            }}
          >
            <div style={cardBase}>
              <Eyebrow style={{ marginBottom: 12 }}>
                PATIENT · {patient.pronouns.toUpperCase()}
              </Eyebrow>
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                  gap: 16,
                }}
              >
                <div>
                  <div
                    style={{
                      fontFamily: "var(--font-display)",
                      fontSize: 36,
                      fontWeight: 700,
                      letterSpacing: "-0.025em",
                      lineHeight: 1.05,
                    }}
                  >
                    {patient.name}
                  </div>
                  <div
                    style={{
                      marginTop: 8,
                      display: "flex",
                      gap: 18,
                      flexWrap: "wrap",
                    }}
                  >
                    <KeyVal k="AGE" v={`${patient.age} yr`} />
                    <KeyVal k="DOB" v={patient.dob} />
                    <KeyVal
                      k="BLOOD"
                      v={patient.bloodType}
                      tip={{
                        label: "BLOOD TYPE",
                        range: "O+ · universal red-cell donor",
                        hint: "Compatible with all positive recipients.",
                        source: "TYPE & SCREEN · 2018",
                      }}
                    />
                    <KeyVal k="HT" v={patient.height} />
                    <KeyVal
                      k="WT"
                      v={patient.weight}
                      tip={{
                        label: "WEIGHT",
                        range: vitals.weight.range,
                        hint: vitals.weight.hint,
                        source: vitals.weight.source,
                      }}
                    />
                    <KeyVal k="PCP" v={patient.primaryCare} />
                  </div>
                </div>
                <Stripe width={48} height={3} />
              </div>
              {/* Active flags */}
              <div
                style={{
                  marginTop: 18,
                  paddingTop: 16,
                  borderTop: "1px solid rgba(255,255,255,0.06)",
                  display: "flex",
                  gap: 8,
                  flexWrap: "wrap",
                }}
              >
                <Tag accent={true}>
                  <Dot size={5} color="var(--accent)" glow={true} /> 2 ACTIVE CONDITIONS
                </Tag>
                <Tag>
                  <Dot size={5} color="rgba(255,255,255,0.5)" /> 3 RX ACTIVE
                </Tag>
                <Tag>
                  <Dot size={5} color="rgba(255,255,255,0.5)" /> NKA · NO ALLERGIES
                </Tag>
                <Tag color="#ffaa00">
                  <Dot size={5} color="#ffaa00" /> VIT D LOW · 28 ng/mL
                </Tag>
              </div>
            </div>

            {/* Signal of the day */}
            <div style={{ ...cardBase, position: "relative", overflow: "hidden" }}>
              <Eyebrow style={{ marginBottom: 12 }}>SIGNAL · TODAY</Eyebrow>
              <div
                style={{
                  fontFamily: "var(--font-display)",
                  fontSize: 22,
                  fontWeight: 600,
                  lineHeight: 1.25,
                  letterSpacing: "-0.015em",
                }}
              >
                Your LDL dropped <span style={{ color: "var(--accent)" }}>50 points</span> in 24 months. The statin is working.
              </div>
              <div style={{ marginTop: 16, display: "flex", alignItems: "flex-end", gap: 16 }}>
                <div>
                  <div
                    style={{
                      fontFamily: "var(--font-display)",
                      fontSize: 48,
                      fontWeight: 700,
                      letterSpacing: "-0.03em",
                      lineHeight: 1,
                      color: "var(--accent)",
                    }}
                  >
                    92
                  </div>
                  <Mono size={9} style={{ marginTop: 4, display: "block" }}>
                    MG/DL · TARGET &lt;100
                  </Mono>
                </div>
                <div style={{ flex: 1, paddingBottom: 4, color: "var(--accent)" }}>
                  <Sparkline
                    data={labs[0].trend}
                    w={180}
                    h={56}
                    stroke="var(--accent)"
                    fill="var(--accent)"
                    strokeWidth={1.75}
                  />
                </div>
              </div>
              <Mono
                size={9}
                style={{ marginTop: 12, display: "block" }}
                color="rgba(255,255,255,0.35)"
              >
                ↳ FLAGGED BY 4UWHAt · 4H AGO
              </Mono>
            </div>
          </div>

          {/* Vitals strip */}
          <div style={{ ...cardBase, marginBottom: 16, padding: 0 }}>
            <div
              style={{
                padding: "14px 20px",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <Eyebrow>VITALS · LAST 12 READINGS</Eyebrow>
              <Mono>FROM APPLE WATCH · OURA · CLINIC</Mono>
            </div>
            <div
              style={{
                display: "grid",
                // minmax(0, 1fr) prevents a single oversized vital reading
                // (e.g. a long unit string) from squeezing the other five.
                gridTemplateColumns: "repeat(6, minmax(0, 1fr))",
                minWidth: 0,
              }}
            >
              {Object.values(vitals).map((v, i) => (
                <div
                  key={v.label}
                  style={{
                    padding: "16px 20px",
                    borderRight: i < 5 ? "1px solid rgba(255,255,255,0.06)" : "none",
                    minWidth: 0,
                    // Sparkline below uses w={120} which is wider than the
                    // column at sub-1280px viewports; clip rather than
                    // expand the cell.
                    overflow: "hidden",
                  }}
                >
                  <Mono size={9}>{v.label.toUpperCase()}</Mono>
                  <Tooltip label={v.label} range={v.range} hint={v.hint} source={v.source}>
                    <span style={{ display: "flex", alignItems: "baseline", gap: 5, marginTop: 6 }}>
                      <span
                        style={{
                          fontFamily: "var(--font-display)",
                          fontSize: 24,
                          fontWeight: 700,
                          letterSpacing: "-0.02em",
                        }}
                      >
                        {v.value}
                      </span>
                      <Mono size={9} style={{ fontWeight: 500 }}>
                        {v.unit}
                      </Mono>
                    </span>
                  </Tooltip>
                  <div style={{ marginTop: 8, height: 22, color: "var(--accent)" }}>
                    <Sparkline data={v.spark} w={120} h={22} stroke="var(--accent)" strokeWidth={1.25} />
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginTop: 4,
                    }}
                  >
                    <Mono size={9} color="var(--accent)">
                      {v.delta}
                    </Mono>
                    <Mono size={9} color="rgba(255,255,255,0.3)">
                      {v.range.split(" · ")[0].split(" typical")[0]}
                    </Mono>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Labs + Meds + Conditions row */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)",
              gap: 16,
              marginBottom: 16,
              minWidth: 0,
            }}
          >
            {/* Labs table */}
            <div style={cardBase}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 14,
                }}
              >
                <Eyebrow>LATEST LABS · APR 22</Eyebrow>
                <Mono color="rgba(255,255,255,0.4)">8 RESULTS · 1 OUT OF RANGE</Mono>
              </div>
              <div style={{ display: "flex", flexDirection: "column" }}>
                <div style={tableHead}>
                  <span>ANALYTE</span>
                  <span style={{ textAlign: "right" }}>VALUE</span>
                  <span>RANGE</span>
                  <span>TREND</span>
                  <span style={{ textAlign: "right" }}>FLAG</span>
                </div>
                {labs.map((l) => (
                  <div key={l.id} style={tableRow}>
                    <Tooltip
                      label={l.name}
                      range={`Reference: ${l.range} ${l.unit}`}
                      hint={l.hint}
                      source={l.source}
                    >
                      <span style={{ fontWeight: 500 }}>{l.name}</span>
                    </Tooltip>
                    <span
                      style={{
                        textAlign: "right",
                        fontFamily: "var(--font-mono)",
                        fontVariantNumeric: "tabular-nums",
                        color: l.flag === "low" ? "#ffaa00" : "#fff",
                      }}
                    >
                      {l.value} <Mono size={9}>{l.unit}</Mono>
                    </span>
                    <Mono>{l.range}</Mono>
                    <span
                      style={{
                        color: l.flag === "low" ? "#ffaa00" : "var(--accent)",
                        display: "inline-block",
                        minWidth: 0,
                        overflow: "hidden",
                      }}
                    >
                      <Sparkline
                        data={l.trend}
                        w={70}
                        h={18}
                        stroke="currentColor"
                        strokeWidth={1.25}
                        dot={false}
                      />
                    </span>
                    <span style={{ textAlign: "right" }}>
                      {l.flag === "normal" ? (
                        <Mono size={9} color="rgba(255,255,255,0.45)">
                          OK
                        </Mono>
                      ) : (
                        <Tag color="#ffaa00" style={{ fontSize: 8.5, padding: "2px 5px" }}>
                          LOW
                        </Tag>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Meds + conditions stacked */}
            <div
              style={{
                display: "grid",
                gridTemplateRows: "minmax(0, 1fr) minmax(0, 1fr)",
                gap: 16,
                minWidth: 0,
              }}
            >
              <div style={cardBase}>
                <Eyebrow style={{ marginBottom: 12 }}>MEDICATIONS · 4 ACTIVE</Eyebrow>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {meds.slice(0, 4).map((m) => (
                    <div key={m.id} style={medRow}>
                      <div
                        style={{
                          flex: 1,
                          minWidth: 0,
                          overflowWrap: "anywhere",
                          wordBreak: "break-word",
                        }}
                      >
                        <Tooltip
                          label={m.name}
                          range={`${m.dose} · ${m.freq}`}
                          hint={`Prescriber: ${m.prescriber} · since ${m.since}`}
                          source={m.refills != null ? `${m.refills} refills remaining` : "OTC"}
                        >
                          <span style={{ fontWeight: 600, fontSize: 13 }}>{m.name}</span>
                        </Tooltip>
                        <Mono size={9} style={{ marginTop: 2, display: "block" }}>
                          {m.dose} · {m.freq}
                        </Mono>
                      </div>
                      {m.adherence != null && (
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <Mono size={9}>{Math.round(m.adherence * 100)}%</Mono>
                          <div
                            style={{
                              width: 36,
                              height: 3,
                              background: "rgba(255,255,255,0.08)",
                            }}
                          >
                            <div
                              style={{
                                width: `${m.adherence * 100}%`,
                                height: "100%",
                                background: "var(--accent)",
                              }}
                            />
                          </div>
                        </div>
                      )}
                      {m.refills != null && (
                        <Mono
                          size={9}
                          color={m.refills <= 1 ? "#ffaa00" : "rgba(255,255,255,0.45)"}
                        >
                          {m.refills} REFILL{m.refills !== 1 ? "S" : ""}
                        </Mono>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div style={cardBase}>
                <Eyebrow style={{ marginBottom: 12 }}>PROBLEM LIST</Eyebrow>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  {conditions.map((c) => (
                    <div
                      key={c.id}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "12px minmax(0, 1fr) auto auto",
                        alignItems: "center",
                        gap: 10,
                        padding: "8px 0",
                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                        minWidth: 0,
                      }}
                    >
                      <Dot
                        color={
                          c.status === "active"
                            ? "var(--accent)"
                            : "rgba(255,255,255,0.3)"
                        }
                        size={6}
                        glow={c.status === "active"}
                      />
                      <span
                        style={{
                          fontSize: 13,
                          color: c.status === "active" ? "#fff" : "rgba(255,255,255,0.5)",
                          minWidth: 0,
                          overflowWrap: "anywhere",
                          wordBreak: "break-word",
                        }}
                      >
                        {c.name}
                      </span>
                      <Mono size={9}>{c.icd}</Mono>
                      <Mono
                        size={9}
                        color={
                          c.status === "active"
                            ? "var(--accent)"
                            : "rgba(255,255,255,0.4)"
                        }
                      >
                        SINCE {c.onset}
                      </Mono>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Bottom row: timeline preview + sharing */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)",
              gap: 16,
              minWidth: 0,
            }}
          >
            <div style={cardBase}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 14,
                }}
              >
                <Eyebrow>TIMELINE · LAST 6 MONTHS</Eyebrow>
                <Mono>SCROLL FOR FULL HISTORY ↓</Mono>
              </div>
              <div style={{ display: "flex", flexDirection: "column" }}>
                {timeline.slice(0, 6).map((e, i) => (
                  <div key={`${e.date}-${i}`} style={timelineRow}>
                    <Mono
                      size={10}
                      color="rgba(255,255,255,0.4)"
                      style={{ width: 78, flexShrink: 0 }}
                    >
                      {e.date.slice(5).replace("-", "/")}
                    </Mono>
                    <Tag
                      color={e.kind === "visit" ? "var(--accent)" : "rgba(255,255,255,0.3)"}
                      accent={e.kind === "visit"}
                      style={{ width: 50, justifyContent: "center" }}
                    >
                      {e.tag}
                    </Tag>
                    <span
                      style={{
                        fontSize: 13,
                        fontWeight: 500,
                        flex: 1,
                        minWidth: 0,
                        overflowWrap: "anywhere",
                        wordBreak: "break-word",
                      }}
                    >
                      {e.title}
                    </span>
                    <Mono color="rgba(255,255,255,0.45)">{e.who}</Mono>
                  </div>
                ))}
              </div>
            </div>

            <div style={cardBase}>
              <Eyebrow style={{ marginBottom: 12 }}>WHO HAS ACCESS · {shares.length}</Eyebrow>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {shares.map((s, i) => (
                  <div
                    key={`${s.who}-${i}`}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 10,
                      padding: "10px 12px",
                      border: "1px solid rgba(255,255,255,0.06)",
                      minWidth: 0,
                    }}
                  >
                    <div
                      style={{
                        flex: 1,
                        minWidth: 0,
                        overflowWrap: "anywhere",
                        wordBreak: "break-word",
                      }}
                    >
                      <div style={{ fontSize: 13, fontWeight: 500 }}>{s.who}</div>
                      <Mono size={9}>{s.scope.toUpperCase()}</Mono>
                    </div>
                    <button
                      type="button"
                      style={{ ...btnGhost, padding: "4px 8px", fontSize: 10, flexShrink: 0 }}
                    >
                      REVOKE
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── RIGHT RAIL — AI command bar + activity ─────────────────── */}
      <aside
        style={{
          borderLeft: "1px solid #1f1f1f",
          display: "flex",
          flexDirection: "column",
          background: "#000",
          // Bound the rail to the grid track height so its three sections
          // (chat, care team, recent activity) can each scroll internally
          // without pushing the page taller than the viewport.
          minHeight: 0,
          minWidth: 0,
          overflow: "hidden",
        }}
      >
        {/* Ask your record (AI) — flex:1 so the chat panel takes the
            largest share of the rail; AskYourRecord itself caps to a
            sensible viewport-bounded max. */}
        <div
          style={{
            padding: "20px 22px",
            borderBottom: "1px solid #1f1f1f",
            display: "flex",
            flexDirection: "column",
            flex: "1 1 auto",
            minHeight: 0,
            minWidth: 0,
          }}
        >
          <AskYourRecord suggestions={SAMPLE_AI_SUGGESTIONS} />
        </div>

        {/* Care team */}
        <div
          style={{
            padding: "18px 22px",
            borderBottom: "1px solid #1f1f1f",
            flexShrink: 0,
            minWidth: 0,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
            <Eyebrow>CARE TEAM</Eyebrow>
            <Mono>{careTeam.filter((c) => c.online).length} ONLINE</Mono>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {careTeam.map((c) => (
              <div
                key={c.id}
                style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}
              >
                <Avatar initials={c.avatar} online={c.online} size={28} />
                <div
                  style={{
                    flex: 1,
                    minWidth: 0,
                    overflowWrap: "anywhere",
                    wordBreak: "break-word",
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 600, lineHeight: 1.3 }}>
                    {c.name}
                  </div>
                  <Mono size={9}>{c.role.toUpperCase()}</Mono>
                </div>
                <button
                  type="button"
                  style={{ ...btnGhost, padding: "3px 7px", fontSize: 10, flexShrink: 0 }}
                >
                  MSG
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Recent activity */}
        <div
          style={{
            padding: "18px 22px",
            // Cap at a fraction of the rail so it doesn't crowd out the
            // chat panel. Internal scroll lets the activity list grow.
            flex: "0 1 auto",
            maxHeight: "30vh",
            overflowY: "auto",
            overflowX: "hidden",
            minWidth: 0,
          }}
        >
          <Eyebrow style={{ marginBottom: 12 }}>RECENT ACTIVITY</Eyebrow>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <ActivityRow time="2 min" actor="4UWHAt" verb="synced" obj="Apple Health · 47 entries" />
            <ActivityRow time="11 min" actor="Dr. Patel" verb="signed" obj="Visit note · Feb 19" accent />
            <ActivityRow time="4h" actor="4UWHAt" verb="flagged" obj="Vitamin D · 28 ng/mL · low" />
            <ActivityRow time="1d" actor="Quest" verb="released" obj="Lipid panel + CBC + CMP" />
            <ActivityRow time="2d" actor="You" verb="shared" obj="Pulmonary record → Dr. Patel" />
            <ActivityRow time="6d" actor="CVS" verb="filled" obj="Rosuvastatin · 30 day" />
          </div>
        </div>

        <div style={{ height: 4, background: "var(--accent)", flexShrink: 0 }} />
      </aside>
    </div>
  );
}
