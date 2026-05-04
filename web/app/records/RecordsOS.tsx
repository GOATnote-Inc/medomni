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

import {
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import {
  Avatar,
  DetailDrawer,
  Dot,
  Eyebrow,
  Mono,
  RangeBar,
  ShareDrawer,
  Sparkline,
  Stripe,
  Tag,
  Tooltip,
  TrendChart,
  Wordmark,
} from "@/components/4uwhat";
import { AskYourRecord } from "@/components/4uwhat/AskYourRecord";
import { Resizer, loadPersistedVar } from "@/components/4uwhat/Resizer";
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
  type SampleCareTeamMember,
  type SampleCondition,
  type SampleLab,
  type SampleMed,
  type SampleTimelineEntry,
  type SampleVital,
} from "@/lib/4uwhat/sample-data";

// ── Shared inline styles (copied from variant-records-os.jsx) ─────────
const cardBase: CSSProperties = {
  background: "var(--p42-bg-card, #0e0e0e)",
  border: "1px solid rgba(255,255,255,0.07)",
  padding: 18,
};

const tableHead: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 0.9fr) minmax(0, 0.9fr) minmax(0, 0.5fr)",
  gap: 12,
  padding: "8px 0",
  borderBottom: "1px solid rgba(255,255,255,0.08)",
  fontFamily: "var(--font-mono)",
  fontSize: 9,
  fontWeight: 700,
  letterSpacing: "0.12em",
  color: "rgba(255,255,255,0.4)",
  textTransform: "uppercase",
};

const tableRow: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 0.9fr) minmax(0, 0.9fr) minmax(0, 0.5fr)",
  gap: 12,
  padding: "10px 0",
  borderBottom: "1px solid rgba(255,255,255,0.04)",
  fontSize: 12.5,
  alignItems: "center",
};

const medRow: CSSProperties = {
  display: "flex",
  gap: 10,
  alignItems: "center",
  padding: "8px 10px",
  border: "1px solid rgba(255,255,255,0.06)",
};

const timelineRow: CSSProperties = {
  display: "flex",
  gap: 12,
  alignItems: "center",
  padding: "10px 0",
  borderBottom: "1px solid rgba(255,255,255,0.05)",
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
  /** When set, the row becomes clickable + opens the supplied drawer target. */
  onClick?: () => void;
}

function ActivityRow({
  time,
  actor,
  verb,
  obj,
  accent = false,
  onClick,
}: ActivityRowProps) {
  const [hover, setHover] = useState(false);
  const interactive = !!onClick;
  const inner = (
    <div
      style={{
        display: "flex",
        gap: 10,
        alignItems: "flex-start",
        padding: "2px 4px",
        background:
          interactive && hover ? "rgba(255,0,150,0.06)" : "transparent",
        outline:
          interactive && hover
            ? "1px solid rgba(255,0,150,0.18)"
            : "1px solid transparent",
        transition: "background 120ms ease, outline-color 120ms ease",
        opacity: interactive ? 1 : 0.78,
      }}
    >
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
  if (!interactive) return inner;
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      style={{
        all: "unset",
        cursor: "pointer",
        display: "block",
        width: "100%",
      }}
      aria-label={`${actor} ${verb} ${obj}`}
    >
      {inner}
    </button>
  );
}

// ── Drawer target discriminated union ────────────────────────────────
// Centralizing the open-drawer-target state keeps RecordsOS rendering
// only one DetailDrawer at a time. `null` = closed.
type DrawerTarget =
  | { kind: "lab"; id: string }
  | { kind: "vital"; id: string }
  | { kind: "med"; id: string }
  | { kind: "condition"; id: string }
  | { kind: "careTeam"; id: string }
  | null;

// ── Hover-affordance interactive wrapper ────────────────────────────
// Adds cursor:pointer + a subtle hover tint without re-implementing
// every clickable surface. Renders a <button> for keyboard a11y.
interface ClickableProps {
  onClick: () => void;
  children: ReactNode;
  ariaLabel: string;
  style?: CSSProperties;
  /** Hover tint color override; defaults to a faint magenta. */
  tint?: string;
}

function Clickable({
  onClick,
  children,
  ariaLabel,
  style,
  tint = "rgba(255,0,150,0.06)",
}: ClickableProps) {
  const [hover, setHover] = useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      aria-label={ariaLabel}
      style={{
        // Reset native button chrome so it inherits its parent's slot.
        all: "unset",
        boxSizing: "border-box",
        cursor: "pointer",
        display: "block",
        width: "100%",
        background: hover ? tint : "transparent",
        outline: hover ? "1px solid rgba(255,0,150,0.18)" : "1px solid transparent",
        transition: "background 120ms ease, outline-color 120ms ease",
        ...style,
      }}
    >
      {children}
    </button>
  );
}

// ── Drawer body renderers ───────────────────────────────────────────
// Each renderer takes the resolved entity and returns ready-to-render
// JSX scoped to the drawer surface. Sample-data only.

const drawerSection: CSSProperties = {
  marginBottom: 18,
};

const drawerLabel: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 9,
  fontWeight: 700,
  letterSpacing: "0.16em",
  color: "rgba(255,255,255,0.4)",
  textTransform: "uppercase",
  marginBottom: 8,
  display: "block",
};

function LabDrawerBody({ lab }: { lab: SampleLab }) {
  // Parse range string ("<100", ">40", "30–100", "0.4–4.0") into
  // numeric bounds for RangeBar. Unknown shapes degrade gracefully.
  const bounds = useMemo(() => {
    const r = lab.range.trim();
    const dash = r.match(/([\d.]+)\s*[–-]\s*([\d.]+)/);
    if (dash) return { low: parseFloat(dash[1]), high: parseFloat(dash[2]) };
    const lt = r.match(/^<\s*([\d.]+)/);
    if (lt) return { low: 0, high: parseFloat(lt[1]) };
    const gt = r.match(/^>\s*([\d.]+)/);
    if (gt) return { low: parseFloat(gt[1]), high: parseFloat(gt[1]) * 2 };
    return null;
  }, [lab.range]);

  const trendRows = lab.trend.map((v, i) => ({
    value: v,
    date: lab.dates?.[i] ?? `T-${lab.trend.length - 1 - i}`,
  }));

  return (
    <div>
      <div style={drawerSection}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 10,
            marginBottom: 6,
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 40,
              fontWeight: 700,
              letterSpacing: "-0.025em",
              color: lab.flag === "low" ? "#ffaa00" : "#fff",
            }}
          >
            {lab.value}
          </span>
          <Mono size={10} color="rgba(255,255,255,0.55)">
            {lab.unit} · ref {lab.range}
          </Mono>
        </div>
        {bounds ? (
          <div style={{ marginTop: 4 }}>
            <RangeBar
              value={lab.value}
              low={bounds.low}
              high={bounds.high}
              w={420}
            />
          </div>
        ) : null}
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Trend · {lab.trend.length} draws</span>
        <div style={{ color: "var(--accent)" }}>
          <TrendChart
            data={lab.trend}
            dates={lab.dates}
            w={420}
            h={130}
            accent="#ff0096"
          />
        </div>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Recent values</span>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {[...trendRows]
            .reverse()
            .slice(0, 5)
            .map((row, i) => (
              <div
                key={`${row.date}-${i}`}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "6px 0",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                <span style={{ color: "rgba(255,255,255,0.55)" }}>
                  {row.date}
                </span>
                <span style={{ color: "#fff" }}>
                  {row.value} {lab.unit}
                </span>
              </div>
            ))}
        </div>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>AI insight</span>
        <p
          style={{
            margin: 0,
            fontSize: 13,
            lineHeight: 1.5,
            color: "rgba(255,255,255,0.85)",
          }}
        >
          {lab.hint}
        </p>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Source</span>
        <Mono size={11} color="rgba(255,255,255,0.7)">
          {lab.source}
        </Mono>
      </div>
    </div>
  );
}

function VitalDrawerBody({ vital }: { vital: SampleVital }) {
  return (
    <div>
      <div style={drawerSection}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 10,
            marginBottom: 6,
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 40,
              fontWeight: 700,
              letterSpacing: "-0.025em",
            }}
          >
            {vital.value}
          </span>
          <Mono size={10} color="rgba(255,255,255,0.55)">
            {vital.unit} · {vital.delta}
          </Mono>
        </div>
        <Mono size={10} color="rgba(255,255,255,0.45)">
          Reference: {vital.range}
        </Mono>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Last {vital.spark.length} readings</span>
        <div style={{ color: "var(--accent)" }}>
          <TrendChart data={vital.spark} w={420} h={120} accent="#ff0096" />
        </div>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Recent values</span>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {[...vital.spark]
            .map((v, i) => ({
              v,
              tMinus: vital.spark.length - 1 - i,
            }))
            .reverse()
            .slice(0, 7)
            .map((row, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "6px 0",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                }}
              >
                <span style={{ color: "rgba(255,255,255,0.55)" }}>
                  {row.tMinus === 0 ? "now" : `t-${row.tMinus}`}
                </span>
                <span style={{ color: "#fff" }}>
                  {row.v} {vital.unit}
                </span>
              </div>
            ))}
        </div>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>What this means</span>
        <p
          style={{
            margin: 0,
            fontSize: 13,
            lineHeight: 1.5,
            color: "rgba(255,255,255,0.85)",
          }}
        >
          {vital.hint}
        </p>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Source</span>
        <Mono size={11} color="rgba(255,255,255,0.7)">
          {vital.source}
        </Mono>
      </div>
    </div>
  );
}

function MedDrawerBody({ med }: { med: SampleMed }) {
  return (
    <div>
      <div style={drawerSection}>
        <div
          style={{
            fontFamily: "var(--font-display)",
            fontSize: 22,
            fontWeight: 700,
            marginBottom: 4,
          }}
        >
          {med.dose} · {med.freq}
        </div>
        <Mono size={10} color="rgba(255,255,255,0.55)">
          Since {med.since}
        </Mono>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Adherence</span>
        {med.adherence != null ? (
          <div>
            <div
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 28,
                fontWeight: 700,
                color: med.adherence >= 0.85 ? "var(--accent)" : "#ffaa00",
                marginBottom: 6,
              }}
            >
              {Math.round(med.adherence * 100)}%
            </div>
            <div
              style={{
                width: "100%",
                height: 4,
                background: "rgba(255,255,255,0.08)",
              }}
            >
              <div
                style={{
                  width: `${med.adherence * 100}%`,
                  height: "100%",
                  background:
                    med.adherence >= 0.85 ? "var(--accent)" : "#ffaa00",
                }}
              />
            </div>
          </div>
        ) : (
          <Mono size={11} color="rgba(255,255,255,0.55)">
            Not tracked (PRN / OTC)
          </Mono>
        )}
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Refills</span>
        <Mono
          size={12}
          color={
            med.refills != null && med.refills <= 1
              ? "#ffaa00"
              : "rgba(255,255,255,0.85)"
          }
        >
          {med.refills != null
            ? `${med.refills} remaining`
            : "OTC · no refill schedule"}
        </Mono>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Prescriber</span>
        <Mono size={12} color="rgba(255,255,255,0.85)">
          {med.prescriber}
        </Mono>
      </div>
    </div>
  );
}

function ConditionDrawerBody({
  condition,
  timeline,
}: {
  condition: SampleCondition;
  timeline: SampleTimelineEntry[];
}) {
  // Surface timeline entries that mention this condition by name token.
  const token = condition.name.split(",")[0].split(" ")[0].toLowerCase();
  const related = timeline.filter((e) =>
    `${e.title} ${e.who} ${e.loc}`.toLowerCase().includes(token),
  );

  return (
    <div>
      <div style={drawerSection}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 4,
          }}
        >
          <Dot
            color={
              condition.status === "active"
                ? "var(--accent)"
                : "rgba(255,255,255,0.35)"
            }
            size={8}
            glow={condition.status === "active"}
          />
          <Mono size={10} color="rgba(255,255,255,0.55)">
            {condition.status.toUpperCase()}
          </Mono>
        </div>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Onset</span>
        <Mono size={12} color="rgba(255,255,255,0.85)">
          {condition.onset}
        </Mono>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>ICD-10</span>
        <Mono size={12} color="rgba(255,255,255,0.85)">
          {condition.icd}
        </Mono>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>
          Recent encounters · {related.length}
        </span>
        {related.length === 0 ? (
          <Mono size={11} color="rgba(255,255,255,0.45)">
            No recent encounters reference this condition.
          </Mono>
        ) : (
          <div style={{ display: "flex", flexDirection: "column" }}>
            {related.slice(0, 6).map((e, i) => (
              <div
                key={`${e.date}-${i}`}
                style={{
                  display: "flex",
                  gap: 10,
                  padding: "6px 0",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  fontSize: 12,
                  alignItems: "baseline",
                }}
              >
                <Mono
                  size={10}
                  color="rgba(255,255,255,0.45)"
                  style={{ width: 70, flexShrink: 0 }}
                >
                  {e.date}
                </Mono>
                <span style={{ color: "#fff", flex: 1 }}>{e.title}</span>
                <Mono size={10} color="rgba(255,255,255,0.45)">
                  {e.who}
                </Mono>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CareTeamDrawerBody({
  member,
  onClose,
}: {
  member: SampleCareTeamMember;
  onClose: () => void;
}) {
  const [subject, setSubject] = useState(`Question for ${member.name}`);
  const [body, setBody] = useState("");

  const inputBase: CSSProperties = {
    width: "100%",
    background: "#000",
    color: "#fff",
    border: "1px solid rgba(255,255,255,0.12)",
    padding: "8px 10px",
    fontFamily: "var(--font-display)",
    fontSize: 13,
    boxSizing: "border-box",
  };

  // Format ISO dates ("2026-10-21") as "Oct 21, 2026". Locale-en for the
  // demo; if i18n lands later, swap to Intl with the active locale.
  const apptDateLabel = member.nextAppointment
    ? new Date(`${member.nextAppointment.date}T00:00:00`).toLocaleDateString(
        "en-US",
        { year: "numeric", month: "short", day: "numeric" },
      )
    : null;

  return (
    <div>
      <div style={drawerSection}>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <Avatar initials={member.avatar} size={40} online={member.online} />
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>{member.name}</div>
            <Mono size={10} color="rgba(255,255,255,0.55)">
              {member.role.toUpperCase()} · {member.org}
            </Mono>
          </div>
        </div>
      </div>

      {/* Next appointment — always rendered (renders "(none scheduled)"
          when the member has no nextAppointment). The Reschedule button
          is intentionally disabled in demo mode; the tooltip explains. */}
      <div style={drawerSection}>
        <span style={drawerLabel}>Next appointment</span>
        {member.nextAppointment ? (
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.03)",
              padding: "10px 12px",
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#fff" }}>
                {apptDateLabel}
              </div>
              <div
                style={{
                  marginTop: 2,
                  fontSize: 12,
                  color: "rgba(255,255,255,0.7)",
                  fontFamily: "var(--font-display)",
                }}
              >
                {member.nextAppointment.type}
              </div>
            </div>
            <Tooltip
              label="DEMO MODE"
              range="Reschedule disabled"
              hint="Demo mode — reschedule lands when SMART scheduling is wired."
              source="4UWHAt · sample data"
            >
              <button
                type="button"
                disabled
                aria-disabled="true"
                style={{
                  background: "rgba(255,255,255,0.08)",
                  color: "rgba(255,255,255,0.45)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  padding: "6px 10px",
                  fontFamily: "var(--font-display)",
                  fontWeight: 600,
                  fontSize: 11,
                  cursor: "not-allowed",
                }}
              >
                Reschedule
              </button>
            </Tooltip>
          </div>
        ) : (
          <div
            style={{
              border: "1px dashed rgba(255,255,255,0.12)",
              padding: "10px 12px",
              fontSize: 12,
              color: "rgba(255,255,255,0.5)",
              fontStyle: "italic",
            }}
          >
            (none scheduled)
          </div>
        )}
      </div>

      {/* Items to watch — bullet list of open clinical threads sourced
          from sample-data.ts. Hidden only when the array is empty (an
          empty list adds noise; absent appointment is meaningful, but
          "no items to watch" reads as a status the team didn't author). */}
      {member.itemsToWatch && member.itemsToWatch.length > 0 && (
        <div style={drawerSection}>
          <span style={drawerLabel}>Items to watch</span>
          <ul
            style={{
              listStyle: "none",
              padding: 0,
              margin: 0,
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            {member.itemsToWatch.map((item, i) => (
              <li
                key={i}
                style={{
                  display: "flex",
                  gap: 10,
                  fontSize: 12.5,
                  lineHeight: 1.45,
                  color: "rgba(255,255,255,0.85)",
                }}
              >
                <span
                  aria-hidden
                  style={{
                    color: "var(--accent)",
                    flexShrink: 0,
                    marginTop: 2,
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  ◆
                </span>
                <span style={{ flex: 1 }}>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div style={drawerSection}>
        <span style={drawerLabel}>Subject</span>
        <input
          type="text"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          style={inputBase}
        />
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Message</span>
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={6}
          placeholder="Write a short note. Don't include sensitive details until SMART OAuth ships."
          style={{ ...inputBase, resize: "vertical", minHeight: 120 }}
        />
      </div>

      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "center",
          marginTop: 18,
        }}
      >
        <Tooltip
          label="DEMO MODE"
          range="Send disabled"
          hint="Demo mode — enable when SMART OAuth lands."
          source="4UWHAt · sample data"
        >
          <button
            type="button"
            disabled
            style={{
              background: "rgba(255,255,255,0.08)",
              color: "rgba(255,255,255,0.45)",
              border: "1px solid rgba(255,255,255,0.12)",
              padding: "8px 14px",
              fontFamily: "var(--font-display)",
              fontWeight: 600,
              fontSize: 12,
              cursor: "not-allowed",
            }}
          >
            Send (demo)
          </button>
        </Tooltip>
        <button
          type="button"
          onClick={onClose}
          style={{
            background: "transparent",
            color: "rgba(255,255,255,0.7)",
            border: "1px solid #2a2a2a",
            padding: "8px 14px",
            fontFamily: "var(--font-display)",
            fontWeight: 600,
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          Discard
        </button>
      </div>
    </div>
  );
}

export function RecordsOS() {
  const [activeModule, setActiveModule] = useState("overview");
  const [drawer, setDrawer] = useState<DrawerTarget>(null);
  const [shareOpen, setShareOpen] = useState(false);
  const closeDrawer = () => setDrawer(null);

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

  // Persisted column widths. SSR-safe: we render the design defaults
  // first, then hydrate the user's stored sizes from localStorage on
  // mount via the effect below. Avoids a server/client mismatch.
  const COL_LEFT_DEFAULT = "220px";
  const COL_RIGHT_DEFAULT = "360px";
  useEffect(() => {
    const grid = document.querySelector(
      "[data-records-os-grid]",
    ) as HTMLElement | null;
    if (!grid) return;
    grid.style.setProperty(
      "--col-left",
      loadPersistedVar("4uwhat:col-left", COL_LEFT_DEFAULT),
    );
    grid.style.setProperty(
      "--col-right",
      loadPersistedVar("4uwhat:col-right", COL_RIGHT_DEFAULT),
    );
  }, []);

  return (
    <div
      data-records-os-grid
      style={
        {
          background: "#000",
          color: "#fff",
          display: "grid",
          // Two 4px tracks for the drag handles between rails.
          gridTemplateColumns:
            "var(--col-left) 4px minmax(0, 1fr) 4px var(--col-right)",
          // CSS variable defaults — overridden in the mount effect once
          // localStorage is available.
          ["--col-left" as string]: COL_LEFT_DEFAULT,
          ["--col-right" as string]: COL_RIGHT_DEFAULT,
          fontFamily: "var(--font-display)",
          minHeight: "calc(100vh - 40px)",
          borderTop: "1px solid #1f1f1f",
        } as CSSProperties
      }
    >
      {/* ── LEFT RAIL ───────────────────────────────────────────────── */}
      <aside
        style={{
          borderRight: "1px solid #1f1f1f",
          display: "flex",
          flexDirection: "column",
          background: "#000",
        }}
      >
        <div style={{ padding: "20px 20px 16px", borderBottom: "1px solid #1f1f1f" }}>
          <Wordmark size={13} />
          <Mono size={9} style={{ marginTop: 10, display: "block" }} color="rgba(255,255,255,0.35)">
            HEALTH / v4.2
          </Mono>
        </div>

        {/* Patient picker */}
        <div style={{ padding: "12px 20px", borderBottom: "1px solid #1f1f1f" }}>
          <Mono size={9} color="rgba(255,255,255,0.4)" style={{ marginBottom: 6, display: "block" }}>
            PATIENT
          </Mono>
          <PatientPicker />
        </div>

        {/* Patient identity */}
        <div style={{ padding: "16px 20px", borderBottom: "1px solid #1f1f1f" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
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
            <div>
              <div style={{ fontWeight: 600, fontSize: 13, lineHeight: 1.2 }}>{patient.name}</div>
              <Mono size={9} color="rgba(255,255,255,0.4)">
                {patient.mrn}
              </Mono>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ padding: "8px 8px", flex: 1, overflowY: "auto" }}>
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
        <div style={{ padding: "12px 20px", borderTop: "1px solid #1f1f1f" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Dot color="var(--accent)" glow={true} size={6} />
            <Mono size={9}>SYNCED · 2 MIN AGO</Mono>
          </div>
        </div>
      </aside>

      {/* Drag handle between left rail and main content */}
      <Resizer
        targetSelector="[data-records-os-grid]"
        varName="--col-left"
        edge="right"
        min={180}
        max={280}
        storageKey="4uwhat:col-left"
        label="Resize navigation rail"
      />

      {/* ── MAIN ────────────────────────────────────────────────────── */}
      <main style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
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
            {/* B2 — Single working Share button. Replaces the prior
                cosmetic Export+Share pair. Opens ShareDrawer with a
                live FHIR R4 Bundle assembled from sample-data. */}
            <button
              type="button"
              style={btnPrimary}
              onClick={() => setShareOpen(true)}
              aria-label="Share your FHIR record"
            >
              Share
            </button>
            <Tooltip
              label="DEMO MODE"
              range="New entry disabled"
              hint="v1.1: surfaces FHIR data-entry forms (vital, lab, message). Sharing the existing record works today via Share → above."
              source="medomni · sample data"
            >
              <button
                type="button"
                disabled
                aria-disabled="true"
                style={{ ...btnGhost, cursor: "not-allowed", opacity: 0.55 }}
              >
                + New entry (demo)
              </button>
            </Tooltip>
          </div>
        </div>

        {/* Content scroll area */}
        <div style={{ padding: "24px 28px", flex: 1, minHeight: 0 }}>
          {/* Hero row */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 1fr)",
              gap: 16,
              marginBottom: 16,
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
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6, minmax(0, 1fr))" }}>
              {Object.entries(vitals).map(([key, v], i) => {
                // Build trend hint: last 7 readings with relative time labels.
                const last7 = [...v.spark]
                  .map((value, idx) => ({
                    value,
                    tMinus: v.spark.length - 1 - idx,
                  }))
                  .reverse()
                  .slice(0, 7)
                  .map((row) =>
                    row.tMinus === 0
                      ? `${row.value} (now)`
                      : `${row.value} (t-${row.tMinus})`,
                  )
                  .join(" ← ");
                return (
                  <Clickable
                    key={key}
                    onClick={() => setDrawer({ kind: "vital", id: key })}
                    ariaLabel={`Open ${v.label} details`}
                    style={{
                      borderRight:
                        i < 5 ? "1px solid rgba(255,255,255,0.06)" : "none",
                    }}
                  >
                    <div style={{ padding: "16px 20px" }}>
                      <Mono size={9}>{v.label.toUpperCase()}</Mono>
                      <Tooltip
                        label={v.label}
                        range={v.range}
                        hint={`${v.hint}\nLast 7: ${last7}`}
                        source={`${v.source} · click to expand`}
                      >
                        <span
                          style={{
                            display: "flex",
                            alignItems: "baseline",
                            gap: 5,
                            marginTop: 6,
                          }}
                        >
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
                      <div
                        style={{
                          marginTop: 8,
                          height: 22,
                          color: "var(--accent)",
                        }}
                      >
                        <Sparkline
                          data={v.spark}
                          w={120}
                          h={22}
                          stroke="var(--accent)"
                          strokeWidth={1.25}
                        />
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
                  </Clickable>
                );
              })}
            </div>
          </div>

          {/* Labs + Meds + Conditions row */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)",
              gap: 16,
              marginBottom: 16,
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
                {labs.map((l) => {
                  // Build a richer hover hint: trend values paired with dates
                  // (or T-N labels), most-recent first.
                  const trendLine = [...l.trend]
                    .map((v, i) => ({
                      v,
                      d: l.dates?.[i] ?? `T-${l.trend.length - 1 - i}`,
                    }))
                    .reverse()
                    .slice(0, 4)
                    .map((row) => `${row.v} (${row.d})`)
                    .join(" ← ");
                  return (
                    <Clickable
                      key={l.id}
                      onClick={() => setDrawer({ kind: "lab", id: l.id })}
                      ariaLabel={`Open ${l.name} details`}
                      style={{ display: "block" }}
                    >
                      <div style={tableRow}>
                        <Tooltip
                          label={l.name}
                          range={`Reference: ${l.range} ${l.unit}`}
                          hint={`${l.hint}\nTrend: ${trendLine}`}
                          source={`${l.source} · click to expand`}
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
                            <Tag
                              color="#ffaa00"
                              style={{ fontSize: 8.5, padding: "2px 5px" }}
                            >
                              LOW
                            </Tag>
                          )}
                        </span>
                      </div>
                    </Clickable>
                  );
                })}
              </div>
            </div>

            {/* Meds + conditions stacked */}
            <div style={{ display: "grid", gridTemplateRows: "1fr 1fr", gap: 16 }}>
              <div style={cardBase}>
                <Eyebrow style={{ marginBottom: 12 }}>MEDICATIONS · 4 ACTIVE</Eyebrow>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {meds.slice(0, 4).map((m) => (
                    <Clickable
                      key={m.id}
                      onClick={() => setDrawer({ kind: "med", id: m.id })}
                      ariaLabel={`Open ${m.name} details`}
                    >
                      <div style={medRow}>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <Tooltip
                            label={m.name}
                            range={`${m.dose} · ${m.freq}`}
                            hint={`Prescriber: ${m.prescriber} · since ${m.since}${
                              m.adherence != null
                                ? ` · adherence ${Math.round(m.adherence * 100)}%`
                                : ""
                            }`}
                            source={
                              m.refills != null
                                ? `${m.refills} refills · click to expand`
                                : "OTC · click to expand"
                            }
                          >
                            <span style={{ fontWeight: 600, fontSize: 13 }}>
                              {m.name}
                            </span>
                          </Tooltip>
                          <Mono
                            size={9}
                            style={{ marginTop: 2, display: "block" }}
                          >
                            {m.dose} · {m.freq}
                          </Mono>
                        </div>
                        {m.adherence != null && (
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                            }}
                          >
                            <Mono size={9}>
                              {Math.round(m.adherence * 100)}%
                            </Mono>
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
                            color={
                              m.refills <= 1 ? "#ffaa00" : "rgba(255,255,255,0.45)"
                            }
                          >
                            {m.refills} REFILL{m.refills !== 1 ? "S" : ""}
                          </Mono>
                        )}
                      </div>
                    </Clickable>
                  ))}
                </div>
              </div>

              <div style={cardBase}>
                <Eyebrow style={{ marginBottom: 12 }}>PROBLEM LIST</Eyebrow>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  {conditions.map((c) => (
                    <Clickable
                      key={c.id}
                      onClick={() =>
                        setDrawer({ kind: "condition", id: c.id })
                      }
                      ariaLabel={`Open ${c.name} details`}
                    >
                      <Tooltip
                        label={c.name}
                        range={`ICD-10 ${c.icd} · onset ${c.onset}`}
                        hint={`Status: ${c.status}. Click to see related encounters.`}
                        source="Problem list"
                      >
                        <span
                          style={{
                            display: "grid",
                            gridTemplateColumns: "12px minmax(0, 1fr) auto auto",
                            alignItems: "center",
                            gap: 10,
                            padding: "8px 0",
                            borderBottom:
                              "1px solid rgba(255,255,255,0.05)",
                            width: "100%",
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
                              color:
                                c.status === "active"
                                  ? "#fff"
                                  : "rgba(255,255,255,0.5)",
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
                        </span>
                      </Tooltip>
                    </Clickable>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Bottom row: timeline preview + sharing */}
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)", gap: 16 }}>
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
                    <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>{e.title}</span>
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
                      padding: "10px 12px",
                      border: "1px solid rgba(255,255,255,0.06)",
                    }}
                  >
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 500 }}>{s.who}</div>
                      <Mono size={9}>{s.scope.toUpperCase()}</Mono>
                    </div>
                    <button type="button" style={{ ...btnGhost, padding: "4px 8px", fontSize: 10 }}>
                      REVOKE
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Drag handle between main content and right rail */}
      <Resizer
        targetSelector="[data-records-os-grid]"
        varName="--col-right"
        edge="left"
        min={280}
        max={560}
        storageKey="4uwhat:col-right"
        label="Resize AI rail"
      />

      {/* ── RIGHT RAIL — AI command bar + activity ─────────────────── */}
      <aside
        style={{
          borderLeft: "1px solid #1f1f1f",
          display: "flex",
          flexDirection: "column",
          background: "#000",
        }}
      >
        {/* Ask your record (AI) */}
        <div style={{ padding: "20px 22px", borderBottom: "1px solid #1f1f1f" }}>
          <AskYourRecord suggestions={SAMPLE_AI_SUGGESTIONS} />
        </div>

        {/* Care team */}
        <div style={{ padding: "18px 22px", borderBottom: "1px solid #1f1f1f" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
            <Eyebrow>CARE TEAM</Eyebrow>
            <Mono>{careTeam.filter((c) => c.online).length} ONLINE</Mono>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {careTeam.map((c) => (
              <Clickable
                key={c.id}
                onClick={() => setDrawer({ kind: "careTeam", id: c.id })}
                ariaLabel={`Message ${c.name}`}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "4px 6px",
                  }}
                >
                  <Tooltip
                    label={c.name}
                    range={`${c.role} · ${c.org}`}
                    hint={
                      c.online
                        ? "Online now. Click to draft a message."
                        : "Offline. Click to draft a message."
                    }
                    source={c.online ? "Available" : "Last seen recently"}
                  >
                    <Avatar
                      initials={c.avatar}
                      online={c.online}
                      size={28}
                    />
                  </Tooltip>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        lineHeight: 1.3,
                      }}
                    >
                      {c.name}
                    </div>
                    <Mono size={9}>{c.role.toUpperCase()}</Mono>
                  </div>
                  <span
                    style={{
                      ...btnGhost,
                      padding: "3px 7px",
                      fontSize: 10,
                      // Render as a static label inside the Clickable wrapper
                      // (nested <button> inside <button> is invalid HTML).
                      pointerEvents: "none",
                    }}
                  >
                    MSG
                  </span>
                </div>
              </Clickable>
            ))}
          </div>
        </div>

        {/* Recent activity */}
        <div style={{ padding: "18px 22px", flex: 1, overflowY: "auto" }}>
          <Eyebrow style={{ marginBottom: 12 }}>RECENT ACTIVITY</Eyebrow>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <ActivityRow
              time="2 min"
              actor="4UWHAt"
              verb="synced"
              obj="Apple Health · 47 entries"
              onClick={() => setDrawer({ kind: "vital", id: "hr" })}
            />
            <ActivityRow
              time="11 min"
              actor="Dr. Patel"
              verb="signed"
              obj="Visit note · Feb 19"
              accent
            />
            <ActivityRow
              time="4h"
              actor="4UWHAt"
              verb="flagged"
              obj="Vitamin D · 28 ng/mL · low"
              onClick={() => setDrawer({ kind: "lab", id: "vitd" })}
            />
            <ActivityRow
              time="1d"
              actor="Quest"
              verb="released"
              obj="Lipid panel + CBC + CMP"
              onClick={() => setDrawer({ kind: "lab", id: "ldl" })}
            />
            <ActivityRow
              time="2d"
              actor="You"
              verb="shared"
              obj="Pulmonary record → Dr. Patel"
            />
            <ActivityRow
              time="6d"
              actor="CVS"
              verb="filled"
              obj="Rosuvastatin · 30 day"
              onClick={() => setDrawer({ kind: "med", id: "rosu" })}
            />
          </div>
        </div>

        <div style={{ height: 4, background: "var(--accent)" }} />
      </aside>

      {/* ── DETAIL DRAWER ─────────────────────────────────────────────
        Renders into a portal at document.body, so it does not perturb
        the three-column grid layout (A1's domain). One drawer at a time;
        target selected by `drawer` state above. */}
      {drawer?.kind === "lab" &&
        (() => {
          const lab = labs.find((l) => l.id === drawer.id);
          if (!lab) return null;
          return (
            <DetailDrawer
              open={true}
              onClose={closeDrawer}
              title={lab.name}
              kicker={`Lab · ${lab.date} · ${lab.flag.toUpperCase()}`}
            >
              <LabDrawerBody lab={lab} />
            </DetailDrawer>
          );
        })()}
      {drawer?.kind === "vital" &&
        (() => {
          const vital = vitals[drawer.id];
          if (!vital) return null;
          return (
            <DetailDrawer
              open={true}
              onClose={closeDrawer}
              title={vital.label}
              kicker={`Vital · ${vital.source}`}
            >
              <VitalDrawerBody vital={vital} />
            </DetailDrawer>
          );
        })()}
      {drawer?.kind === "med" &&
        (() => {
          const med = meds.find((m) => m.id === drawer.id);
          if (!med) return null;
          return (
            <DetailDrawer
              open={true}
              onClose={closeDrawer}
              title={med.name}
              kicker={`Medication · since ${med.since}`}
            >
              <MedDrawerBody med={med} />
            </DetailDrawer>
          );
        })()}
      {drawer?.kind === "condition" &&
        (() => {
          const cond = conditions.find((c) => c.id === drawer.id);
          if (!cond) return null;
          return (
            <DetailDrawer
              open={true}
              onClose={closeDrawer}
              title={cond.name}
              kicker={`Condition · ${cond.icd}`}
            >
              <ConditionDrawerBody condition={cond} timeline={timeline} />
            </DetailDrawer>
          );
        })()}
      {drawer?.kind === "careTeam" &&
        (() => {
          const member = careTeam.find((c) => c.id === drawer.id);
          if (!member) return null;
          return (
            <DetailDrawer
              open={true}
              onClose={closeDrawer}
              title={`Message ${member.name}`}
              kicker="Demo · SMART OAuth pending"
            >
              <CareTeamDrawerBody member={member} onClose={closeDrawer} />
            </DetailDrawer>
          );
        })()}

      {/* B2 — FHIR R4 Bundle share surface. Independent of the
          per-entity DetailDrawer state above so the user can open
          Share without losing a focused detail context. */}
      <ShareDrawer open={shareOpen} onClose={() => setShareOpen(false)} />
    </div>
  );
}
