"use client";

// 4UWHAt — ImagingPanel
// Click-to-view gallery of imaging studies. Renders SAMPLE_IMAGING (or any
// SampleImaging[]) as 3 cards in a 1fr × 3 grid above the fold; each card
// opens a DetailDrawer with the full read + a larger version of the same
// stylized SVG graphic.
//
// Hard constraints honored:
//   - No new SampleImaging fields. Facility, study-id, and the synthetic
//     intensity histogram are derived locally from the existing shape.
//   - No real DICOM imagery (IP/license risk). All graphics are pure
//     inline SVG, magenta-tinted line art on the dark canvas.
//   - Inline-style + token pattern matches the rest of RecordsOS;
//     no new globals introduced.
//
// "Reorder" is intentionally cosmetic + disabled (Tooltip explains: "Demo
// mode"). The real ordering flow ships with the booking wave.

import {
  useMemo,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

import { DetailDrawer } from "./DetailDrawer";
import { Eyebrow } from "./Eyebrow";
import { Mono } from "./Mono";
import { Sparkline } from "./Sparkline";
import { Tooltip } from "./Tooltip";
import type { SampleImaging } from "@/lib/4uwhat/sample-data";

// ── Modality classification ────────────────────────────────────────────
// Map free-text `kind` (X-ray / MRI / Panoramic / future Ultrasound /
// CT) onto a small enum used by the SVG renderer. Anything we don't
// recognize falls back to a generic scan-frame placeholder so a future
// modality added by track D doesn't crash the gallery.

type ModalityKey = "xray" | "mri" | "panoramic" | "default";

function classifyModality(kind: string): ModalityKey {
  const k = kind.toLowerCase();
  if (k.includes("x-ray") || k.includes("xray") || k === "x ray") return "xray";
  if (k.includes("mri")) return "mri";
  if (k.includes("panoramic") || k.includes("dental")) return "panoramic";
  return "default";
}

// ── Reference imagery ──────────────────────────────────────────────────
// Public-domain / CC reference images bundled in /public/4uwhat-demo/imaging/.
// These are NOT this patient's actual films — they're representative
// reference images sourced from Wikimedia Commons and labeled as such in
// the drawer. The ScanGraphic component overlays a magenta crosshair
// frame so the look stays consistent with the rest of Records OS.
//
// Sources (all licenses honored in the drawer footer):
//   cxr-pa.jpg   — CC0 by Mikael Häggström, Wikimedia Commons
//   mri-knee.jpg — CC BY-SA 4.0 by Ptrump16, Wikimedia Commons
//   panoramic.jpg — CC BY 3.0 (Wikimedia Commons)

import { BASE_PATH } from "@/lib/basePath";

const IMAGE_HREF: Record<ModalityKey, string | null> = {
  xray: `${BASE_PATH}/4uwhat-demo/imaging/cxr-pa.jpg`,
  mri: `${BASE_PATH}/4uwhat-demo/imaging/mri-knee.jpg`,
  panoramic: `${BASE_PATH}/4uwhat-demo/imaging/panoramic.jpg`,
  default: null,
};

const IMAGE_CREDIT: Record<ModalityKey, string | null> = {
  xray: "Reference image: PA chest radiograph, CC0 (Mikael Häggström, Wikimedia Commons)",
  mri: "Reference image: T1 TSE sagittal knee MRI, CC BY-SA 4.0 (Ptrump16, Wikimedia Commons)",
  panoramic: "Reference image: panoramic dental radiograph, CC BY 3.0 (Wikimedia Commons)",
  default: null,
};

// ── Scan graphic ───────────────────────────────────────────────────────
// Renders the public-domain reference image inside a magenta crosshair
// frame at any size. Stylized SVG glyph functions below are kept as a
// fallback for modalities without a bundled image.

interface ScanGraphicProps {
  modality: ModalityKey;
  w?: number;
  h?: number;
}

function ScanGraphic({ modality, w = 200, h = 150 }: ScanGraphicProps) {
  const href = IMAGE_HREF[modality];
  return (
    <svg
      viewBox="0 0 200 150"
      width={w}
      height={h}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`${modality} reference image`}
      style={{ display: "block" }}
    >
      {/* Frame: dark canvas */}
      <rect
        x="0.5"
        y="0.5"
        width="199"
        height="149"
        fill="#0b0b0b"
        stroke="rgba(255,255,255,0.06)"
      />

      {/* Real reference image, centered, contain-fit */}
      {href ? (
        <image
          href={href}
          x="6"
          y="6"
          width="188"
          height="138"
          preserveAspectRatio="xMidYMid meet"
        />
      ) : null}

      {/* Crosshair corners — two short strokes per corner */}
      {(
        [
          [4, 4, 14, 4, 4, 4, 4, 14],
          [196, 4, 186, 4, 196, 4, 196, 14],
          [4, 146, 14, 146, 4, 146, 4, 136],
          [196, 146, 186, 146, 196, 146, 196, 136],
        ] as const
      ).map((c, i) => (
        <g key={`c-${i}`} stroke="var(--accent)" strokeWidth="1" opacity="0.7">
          <line x1={c[0]} y1={c[1]} x2={c[2]} y2={c[3]} />
          <line x1={c[4]} y1={c[5]} x2={c[6]} y2={c[7]} />
        </g>
      ))}

      {/* Stylized glyph fallback for modalities without a bundled image */}
      {!href && modality === "xray" ? <XrayGlyph /> : null}
      {!href && modality === "mri" ? <MriGlyph /> : null}
      {!href && modality === "panoramic" ? <PanoramicGlyph /> : null}
      {!href && modality === "default" ? <DefaultGlyph /> : null}
    </svg>
  );
}

function XrayGlyph() {
  // Stylized rib silhouette — paired curved lines on either side of the
  // sternum, with a magenta "lung field" overlay. Monochrome, thin
  // strokes — evocative, not literal.
  const ribs: ReactNode[] = [];
  for (let i = 0; i < 6; i += 1) {
    const y = 50 + i * 11;
    // Left side ribs
    ribs.push(
      <path
        key={`rl-${i}`}
        d={`M 100 ${y} Q 70 ${y + 6} 50 ${y + 14}`}
        fill="none"
        stroke="rgba(255,255,255,0.45)"
        strokeWidth="1"
      />,
    );
    // Right side ribs
    ribs.push(
      <path
        key={`rr-${i}`}
        d={`M 100 ${y} Q 130 ${y + 6} 150 ${y + 14}`}
        fill="none"
        stroke="rgba(255,255,255,0.45)"
        strokeWidth="1"
      />,
    );
  }
  return (
    <g>
      {/* Magenta lung-field tint */}
      <ellipse
        cx="78"
        cy="80"
        rx="22"
        ry="32"
        fill="rgba(255,0,150,0.10)"
        stroke="var(--accent)"
        strokeWidth="0.75"
        opacity="0.85"
      />
      <ellipse
        cx="122"
        cy="80"
        rx="22"
        ry="32"
        fill="rgba(255,0,150,0.10)"
        stroke="var(--accent)"
        strokeWidth="0.75"
        opacity="0.85"
      />
      {/* Sternum */}
      <line
        x1="100"
        y1="44"
        x2="100"
        y2="120"
        stroke="rgba(255,255,255,0.35)"
        strokeWidth="1.2"
      />
      {/* Spine column dots */}
      {[48, 60, 72, 84, 96, 108, 120].map((y) => (
        <circle
          key={`sp-${y}`}
          cx="100"
          cy={y}
          r="2"
          fill="rgba(255,255,255,0.55)"
        />
      ))}
      {ribs}
    </g>
  );
}

function MriGlyph() {
  // Cross-section circle with concentric magenta gradients — evocative
  // of a single MRI slice. No anatomical fidelity claimed.
  return (
    <g>
      <circle
        cx="100"
        cy="75"
        r="50"
        fill="rgba(255,255,255,0.04)"
        stroke="rgba(255,255,255,0.25)"
        strokeWidth="1"
      />
      <circle
        cx="100"
        cy="75"
        r="40"
        fill="none"
        stroke="rgba(255,0,150,0.55)"
        strokeWidth="0.75"
      />
      <circle
        cx="100"
        cy="75"
        r="30"
        fill="none"
        stroke="rgba(255,0,150,0.45)"
        strokeWidth="0.75"
      />
      <circle
        cx="100"
        cy="75"
        r="20"
        fill="none"
        stroke="rgba(255,0,150,0.6)"
        strokeWidth="0.75"
      />
      <circle
        cx="100"
        cy="75"
        r="10"
        fill="rgba(255,0,150,0.18)"
        stroke="var(--accent)"
        strokeWidth="0.75"
      />
      {/* Slice marker */}
      <line
        x1="50"
        y1="75"
        x2="150"
        y2="75"
        stroke="rgba(255,0,150,0.35)"
        strokeWidth="0.5"
        strokeDasharray="3 3"
      />
      <line
        x1="100"
        y1="25"
        x2="100"
        y2="125"
        stroke="rgba(255,0,150,0.35)"
        strokeWidth="0.5"
        strokeDasharray="3 3"
      />
    </g>
  );
}

function PanoramicGlyph() {
  // Jaw arc with tooth-tick marks — symmetric upper + lower arch.
  const upperTicks: ReactNode[] = [];
  const lowerTicks: ReactNode[] = [];
  // 14 ticks across the upper jaw (180° → 0°), plus matching lower.
  for (let i = 0; i <= 14; i += 1) {
    const t = i / 14;
    const angle = Math.PI * (1 - t); // π → 0
    const cxArc = 100;
    const cyArc = 90;
    const rxArc = 70;
    const ryArc = 32;
    const x = cxArc + Math.cos(angle) * rxArc;
    const yU = cyArc - Math.sin(angle) * ryArc;
    const yL = cyArc + Math.sin(angle) * (ryArc * 0.85);
    upperTicks.push(
      <line
        key={`tu-${i}`}
        x1={x}
        y1={yU}
        x2={x}
        y2={yU - 6}
        stroke="rgba(255,255,255,0.7)"
        strokeWidth="1.2"
      />,
    );
    lowerTicks.push(
      <line
        key={`tl-${i}`}
        x1={x}
        y1={yL}
        x2={x}
        y2={yL + 6}
        stroke="rgba(255,255,255,0.7)"
        strokeWidth="1.2"
      />,
    );
  }
  return (
    <g>
      {/* Magenta arc backplate */}
      <path
        d="M 30 90 A 70 32 0 0 1 170 90"
        fill="none"
        stroke="rgba(255,0,150,0.45)"
        strokeWidth="1.2"
      />
      <path
        d="M 30 90 A 70 27 0 0 0 170 90"
        fill="none"
        stroke="rgba(255,0,150,0.45)"
        strokeWidth="1.2"
      />
      {/* Centerline */}
      <line
        x1="100"
        y1="55"
        x2="100"
        y2="125"
        stroke="rgba(255,255,255,0.18)"
        strokeWidth="0.5"
        strokeDasharray="2 3"
      />
      {upperTicks}
      {lowerTicks}
    </g>
  );
}

function DefaultGlyph() {
  // Generic scan-frame placeholder: empty grid with a magenta scan line.
  const lines: ReactNode[] = [];
  for (let i = 1; i < 8; i += 1) {
    const x = (200 / 8) * i;
    lines.push(
      <line
        key={`gx-${i}`}
        x1={x}
        y1="20"
        x2={x}
        y2="130"
        stroke="rgba(255,255,255,0.05)"
        strokeWidth="0.5"
      />,
    );
  }
  for (let i = 1; i < 6; i += 1) {
    const y = 20 + (110 / 6) * i;
    lines.push(
      <line
        key={`gy-${i}`}
        x1="20"
        y1={y}
        x2="180"
        y2={y}
        stroke="rgba(255,255,255,0.05)"
        strokeWidth="0.5"
      />,
    );
  }
  return (
    <g>
      {lines}
      <line
        x1="20"
        y1="75"
        x2="180"
        y2="75"
        stroke="var(--accent)"
        strokeWidth="1"
        opacity="0.7"
      />
      <text
        x="100"
        y="110"
        textAnchor="middle"
        fill="rgba(255,255,255,0.35)"
        fontFamily="var(--font-mono)"
        fontSize="9"
        letterSpacing="2"
      >
        SCAN
      </text>
    </g>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────

function formatDate(iso: string): string {
  // "2026-03-08" → "Mar 8, 2026". Locale-stable ASCII only.
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return iso;
  const yr = m[1];
  const mo = months[parseInt(m[2], 10) - 1] ?? m[2];
  const day = parseInt(m[3], 10);
  return `${mo} ${day}, ${yr}`;
}

function truncate(text: string, max = 60): string {
  if (text.length <= max) return text;
  // Trim back to the last word boundary that still fits.
  const slice = text.slice(0, max - 1);
  const lastSpace = slice.lastIndexOf(" ");
  return `${slice.slice(0, lastSpace > 30 ? lastSpace : slice.length).trimEnd()}…`;
}

// Synthetic facility lookup — derived from the existing study id so we
// don't add fields to SampleImaging. Track D may later replace this with
// a real `facility` field; this map will lose to that without conflict.
const FACILITY_BY_ID: Record<string, string> = {
  cxr: "Riverside Imaging",
  mri: "Westside MRI",
  pan: "Cohen Oral Surgery",
};

function facilityFor(id: string, kind: string): string {
  if (FACILITY_BY_ID[id]) return FACILITY_BY_ID[id];
  // Fallback by modality so unknown ids still get a plausible label.
  const m = classifyModality(kind);
  if (m === "xray") return "Riverside Imaging";
  if (m === "mri") return "Westside MRI";
  if (m === "panoramic") return "Cohen Oral Surgery";
  return "External Imaging";
}

// Synthetic per-study intensity histogram (12 bins). Pure visual flair
// so the drawer feels like a scan viewer rather than a paragraph dump.
// Seeded by the study id char codes — stable across renders, no data.
function syntheticHistogram(seed: string): number[] {
  let h = 0;
  for (let i = 0; i < seed.length; i += 1) h = (h * 31 + seed.charCodeAt(i)) | 0;
  const out: number[] = [];
  for (let i = 0; i < 12; i += 1) {
    h = (h * 1103515245 + 12345) | 0;
    const v = Math.abs(h % 100);
    out.push(20 + v * 0.8);
  }
  return out;
}

// ── Internal Clickable wrapper ─────────────────────────────────────────
// Mirrors RecordsOS's internal Clickable pattern (cursor:pointer +
// magenta hover tint) without exporting it. Keeps ImagingPanel
// self-contained so the public barrel only widens by ImagingPanel.

interface ClickableCardProps {
  onClick: () => void;
  ariaLabel: string;
  children: ReactNode;
  style?: CSSProperties;
}

function ClickableCard({
  onClick,
  ariaLabel,
  children,
  style,
}: ClickableCardProps) {
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
        all: "unset",
        boxSizing: "border-box",
        cursor: "pointer",
        display: "block",
        width: "100%",
        background: hover
          ? "rgba(255,0,150,0.06)"
          : "var(--p42-bg-card, #0e0e0e)",
        outline: hover
          ? "1px solid rgba(255,0,150,0.35)"
          : "1px solid rgba(255,255,255,0.07)",
        transition: "background 120ms ease, outline-color 120ms ease",
        ...style,
      }}
    >
      {children}
    </button>
  );
}

// ── Card ───────────────────────────────────────────────────────────────

interface ImagingCardProps {
  study: SampleImaging;
  onClick: () => void;
}

function ImagingCard({ study, onClick }: ImagingCardProps) {
  const modality = classifyModality(study.kind);
  return (
    <ClickableCard
      onClick={onClick}
      ariaLabel={`${study.kind} of ${study.region}, ${formatDate(study.date)} — open details`}
    >
      <div
        style={{
          padding: 14,
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {/* SVG canvas */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            background: "#070707",
            border: "1px solid rgba(255,255,255,0.05)",
            padding: "10px 8px",
          }}
        >
          <ScanGraphic modality={modality} w={200} h={150} />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <Mono size={9} color="var(--accent)">
            {study.kind.toUpperCase()} · {study.region.toUpperCase()}
          </Mono>
          <div
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 13,
              fontWeight: 600,
              lineHeight: 1.3,
              color: "rgba(255,255,255,0.95)",
            }}
          >
            {formatDate(study.date)}
          </div>
          <Mono size={10} color="rgba(255,255,255,0.55)">
            {study.radiologist}
          </Mono>
        </div>

        <p
          style={{
            margin: 0,
            fontSize: 12,
            lineHeight: 1.45,
            color: "rgba(255,255,255,0.7)",
          }}
        >
          {truncate(study.read, 60)}
        </p>
      </div>
    </ClickableCard>
  );
}

// ── Drawer body ────────────────────────────────────────────────────────

const drawerSection: CSSProperties = { marginBottom: 18 };

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

interface ImagingDrawerBodyProps {
  study: SampleImaging;
}

function ImagingDrawerBody({ study }: ImagingDrawerBodyProps) {
  const modality = classifyModality(study.kind);
  const facility = facilityFor(study.id, study.kind);
  const histogram = useMemo(() => syntheticHistogram(study.id), [study.id]);
  const credit = IMAGE_CREDIT[modality];

  return (
    <div>
      <div style={drawerSection}>
        <div
          style={{
            background: "#070707",
            border: "1px solid rgba(255,255,255,0.06)",
            padding: "16px 12px",
            display: "flex",
            justifyContent: "center",
          }}
        >
          <ScanGraphic modality={modality} w={460} h={345} />
        </div>
        {credit ? (
          <Mono size={9} color="rgba(255,255,255,0.35)" style={{ marginTop: 6, display: "block" }}>
            {credit}
          </Mono>
        ) : null}
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Read</span>
        <p
          style={{
            margin: 0,
            fontSize: 13.5,
            lineHeight: 1.55,
            color: "rgba(255,255,255,0.92)",
          }}
        >
          {study.read}
        </p>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Study</span>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)",
            gap: 12,
          }}
        >
          <Field k="MODALITY" v={study.kind} />
          <Field k="REGION" v={study.region} />
          <Field k="DATE" v={formatDate(study.date)} />
          <Field k="RADIOLOGIST" v={study.radiologist} />
          <Field k="FACILITY" v={facility} />
          <Field k="STUDY ID" v={study.id.toUpperCase()} />
        </div>
      </div>

      <div style={drawerSection}>
        <span style={drawerLabel}>Intensity histogram (synthetic)</span>
        <div style={{ color: "var(--accent)" }}>
          <Sparkline
            data={histogram}
            w={420}
            h={48}
            stroke="var(--accent)"
            strokeWidth={1.25}
          />
        </div>
        <Mono size={9} color="rgba(255,255,255,0.35)" style={{ marginTop: 6, display: "block" }}>
          DEMO MODE · NOT DERIVED FROM PIXEL DATA
        </Mono>
      </div>

      <div style={drawerSection}>
        <Tooltip
          label="Reorder"
          range="Demo mode"
          hint="The reorder + booking flow lands with the scheduling wave. The demo build skips the round-trip to the imaging center."
        >
          <button
            type="button"
            disabled
            aria-disabled="true"
            style={{
              background: "transparent",
              color: "rgba(255,255,255,0.45)",
              border: "1px solid #2a2a2a",
              padding: "7px 14px",
              fontFamily: "var(--font-display)",
              fontWeight: 600,
              fontSize: 11.5,
              cursor: "not-allowed",
            }}
          >
            Reorder this study
          </button>
        </Tooltip>
      </div>
    </div>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Mono size={9}>{k}</Mono>
      <span style={{ fontSize: 13, fontWeight: 500, color: "#fff" }}>{v}</span>
    </div>
  );
}

// ── Panel ──────────────────────────────────────────────────────────────

export interface ImagingPanelProps {
  imaging: SampleImaging[];
}

export function ImagingPanel({ imaging }: ImagingPanelProps) {
  const [openId, setOpenId] = useState<string | null>(null);
  const open = openId
    ? (imaging.find((s) => s.id === openId) ?? null)
    : null;

  return (
    <div
      style={{
        background: "var(--p42-bg-card, #0e0e0e)",
        border: "1px solid rgba(255,255,255,0.07)",
        padding: 18,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <Eyebrow>IMAGING · {imaging.length} STUDIES</Eyebrow>
        <Mono color="rgba(255,255,255,0.4)">CLICK A STUDY TO EXPAND</Mono>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
          gap: 12,
        }}
      >
        {imaging.map((study) => (
          <ImagingCard
            key={study.id}
            study={study}
            onClick={() => setOpenId(study.id)}
          />
        ))}
      </div>

      {open ? (
        <DetailDrawer
          open={true}
          onClose={() => setOpenId(null)}
          title={`${open.kind} · ${open.region}`}
          kicker={`Imaging · ${formatDate(open.date)} · ${open.radiologist}`}
          width={520}
        >
          <ImagingDrawerBody study={open} />
        </DetailDrawer>
      ) : null}
    </div>
  );
}
