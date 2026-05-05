"use client";

// 4UWHAt — ShareDrawer
//
// Replaces the cosmetic Export+Share header buttons on the Records OS
// with a working FHIR R4 Bundle export surface. Three sections:
//
//   A. Download FHIR Bundle — Blob + <a download> for a real file save
//   B. Copy to clipboard    — navigator.clipboard.writeText
//   C. Send to receiving system (preview) — disabled select + consent
//      checkbox, foreshadowing the FHIR Bulk Data POST that will land
//      in v1.1. The architecture (Bundle assembled, consent captured)
//      is what's load-bearing for the demo's "patient owns their record
//      and can take it anywhere FHIR is spoken" story; the wire format
//      is the easy part.
//
// All synthetic data — Maya Okafor — see lib/4uwhat/sample-data.ts.
// Bundle assembly lives in lib/4uwhat/fhir-export.ts (pure, tested).

import { useMemo, useState, type CSSProperties } from "react";
import { DetailDrawer } from "./DetailDrawer";
import { Mono } from "./Mono";
import { Tooltip } from "./Tooltip";
import { buildFhirBundle, formatBundleSummary } from "@/lib/4uwhat/fhir-export";
import { SAMPLE_PATIENT } from "@/lib/4uwhat/sample-data";

interface ShareDrawerProps {
  open: boolean;
  onClose: () => void;
}

const DESTINATIONS = [
  { id: "charite", label: "Charité – Universitätsmedizin Berlin (Germany trauma center)" },
  { id: "mayo", label: "Mayo Clinic Rochester (US)" },
  { id: "sgh", label: "Singapore General Hospital" },
] as const;

const sectionStyle: CSSProperties = {
  border: "1px solid rgba(255,255,255,0.07)",
  padding: 14,
  marginBottom: 14,
};

const sectionLabel: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 9,
  fontWeight: 700,
  letterSpacing: "0.16em",
  color: "var(--accent)",
  textTransform: "uppercase",
  marginBottom: 8,
  display: "block",
};

const helpText: CSSProperties = {
  margin: "0 0 12px 0",
  fontSize: 12.5,
  lineHeight: 1.45,
  color: "rgba(255,255,255,0.65)",
};

const btnPrimary: CSSProperties = {
  background: "var(--accent)",
  color: "#fff",
  border: "1px solid var(--accent)",
  padding: "8px 14px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 12,
  cursor: "pointer",
};

const btnGhost: CSSProperties = {
  background: "transparent",
  color: "rgba(255,255,255,0.85)",
  border: "1px solid rgba(255,255,255,0.18)",
  padding: "8px 14px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 12,
  cursor: "pointer",
};

const btnDisabled: CSSProperties = {
  background: "rgba(255,255,255,0.06)",
  color: "rgba(255,255,255,0.4)",
  border: "1px solid rgba(255,255,255,0.10)",
  padding: "8px 14px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 12,
  cursor: "not-allowed",
};

const selectStyle: CSSProperties = {
  width: "100%",
  background: "#000",
  color: "#fff",
  border: "1px solid rgba(255,255,255,0.18)",
  padding: "8px 10px",
  fontFamily: "var(--font-display)",
  fontSize: 12.5,
  boxSizing: "border-box",
  marginBottom: 10,
};

function fileSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)+/g, "");
}

export function ShareDrawer({ open, onClose }: ShareDrawerProps) {
  // Build the bundle once when the drawer opens. Rebuilding on every
  // render would re-roll all of the urn:uuid: ids, which is wasteful
  // and would also mean Download/Copy disagree if the user clicks one
  // then the other. The bundle is regenerated each open() so the
  // export timestamp + ids match the moment the user clicked Share.
  const bundle = useMemo(() => (open ? buildFhirBundle() : null), [open]);
  const summary = useMemo(
    () => (bundle ? formatBundleSummary(bundle) : ""),
    [bundle],
  );

  const [copied, setCopied] = useState(false);
  const [destination, setDestination] = useState<string>(DESTINATIONS[0].id);
  const [consented, setConsented] = useState(false);

  const filename = `${fileSlug(SAMPLE_PATIENT.name)}-fhir.json`;

  function handleDownload(): void {
    if (!bundle) return;
    const json = JSON.stringify(bundle, null, 2);
    const blob = new Blob([json], { type: "application/fhir+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Defer revoke so Safari has a chance to start the download.
    setTimeout(() => URL.revokeObjectURL(url), 1_000);
  }

  async function handleCopy(): Promise<void> {
    if (!bundle) return;
    const json = JSON.stringify(bundle, null, 2);
    try {
      await navigator.clipboard.writeText(json);
      setCopied(true);
      setTimeout(() => setCopied(false), 1_500);
    } catch (e) {
      // Surface as a console error — clipboard API rejects in some
      // permission-restricted contexts. The download path is the
      // documented fallback.
      console.error("[ShareDrawer] clipboard write failed:", e);
    }
  }

  return (
    <DetailDrawer
      open={open}
      onClose={onClose}
      title="Share your record"
      kicker="FHIR R4 Bundle · synthetic patient"
      width={520}
    >
      {/* Section A — Download FHIR Bundle ─────────────────────── */}
      <div style={sectionStyle}>
        <span style={sectionLabel}>A · Download FHIR Bundle</span>
        <p style={helpText}>
          Your record assembled as a FHIR R4 collection Bundle. Any system
          that speaks FHIR (Epic, Cerner, Charité, Mayo, your phone&apos;s
          health app) can ingest this file directly — it&apos;s the open
          standard for clinical interchange.
        </p>
        <div style={{ marginBottom: 10 }}>
          <Mono size={10} color="rgba(255,255,255,0.55)">
            {summary || "…"}
          </Mono>
        </div>
        <button type="button" style={btnPrimary} onClick={handleDownload}>
          Download {filename}
        </button>
      </div>

      {/* Section B — Copy to clipboard ────────────────────────── */}
      <div style={sectionStyle}>
        <span style={sectionLabel}>B · Copy to clipboard</span>
        <p style={helpText}>
          Paste the Bundle JSON into a clinician&apos;s EHR test harness, an
          interoperability sandbox, or any FHIR validator.
        </p>
        <button
          type="button"
          style={copied ? btnGhost : btnPrimary}
          onClick={handleCopy}
          aria-live="polite"
        >
          {copied ? "Copied" : "Copy FHIR Bundle JSON"}
        </button>
      </div>

      {/* Section C — Send to receiving system (preview) ────────── */}
      <div style={sectionStyle}>
        <span style={sectionLabel}>C · Send to receiving system (preview)</span>
        <p style={helpText}>
          The Germany trauma-center scenario: pick a receiver, consent, ship.
          v1.1 turns this into a real FHIR Bulk Data POST with SMART backend
          auth. The architecture (Bundle assembled, consent captured) is
          ready today — see Download or Copy above to use it now.
        </p>
        <label
          style={{
            display: "block",
            marginBottom: 6,
            fontSize: 11,
            color: "rgba(255,255,255,0.55)",
            fontFamily: "var(--font-mono)",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          Receiver
        </label>
        <select
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
          style={selectStyle}
          aria-label="Receiving system"
        >
          {DESTINATIONS.map((d) => (
            <option key={d.id} value={d.id}>
              {d.label}
            </option>
          ))}
        </select>
        <label
          style={{
            display: "flex",
            gap: 8,
            alignItems: "flex-start",
            margin: "10px 0 14px",
            fontSize: 12.5,
            lineHeight: 1.45,
            color: "rgba(255,255,255,0.85)",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={consented}
            onChange={(e) => setConsented(e.target.checked)}
            style={{ marginTop: 3 }}
          />
          <span>
            I consent to share my FHIR Bundle with the selected provider for
            clinical care purposes.
          </span>
        </label>
        <Tooltip
          label="PREVIEW"
          range="v1.1: real FHIR Bulk Data POST"
          hint="In v1.1 this becomes a real FHIR Bulk Data POST to the selected receiver. For now, see Download or Copy above."
          source="medomni · synthetic patient"
        >
          <button
            type="button"
            disabled
            aria-disabled="true"
            style={btnDisabled}
          >
            Send (preview) →
          </button>
        </Tooltip>
        <Mono
          size={9}
          color="rgba(255,255,255,0.4)"
          style={{ marginTop: 12, display: "block" }}
        >
          {consented
            ? "CONSENT CAPTURED · v1.1 BULK-DATA UPLOAD GATED ON SMART OAUTH"
            : "AWAITING CONSENT"}
        </Mono>
      </div>
    </DetailDrawer>
  );
}
