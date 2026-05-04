// 4UWHAt — PatientPicker
// Dropdown of patients fetched from the BFF route at
// `${BASE_PATH}/api/patients`. Calls setPatientId() on the SessionProvider
// when the user picks an entry. Auto-selects the design-fallback entry
// on first mount when the API returns the synthetic Maya Okafor record
// (or when the fetch fails) so the page is never patientless.
//
// Uses BASE_PATH to avoid the basePath-prefix gotcha documented in
// web/lib/basePath.ts — `fetch("/api/patients")` would resolve against
// the document origin and bypass the /4UWHAt rewrite when proxied.

"use client";

import { useEffect, useState } from "react";
import { BASE_PATH } from "@/lib/basePath";
import { usePatientId } from "@/hooks/usePatientId";

interface PatientSummary {
  id: string;
  displayName: string;
  dob: string;
}

const DESIGN_FALLBACK_ID = "design-sample-patient";
const DESIGN_FALLBACK: PatientSummary[] = [
  { id: DESIGN_FALLBACK_ID, displayName: "Maya Okafor", dob: "1991-04-12" },
];

export interface PatientPickerProps {
  className?: string;
  /** Optional aria-label override; default "Select patient". */
  label?: string;
}

export function PatientPicker({
  className,
  label = "Select patient",
}: PatientPickerProps) {
  const [patientId, setPatientId] = usePatientId();
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [loadFailed, setLoadFailed] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(`${BASE_PATH}/api/patients`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return (await res.json()) as PatientSummary[];
      })
      .then((rows) => {
        if (cancelled) return;
        const safe = Array.isArray(rows) && rows.length > 0 ? rows : DESIGN_FALLBACK;
        setPatients(safe);
        setLoadFailed(false);
        // Auto-select on first mount when:
        //   (a) no patientId is currently selected, OR
        //   (b) the API returned exactly the design-fallback entry (so the
        //       page is never patientless in design mode).
        const onlyFallback =
          safe.length === 1 && safe[0].id === DESIGN_FALLBACK_ID;
        if (!patientId || onlyFallback) {
          setPatientId(safe[0].id);
        }
      })
      .catch(() => {
        if (cancelled) return;
        setPatients(DESIGN_FALLBACK);
        setLoadFailed(true);
        setPatientId(DESIGN_FALLBACK_ID);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // Mount-only fetch. The picker is not expected to live-refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      className={className}
      style={{
        display: "inline-flex",
        flexDirection: "column",
        gap: 4,
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        color: "rgba(255,255,255,0.85)",
      }}
    >
      <select
        aria-label={label}
        value={patientId ?? ""}
        disabled={loading}
        onChange={(e) => {
          const v = e.target.value;
          setPatientId(v === "" ? null : v);
        }}
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: "0.04em",
          padding: "6px 10px",
          background: "var(--p42-bg-card, #1f1f1f)",
          color: "rgba(255,255,255,0.92)",
          border: "1px solid rgba(255,255,255,0.18)",
          borderRadius: 2,
          minWidth: 240,
          cursor: loading ? "wait" : "pointer",
        }}
      >
        {patients.length === 0 && (
          <option value="" disabled>
            (loading patients…)
          </option>
        )}
        {patients.map((p) => (
          <option key={p.id} value={p.id}>
            {p.displayName}
            {p.dob ? ` (DOB: ${p.dob})` : ""}
          </option>
        ))}
      </select>
      {loadFailed && (
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 9.5,
            fontWeight: 600,
            letterSpacing: "0.08em",
            color: "rgba(255,255,255,0.55)",
            textTransform: "uppercase",
          }}
        >
          (could not load patients — using sample data)
        </span>
      )}
    </div>
  );
}
