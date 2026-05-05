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
//
// Layout containment (B1):
//   The picker lives in the 220px-wide left rail, inside a column that
//   is also home to the Records OS nav tabs. The native <select> trigger
//   was previously rendered with a `minWidth: 240` that overflowed the
//   rail (the trailing ")" of "Maya Okafor (DOB: 1991-04-12)" bled into
//   the Overview tab). The trigger now fills its container (width:100%)
//   with `box-sizing: border-box`, clips long values with ellipsis, and
//   uses an `aria-label` that captures the full active label so screen
//   readers still hear the DOB. Native <option> rendering is OS-painted
//   and not styleable from CSS — we instead pass a pre-truncated label
//   so the OS popup doesn't paint past the trigger boundary either.

"use client";

import { useEffect, useMemo, useState } from "react";
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

/** Build the select option label. The DOB is shortened to YYYY in the
 *  visible label (full ISO DOB is preserved on the option's `title` attr
 *  for hover and on the trigger's `aria-label` for accessibility). This
 *  gives the OS-painted dropdown a label short enough to fit the 220px
 *  left rail without overflowing into the next column. */
function shortLabel(p: PatientSummary): string {
  if (!p.dob) return p.displayName;
  const year = p.dob.slice(0, 4);
  return `${p.displayName} (${year})`;
}

export function PatientPicker({
  className,
  label = "Select patient",
}: PatientPickerProps) {
  const [patientId, setPatientId] = usePatientId();
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [loadFailed, setLoadFailed] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  // Full-fidelity label for the active patient — used as aria-label so
  // assistive tech still reads "Maya Okafor (DOB: 1991-04-12)" even
  // though the visible trigger shows the short form.
  const activeLabel = useMemo(() => {
    const active = patients.find((p) => p.id === patientId);
    if (!active) return label;
    return active.dob
      ? `${active.displayName} (DOB: ${active.dob})`
      : active.displayName;
  }, [patients, patientId, label]);

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
        // Block-level + width:100% so the picker fills the rail's content
        // box exactly, never overflowing into the next grid column.
        display: "flex",
        flexDirection: "column",
        gap: 4,
        width: "100%",
        minWidth: 0,
        maxWidth: "100%",
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        color: "rgba(255,255,255,0.85)",
      }}
    >
      <select
        aria-label={activeLabel}
        title={activeLabel}
        value={patientId ?? ""}
        disabled={loading}
        onChange={(e) => {
          const v = e.target.value;
          setPatientId(v === "" ? null : v);
        }}
        style={{
          // Containment: fill the rail, never escape it.
          width: "100%",
          maxWidth: "100%",
          boxSizing: "border-box",
          // Clip long active values rather than pushing the trigger
          // past its container edge.
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          // Stack above adjacent column borders / hover affordances so
          // the trigger always paints on top.
          position: "relative",
          zIndex: 50,
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: "0.04em",
          padding: "6px 10px",
          background: "var(--p42-bg-card, #1f1f1f)",
          color: "rgba(255,255,255,0.92)",
          border: "1px solid rgba(255,255,255,0.18)",
          borderRadius: 2,
          // Subtle elevation so the trigger reads as a distinct surface
          // when the OS-painted option list opens against the dark rail.
          boxShadow: "0 1px 0 rgba(0,0,0,0.4)",
          cursor: loading ? "wait" : "pointer",
        }}
      >
        {patients.length === 0 && (
          <option value="" disabled>
            (loading patients…)
          </option>
        )}
        {patients.map((p) => {
          const full = p.dob
            ? `${p.displayName} (DOB: ${p.dob})`
            : p.displayName;
          return (
            <option
              key={p.id}
              value={p.id}
              // Native <option> rendering is OS-painted; we cannot apply
              // ellipsis with CSS. Pass a pre-shortened label so the
              // popup column doesn't paint past the trigger width, and
              // surface the full DOB on hover via `title`.
              title={full}
            >
              {shortLabel(p)}
            </option>
          );
        })}
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
