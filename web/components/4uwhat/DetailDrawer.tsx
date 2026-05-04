"use client";

// 4UWHAt — DetailDrawer
// Right-anchored side drawer for click-to-expand details on labs/vitals/
// meds/conditions/care-team. Owns: ESC-to-close, click-on-scrim-to-close,
// scroll lock on body while open, restore-focus on close, autofocus the
// close button on open. No external deps; portals into document.body so
// the drawer escapes any positioned/clipped ancestor.
//
// Layout primitives (display, grid, flex on the dashboard panels) are
// owned by A1. This component renders into a portal at <body> root and
// does not perturb any layout in the page tree.

import {
  useEffect,
  useRef,
  useSyncExternalStore,
  type CSSProperties,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

const subscribe = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

interface DetailDrawerProps {
  open: boolean;
  onClose: () => void;
  /** Short, ALL-CAPS-ish title shown in the drawer eyebrow. */
  title: string;
  /** Optional kicker rendered under the title (e.g. "LIPID PANEL · APR 22"). */
  kicker?: string;
  /** Optional override width. Default 480px. */
  width?: number;
  children: ReactNode;
}

const scrimStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.55)",
  zIndex: 9990,
  // Backdrop blur is cheap on modern browsers; gracefully ignored elsewhere.
  backdropFilter: "blur(2px)",
};

const drawerBase: CSSProperties = {
  position: "fixed",
  top: 0,
  right: 0,
  bottom: 0,
  background: "#0b0b0b",
  borderLeft: "1px solid rgba(255,255,255,0.08)",
  boxShadow: "-24px 0 48px rgba(0,0,0,0.6)",
  color: "#fff",
  fontFamily: "var(--font-display)",
  zIndex: 9991,
  display: "flex",
  flexDirection: "column",
  // Hairline accent stripe along the leading edge to echo the design system.
  borderTop: "2px solid var(--accent)",
};

const headerStyle: CSSProperties = {
  padding: "18px 22px 14px",
  borderBottom: "1px solid rgba(255,255,255,0.08)",
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: 12,
  flexShrink: 0,
};

const closeBtn: CSSProperties = {
  background: "transparent",
  color: "rgba(255,255,255,0.7)",
  border: "1px solid rgba(255,255,255,0.18)",
  width: 28,
  height: 28,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  fontFamily: "var(--font-mono)",
  fontSize: 14,
  lineHeight: 1,
  flexShrink: 0,
};

export function DetailDrawer({
  open,
  onClose,
  title,
  kicker,
  width = 480,
  children,
}: DetailDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  const mounted = useSyncExternalStore(
    subscribe,
    getClientSnapshot,
    getServerSnapshot,
  );

  // ESC closes; capture-phase so it beats any inner key handler.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, onClose]);

  // Body scroll lock + focus management.
  useEffect(() => {
    if (!open) return;
    previousFocus.current =
      (document.activeElement as HTMLElement | null) ?? null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    // Defer focus to next tick so the close button is mounted.
    const t = window.setTimeout(() => closeRef.current?.focus(), 0);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.clearTimeout(t);
      previousFocus.current?.focus?.();
    };
  }, [open]);

  if (!mounted || !open) return null;

  return createPortal(
    <>
      <div
        style={scrimStyle}
        onClick={onClose}
        // Scrim is decorative; the close button + ESC are the a11y exits.
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        style={{ ...drawerBase, width }}
      >
        <div style={headerStyle}>
          <div style={{ minWidth: 0 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                fontWeight: 700,
                letterSpacing: "0.16em",
                color: "var(--accent)",
                textTransform: "uppercase",
                marginBottom: 6,
              }}
            >
              DETAIL · DRAWER
            </div>
            <div
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 18,
                fontWeight: 700,
                letterSpacing: "-0.015em",
                lineHeight: 1.2,
                wordBreak: "break-word",
              }}
            >
              {title}
            </div>
            {kicker ? (
              <div
                style={{
                  marginTop: 4,
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  letterSpacing: "0.08em",
                  color: "rgba(255,255,255,0.5)",
                  textTransform: "uppercase",
                }}
              >
                {kicker}
              </div>
            ) : null}
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close drawer"
            style={closeBtn}
          >
            ×
          </button>
        </div>
        <div
          style={{
            flex: 1,
            minHeight: 0,
            overflowY: "auto",
            padding: "18px 22px 24px",
          }}
        >
          {children}
        </div>
        <div
          style={{
            padding: "10px 22px",
            borderTop: "1px solid rgba(255,255,255,0.08)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexShrink: 0,
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.12em",
              color: "rgba(255,255,255,0.35)",
            }}
          >
            ESC TO CLOSE
          </span>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: "transparent",
              color: "rgba(255,255,255,0.7)",
              border: "1px solid #2a2a2a",
              padding: "6px 12px",
              fontFamily: "var(--font-display)",
              fontWeight: 600,
              fontSize: 11,
              cursor: "pointer",
            }}
          >
            Close
          </button>
        </div>
      </div>
    </>,
    document.body,
  );
}
