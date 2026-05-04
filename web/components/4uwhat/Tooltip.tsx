"use client";

// 4UWHAt — Tooltip
// Hover-revealed reference info on any measurement. Black surface, hairline
// magenta top stripe, mono body. Anchored above (or below) the wrapped element.
// 80ms hover delay in, no fade-out delay.
// Source: assets/app/atoms.jsx (function Tooltip).

import { useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { createPortal } from "react-dom";

// SSR-safe mount detector. `useSyncExternalStore` returns the server snapshot
// on the server and the client snapshot after hydration — no setState-in-effect
// (which the project's eslint config bans).
const subscribe = () => () => {};
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

interface TooltipProps {
  children: ReactNode;
  label?: string;
  range?: string;
  source?: string;
  hint?: string;
  side?: "top" | "bottom";
  delay?: number;
}

export function Tooltip({
  children,
  label,
  range,
  source,
  hint,
  side = "top",
  delay = 80,
}: TooltipProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const ref = useRef<HTMLSpanElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useSyncExternalStore(
    subscribe,
    getClientSnapshot,
    getServerSnapshot,
  );

  const show = () => {
    timer.current = setTimeout(() => {
      const r = ref.current?.getBoundingClientRect();
      if (!r) return;
      setPos({
        x: r.left + r.width / 2,
        y: side === "top" ? r.top : r.bottom,
      });
      setOpen(true);
    }, delay);
  };

  const hide = () => {
    if (timer.current) clearTimeout(timer.current);
    setOpen(false);
  };

  return (
    <>
      <span
        ref={ref}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        tabIndex={0}
        style={{
          display: "inline-flex",
          alignItems: "baseline",
          borderBottom: "1px dotted rgba(255,255,255,0.18)",
          cursor: "help",
          gap: "inherit",
        }}
      >
        {children}
      </span>
      {mounted && open
        ? createPortal(
            <div
              role="tooltip"
              style={{
                position: "fixed",
                left: pos.x,
                top: pos.y,
                transform:
                  side === "top"
                    ? "translate(-50%, calc(-100% - 8px))"
                    : "translate(-50%, 8px)",
                zIndex: 9999,
                pointerEvents: "none",
                background: "#000",
                border: "1px solid var(--accent)",
                boxShadow:
                  "0 8px 24px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,0,150,0.2)",
                padding: "10px 12px",
                minWidth: 180,
                maxWidth: 260,
                fontFamily: "var(--font-mono)",
              }}
            >
              <div
                aria-hidden="true"
                style={{
                  position: "absolute",
                  top: -1,
                  left: 0,
                  right: 0,
                  height: 2,
                  background: "var(--accent)",
                }}
              />
              {label ? (
                <div
                  style={{
                    fontSize: 9,
                    fontWeight: 700,
                    letterSpacing: "0.16em",
                    color: "var(--accent)",
                    textTransform: "uppercase",
                    marginBottom: 6,
                  }}
                >
                  {label}
                </div>
              ) : null}
              {range ? (
                <div
                  style={{
                    fontSize: 12,
                    color: "#fff",
                    fontWeight: 500,
                    marginBottom: hint ? 4 : 0,
                  }}
                >
                  {range}
                </div>
              ) : null}
              {hint ? (
                <div
                  style={{
                    fontSize: 10.5,
                    color: "rgba(255,255,255,0.65)",
                    lineHeight: 1.4,
                  }}
                >
                  {hint}
                </div>
              ) : null}
              {source ? (
                <div
                  style={{
                    marginTop: 6,
                    paddingTop: 6,
                    borderTop: "1px solid rgba(255,255,255,0.08)",
                    fontSize: 9,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "rgba(255,255,255,0.4)",
                  }}
                >
                  {source}
                </div>
              ) : null}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
