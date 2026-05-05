// 4UWHAt — Resizer
// 4px-wide vertical drag handle for resizing the rails of the Records OS
// three-column grid. Updates a CSS custom property on the closest grid
// container (passed via `targetSelector`) and persists the value to
// localStorage on mouseup. No external deps.
//
// Usage:
//   <Resizer
//     targetSelector="[data-records-os-grid]"
//     varName="--col-left"
//     edge="right"   // dragging right grows the left rail
//     min={180}
//     max={280}
//     storageKey="4uwhat:col-left"
//   />
//
// `edge="right"`  : the rail being sized lives to the LEFT of the handle.
//                   dragging right -> larger value.
// `edge="left"`   : the rail being sized lives to the RIGHT of the handle.
//                   dragging right -> smaller value.

"use client";

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

export interface ResizerProps {
  targetSelector: string;
  varName: string;
  edge: "left" | "right";
  min: number;
  max: number;
  storageKey?: string;
  /** Optional aria label override. */
  label?: string;
}

export function Resizer({
  targetSelector,
  varName,
  edge,
  min,
  max,
  storageKey,
  label = "Resize column",
}: ResizerProps) {
  const [dragging, setDragging] = useState(false);
  const [hover, setHover] = useState(false);
  // Snapshot of state at mousedown so mousemove math is stable.
  const startRef = useRef<{ x: number; size: number } | null>(null);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const target = document.querySelector(targetSelector) as HTMLElement | null;
      if (!target) return;
      const cur = parseFloat(
        getComputedStyle(target).getPropertyValue(varName) || "0",
      );
      startRef.current = { x: e.clientX, size: cur };
      setDragging(true);
    },
    [targetSelector, varName],
  );

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      const start = startRef.current;
      if (!start) return;
      const target = document.querySelector(targetSelector) as HTMLElement | null;
      if (!target) return;
      const dx = e.clientX - start.x;
      const delta = edge === "right" ? dx : -dx;
      const next = Math.max(min, Math.min(max, start.size + delta));
      target.style.setProperty(varName, `${next}px`);
    };
    const onUp = () => {
      setDragging(false);
      startRef.current = null;
      if (!storageKey) return;
      const target = document.querySelector(targetSelector) as HTMLElement | null;
      if (!target) return;
      const finalVal = target.style.getPropertyValue(varName);
      try {
        if (finalVal) localStorage.setItem(storageKey, finalVal);
      } catch {
        // localStorage may be disabled (private mode); silently skip.
      }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    // Block text selection while dragging — and switch the global cursor
    // so it doesn't flicker as the pointer crosses sibling elements.
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
    };
  }, [dragging, targetSelector, varName, edge, min, max, storageKey]);

  const baseStyle: CSSProperties = {
    width: 4,
    cursor: "col-resize",
    background: dragging || hover ? "rgba(255,0,150,0.35)" : "#1f1f1f",
    transition: "background 120ms ease",
    flexShrink: 0,
    userSelect: "none",
    // Sit above adjacent borders so the handle hit-target is reliable.
    position: "relative",
    zIndex: 30,
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      onMouseDown={onMouseDown}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={baseStyle}
    />
  );
}

/** Read a previously-persisted CSS variable value from localStorage and
 *  apply it to the inline style of the target element. Call from a
 *  client-side useEffect on mount. Returns the loaded value (or null). */
export function loadPersistedVar(
  storageKey: string,
  fallback: string,
): string {
  try {
    return localStorage.getItem(storageKey) ?? fallback;
  } catch {
    return fallback;
  }
}
