// 4UWHAt — Dot
// Status dot with optional magenta glow. Used in nav, problem list, timeline.
// Source: assets/app/atoms.jsx (function Dot).

import type { CSSProperties } from "react";

interface DotProps {
  color?: string;
  size?: number;
  glow?: boolean;
  style?: CSSProperties;
  className?: string;
}

export function Dot({
  color = "var(--accent)",
  size = 6,
  glow = false,
  style,
  className,
}: DotProps) {
  return (
    <span
      aria-hidden="true"
      className={className}
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: 999,
        background: color,
        boxShadow: glow ? `0 0 10px ${color}` : "none",
        ...style,
      }}
    />
  );
}
